"""Curriculum helpers for satellite-aware DCVC-FM training.

The legacy A/B/C script is intentionally kept for compatibility.  This module
defines a more surgical curriculum for the satellite extension:

1. warm up the official Slot Attention adapter;
2. train differentiable semantic token selection without channel noise;
3. calibrate continuous bandwidth response on paired conditions;
4. train robust satellite reconstruction;
5. conservatively fine-tune selected DCVC-FM modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F

from src.models.satellite import FrameForwardOutput, SatelliteDCVCFM, SlotAdapterOutput
from src.models.satellite.losses import (
    channel_robustness_loss,
    reconstruction_distortion,
    temporal_consistency_loss,
    token_selection_regularization,
)
from src.models.satellite.capacity_controller import rate_budget_interval_loss


PHASES = (
    "baseline",
    "slot_warmup",
    "selection_warmup",
    "capacity_calibration",
    "robust_curriculum",
    "joint_finetune",
)


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    stage_alias: str
    enable_satellite: bool
    paired_bandwidth: bool
    channel_type_hint: str
    description: str


@dataclass
class ConditionBatch:
    snr_db: torch.Tensor
    bandwidth_mbps: torch.Tensor
    packet_loss_rate: torch.Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "snr_db": float(self.snr_db.detach().mean().cpu().item()),
            "bandwidth_mbps": float(self.bandwidth_mbps.detach().mean().cpu().item()),
            "packet_loss_rate": float(self.packet_loss_rate.detach().mean().cpu().item()),
        }


PHASE_SPECS: dict[str, PhaseSpec] = {
    "baseline": PhaseSpec(
        name="baseline",
        stage_alias="A",
        enable_satellite=False,
        paired_bandwidth=False,
        channel_type_hint="identity",
        description="Pure DCVC-FM wrapper check. No satellite adapter path is enabled.",
    ),
    "slot_warmup": PhaseSpec(
        name="slot_warmup",
        stage_alias="B",
        enable_satellite=False,
        paired_bandwidth=False,
        channel_type_hint="identity",
        description="Train only the official Slot Attention adapter on frame reconstruction and mask regularity.",
    ),
    "selection_warmup": PhaseSpec(
        name="selection_warmup",
        stage_alias="B",
        enable_satellite=True,
        paired_bandwidth=False,
        channel_type_hint="identity",
        description="Freeze DCVC-FM and train Slot/Token/Capacity adapters with an identity channel.",
    ),
    "capacity_calibration": PhaseSpec(
        name="capacity_calibration",
        stage_alias="B",
        enable_satellite=True,
        paired_bandwidth=True,
        channel_type_hint="identity",
        description="Use paired bandwidth batches to enforce monotonic bpp and keep-ratio response.",
    ),
    "robust_curriculum": PhaseSpec(
        name="robust_curriculum",
        stage_alias="B",
        enable_satellite=True,
        paired_bandwidth=False,
        channel_type_hint="satellite",
        description="Train frozen-backbone adapters under progressive SNR/PLR satellite corruption.",
    ),
    "joint_finetune": PhaseSpec(
        name="joint_finetune",
        stage_alias="C",
        enable_satellite=True,
        paired_bandwidth=False,
        channel_type_hint="satellite",
        description="Small-LR fine-tune of selected DCVC-FM priors/modulation/decoder layers plus adapters.",
    ),
}


def get_phase_spec(name: str) -> PhaseSpec:
    if name not in PHASE_SPECS:
        raise ValueError(f"unknown curriculum phase {name!r}; expected one of {', '.join(PHASES)}")
    return PHASE_SPECS[name]


def parse_float_list(raw: str | Iterable[float]) -> list[float]:
    if isinstance(raw, str):
        vals = [float(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        vals = [float(v) for v in raw]
    if not vals:
        raise ValueError("expected at least one numeric value")
    return vals


def configure_trainable_parameters(model: SatelliteDCVCFM, phase: str) -> list[nn.Parameter]:
    """Set requires_grad for the selected curriculum phase."""

    spec = get_phase_spec(phase)
    for param in model.parameters():
        param.requires_grad = False

    if spec.name == "baseline":
        return []

    if spec.name == "slot_warmup":
        for param in model.slot_adapter.parameters():
            param.requires_grad = True
    elif spec.name in {"selection_warmup", "capacity_calibration", "robust_curriculum"}:
        model.freeze_dcvcfm_backbone()
    elif spec.name == "joint_finetune":
        model.unfreeze_stage_c()
    else:  # pragma: no cover - guarded by get_phase_spec
        raise ValueError(spec.name)

    return [param for param in model.parameters() if param.requires_grad]


def _uniform(batch_size: int, device: torch.device, low: float, high: float) -> torch.Tensor:
    if high < low:
        low, high = high, low
    if high == low:
        return torch.full((batch_size,), float(low), device=device)
    return torch.empty(batch_size, device=device).uniform_(float(low), float(high))


def sample_conditions_for_phase(
    args: Any,
    *,
    phase: str,
    batch_size: int,
    device: torch.device,
    global_step: int = 0,
    max_steps: int = 1,
) -> ConditionBatch:
    """Sample channel conditions according to the current phase."""

    progress = 0.0 if max_steps <= 0 else min(max(global_step / float(max_steps), 0.0), 1.0)
    if phase in {"baseline", "slot_warmup"}:
        return ConditionBatch(
            snr_db=torch.full((batch_size,), 30.0, device=device),
            bandwidth_mbps=torch.full((batch_size,), 25.0, device=device),
            packet_loss_rate=torch.zeros(batch_size, device=device),
        )

    if phase == "selection_warmup":
        bandwidths = torch.tensor(parse_float_list(args.bandwidth_grid), device=device)
        idx = torch.randint(0, bandwidths.numel(), (batch_size,), device=device)
        return ConditionBatch(
            snr_db=torch.full((batch_size,), 30.0, device=device),
            bandwidth_mbps=bandwidths[idx],
            packet_loss_rate=torch.zeros(batch_size, device=device),
        )

    if phase == "capacity_calibration":
        bandwidths = torch.tensor(parse_float_list(args.bandwidth_grid), device=device)
        if bandwidths.numel() != batch_size:
            repeats = (batch_size + bandwidths.numel() - 1) // bandwidths.numel()
            bandwidths = bandwidths.repeat(repeats)[:batch_size]
        return ConditionBatch(
            snr_db=torch.full((batch_size,), float(args.capacity_calibration_snr_db), device=device),
            bandwidth_mbps=bandwidths,
            packet_loss_rate=torch.full((batch_size,), float(args.capacity_calibration_plr), device=device),
        )

    if phase == "robust_curriculum":
        snr_min = (1.0 - progress) * max(float(args.robust_snr_mid), float(args.snr_min)) + progress * float(args.snr_min)
        plr_max = (1.0 - progress) * float(args.robust_plr_warmup) + progress * float(args.pkt_loss_max)
        return ConditionBatch(
            snr_db=_uniform(batch_size, device, snr_min, float(args.snr_max)),
            bandwidth_mbps=_uniform(batch_size, device, float(args.bandwidth_min_mbps), float(args.bandwidth_max_mbps)),
            packet_loss_rate=_uniform(batch_size, device, 0.0, plr_max),
        )

    if phase == "joint_finetune":
        return ConditionBatch(
            snr_db=_uniform(batch_size, device, float(args.snr_min), float(args.snr_max)),
            bandwidth_mbps=_uniform(batch_size, device, float(args.bandwidth_min_mbps), float(args.bandwidth_max_mbps)),
            packet_loss_rate=_uniform(batch_size, device, 0.0, float(args.pkt_loss_max)),
        )

    raise ValueError(f"unknown phase: {phase}")


def repeat_clip_for_paired_bandwidth(clip: torch.Tensor, bandwidth_grid: str) -> torch.Tensor:
    bandwidth_count = len(parse_float_list(bandwidth_grid))
    return clip.repeat_interleave(bandwidth_count, dim=0)


def slot_adapter_auxiliary_loss(
    slot_out: SlotAdapterOutput,
    target: torch.Tensor,
    *,
    recon_weight: float = 1.0,
    entropy_weight: float = 0.05,
    balance_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Slot warm-up objective with reconstruction, confidence, and balance."""

    if slot_out.recon_image is None:
        raise ValueError("slot adapter output does not include recon_image")
    recon = slot_out.recon_image.clamp(0.0, 1.0)
    recon_loss = F.mse_loss(recon, target)

    masks = slot_out.masks.clamp_min(1e-8)
    entropy = -(masks * masks.log()).sum(dim=1).mean() / float(masks.shape[1])
    slot_mass = masks.mean(dim=(2, 3))
    target_mass = torch.full_like(slot_mass, 1.0 / float(masks.shape[1]))
    balance = F.smooth_l1_loss(slot_mass, target_mass)
    total = recon_weight * recon_loss + entropy_weight * entropy + balance_weight * balance
    logs = {
        "slot_recon": float(recon_loss.detach().cpu().item()),
        "slot_entropy": float(entropy.detach().cpu().item()),
        "slot_balance": float(balance.detach().cpu().item()),
    }
    return total, logs


def slot_temporal_stability_loss(slot_outputs: list[SlotAdapterOutput]) -> torch.Tensor:
    if len(slot_outputs) < 2:
        ref = slot_outputs[0].masks if slot_outputs else None
        return torch.tensor(0.0, device=ref.device if ref is not None else "cpu")
    losses = []
    for prev, curr in zip(slot_outputs[:-1], slot_outputs[1:]):
        losses.append(F.l1_loss(curr.object_importance, prev.object_importance.detach()))
    return torch.stack(losses).mean() if losses else slot_outputs[0].masks.new_tensor(0.0)


def differentiable_p_frame_losses(
    frame_outputs: list[FrameForwardOutput],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean rate-budget, token-regularization, and monotonic losses."""

    rate_losses = []
    token_losses = []
    for frame in frame_outputs:
        if frame.frame_type != "P" or frame.control is None:
            continue
        rate_loss, _ = rate_budget_interval_loss(frame.bpp, frame.control)
        rate_losses.append(rate_loss)
        token_losses.append(token_selection_regularization(frame.selection, frame.control))
    if not rate_losses:
        ref = frame_outputs[0].x_hat if frame_outputs else torch.tensor(0.0)
        zero = ref.new_tensor(0.0)
        return zero, zero, zero
    rate = torch.stack(rate_losses).mean()
    token = torch.stack(token_losses).mean()
    mono = bandwidth_monotonic_loss(frame_outputs)
    return rate, token, mono


def bandwidth_monotonic_loss(
    frame_outputs: list[FrameForwardOutput],
    *,
    margin: float = 0.0,
) -> torch.Tensor:
    """Penalize paired samples where higher capacity uses lower bpp/keep ratio."""

    penalties = []
    for frame in frame_outputs:
        if frame.frame_type != "P" or frame.control is None or frame.selection is None:
            continue
        capacity = frame.control.capacity_mbps.reshape(-1)
        bpp = frame.bpp.reshape(-1).to(capacity.dtype)
        keep = frame.selection.expected_keep_ratio.reshape(-1).to(capacity.dtype)
        if capacity.numel() < 2:
            continue
        diff_cap = capacity.unsqueeze(1) - capacity.unsqueeze(0)
        higher = diff_cap > 1e-6
        if not higher.any():
            continue
        diff_bpp = bpp.unsqueeze(1) - bpp.unsqueeze(0)
        diff_keep = keep.unsqueeze(1) - keep.unsqueeze(0)
        penalties.append(torch.relu(-diff_bpp[higher] + margin).mean())
        penalties.append(torch.relu(-diff_keep[higher] + margin).mean())
    if penalties:
        return torch.stack(penalties).mean()
    ref = frame_outputs[0].x_hat if frame_outputs else torch.tensor(0.0)
    return ref.new_tensor(0.0)


def temporal_sequence_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if recon.shape[1] < 2:
        return recon.new_tensor(0.0)
    losses = []
    for idx in range(1, recon.shape[1]):
        losses.append(temporal_consistency_loss(recon[:, idx - 1], recon[:, idx], target[:, idx - 1], target[:, idx]))
    return torch.stack(losses).mean() if losses else recon.new_tensor(0.0)


def compute_satellite_sequence_loss(
    model: SatelliteDCVCFM,
    clip: torch.Tensor,
    *,
    conditions: ConditionBatch,
    args: Any,
    enable_satellite: bool,
    include_channel_teacher: bool = False,
) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
    """Compute the main differentiable video objective for non-slot phases."""

    out = model.forward_sequence(
        clip,
        snr_db=conditions.snr_db,
        bandwidth_mbps=conditions.bandwidth_mbps,
        packet_loss_rate=conditions.packet_loss_rate,
        q_index_i=args.q_index_i,
        q_index_p=None if args.capacity_q_index else args.q_index_p,
        intra_period=args.intra_period,
        enable_satellite=enable_satellite,
    )
    recon = out["x_hat"].clamp(0.0, 1.0)
    recon_loss = reconstruction_distortion(recon, clip, mse_weight=args.lambda_mse, l1_weight=args.lambda_l1)
    rate_loss, token_loss, mono_loss = differentiable_p_frame_losses(out["frames"])
    temporal = temporal_sequence_loss(recon, clip)

    channel_loss = recon.new_tensor(0.0)
    if include_channel_teacher and args.lambda_channel > 0:
        with torch.no_grad():
            clean = model.forward_sequence(
                clip,
                snr_db=conditions.snr_db,
                bandwidth_mbps=conditions.bandwidth_mbps,
                packet_loss_rate=conditions.packet_loss_rate,
                q_index_i=args.q_index_i,
                q_index_p=None if args.capacity_q_index else args.q_index_p,
                intra_period=args.intra_period,
                enable_satellite=False,
            )["x_hat"].clamp(0.0, 1.0)
        channel_loss = channel_robustness_loss(clean, recon)

    total = (
        args.lambda_recon * recon_loss
        + args.lambda_rate_budget * rate_loss
        + args.lambda_token * token_loss
        + args.lambda_temporal * temporal
        + args.lambda_monotonic * mono_loss
        + args.lambda_channel * channel_loss
    )
    metrics = out.get("metrics", {})
    logs = {
        "loss": float(total.detach().cpu().item()),
        "recon": float(recon_loss.detach().cpu().item()),
        "weighted_recon": float((args.lambda_recon * recon_loss).detach().cpu().item()),
        "rate_budget": float(rate_loss.detach().cpu().item()),
        "weighted_rate": float((args.lambda_rate_budget * rate_loss).detach().cpu().item()),
        "token": float(token_loss.detach().cpu().item()),
        "temporal": float(temporal.detach().cpu().item()),
        "monotonic": float(mono_loss.detach().cpu().item()),
        "channel": float(channel_loss.detach().cpu().item()),
        "bpp": float(metrics.get("bpp", 0.0)),
        "target_bpp": float(metrics.get("target_bpp", 0.0)),
        "capacity": float(metrics.get("capacity_mbps", 0.0)),
        "keep": float(metrics.get("keep_ratio", 0.0)),
        "base": float(metrics.get("base_layer_ratio", 0.0)),
        "enh": float(metrics.get("enhancement_layer_ratio", 0.0)),
        "tx_ms": float(metrics.get("tx_time_ms", 0.0)),
        "proc_ms": float(metrics.get("proc_time_ms", 0.0)),
    }
    return total, logs, out


def bandwidth_scan_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute monotonic diagnostics from eval-suite condition summaries."""

    ordered = sorted(items, key=lambda item: float(item["bandwidth_mbps"]))
    bpps = [float(item.get("bpp", 0.0)) for item in ordered]
    keeps = [float(item.get("keep_ratio", 0.0)) for item in ordered]
    psnrs = [float(item.get("PSNR", 0.0)) for item in ordered]
    bws = [float(item["bandwidth_mbps"]) for item in ordered]

    def violations(values: list[float]) -> int:
        return sum(1 for a, b in zip(values[:-1], values[1:]) if b + 1e-6 < a)

    ratio = None
    if bpps and bpps[0] > 1e-8:
        ratio = bpps[-1] / bpps[0]
    return {
        "bandwidth_mbps": bws,
        "bpp": bpps,
        "keep_ratio": keeps,
        "PSNR": psnrs,
        "bpp_monotonic_violations": violations(bpps),
        "keep_monotonic_violations": violations(keeps),
        "psnr_monotonic_violations": violations(psnrs),
        "bpp_bw_max_over_min": ratio,
        "passes_bw_response_gate": bool(
            violations(bpps) == 0
            and violations(keeps) == 0
            and ratio is not None
            and ratio >= 2.5
        ),
    }


__all__ = [
    "PHASES",
    "PhaseSpec",
    "ConditionBatch",
    "get_phase_spec",
    "parse_float_list",
    "configure_trainable_parameters",
    "sample_conditions_for_phase",
    "repeat_clip_for_paired_bandwidth",
    "slot_adapter_auxiliary_loss",
    "slot_temporal_stability_loss",
    "bandwidth_monotonic_loss",
    "temporal_sequence_loss",
    "compute_satellite_sequence_loss",
    "bandwidth_scan_diagnostics",
]

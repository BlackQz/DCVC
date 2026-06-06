"""Continuous capacity control for satellite-aware DCVC-FM.

The controller intentionally does not use a four-tier keep-ratio table as the
main policy.  It maps measured channel state to continuous rate, quantization,
base-layer, and enhancement-layer budgets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CapacityControl:
    snr_db: torch.Tensor
    bandwidth_mbps: torch.Tensor
    packet_loss_rate: torch.Tensor
    capacity_mbps: torch.Tensor
    capacity_norm: torch.Tensor
    target_bpp: torch.Tensor
    target_bpp_low: torch.Tensor
    target_bpp_high: torch.Tensor
    q_index: torch.Tensor
    lambda_rd: torch.Tensor
    lambda_rate_over: torch.Tensor
    lambda_rate_under: torch.Tensor
    total_keep_ratio: torch.Tensor
    base_keep_ratio: torch.Tensor
    enhancement_keep_ratio: torch.Tensor
    base_layer_budget_bits: torch.Tensor
    enhancement_layer_budget_bits: torch.Tensor


def _as_batch_tensor(
    value: float | int | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.to(device=device, dtype=dtype).reshape(-1)
    else:
        out = torch.full((1,), float(value), device=device, dtype=dtype)
    if out.numel() == 1:
        return out.expand(batch_size)
    if out.numel() != batch_size:
        raise ValueError(f"{name} must contain 1 or {batch_size} values, got {out.numel()}.")
    return out


class ContinuousCapacityController(nn.Module):
    """Map SNR/BW/PLR to continuous codec and selection controls.

    capacity = bandwidth_mbps * (1 - packet_loss_rate) * log2(1 + snr_linear)

    The output is monotonic in capacity by construction.  At low capacity the
    enhancement budget is zero; at high capacity both target_bpp and keep_ratio
    increase so the rate loss cannot collapse every condition to a low-rate
    solution.
    """

    def __init__(
        self,
        *,
        min_capacity_mbps: float = 0.5,
        max_capacity_mbps: float = 110.0,
        min_target_bpp: float = 0.035,
        max_target_bpp: float = 0.260,
        target_tolerance_low: float = 0.18,
        target_tolerance_high: float = 0.25,
        min_keep_ratio: float = 0.16,
        max_keep_ratio: float = 0.92,
        base_min_keep_ratio: float = 0.16,
        base_max_keep_ratio: float = 0.46,
        enhancement_start_norm: float = 0.28,
        keep_gamma: float = 0.85,
        q_index_low_capacity: float = 0.0,
        q_index_high_capacity: float = 63.0,
        lambda_rd_low_capacity: float = 2048.0,
        lambda_rd_high_capacity: float = 256.0,
        lambda_rate_over: float = 1.0,
        lambda_rate_under: float = 0.35,
        learnable_offsets: bool = True,
    ) -> None:
        super().__init__()
        if min_capacity_mbps <= 0 or max_capacity_mbps <= min_capacity_mbps:
            raise ValueError("capacity range must satisfy 0 < min < max.")
        if min_target_bpp <= 0 or max_target_bpp < min_target_bpp:
            raise ValueError("target bpp range must satisfy 0 < min <= max.")
        if not 0 <= min_keep_ratio <= max_keep_ratio <= 1:
            raise ValueError("keep ratios must satisfy 0 <= min <= max <= 1.")

        self.min_capacity_mbps = float(min_capacity_mbps)
        self.max_capacity_mbps = float(max_capacity_mbps)
        self.min_target_bpp = float(min_target_bpp)
        self.max_target_bpp = float(max_target_bpp)
        self.target_tolerance_low = float(target_tolerance_low)
        self.target_tolerance_high = float(target_tolerance_high)
        self.min_keep_ratio = float(min_keep_ratio)
        self.max_keep_ratio = float(max_keep_ratio)
        self.base_min_keep_ratio = float(base_min_keep_ratio)
        self.base_max_keep_ratio = float(base_max_keep_ratio)
        self.enhancement_start_norm = float(enhancement_start_norm)
        self.keep_gamma = float(keep_gamma)
        self.q_index_low_capacity = float(q_index_low_capacity)
        self.q_index_high_capacity = float(q_index_high_capacity)
        self.lambda_rd_low_capacity = float(lambda_rd_low_capacity)
        self.lambda_rd_high_capacity = float(lambda_rd_high_capacity)
        self.lambda_rate_over_value = float(lambda_rate_over)
        self.lambda_rate_under_value = float(lambda_rate_under)
        self.learnable_offsets = bool(learnable_offsets)
        if self.learnable_offsets:
            self.target_bpp_log_gain = nn.Parameter(torch.zeros(()))
            self.keep_logit_bias = nn.Parameter(torch.zeros(()))
            self.base_logit_bias = nn.Parameter(torch.zeros(()))
            self.q_index_bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("target_bpp_log_gain", torch.zeros(()), persistent=True)
            self.register_buffer("keep_logit_bias", torch.zeros(()), persistent=True)
            self.register_buffer("base_logit_bias", torch.zeros(()), persistent=True)
            self.register_buffer("q_index_bias", torch.zeros(()), persistent=True)

    @staticmethod
    def shannon_capacity_mbps(
        snr_db: torch.Tensor,
        bandwidth_mbps: torch.Tensor,
        packet_loss_rate: torch.Tensor,
    ) -> torch.Tensor:
        snr_linear = torch.pow(10.0, snr_db / 10.0)
        plr_factor = (1.0 - packet_loss_rate.clamp(0.0, 0.999)).clamp_min(0.0)
        return bandwidth_mbps.clamp_min(1e-6) * plr_factor * torch.log2(1.0 + snr_linear)

    def _normalize_capacity(self, capacity_mbps: torch.Tensor) -> torch.Tensor:
        lo = torch.log1p(capacity_mbps.new_tensor(self.min_capacity_mbps))
        hi = torch.log1p(capacity_mbps.new_tensor(self.max_capacity_mbps))
        return ((torch.log1p(capacity_mbps.clamp_min(0.0)) - lo) / (hi - lo)).clamp(0.0, 1.0)

    @staticmethod
    def _monotone_shift_ratio(ratio: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        ratio = ratio.clamp(1e-4, 1.0 - 1e-4)
        return torch.sigmoid(torch.logit(ratio) + bias)

    def forward(
        self,
        *,
        snr_db: float | torch.Tensor,
        bandwidth_mbps: float | torch.Tensor,
        packet_loss_rate: float | torch.Tensor,
        num_pixels: int | torch.Tensor,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> CapacityControl:
        if batch_size is None:
            candidates = [
                v.numel() for v in (snr_db, bandwidth_mbps, packet_loss_rate)
                if torch.is_tensor(v)
            ]
            batch_size = max(candidates) if candidates else 1
        if device is None:
            tensor_inputs = [v for v in (snr_db, bandwidth_mbps, packet_loss_rate) if torch.is_tensor(v)]
            device = tensor_inputs[0].device if tensor_inputs else torch.device("cpu")

        snr = _as_batch_tensor(snr_db, batch_size=batch_size, device=device, dtype=dtype, name="snr_db")
        bw = _as_batch_tensor(
            bandwidth_mbps, batch_size=batch_size, device=device, dtype=dtype, name="bandwidth_mbps"
        )
        plr = _as_batch_tensor(
            packet_loss_rate, batch_size=batch_size, device=device, dtype=dtype, name="packet_loss_rate"
        ).clamp(0.0, 0.999)

        capacity = self.shannon_capacity_mbps(snr, bw, plr)
        cap_norm = self._normalize_capacity(capacity)
        keep_progress = torch.pow(cap_norm, self.keep_gamma)

        target_bpp = self.min_target_bpp + cap_norm * (self.max_target_bpp - self.min_target_bpp)
        target_bpp = target_bpp * torch.exp(self.target_bpp_log_gain.clamp(-0.5, 0.5))
        target_bpp_low = target_bpp * (1.0 - self.target_tolerance_low * (0.5 + 0.5 * cap_norm))
        target_bpp_high = target_bpp * (1.0 + self.target_tolerance_high)

        total_keep = self.min_keep_ratio + keep_progress * (self.max_keep_ratio - self.min_keep_ratio)
        base_keep = self.base_min_keep_ratio + cap_norm * (self.base_max_keep_ratio - self.base_min_keep_ratio)
        total_keep = self._monotone_shift_ratio(total_keep, self.keep_logit_bias.clamp(-2.0, 2.0))
        base_keep = self._monotone_shift_ratio(base_keep, self.base_logit_bias.clamp(-2.0, 2.0))
        total_keep = total_keep.clamp(self.min_keep_ratio, self.max_keep_ratio)
        base_keep = base_keep.clamp(self.base_min_keep_ratio, self.base_max_keep_ratio)
        base_keep = torch.minimum(base_keep, total_keep)

        enh_gate = ((cap_norm - self.enhancement_start_norm) / (1.0 - self.enhancement_start_norm)).clamp(0.0, 1.0)
        enhancement_keep = (total_keep - base_keep).clamp_min(0.0) * enh_gate
        total_keep = (base_keep + enhancement_keep).clamp(self.min_keep_ratio, self.max_keep_ratio)

        q_index = self.q_index_low_capacity + cap_norm * (
            self.q_index_high_capacity - self.q_index_low_capacity
        )
        q_index = (q_index + self.q_index_bias.clamp(-4.0, 4.0)).clamp(
            min(self.q_index_low_capacity, self.q_index_high_capacity),
            max(self.q_index_low_capacity, self.q_index_high_capacity),
        )
        lambda_rd = self.lambda_rd_low_capacity + cap_norm * (
            self.lambda_rd_high_capacity - self.lambda_rd_low_capacity
        )
        lambda_rate_over = self.lambda_rate_over_value * (1.10 - 0.45 * cap_norm)
        lambda_rate_under = self.lambda_rate_under_value * cap_norm

        if torch.is_tensor(num_pixels):
            pixels = num_pixels.to(device=device, dtype=dtype).reshape(-1)
            if pixels.numel() == 1:
                pixels = pixels.expand(batch_size)
        else:
            pixels = torch.full((batch_size,), float(num_pixels), device=device, dtype=dtype)

        target_bits = target_bpp * pixels
        base_budget = target_bits * (base_keep / total_keep.clamp_min(1e-6))
        enhancement_budget = (target_bits - base_budget).clamp_min(0.0)

        return CapacityControl(
            snr_db=snr,
            bandwidth_mbps=bw,
            packet_loss_rate=plr,
            capacity_mbps=capacity,
            capacity_norm=cap_norm,
            target_bpp=target_bpp,
            target_bpp_low=target_bpp_low,
            target_bpp_high=target_bpp_high,
            q_index=q_index,
            lambda_rd=lambda_rd,
            lambda_rate_over=lambda_rate_over,
            lambda_rate_under=lambda_rate_under,
            total_keep_ratio=total_keep,
            base_keep_ratio=base_keep,
            enhancement_keep_ratio=enhancement_keep,
            base_layer_budget_bits=base_budget,
            enhancement_layer_budget_bits=enhancement_budget,
        )


def rate_budget_interval_loss(
    actual_bpp: torch.Tensor,
    control: CapacityControl,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize bpp outside a capacity-dependent interval.

    Low-capacity samples mainly penalize overshoot.  High-capacity samples also
    receive an under-use penalty so optimization does not learn to save bitrate
    for every bandwidth condition.
    """

    actual = actual_bpp.reshape(-1).to(control.target_bpp.dtype)
    over = torch.relu(actual - control.target_bpp_high)
    under = torch.relu(control.target_bpp_low - actual)
    over_loss = control.lambda_rate_over * over.square()
    under_loss = control.lambda_rate_under * under.square()
    loss = (over_loss + under_loss).mean()
    parts = {
        "rate_over": over.mean().detach(),
        "rate_under": under.mean().detach(),
        "target_bpp": control.target_bpp.mean().detach(),
        "target_bpp_low": control.target_bpp_low.mean().detach(),
        "target_bpp_high": control.target_bpp_high.mean().detach(),
    }
    return loss, parts

"""Satellite-aware wrapper around the original DCVC-FM codec.

This module keeps DCVC-FM as the main video codec.  It reuses DMCI/DMC modules
for motion estimation, contextual coding, entropy priors, feature modulation,
quantization scaling, and reconstruction.  The satellite path is an additional
training/evaluation forward path over the decoded latents, while the original
bitstream compress/decompress functions remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ..image_model import DMCI
from ..video_model import DMC
from ...utils.stream_helper import get_state_dict
from .capacity_controller import CapacityControl, ContinuousCapacityController
from .channel import LayeredSatelliteChannel
from .slot_adapter import SatelliteSlotAdapter, SlotAdapterOutput
from .token_selector import SatelliteTokenSelector, SelectionOutput


@dataclass
class SatelliteForwardState:
    dpb: dict[str, torch.Tensor | None]
    slot: SlotAdapterOutput | None = None


@dataclass
class FrameForwardOutput:
    x_hat: torch.Tensor
    bit: torch.Tensor
    bpp: torch.Tensor
    state: SatelliteForwardState
    frame_type: str
    control: CapacityControl | None
    selection: SelectionOutput | None
    mv_selection: SelectionOutput | None
    metrics: dict[str, float] = field(default_factory=dict)


def _tensor_mean_float(x: torch.Tensor | None) -> float:
    if x is None:
        return 0.0
    return float(x.detach().to(torch.float32).mean().cpu().item())


def _as_condition_tensor(
    value: float | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.to(device=device, dtype=dtype).reshape(-1)
    else:
        out = torch.full((1,), float(value), device=device, dtype=dtype)
    if out.numel() == 1:
        return out.expand(batch_size)
    if out.numel() != batch_size:
        raise ValueError(f"condition must contain 1 or {batch_size} values, got {out.numel()}.")
    return out


def _pad_video_to_multiple(
    frames: torch.Tensor,
    multiple: int = 16,
) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = frames.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return frames, (0, 0)
    b, t, c, _, _ = frames.shape
    flat = frames.reshape(b * t, c, h, w)
    flat = F.pad(flat, (0, pad_w, 0, pad_h), mode="replicate")
    return flat.reshape(b, t, c, h + pad_h, w + pad_w), (pad_h, pad_w)


class SlotLatentModulator(nn.Module):
    """Lightweight FiLM modulation from object slots to DCVC residual latents."""

    def __init__(self, slot_dim: int, latent_channels: int) -> None:
        super().__init__()
        hidden = max(slot_dim, latent_channels)
        self.net = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, latent_channels * 2),
        )
        self.logit_strength = nn.Parameter(torch.tensor(-4.0))

    def forward(self, latent: torch.Tensor, slots: torch.Tensor | None) -> torch.Tensor:
        if slots is None:
            return latent
        pooled = slots.mean(dim=1).to(device=latent.device, dtype=latent.dtype)
        gamma, beta = self.net(pooled).view(latent.shape[0], 2, latent.shape[1], 1, 1).unbind(dim=1)
        strength = torch.sigmoid(self.logit_strength).to(dtype=latent.dtype)
        return latent + strength * (torch.tanh(gamma) * latent + beta)


class SatelliteDCVCFM(nn.Module):
    """DCVC-FM backbone with satellite semantic transmission adapters."""

    def __init__(
        self,
        *,
        i_frame_net: DMCI | None = None,
        p_frame_net: DMC | None = None,
        channel_type: str = "satellite",
        slot_num: int = 7,
        slot_dim: int = 64,
        slot_output_dim: int = 128,
        slot_iterations: int = 3,
        slot_adapter_resolution: tuple[int, int] = (128, 128),
        update_slots_on_p: bool = True,
        eval_noisy: bool = True,
        controller_kwargs: dict[str, Any] | None = None,
        selector_kwargs: dict[str, Any] | None = None,
        channel_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.i_frame_net = i_frame_net if i_frame_net is not None else DMCI()
        self.p_frame_net = p_frame_net if p_frame_net is not None else DMC()
        self.capacity_controller = ContinuousCapacityController(**(controller_kwargs or {}))
        self.slot_adapter = SatelliteSlotAdapter(
            num_slots=slot_num,
            slot_dim=slot_dim,
            output_slot_dim=slot_output_dim,
            num_iterations=slot_iterations,
            adapter_resolution=slot_adapter_resolution,
        )
        self.token_selector = SatelliteTokenSelector(**(selector_kwargs or {}))
        self.slot_y_modulator = SlotLatentModulator(slot_output_dim, 128)
        channel_args = dict(channel_kwargs or {})
        channel_args.setdefault("channel_type", channel_type)
        channel_args.setdefault("eval_noisy", eval_noisy)
        self.channel = LayeredSatelliteChannel(**channel_args)
        self.update_slots_on_p = bool(update_slots_on_p)

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_path_i: str | None,
        model_path_p: str | None,
        strict: bool = True,
        **kwargs: Any,
    ) -> "SatelliteDCVCFM":
        model = cls(**kwargs)
        if model_path_i:
            model.i_frame_net.load_state_dict(get_state_dict(model_path_i), strict=strict)
        if model_path_p:
            model.p_frame_net.load_state_dict(get_state_dict(model_path_p), strict=strict)
        return model

    @staticmethod
    def make_empty_dpb() -> dict[str, None]:
        return {
            "ref_frame": None,
            "ref_feature": None,
            "ref_mv_feature": None,
            "ref_y": None,
            "ref_mv_y": None,
        }

    def freeze_dcvcfm_backbone(self) -> None:
        for param in self.i_frame_net.parameters():
            param.requires_grad = False
        for param in self.p_frame_net.parameters():
            param.requires_grad = False
        for module in (self.slot_adapter, self.token_selector, self.capacity_controller,
                       self.channel, self.slot_y_modulator):
            for param in module.parameters():
                param.requires_grad = True

    def unfreeze_stage_c(self) -> None:
        """Unfreeze conservative DCVC-FM parts for small-LR joint tuning."""

        self.freeze_dcvcfm_backbone()
        names = (
            "feature_adaptor_I",
            "feature_adaptor",
            "contextual_decoder",
            "recon_generation_net",
            "temporal_prior_encoder",
            "y_prior_fusion_adaptor_0",
            "y_prior_fusion_adaptor_1",
            "y_prior_fusion",
            "mv_y_prior_fusion_adaptor_0",
            "mv_y_prior_fusion_adaptor_1",
            "mv_y_prior_fusion",
            "y_q_enc",
            "y_q_dec",
            "mv_y_q_enc",
            "mv_y_q_dec",
        )
        for name, module_or_param in self.p_frame_net.named_children():
            if name in names:
                for param in module_or_param.parameters():
                    param.requires_grad = True
        for name in ("y_q_enc", "y_q_dec", "mv_y_q_enc", "mv_y_q_dec"):
            param = getattr(self.p_frame_net, name, None)
            if isinstance(param, nn.Parameter):
                param.requires_grad = True
        for name in ("q_scale_enc", "q_scale_dec", "dec", "refine"):
            module_or_param = getattr(self.i_frame_net, name, None)
            if isinstance(module_or_param, nn.Parameter):
                module_or_param.requires_grad = True
            elif isinstance(module_or_param, nn.Module):
                for param in module_or_param.parameters():
                    param.requires_grad = True

    def adapter_parameters(self):
        for module in (self.slot_adapter, self.token_selector, self.capacity_controller,
                       self.channel, self.slot_y_modulator):
            yield from module.parameters()

    def _build_control(
        self,
        x: torch.Tensor,
        *,
        snr_db: float | torch.Tensor,
        bandwidth_mbps: float | torch.Tensor,
        packet_loss_rate: float | torch.Tensor,
    ) -> CapacityControl:
        return self.capacity_controller(
            snr_db=snr_db,
            bandwidth_mbps=bandwidth_mbps,
            packet_loss_rate=packet_loss_rate,
            num_pixels=x.shape[-2] * x.shape[-1],
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype if x.is_floating_point() else torch.float32,
        )

    @staticmethod
    def _control_q_index(control: CapacityControl, fallback: int | None) -> int | list[int]:
        if fallback is not None:
            return int(fallback)
        q = control.q_index.detach().to(torch.float32).round().clamp(0, DMC.get_qp_num() - 1)
        q_list = [int(v) for v in q.cpu().tolist()]
        return q_list[0] if len(q_list) == 1 else q_list

    def _slot_from_frame(
        self,
        frame: torch.Tensor,
        *,
        latent_size: tuple[int, int] | None = None,
    ) -> SlotAdapterOutput:
        return self.slot_adapter(frame.detach() if not self.training else frame, latent_size=latent_size)

    def forward_i_frame(
        self,
        x: torch.Tensor,
        *,
        q_index: int = 63,
        control: CapacityControl | None = None,
    ) -> FrameForwardOutput:
        encoded = self.i_frame_net.forward_one_frame(x, q_index=q_index)
        x_hat = encoded["x_hat"].clamp(0, 1)
        bit = encoded["bit"].reshape(1).to(device=x.device, dtype=x.dtype)
        bpp = bit / float(x.shape[-2] * x.shape[-1] * x.shape[0])
        dpb = {
            "ref_frame": x_hat,
            "ref_feature": None,
            "ref_mv_feature": None,
            "ref_y": None,
            "ref_mv_y": None,
        }
        slot = self._slot_from_frame(x_hat)
        metrics = {
            "bpp": _tensor_mean_float(bpp),
            "pixel_num": float(x.shape[-2] * x.shape[-1]),
            "keep_ratio": 1.0,
            "base_layer_ratio": 1.0,
            "enhancement_layer_ratio": 0.0,
            "target_bpp": _tensor_mean_float(control.target_bpp) if control is not None else 0.0,
            "capacity_mbps": _tensor_mean_float(control.capacity_mbps) if control is not None else 0.0,
            "tx_time_ms": 0.0,
        }
        return FrameForwardOutput(
            x_hat=x_hat,
            bit=bit,
            bpp=bpp.reshape(1),
            state=SatelliteForwardState(dpb=dpb, slot=slot),
            frame_type="I",
            control=control,
            selection=None,
            mv_selection=None,
            metrics=metrics,
        )

    def _select_and_transmit(
        self,
        latent: torch.Tensor,
        *,
        control: CapacityControl,
        residual: torch.Tensor | None,
        reference_latent: torch.Tensor | None,
        object_importance: torch.Tensor | None,
        temporal_uncertainty: torch.Tensor | None,
        enable_satellite: bool,
    ) -> tuple[torch.Tensor, SelectionOutput]:
        selection = self.token_selector(
            latent,
            control=control,
            residual=residual,
            reference_latent=reference_latent,
            object_importance=object_importance,
            temporal_uncertainty=temporal_uncertainty,
        )
        channel_out = self.channel.forward_layered_latent(
            latent,
            base_mask=selection.base_mask,
            enhancement_mask=selection.enhancement_mask,
            snr_db=control.snr_db,
            packet_loss_rate=control.packet_loss_rate,
            fallback=reference_latent,
        )
        return channel_out.latent, selection

    def forward_p_frame(
        self,
        x: torch.Tensor,
        state: SatelliteForwardState,
        *,
        snr_db: float | torch.Tensor = 10.0,
        bandwidth_mbps: float | torch.Tensor = 10.0,
        packet_loss_rate: float | torch.Tensor = 0.0,
        q_index: int | None = None,
        fa_idx: int = 0,
        enable_satellite: bool = True,
    ) -> FrameForwardOutput:
        if state.dpb.get("ref_frame") is None:
            raise ValueError("P-frame forward requires an initialized DPB from an I-frame.")
        begin = time.perf_counter()
        p = self.p_frame_net
        control = self._build_control(
            x,
            snr_db=snr_db,
            bandwidth_mbps=bandwidth_mbps,
            packet_loss_rate=packet_loss_rate,
        )
        q_idx = self._control_q_index(control, q_index)
        mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec = p.get_all_q(q_idx)
        index = p.get_index_tensor(0, x.device)
        dpb = state.dpb

        est_mv = p.optic_flow(x, dpb["ref_frame"])
        mv_y = p.mv_encoder(est_mv, dpb["ref_mv_feature"], mv_y_q_enc)
        mv_y_pad, mv_slice_shape = p.pad_for_y(mv_y)
        mv_z = p.mv_hyper_prior_encoder(mv_y_pad)
        mv_z_hat = p.quant(mv_z)
        mv_params = p.mv_prior_param_decoder(mv_z_hat, dpb, mv_slice_shape)
        mv_y_res, mv_y_q, mv_y_hat, mv_scales_hat = p.forward_four_part_prior(
            mv_y,
            mv_params,
            p.mv_y_spatial_prior_adaptor_1,
            p.mv_y_spatial_prior_adaptor_2,
            p.mv_y_spatial_prior_adaptor_3,
            p.mv_y_spatial_prior,
        )

        mv_object = None
        if state.slot is not None:
            mv_object = state.slot.object_importance
        if enable_satellite:
            mv_y_rx, mv_selection = self._select_and_transmit(
                mv_y_hat,
                control=control,
                residual=mv_y_res,
                reference_latent=dpb["ref_mv_y"],
                object_importance=mv_object,
                temporal_uncertainty=mv_scales_hat,
                enable_satellite=True,
            )
        else:
            mv_y_rx, mv_selection = mv_y_hat, None
        mv_hat, mv_feature = p.mv_decoder(mv_y_rx, mv_y_q_dec)
        context1, context2, context3, _ = p.motion_compensation(dpb, mv_hat, fa_idx)

        y = p.contextual_encoder(x, context1, context2, context3, y_q_enc)
        y_pad, y_slice_shape = p.pad_for_y(y)
        z = p.contextual_hyper_prior_encoder(y_pad)
        z_hat = p.quant(z)
        params = p.contextual_prior_param_decoder(z_hat, dpb, context3, y_slice_shape)
        y_res, y_q, y_hat, scales_hat = p.forward_four_part_prior(
            y,
            params,
            p.y_spatial_prior_adaptor_1,
            p.y_spatial_prior_adaptor_2,
            p.y_spatial_prior_adaptor_3,
            p.y_spatial_prior,
        )

        slot_latent_object = None
        if state.slot is not None:
            slot_latent_object = F.interpolate(
                state.slot.object_importance,
                size=y_hat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if enable_satellite:
            y_rx, selection = self._select_and_transmit(
                y_hat,
                control=control,
                residual=y_res,
                reference_latent=dpb["ref_y"],
                object_importance=slot_latent_object,
                temporal_uncertainty=scales_hat,
                enable_satellite=True,
            )
            if state.slot is not None:
                y_rx = self.slot_y_modulator(y_rx, state.slot.slots)
        else:
            y_rx, selection = y_hat, None

        x_hat, feature = p.get_recon_and_feature(y_rx, context1, context2, context3, y_q_dec)
        x_hat = x_hat.clamp(0, 1)

        pixel_num = x.shape[-2] * x.shape[-1]
        # Entropy/bit estimation goes through Laplace/Gaussian CDFs that are
        # numerically unstable in fp16, so force fp32 even under AMP autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            bits_y = p.get_y_laplace_bits(y_q.float(), scales_hat.float())
            bits_mv_y = p.get_y_laplace_bits(mv_y_q.float(), mv_scales_hat.float())
            bits_z = p.get_z_bits(z_hat.float(), p.bit_estimator_z, index)
            bits_mv_z = p.get_z_bits(mv_z_hat.float(), p.bit_estimator_z_mv, index)
        if selection is not None:
            bits_y = bits_y * selection.keep_mask.float()
        if mv_selection is not None:
            bits_mv_y = bits_mv_y * mv_selection.keep_mask.float()
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_mv_y = torch.sum(bits_mv_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp_mv_z = torch.sum(bits_mv_z, dim=(1, 2, 3)) / pixel_num
        bpp = bpp_y + bpp_mv_y + bpp_z + bpp_mv_z
        bit = bpp.sum() * pixel_num

        next_dpb = {
            "ref_frame": x_hat,
            "ref_feature": feature,
            "ref_mv_feature": mv_feature,
            "ref_y": y_rx,
            "ref_mv_y": mv_y_rx,
        }
        next_slot = state.slot
        if self.update_slots_on_p:
            next_slot = self._slot_from_frame(x_hat)
        next_state = SatelliteForwardState(dpb=next_dpb, slot=next_slot)

        proc_ms = (time.perf_counter() - begin) * 1000.0
        capacity_bps = control.capacity_mbps.clamp_min(1e-6) * 1_000_000.0
        tx_ms = (bpp.detach() * pixel_num / capacity_bps * 1000.0).mean()
        metrics = {
            "bpp": _tensor_mean_float(bpp),
            "pixel_num": float(pixel_num),
            "bpp_y": _tensor_mean_float(bpp_y),
            "bpp_z": _tensor_mean_float(bpp_z),
            "bpp_mv_y": _tensor_mean_float(bpp_mv_y),
            "bpp_mv_z": _tensor_mean_float(bpp_mv_z),
            "keep_ratio": _tensor_mean_float(selection.keep_ratio) if selection is not None else 1.0,
            "base_layer_ratio": _tensor_mean_float(selection.base_ratio) if selection is not None else 1.0,
            "enhancement_layer_ratio": _tensor_mean_float(selection.enhancement_ratio) if selection is not None else 0.0,
            "expected_keep_ratio": _tensor_mean_float(selection.expected_keep_ratio) if selection is not None else 1.0,
            "mv_keep_ratio": _tensor_mean_float(mv_selection.keep_ratio) if mv_selection is not None else 1.0,
            "target_bpp": _tensor_mean_float(control.target_bpp),
            "target_bpp_low": _tensor_mean_float(control.target_bpp_low),
            "target_bpp_high": _tensor_mean_float(control.target_bpp_high),
            "capacity_mbps": _tensor_mean_float(control.capacity_mbps),
            "capacity_norm": _tensor_mean_float(control.capacity_norm),
            "q_index": float(sum(q_idx) / len(q_idx)) if isinstance(q_idx, list) else float(q_idx),
            "proc_time_ms": proc_ms,
            "tx_time_ms": _tensor_mean_float(tx_ms),
        }

        return FrameForwardOutput(
            x_hat=x_hat,
            bit=bit.reshape(1),
            bpp=bpp,
            state=next_state,
            frame_type="P",
            control=control,
            selection=selection,
            mv_selection=mv_selection,
            metrics=metrics,
        )

    def forward_sequence(
        self,
        frames: torch.Tensor,
        *,
        snr_db: float | torch.Tensor = 10.0,
        bandwidth_mbps: float | torch.Tensor = 10.0,
        packet_loss_rate: float | torch.Tensor = 0.0,
        q_index_i: int = 63,
        q_index_p: int | None = None,
        intra_period: int = 9999,
        rate_gop_size: int = 8,
        enable_satellite: bool = True,
    ) -> dict[str, Any]:
        if frames.ndim != 5:
            raise ValueError(f"frames must be (B,T,C,H,W), got {tuple(frames.shape)}")
        orig_h, orig_w = frames.shape[-2:]
        frames_proc, (pad_h, pad_w) = _pad_video_to_multiple(frames, 16)
        b, t, _, _, _ = frames_proc.shape
        device = frames.device
        dtype = frames.dtype
        snr = _as_condition_tensor(snr_db, batch_size=b, device=device, dtype=dtype)
        bw = _as_condition_tensor(bandwidth_mbps, batch_size=b, device=device, dtype=dtype)
        plr = _as_condition_tensor(packet_loss_rate, batch_size=b, device=device, dtype=dtype)

        outputs: list[torch.Tensor] = []
        frame_outputs: list[FrameForwardOutput] = []
        state: SatelliteForwardState | None = None
        index_map = [0, 1, 0, 2, 0, 2, 0, 2]
        for frame_idx in range(t):
            x = frames_proc[:, frame_idx]
            if frame_idx % intra_period == 0 or state is None:
                control = self._build_control(x, snr_db=snr, bandwidth_mbps=bw, packet_loss_rate=plr)
                out = self.forward_i_frame(x, q_index=q_index_i, control=control)
            else:
                fa_idx = index_map[frame_idx % rate_gop_size]
                out = self.forward_p_frame(
                    x,
                    state,
                    snr_db=snr,
                    bandwidth_mbps=bw,
                    packet_loss_rate=plr,
                    q_index=q_index_p,
                    fa_idx=fa_idx,
                    enable_satellite=enable_satellite,
                )
            state = out.state
            x_hat_out = out.x_hat
            if pad_h > 0 or pad_w > 0:
                x_hat_out = x_hat_out[..., :orig_h, :orig_w]
                orig_pixel_num = float(orig_h * orig_w)
                out.bpp = out.bit.reshape(-1) / orig_pixel_num
                out.metrics["bpp"] = _tensor_mean_float(out.bpp)
                out.metrics["pixel_num"] = orig_pixel_num
            outputs.append(x_hat_out)
            frame_outputs.append(out)

        recon = torch.stack(outputs, dim=1)
        bpps = torch.stack([
            out.bpp.mean().to(device=device, dtype=dtype).reshape(())
            for out in frame_outputs
        ])
        metrics = self.aggregate_frame_metrics(frame_outputs, fps=30.0)
        return {
            "x_hat": recon,
            "frames": frame_outputs,
            "bpp": bpps.mean(),
            "state": state,
            "metrics": metrics,
        }

    @staticmethod
    def aggregate_frame_metrics(frame_outputs: list[FrameForwardOutput], *, fps: float) -> dict[str, float]:
        if not frame_outputs:
            return {}
        keys: set[str] = set()
        for out in frame_outputs:
            keys.update(out.metrics.keys())
        metrics = {}
        for key in keys:
            vals = [out.metrics[key] for out in frame_outputs if key in out.metrics]
            if vals:
                metrics[key] = float(sum(vals) / len(vals))
        avg_bpp = metrics.get("bpp", 0.0)
        pixel_num = metrics.get("pixel_num", 0.0)
        metrics["kbps"] = avg_bpp * pixel_num * fps / 1000.0 if pixel_num > 0 else 0.0
        return metrics

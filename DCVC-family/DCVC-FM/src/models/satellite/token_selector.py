"""Object-aware base/enhancement latent selection for satellite DCVC-FM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .capacity_controller import CapacityControl


@dataclass
class SelectionOutput:
    keep_score: torch.Tensor
    base_mask: torch.Tensor
    enhancement_mask: torch.Tensor
    keep_mask: torch.Tensor
    drop_mask: torch.Tensor
    soft_base_mask: torch.Tensor
    soft_enhancement_mask: torch.Tensor
    soft_keep_mask: torch.Tensor
    residual_magnitude: torch.Tensor
    novelty: torch.Tensor
    object_importance: torch.Tensor
    temporal_uncertainty: torch.Tensor
    keep_ratio: torch.Tensor
    base_ratio: torch.Tensor
    enhancement_ratio: torch.Tensor
    expected_keep_ratio: torch.Tensor
    expected_base_ratio: torch.Tensor
    expected_enhancement_ratio: torch.Tensor


class SatelliteTokenSelector(nn.Module):
    """Select latent positions using residual, novelty, object, and temporal cues."""

    def __init__(
        self,
        *,
        w_mag: float = 0.45,
        w_novelty: float = 0.25,
        w_obj: float = 0.20,
        w_temporal: float = 0.10,
        min_keep_ratio: float = 0.01,
        ste_temperature: float = 0.08,
        learnable_weights: bool = True,
    ) -> None:
        super().__init__()
        init = torch.tensor([w_mag, w_novelty, w_obj, w_temporal], dtype=torch.float32)
        init = init / init.sum().clamp_min(1e-6)
        if learnable_weights:
            self.logit_weights = nn.Parameter(torch.log(init.clamp_min(1e-6)))
        else:
            self.register_buffer("logit_weights", torch.log(init.clamp_min(1e-6)), persistent=True)
        self.min_keep_ratio = float(min_keep_ratio)
        self.ste_temperature = float(ste_temperature)

    @staticmethod
    def _normalize_map(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2)
        lo = flat.min(dim=-1, keepdim=True).values.unsqueeze(-1)
        hi = flat.max(dim=-1, keepdim=True).values.unsqueeze(-1)
        return (x - lo) / (hi - lo + 1e-8)

    @staticmethod
    def _resize_like(x: torch.Tensor | None, size: tuple[int, int], ref: torch.Tensor) -> torch.Tensor:
        if x is None:
            return torch.zeros(ref.shape[0], 1, size[0], size[1], device=ref.device, dtype=ref.dtype)
        x = x.to(device=ref.device, dtype=ref.dtype)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        return x

    @staticmethod
    def _expand_ratio(ratio: torch.Tensor, batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        out = ratio.to(device=device, dtype=dtype).reshape(-1)
        if out.numel() == 1:
            return out.expand(batch)
        if out.numel() != batch:
            raise ValueError(f"ratio must contain 1 or {batch} values, got {out.numel()}.")
        return out

    def _topk_ste_mask(
        self,
        score: torch.Tensor,
        ratio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return straight-through, hard, and soft top-k masks.

        Forward uses exact hard top-k.  Backward uses a sigmoid relaxation around
        the detached kth-score threshold, so Slot Attention and selector weights
        receive distortion and rate gradients.
        """

        b, _, h, w = score.shape
        n = h * w
        score_flat = score.flatten(1)
        hard = torch.zeros_like(score_flat)
        soft = torch.zeros_like(score_flat)
        k_values = torch.ceil(ratio.clamp(0.0, 1.0) * n).long().clamp(0, n)
        for bi in range(b):
            k = int(k_values[bi].item())
            if k <= 0:
                continue
            values, idx = torch.topk(score_flat[bi], k, sorted=False)
            hard[bi].scatter_(0, idx, 1.0)
            threshold = values.min().detach()
            temp = max(self.ste_temperature, 1e-4)
            soft[bi] = torch.sigmoid((score_flat[bi] - threshold) / temp)
        soft = soft * ratio.view(b, 1).clamp(0.0, 1.0) / soft.mean(dim=1, keepdim=True).clamp_min(1e-6)
        soft = soft.clamp(0.0, 1.0)
        hard = hard.view(b, 1, h, w)
        soft = soft.view(b, 1, h, w)
        st = hard - soft.detach() + soft
        return st, hard, soft

    def forward(
        self,
        latent: torch.Tensor,
        *,
        control: CapacityControl,
        residual: torch.Tensor | None = None,
        reference_latent: torch.Tensor | None = None,
        object_importance: torch.Tensor | None = None,
        temporal_uncertainty: torch.Tensor | None = None,
    ) -> SelectionOutput:
        if latent.ndim != 4:
            raise ValueError(f"latent must be BCHW, got {tuple(latent.shape)}")
        b, _, h, w = latent.shape
        size = (h, w)

        residual_source = latent if residual is None else residual
        if residual_source.shape[-2:] != size:
            residual_source = F.interpolate(residual_source, size=size, mode="bilinear", align_corners=False)
        residual_map = self._normalize_map(residual_source.abs().mean(dim=1, keepdim=True))

        if reference_latent is not None and reference_latent.shape[1] == latent.shape[1]:
            ref = reference_latent.to(device=latent.device, dtype=latent.dtype)
            if ref.shape[-2:] != size:
                ref = F.interpolate(ref, size=size, mode="bilinear", align_corners=False)
            novelty_map = self._normalize_map((latent - ref).abs().mean(dim=1, keepdim=True))
        else:
            novelty_map = torch.zeros_like(residual_map)

        object_map = self._normalize_map(self._resize_like(object_importance, size, latent))
        temporal_map = self._normalize_map(self._resize_like(temporal_uncertainty, size, latent))

        weights = torch.softmax(self.logit_weights.to(device=latent.device, dtype=latent.dtype), dim=0)
        keep_score = (
            weights[0] * residual_map
            + weights[1] * novelty_map
            + weights[2] * object_map
            + weights[3] * temporal_map
        ).clamp(0.0, 1.0)

        total_ratio = self._expand_ratio(
            control.total_keep_ratio, b, device=latent.device, dtype=latent.dtype
        ).clamp(self.min_keep_ratio, 1.0)
        base_ratio = self._expand_ratio(
            control.base_keep_ratio, b, device=latent.device, dtype=latent.dtype
        ).clamp(0.0, 1.0)
        enhancement_ratio = self._expand_ratio(
            control.enhancement_keep_ratio, b, device=latent.device, dtype=latent.dtype
        ).clamp(0.0, 1.0)

        total_ratio = torch.minimum(total_ratio, torch.ones_like(total_ratio))
        base_ratio = torch.minimum(base_ratio, total_ratio)
        total_from_layers = torch.minimum(base_ratio + enhancement_ratio, torch.ones_like(total_ratio))
        total_ratio = torch.maximum(total_ratio, total_from_layers)

        base_mask, base_hard, base_soft = self._topk_ste_mask(keep_score, base_ratio)
        total_mask, total_hard, total_soft = self._topk_ste_mask(keep_score, total_ratio)
        enhancement_hard = (total_hard - base_hard).clamp_min(0.0)
        enhancement_soft = (total_soft - base_soft).clamp_min(0.0)
        enhancement_mask = enhancement_hard - enhancement_soft.detach() + enhancement_soft
        keep_mask = (base_mask + enhancement_mask).clamp(0.0, 1.0)
        drop_mask = 1.0 - keep_mask

        keep_ratio = total_hard.flatten(1).mean(dim=1)
        actual_base_ratio = base_hard.flatten(1).mean(dim=1)
        actual_enhancement_ratio = enhancement_hard.flatten(1).mean(dim=1)
        expected_keep_ratio = total_soft.flatten(1).mean(dim=1)
        expected_base_ratio = base_soft.flatten(1).mean(dim=1)
        expected_enhancement_ratio = enhancement_soft.flatten(1).mean(dim=1)

        return SelectionOutput(
            keep_score=keep_score,
            base_mask=base_mask,
            enhancement_mask=enhancement_mask,
            keep_mask=keep_mask,
            drop_mask=drop_mask,
            soft_base_mask=base_soft,
            soft_enhancement_mask=enhancement_soft,
            soft_keep_mask=total_soft,
            residual_magnitude=residual_map,
            novelty=novelty_map,
            object_importance=object_map,
            temporal_uncertainty=temporal_map,
            keep_ratio=keep_ratio,
            base_ratio=actual_base_ratio,
            enhancement_ratio=actual_enhancement_ratio,
            expected_keep_ratio=expected_keep_ratio,
            expected_base_ratio=expected_base_ratio,
            expected_enhancement_ratio=expected_enhancement_ratio,
        )


def apply_selection_fallback(
    latent: torch.Tensor,
    keep_mask: torch.Tensor,
    fallback: torch.Tensor | None,
) -> torch.Tensor:
    """Fill dropped latent positions from the reference latent or zeros."""

    if fallback is None or fallback.shape != latent.shape:
        fallback = torch.zeros_like(latent)
    return latent * keep_mask + fallback.to(device=latent.device, dtype=latent.dtype) * (1.0 - keep_mask)

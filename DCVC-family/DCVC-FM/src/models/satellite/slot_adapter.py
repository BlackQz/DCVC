"""Slot Attention adapter for DCVC-FM.

The Slot Attention core is not handwritten here.  It is the PyTorch compatibility
layer in `official_slot_attention.py`, ported from the official TensorFlow code
stored at `DCVC-family/slot-attention/model.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .official_slot_attention import SlotAttentionAutoEncoder


@dataclass
class SlotAdapterOutput:
    slots: torch.Tensor
    masks: torch.Tensor
    object_importance: torch.Tensor
    masks_latent: torch.Tensor | None = None
    object_importance_latent: torch.Tensor | None = None
    recon_image: torch.Tensor | None = None
    slot_recons: torch.Tensor | None = None
    # Target for the Slot Attention auto-encoder reconstruction loss.  It is the
    # (detached) image the slots were extracted from, resampled to the same size
    # as ``recon_image`` so object-discovery supervision is independent of any
    # video-codec padding.
    recon_target: torch.Tensor | None = None


class SatelliteSlotAdapter(nn.Module):
    """Adapt the official Slot Attention object-discovery model to DCVC-FM."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        slot_dim: int = 64,
        output_slot_dim: int = 128,
        num_slots: int = 7,
        num_iterations: int = 3,
        adapter_resolution: tuple[int, int] = (128, 128),
        **_: object,
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim)
        self.output_slot_dim = int(output_slot_dim)
        self.adapter_resolution = tuple(adapter_resolution)

        self.official_autoencoder = SlotAttentionAutoEncoder(
            resolution=self.adapter_resolution,
            num_slots=num_slots,
            num_iterations=num_iterations,
            num_channels=in_channels,
            hidden_size=slot_dim,
        )
        self.slot_out = nn.Linear(slot_dim, output_slot_dim)
        self.importance_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, max(16, slot_dim // 2)),
            nn.SiLU(inplace=True),
            nn.Linear(max(16, slot_dim // 2), 1),
        )

    @staticmethod
    def _normalize_map(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2)
        lo = flat.min(dim=-1, keepdim=True).values.unsqueeze(-1)
        hi = flat.max(dim=-1, keepdim=True).values.unsqueeze(-1)
        return (x - lo) / (hi - lo + 1e-8)

    def forward(
        self,
        image_or_feature: torch.Tensor,
        *,
        latent_size: tuple[int, int] | None = None,
    ) -> SlotAdapterOutput:
        if image_or_feature.ndim != 4:
            raise ValueError(f"expected BCHW tensor, got {tuple(image_or_feature.shape)}")

        original_size = image_or_feature.shape[-2:]
        slot_input = image_or_feature
        if original_size != self.adapter_resolution:
            slot_input = F.interpolate(
                slot_input,
                size=self.adapter_resolution,
                mode="bilinear",
                align_corners=False,
            )

        recon_low, slot_recons_low, masks_low_5d, slots_core = self.official_autoencoder(slot_input)
        masks_low = masks_low_5d.squeeze(2)
        slots = self.slot_out(slots_core)

        weights = torch.sigmoid(self.importance_head(slots_core)).view(
            image_or_feature.shape[0], self.num_slots, 1, 1
        )
        weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-6)
        object_low = self._normalize_map((masks_low * weights).sum(dim=1, keepdim=True))

        masks_full = F.interpolate(masks_low, size=original_size, mode="bilinear", align_corners=False)
        masks_full = masks_full / masks_full.sum(dim=1, keepdim=True).clamp_min(1e-6)
        object_full = F.interpolate(object_low, size=original_size, mode="bilinear", align_corners=False)
        object_full = self._normalize_map(object_full)
        recon_full = F.interpolate(recon_low, size=original_size, mode="bilinear", align_corners=False)

        masks_latent = None
        object_latent = None
        if latent_size is not None:
            masks_latent = F.interpolate(masks_low, size=latent_size, mode="bilinear", align_corners=False)
            masks_latent = masks_latent / masks_latent.sum(dim=1, keepdim=True).clamp_min(1e-6)
            object_latent = F.interpolate(object_low, size=latent_size, mode="bilinear", align_corners=False)
            object_latent = self._normalize_map(object_latent)

        return SlotAdapterOutput(
            slots=slots,
            masks=masks_full,
            object_importance=object_full,
            masks_latent=masks_latent,
            object_importance_latent=object_latent,
            recon_image=recon_full,
            slot_recons=slot_recons_low,
            recon_target=image_or_feature.detach(),
        )

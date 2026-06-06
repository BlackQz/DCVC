"""Training losses for satellite-aware DCVC-FM."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .capacity_controller import CapacityControl, rate_budget_interval_loss
from .slot_adapter import SlotAdapterOutput
from .token_selector import SelectionOutput


def reconstruction_distortion(
    recon: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float = 1.0,
    l1_weight: float = 0.0,
) -> torch.Tensor:
    loss = recon.new_tensor(0.0)
    if mse_weight > 0:
        loss = loss + float(mse_weight) * F.mse_loss(recon, target)
    if l1_weight > 0:
        loss = loss + float(l1_weight) * F.l1_loss(recon, target)
    return loss


def temporal_consistency_loss(
    recon_prev: torch.Tensor | None,
    recon_curr: torch.Tensor,
    target_prev: torch.Tensor | None,
    target_curr: torch.Tensor,
) -> torch.Tensor:
    if recon_prev is None or target_prev is None:
        return recon_curr.new_tensor(0.0)
    recon_delta = recon_curr - recon_prev
    target_delta = target_curr - target_prev
    return F.l1_loss(recon_delta, target_delta)


def token_selection_regularization(
    selection: SelectionOutput | None,
    control: CapacityControl,
) -> torch.Tensor:
    if selection is None:
        return control.target_bpp.new_tensor(0.0)
    keep = selection.expected_keep_ratio.to(control.total_keep_ratio.dtype)
    target = control.total_keep_ratio.to(keep.device)
    return F.smooth_l1_loss(keep, target)


def channel_robustness_loss(
    clean_recon: torch.Tensor | None,
    noisy_recon: torch.Tensor,
) -> torch.Tensor:
    if clean_recon is None:
        return noisy_recon.new_tensor(0.0)
    return F.l1_loss(noisy_recon, clean_recon.detach())


def slot_reconstruction_loss(slot_output: "SlotAdapterOutput | None") -> torch.Tensor | None:
    """Object-discovery supervision for the Slot Attention auto-encoder.

    Reconstructs the (detached) frame the slots were extracted from, exactly as
    in the official Slot Attention object-discovery objective.  This keeps the
    slot decoder useful and pushes slots toward genuine object decomposition,
    instead of letting them drift under the compression objective only.
    """

    if slot_output is None or slot_output.recon_image is None or slot_output.recon_target is None:
        return None
    return F.mse_loss(slot_output.recon_image, slot_output.recon_target)


__all__ = [
    "reconstruction_distortion",
    "temporal_consistency_loss",
    "token_selection_regularization",
    "channel_robustness_loss",
    "slot_reconstruction_loss",
    "rate_budget_interval_loss",
]

"""Satellite-aware adapters for DCVC-FM."""

from .capacity_controller import CapacityControl, ContinuousCapacityController, rate_budget_interval_loss
from .channel import LayeredSatelliteChannel
from .dcvc_fm_satellite import FrameForwardOutput, SatelliteDCVCFM, SatelliteForwardState
from .losses import (
    channel_robustness_loss,
    reconstruction_distortion,
    slot_reconstruction_loss,
    temporal_consistency_loss,
    token_selection_regularization,
)
from .slot_adapter import SatelliteSlotAdapter, SlotAdapterOutput
from .token_selector import SatelliteTokenSelector, SelectionOutput

__all__ = [
    "CapacityControl",
    "ContinuousCapacityController",
    "rate_budget_interval_loss",
    "LayeredSatelliteChannel",
    "FrameForwardOutput",
    "SatelliteDCVCFM",
    "SatelliteForwardState",
    "SatelliteSlotAdapter",
    "SlotAdapterOutput",
    "SatelliteTokenSelector",
    "SelectionOutput",
    "reconstruction_distortion",
    "temporal_consistency_loss",
    "token_selection_regularization",
    "channel_robustness_loss",
    "slot_reconstruction_loss",
]

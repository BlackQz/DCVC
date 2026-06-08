"""Continuous capacity control for satellite-aware DCVC-FM.

The controller intentionally does not use a four-tier keep-ratio table as the
main policy.  It maps measured channel state to continuous rate, quantization,
base-layer, and enhancement-layer budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


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
    q_index_base: torch.Tensor
    q_index_residual: torch.Tensor
    q_index_rounded: torch.Tensor
    q_index_proxy_bpp: torch.Tensor | None
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
        q_index_mode: str = "linear",
        q_rd_table_path: str = "",
        q_delta_max: float = 0.0,
        q_mlp_hidden: int = 32,
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
        if q_index_mode not in {"linear", "rd_table"}:
            raise ValueError("q_index_mode must be 'linear' or 'rd_table'.")

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
        self.q_index_mode = str(q_index_mode)
        self.q_rd_table_path = str(q_rd_table_path)
        self.q_delta_max = float(q_delta_max)
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
        if self.learnable_offsets and q_mlp_hidden > 0 and self.q_delta_max > 0:
            self.q_residual_mlp = nn.Sequential(
                nn.Linear(5, q_mlp_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(q_mlp_hidden, q_mlp_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(q_mlp_hidden, 1),
            )
            nn.init.zeros_(self.q_residual_mlp[-1].weight)
            nn.init.zeros_(self.q_residual_mlp[-1].bias)
        else:
            self.q_residual_mlp = None

        self.register_buffer("rd_q_by_bpp", torch.empty(0), persistent=True)
        self.register_buffer("rd_bpp_by_bpp", torch.empty(0), persistent=True)
        self.register_buffer("rd_q_by_index", torch.empty(0), persistent=True)
        self.register_buffer("rd_bpp_by_index", torch.empty(0), persistent=True)
        if self.q_rd_table_path:
            self.load_rd_table(self.q_rd_table_path)
        if self.q_index_mode == "rd_table" and self.rd_bpp_by_bpp.numel() == 0:
            raise ValueError(
                "q_index_mode='rd_table' requires a non-empty RD table, but none was "
                "provided. Pass --q_rd_table_path (build it once with "
                "tools/build_qindex_rd_table.py) or set --q_index_mode linear "
                "explicitly. The controller will NOT silently fall back to linear."
            )

    def load_rd_table(self, path: str | Path) -> None:
        """Load a q-index RD table used to initialise CSI-aware q selection.

        The expected format is a JSON file containing either ``q_points`` or a
        top-level list.  Each point must provide ``q_index`` (or ``q``) and
        ``bpp``.  Repeated q entries are averaged so small validation subsets
        can be concatenated safely.
        """

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        points: Any
        if isinstance(data, dict):
            points = data.get("q_points", data.get("points", data.get("rates", [])))
        else:
            points = data
        if isinstance(points, dict):
            points = [
                {"q_index": key, **value} if isinstance(value, dict) else {"q_index": key, "bpp": value}
                for key, value in points.items()
            ]
        buckets: dict[int, list[float]] = {}
        for item in points:
            if not isinstance(item, dict):
                continue
            q_raw = item.get("q_index", item.get("q", item.get("qp")))
            bpp_raw = item.get("bpp", item.get("mean_bpp"))
            if isinstance(bpp_raw, dict):
                bpp_raw = bpp_raw.get("mean", bpp_raw.get("value"))
            if q_raw is None or bpp_raw is None:
                continue
            q = int(round(float(q_raw)))
            q = max(0, min(self.get_qp_num() - 1, q))
            buckets.setdefault(q, []).append(float(bpp_raw))
        if not buckets:
            raise ValueError(f"no valid q_index/bpp points found in RD table: {path}")
        by_q = sorted((q, sum(vals) / len(vals)) for q, vals in buckets.items())
        q_by_index = torch.tensor([q for q, _ in by_q], dtype=torch.float32)
        bpp_by_index = torch.tensor([bpp for _, bpp in by_q], dtype=torch.float32).clamp_min(1e-8)
        by_bpp = sorted(by_q, key=lambda item: item[1])
        q_by_bpp = torch.tensor([q for q, _ in by_bpp], dtype=torch.float32)
        bpp_by_bpp = torch.tensor([bpp for _, bpp in by_bpp], dtype=torch.float32).clamp_min(1e-8)
        self.rd_q_by_index = q_by_index
        self.rd_bpp_by_index = bpp_by_index
        self.rd_q_by_bpp = q_by_bpp
        self.rd_bpp_by_bpp = bpp_by_bpp

    @staticmethod
    def get_qp_num() -> int:
        return 64

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

    @staticmethod
    def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        if xp.numel() == 0:
            raise ValueError("empty interpolation table")
        if xp.numel() == 1:
            return fp[0].expand_as(x)
        xp = xp.to(device=x.device, dtype=x.dtype)
        fp = fp.to(device=x.device, dtype=x.dtype)
        idx_hi = torch.bucketize(x.contiguous(), xp, right=False).clamp(1, xp.numel() - 1)
        idx_lo = idx_hi - 1
        x_lo = xp[idx_lo]
        x_hi = xp[idx_hi]
        y_lo = fp[idx_lo]
        y_hi = fp[idx_hi]
        alpha = ((x - x_lo) / (x_hi - x_lo).clamp_min(1e-8)).clamp(0.0, 1.0)
        return y_lo + alpha * (y_hi - y_lo)

    def _linear_q_index(self, cap_norm: torch.Tensor) -> torch.Tensor:
        return self.q_index_low_capacity + cap_norm * (
            self.q_index_high_capacity - self.q_index_low_capacity
        )

    def _rd_table_q_index(self, target_bpp: torch.Tensor) -> torch.Tensor:
        if self.rd_bpp_by_bpp.numel() == 0:
            return target_bpp.new_zeros(target_bpp.shape)
        return self._interp1d(target_bpp, self.rd_bpp_by_bpp, self.rd_q_by_bpp)

    def _rd_table_bpp_proxy(self, q_index: torch.Tensor) -> torch.Tensor | None:
        if self.rd_q_by_index.numel() == 0:
            return None
        q_sorted, order = torch.sort(self.rd_q_by_index.to(device=q_index.device, dtype=q_index.dtype))
        bpp_sorted = self.rd_bpp_by_index.to(device=q_index.device, dtype=q_index.dtype)[order]
        return self._interp1d(q_index, q_sorted, bpp_sorted)

    def _q_residual(
        self,
        *,
        cap_norm: torch.Tensor,
        capacity: torch.Tensor,
        target_bpp: torch.Tensor,
        snr: torch.Tensor,
        bw: torch.Tensor,
        plr: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.q_index_bias.clamp(-self.q_delta_max, self.q_delta_max).expand_as(cap_norm)
        if self.q_residual_mlp is None:
            return residual
        target_norm = ((target_bpp - self.min_target_bpp) / (self.max_target_bpp - self.min_target_bpp + 1e-8)).clamp(0.0, 1.0)
        snr_norm = (snr / 30.0).clamp(0.0, 1.5)
        bw_norm = (torch.log1p(bw.clamp_min(0.0)) / torch.log1p(bw.new_tensor(25.0))).clamp(0.0, 1.5)
        cap_log_norm = (
            torch.log1p(capacity.clamp_min(0.0)) / torch.log1p(capacity.new_tensor(self.max_capacity_mbps))
        ).clamp(0.0, 1.5)
        features = torch.stack((cap_norm, cap_log_norm, target_norm, snr_norm, bw_norm * (1.0 - plr)), dim=-1)
        mlp_delta = torch.tanh(self.q_residual_mlp(features).squeeze(-1)) * self.q_delta_max
        return residual + mlp_delta

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

        # ---- Decoupled control ----
        # Rate is owned solely by the original DCVC-FM q_index (computed below).
        # Token layering is now a PURE error-protection partition: every latent
        # position is transmitted (total_keep == 1), and the base/enhancement
        # split only decides unequal error protection under packet loss.  The
        # protected base fraction therefore tracks the channel impairment (PLR)
        # rather than the rate budget, which removes the rate/selection coupling.
        total_keep = torch.ones_like(cap_norm)
        protect_progress = plr.clamp(0.0, 1.0)
        base_keep = self.base_min_keep_ratio + protect_progress * (
            self.base_max_keep_ratio - self.base_min_keep_ratio
        )
        base_keep = self._monotone_shift_ratio(base_keep, self.base_logit_bias.clamp(-2.0, 2.0))
        base_keep = base_keep.clamp(self.base_min_keep_ratio, self.base_max_keep_ratio)
        base_keep = torch.minimum(base_keep, total_keep)
        enhancement_keep = (total_keep - base_keep).clamp_min(0.0)

        linear_q = self._linear_q_index(cap_norm)
        if self.q_index_mode == "rd_table" and self.rd_bpp_by_bpp.numel() > 0:
            q_base = self._rd_table_q_index(target_bpp)
        else:
            q_base = linear_q
        q_residual = self._q_residual(
            cap_norm=cap_norm,
            capacity=capacity,
            target_bpp=target_bpp,
            snr=snr,
            bw=bw,
            plr=plr,
        )
        q_index = (q_base + q_residual).clamp(
            min(self.q_index_low_capacity, self.q_index_high_capacity),
            max(self.q_index_low_capacity, self.q_index_high_capacity),
        )
        q_index_rounded = q_index.round().clamp(0, self.get_qp_num() - 1)
        q_proxy_bpp = self._rd_table_bpp_proxy(q_index)
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
            q_index_base=q_base,
            q_index_residual=q_residual,
            q_index_rounded=q_index_rounded,
            q_index_proxy_bpp=q_proxy_bpp,
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


def q_index_proxy_loss(
    control: CapacityControl,
    *,
    delta_weight: float = 0.02,
) -> torch.Tensor:
    """Cheap differentiable objective for CSI-aware discrete q selection.

    The real codec still executes an integer q_index.  This auxiliary loss uses
    the offline RD table as a smooth proxy so the CSI controller can learn a
    sensible continuous q coordinate without running every possible rate point.
    """

    ref = control.target_bpp
    loss = ref.new_tensor(0.0)
    if control.q_index_proxy_bpp is not None:
        proxy = control.q_index_proxy_bpp.reshape_as(ref).to(dtype=ref.dtype)
        loss = loss + F.smooth_l1_loss(proxy, ref, reduction="mean")
    if delta_weight > 0:
        delta = (control.q_index - control.q_index_base).to(dtype=ref.dtype)
        loss = loss + float(delta_weight) * (delta / 63.0).square().mean()
    return loss

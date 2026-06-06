"""Satellite channel simulation for layered DCVC-FM latents."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class ChannelOutput:
    latent: torch.Tensor
    base_survival_mask: torch.Tensor
    enhancement_survival_mask: torch.Tensor
    base_packet_loss: torch.Tensor
    enhancement_packet_loss: torch.Tensor
    effective_snr_base_db: torch.Tensor
    effective_snr_enhancement_db: torch.Tensor


class PowerNormalizer(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        power = x.square().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.sqrt(power)
        return x / scale, scale


class LayeredSatelliteChannel(nn.Module):
    """AWGN/Rayleigh/Rician satellite channel with row packet loss.

    Base and enhancement latents are transmitted through the same channel model
    but use different SNR offsets and packet-loss scales.  Dropped positions are
    filled from the provided fallback latent, which gives graceful degradation
    during severe packet loss.
    """

    def __init__(
        self,
        *,
        channel_type: str = "satellite",
        eval_noisy: bool = True,
        row_packet_loss: bool = True,
        base_snr_gain_db: float = 6.0,
        enhancement_snr_gain_db: float = 0.0,
        base_plr_scale: float = 0.25,
        enhancement_plr_scale: float = 1.25,
        rician_k_db: float = 10.0,
        shadowing_std_db: float = 3.0,
        beam_switch_prob: float = 0.01,
    ) -> None:
        super().__init__()
        if channel_type not in {"awgn", "rayleigh", "satellite", "identity"}:
            raise ValueError(f"unsupported channel_type: {channel_type}")
        self.channel_type = channel_type
        self.eval_noisy = bool(eval_noisy)
        self.row_packet_loss = bool(row_packet_loss)
        self.base_snr_gain_db = float(base_snr_gain_db)
        self.enhancement_snr_gain_db = float(enhancement_snr_gain_db)
        self.base_plr_scale = float(base_plr_scale)
        self.enhancement_plr_scale = float(enhancement_plr_scale)
        self.rician_k = 10.0 ** (rician_k_db / 10.0)
        self.shadowing_std_db = float(shadowing_std_db)
        self.beam_switch_prob = float(beam_switch_prob)
        self.normalizer = PowerNormalizer()

    @staticmethod
    def _expand(x: torch.Tensor | float, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(x):
            out = x.to(device=device, dtype=dtype).reshape(-1)
        else:
            out = torch.full((1,), float(x), device=device, dtype=dtype)
        if out.numel() == 1:
            return out.expand(batch)
        if out.numel() != batch:
            raise ValueError(f"expected 1 or {batch} values, got {out.numel()}")
        return out

    def _should_perturb(self) -> bool:
        return self.training or self.eval_noisy

    def _awgn(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        snr_linear = torch.pow(10.0, snr_db.view(-1, 1, 1) / 10.0).clamp_min(1e-8)
        noise_std = torch.rsqrt(snr_linear)
        return x + torch.randn_like(x) * noise_std

    def _rayleigh(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h_real = torch.randn(b, n, 1, device=x.device, dtype=x.dtype) * math.sqrt(0.5)
        h_imag = torch.randn(b, n, 1, device=x.device, dtype=x.dtype) * math.sqrt(0.5)
        return self._complex_fading_equalize(x, snr_db, h_real, h_imag)

    def _satellite(self, x: torch.Tensor, snr_db: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = x.shape
        dtype = x.dtype
        device = x.device

        los_amp = math.sqrt(self.rician_k / (self.rician_k + 1.0))
        scatter_std = math.sqrt(1.0 / (2.0 * (self.rician_k + 1.0)))
        h_real = los_amp + torch.randn(b, n, 1, device=device, dtype=dtype) * scatter_std
        h_imag = torch.randn(b, n, 1, device=device, dtype=dtype) * scatter_std

        phase = (torch.rand(b, n, 1, device=device, dtype=dtype) * 2.0 - 1.0) * math.pi
        cos_d = torch.cos(phase)
        sin_d = torch.sin(phase)
        h_real_d = h_real * cos_d - h_imag * sin_d
        h_imag_d = h_real * sin_d + h_imag * cos_d

        shadow_db = torch.randn(b, 1, 1, device=device, dtype=dtype) * self.shadowing_std_db
        shadow_linear = torch.pow(10.0, shadow_db / 10.0)
        beam_switch = (torch.rand(b, 1, 1, device=device, dtype=dtype) < self.beam_switch_prob).to(dtype)
        beam_gain = 1.0 - 0.7 * beam_switch
        effective_snr_db = snr_db + 10.0 * torch.log10((shadow_linear * beam_gain).view(b).clamp_min(1e-8))

        return self._complex_fading_equalize(x, effective_snr_db, h_real_d, h_imag_d), effective_snr_db

    @staticmethod
    def _complex_fading_equalize(
        x: torch.Tensor,
        snr_db: torch.Tensor,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
    ) -> torch.Tensor:
        c = x.shape[-1]
        c_half = c // 2
        if c_half == 0:
            return x
        x_real = x[..., :c_half]
        x_imag = x[..., c_half:2 * c_half]
        y_real = h_real * x_real - h_imag * x_imag
        y_imag = h_real * x_imag + h_imag * x_real

        snr_linear = torch.pow(10.0, snr_db.view(-1, 1, 1) / 10.0).clamp_min(1e-8)
        noise_std = torch.rsqrt(2.0 * snr_linear)
        y_real = y_real + torch.randn_like(y_real) * noise_std
        y_imag = y_imag + torch.randn_like(y_imag) * noise_std

        h_mag_sq = h_real.square() + h_imag.square() + 1e-8
        eq_real = (h_real * y_real + h_imag * y_imag) / h_mag_sq
        eq_imag = (h_real * y_imag - h_imag * y_real) / h_mag_sq
        out = torch.cat([eq_real, eq_imag], dim=-1)
        if c % 2 == 1:
            out = torch.cat([out, x[..., -1:]], dim=-1)
        return out

    def _transmit_tokens(self, tokens: torch.Tensor, snr_db: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.channel_type == "identity" or not self._should_perturb():
            return tokens, snr_db
        tokens_norm, scale = self.normalizer(tokens)
        if self.channel_type == "awgn":
            received = self._awgn(tokens_norm, snr_db)
            eff_snr = snr_db
        elif self.channel_type == "rayleigh":
            received = self._rayleigh(tokens_norm, snr_db)
            eff_snr = snr_db
        else:
            received, eff_snr = self._satellite(tokens_norm, snr_db)
        return received * scale, eff_snr

    def _packet_survival(
        self,
        *,
        batch: int,
        height: int,
        width: int,
        packet_loss_rate: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        plr = packet_loss_rate.clamp(0.0, 1.0)
        if not self._should_perturb() or torch.all(plr <= 0):
            return torch.ones(batch, 1, height, width, device=device, dtype=dtype)
        if self.row_packet_loss:
            row_keep = (torch.rand(batch, height, device=device, dtype=dtype) >= plr.view(batch, 1)).to(dtype)
            return row_keep.view(batch, 1, height, 1).expand(-1, -1, -1, width)
        token_keep = (torch.rand(batch, 1, height, width, device=device, dtype=dtype) >= plr.view(batch, 1, 1, 1))
        return token_keep.to(dtype)

    def forward_layered_latent(
        self,
        latent: torch.Tensor,
        *,
        base_mask: torch.Tensor,
        enhancement_mask: torch.Tensor,
        snr_db: torch.Tensor | float,
        packet_loss_rate: torch.Tensor | float,
        fallback: torch.Tensor | None = None,
    ) -> ChannelOutput:
        if latent.ndim != 4:
            raise ValueError(f"latent must be BCHW, got {tuple(latent.shape)}")
        b, c, h, w = latent.shape
        device = latent.device
        dtype = latent.dtype
        base_mask = base_mask.to(device=device, dtype=dtype)
        enhancement_mask = enhancement_mask.to(device=device, dtype=dtype)
        if fallback is None or fallback.shape != latent.shape:
            fallback = torch.zeros_like(latent)
        else:
            fallback = fallback.to(device=device, dtype=dtype)

        snr = self._expand(snr_db, b, device, dtype)
        plr = self._expand(packet_loss_rate, b, device, dtype).clamp(0.0, 1.0)
        base_snr = snr + self.base_snr_gain_db
        enh_snr = snr + self.enhancement_snr_gain_db
        base_plr = (plr * self.base_plr_scale).clamp(0.0, 1.0)
        enh_plr = (plr * self.enhancement_plr_scale).clamp(0.0, 1.0)

        tokens = latent.flatten(2).transpose(1, 2)
        base_tokens, eff_base_snr = self._transmit_tokens(tokens, base_snr)
        enh_tokens, eff_enh_snr = self._transmit_tokens(tokens, enh_snr)
        base_latent = base_tokens.transpose(1, 2).reshape(b, c, h, w)
        enh_latent = enh_tokens.transpose(1, 2).reshape(b, c, h, w)

        base_survive = self._packet_survival(
            batch=b, height=h, width=w, packet_loss_rate=base_plr, device=device, dtype=dtype
        )
        enh_survive = self._packet_survival(
            batch=b, height=h, width=w, packet_loss_rate=enh_plr, device=device, dtype=dtype
        )

        base_active = base_mask * base_survive
        enh_active = enhancement_mask * enh_survive
        received_mask = (base_active + enh_active).clamp(0.0, 1.0)
        received = base_latent * base_active + enh_latent * enh_active + fallback * (1.0 - received_mask)

        return ChannelOutput(
            latent=received,
            base_survival_mask=base_survive,
            enhancement_survival_mask=enh_survive,
            base_packet_loss=base_plr,
            enhancement_packet_loss=enh_plr,
            effective_snr_base_db=eff_base_snr,
            effective_snr_enhancement_db=eff_enh_snr,
        )

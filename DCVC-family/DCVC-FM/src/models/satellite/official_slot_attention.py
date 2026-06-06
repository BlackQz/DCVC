"""PyTorch compatibility layer for the official Slot Attention code.

The source implementation bundled with this repository is TensorFlow/Keras:
`DCVC-family/slot-attention/model.py`.

This file preserves the same object-discovery structure and math:
`build_grid`, `SoftPositionEmbed`, `spatial_flatten`, `spatial_broadcast`,
`SlotAttention`, and `SlotAttentionAutoEncoder`.  The tensor layout is adapted
from NHWC to PyTorch BCHW so gradients remain inside the DCVC-FM PyTorch graph.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def build_grid(
    resolution: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Equivalent of slot-attention/model.py::build_grid."""

    ranges = [torch.linspace(0.0, 1.0, steps=res, device=device, dtype=dtype) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing="ij")
    grid_t = torch.stack(grid, dim=-1).unsqueeze(0)
    return torch.cat([grid_t, 1.0 - grid_t], dim=-1)


def spatial_flatten(x: torch.Tensor) -> torch.Tensor:
    """BCHW -> B,N,C, equivalent to spatial_flatten on NHWC tensors."""

    return x.flatten(2).transpose(1, 2)


def spatial_broadcast(slots: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
    """Broadcast slot vectors to a 2D grid, matching the official decoder."""

    b, k, d = slots.shape
    return slots.reshape(b * k, d, 1, 1).expand(-1, -1, resolution[0], resolution[1])


def unstack_and_split(
    x: torch.Tensor,
    *,
    batch_size: int,
    num_channels: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Undo slot-batch stacking and split RGB/features from alpha masks."""

    x = x.reshape(batch_size, -1, x.shape[1], x.shape[2], x.shape[3])
    channels = x[:, :, :num_channels]
    masks = x[:, :, num_channels:num_channels + 1]
    return channels, masks


class SoftPositionEmbed(nn.Module):
    """Adds the official learnable projection of [y, x, 1-y, 1-x]."""

    def __init__(self, hidden_size: int, resolution: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.resolution = resolution
        self.dense = nn.Linear(4, hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"SoftPositionEmbed expects BCHW, got {tuple(inputs.shape)}")
        _, _, h, w = inputs.shape
        resolution = self.resolution or (h, w)
        if resolution != (h, w):
            raise ValueError(f"configured resolution {resolution} does not match input {(h, w)}")
        grid = build_grid((h, w), device=inputs.device, dtype=inputs.dtype)
        pos = self.dense(grid).permute(0, 3, 1, 2).contiguous()
        return inputs + pos


class SlotAttention(nn.Module):
    """PyTorch port of slot-attention/model.py::SlotAttention."""

    def __init__(
        self,
        num_iterations: int,
        num_slots: int,
        slot_size: int,
        mlp_hidden_size: int,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_iterations = int(num_iterations)
        self.num_slots = int(num_slots)
        self.slot_size = int(slot_size)
        self.mlp_hidden_size = int(mlp_hidden_size)
        self.epsilon = float(epsilon)

        self.norm_inputs = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        self.slots_mu = nn.Parameter(torch.empty(1, 1, slot_size))
        self.slots_log_sigma = nn.Parameter(torch.empty(1, 1, slot_size))

        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(slot_size, slot_size, bias=False)
        self.project_v = nn.Linear(slot_size, slot_size, bias=False)
        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, slot_size),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_log_sigma)
        for module in (self.project_q, self.project_k, self.project_v):
            nn.init.xavier_uniform_(module.weight)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        mu = self.slots_mu.expand(inputs.shape[0], self.num_slots, -1).to(dtype=inputs.dtype)
        sigma = torch.exp(self.slots_log_sigma).expand_as(mu).to(dtype=inputs.dtype)
        if self.training:
            slots = mu + sigma * torch.randn_like(mu)
        else:
            slots = mu

        attn = None
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)
            q = q * (self.slot_size ** -0.5)
            attn_logits = torch.matmul(k, q.transpose(1, 2))
            attn = F.softmax(attn_logits, dim=-1)
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True).clamp_min(self.epsilon)
            updates = torch.matmul(attn.transpose(1, 2), v)

            slots = self.gru(
                updates.reshape(-1, self.slot_size),
                slots_prev.reshape(-1, self.slot_size),
            )
            slots = slots.reshape(inputs.shape[0], self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))

        assert attn is not None
        return slots, attn.transpose(1, 2)


class SlotAttentionAutoEncoder(nn.Module):
    """Object-discovery autoencoder matching the official Slot Attention model."""

    def __init__(
        self,
        resolution: tuple[int, int],
        num_slots: int,
        num_iterations: int,
        *,
        num_channels: int = 3,
        hidden_size: int = 64,
        decoder_initial_size: tuple[int, int] = (8, 8),
    ) -> None:
        super().__init__()
        self.resolution = tuple(resolution)
        self.num_slots = int(num_slots)
        self.num_iterations = int(num_iterations)
        self.num_channels = int(num_channels)
        self.hidden_size = int(hidden_size)
        self.decoder_initial_size = tuple(decoder_initial_size)

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(num_channels, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, hidden_size, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_size, hidden_size, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_size, hidden_size, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_size, hidden_size, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_size, hidden_size, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_size, num_channels + 1, 3, stride=1, padding=1),
        )
        self.encoder_pos = SoftPositionEmbed(hidden_size, self.resolution)
        self.decoder_pos = SoftPositionEmbed(hidden_size, self.decoder_initial_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )
        self.slot_attention = SlotAttention(
            num_iterations=num_iterations,
            num_slots=num_slots,
            slot_size=hidden_size,
            mlp_hidden_size=128,
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError(f"image must be BCHW, got {tuple(image.shape)}")
        if image.shape[-2:] != self.resolution:
            raise ValueError(f"image resolution must be {self.resolution}, got {tuple(image.shape[-2:])}")

        x = self.encoder_cnn(image)
        x = self.encoder_pos(x)
        x = spatial_flatten(x)
        x = self.mlp(self.layer_norm(x))
        slots, _ = self.slot_attention(x)

        x = spatial_broadcast(slots, self.decoder_initial_size)
        x = self.decoder_pos(x)
        x = self.decoder_cnn(x)
        if x.shape[-2:] != self.resolution:
            x = F.interpolate(x, size=self.resolution, mode="bilinear", align_corners=False)

        recons, masks = unstack_and_split(x, batch_size=image.shape[0], num_channels=self.num_channels)
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks, slots

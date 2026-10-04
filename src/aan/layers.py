"""Reusable building blocks for the educational prototype.

Every block here mirrors a concrete mechanic from the production code base
(``reference/Moore-AnimateAnyone/src/models/``), and the docstring says which:

* ``InflatedConv3d``      — ``resnet.py``: a 2D conv applied frame-by-frame to a
  ``[B, C, F, H, W]`` tensor.  This is how the SD 2D weights are "inflated" to
  video without introducing temporal kernels in the spatial path.
* ``zero_module``         — ``motion_module.py``: zero-initialised output
  projection (ControlNet/AnimateDiff trick; makes the new branch a no-op at
  initialisation, which stabilises two-stage training).
* ``TimeEmbedding``       — sinusoidal timestep embedding + 2-layer MLP, the
  same contract as SD's ``Timesteps`` + ``TimestepEmbedding``.
* ``ResBlock3D``          — GroupNorm/SiLU/conv with additive time embedding and
  residual shortcut (SD "default" time_embedding_norm).
* ``SelfAttention2D``     — 2D spatial self-attention for the ReferenceNet trunk
  (the "writer" path; it does not consume reference features).
* ``Downsample3D`` / ``Upsample3D`` — spatial resolution changes along the UNet.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    "InflatedConv3d",
    "zero_module",
    "sinusoidal_timestep_embedding",
    "TimeEmbedding",
    "ResBlock3D",
    "SelfAttention2D",
    "Downsample3D",
    "Upsample3D",
]


class InflatedConv3d(nn.Conv2d):
    """A ``nn.Conv2d`` applied to every frame of a 5D ``[B, C, F, H, W]`` tensor.

    Mirrors ``InflatedConv3d`` from the reference implementation.  Weights keep
    the 2D shape, so pretrained SD 2D checkpoints load unchanged — this is the
    "inflation" trick that makes video fine-tuning cheap.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise AssertionError(f"InflatedConv3d expects [B,C,F,H,W]; got {tuple(x.shape)}")
        b, c, f, h, w = x.shape
        y = rearrange(x, "b c f h w -> (b f) c h w")
        y = super().forward(y)
        return rearrange(y, "(b f) c h w -> b c f h w", b=b)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero every parameter of ``module`` and return it (reference: zero_module)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Classic sinusoidal embedding (Ho et al. 2020; SD ``Timesteps``)."""
    assert dim % 2 == 0, f"embedding dim must be even, got {dim}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.to(torch.float32)[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding + 2-layer MLP used by every ResBlock."""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, out_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_timestep_embedding(timesteps, self.dim)
        emb = emb.to(self.linear1.weight.dtype)
        return self.linear2(self.act(self.linear1(emb)))


class ResBlock3D(nn.Module):
    """Residual block with additive time embedding (frame-wise 2D convs).

    Channels may change; then a 1x1 ``InflatedConv3d`` shortcut is added.
    """

    def __init__(self, in_channels: int, out_channels: int, temb_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels, eps=1e-5, affine=True)
        self.conv1 = InflatedConv3d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels, eps=1e-5, affine=True)
        self.conv2 = InflatedConv3d(out_channels, out_channels, 3, padding=1)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = InflatedConv3d(in_channels, out_channels, 1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        if x.dim() != 5:
            raise AssertionError(f"ResBlock3D expects [B,C,F,H,W]; got {tuple(x.shape)}")
        h = self.conv1(self.act(self.norm1(x)))
        if temb is not None:
            h = h + self.time_proj(self.act(temb))[:, :, None, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        if self.shortcut is not None:
            x = self.shortcut(x)
        return x + h


class SelfAttention2D(nn.Module):
    """Plain multi-head self-attention over the spatial tokens of a 2D map.

    Used by the ReferenceNet trunk ("writer" side of reference attention).
    In the production code the writer also runs its normal self-attention and
    *additionally* appends the normalised hidden states to a per-module bank
    (see ``mutual_self_attention.py``, ``MODE == "write"`` branch).
    """

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q,k,v: [B, heads, N, head_dim] -> manual scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
        return attn @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``[B, C, H, W]`` 2D feature map (single reference image)."""
        assert x.dim() == 4, f"SelfAttention2D expects [B,C,H,W]; got {tuple(x.shape)}"
        b, c, h, w = x.shape
        tokens = rearrange(x, "b c h w -> b (h w) c")
        n = tokens.shape[1]
        normed = self.norm(tokens)
        q = self.to_q(normed).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(normed).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(normed).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        out = self._attend(q, k, v)                    # [B, heads, N, dh]
        out = out.transpose(1, 2).reshape(b, n, self.dim)
        out = self.to_out(out)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return x + out


class Downsample3D(nn.Module):
    """Spatial downsample by 2 with an optional channel change (SD style)."""

    def __init__(self, in_channels: int, out_channels: Optional[int] = None) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = InflatedConv3d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """Nearest-neighbour spatial upsample by 2 followed by an inflated conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = InflatedConv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = x.shape
        x = F.interpolate(rearrange(x, "b c f h w -> (b f) c h w"), scale_factor=2.0, mode="nearest")
        x = rearrange(x, "(b f) c h w -> b c f h w", b=b)
        return self.conv(x)

        emb = emb.to(self.linear1.weight.dtype)
        return self.linear2(self.act(self.linear1(emb)))

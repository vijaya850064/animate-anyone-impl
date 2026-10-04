"""ReferenceNet: appearance feature extractor + the "bank" of stored features.

Production facts encoded here
-----------------------------
* ReferenceNet has the *same structure as the denoising UNet but no temporal
  layers*, and inherits pretrained SD weights (paper §3.2; the reproduction
  loads it from the same SD checkpoint as the denoising UNet).
* It processes the **reference image only** (a single frame, timestep 0 —
  ``reference_unet(ref_image_latents, torch.zeros_like(timesteps), ...)``), and
  its intermediate features are stored per attention module: in
  ``mutual_self_attention.py`` (write mode) each transformer block runs
  ``self.bank.append(norm_hidden_states.clone())`` — i.e. the **LayerNorm-ed
  hidden states entering attn1**.
* The denoising UNet's attention modules then consume those banked features as
  extra K/V tokens via :class:`~aan.attention.ReferenceSpatialAttention`.
* The reproduction uses ``fusion_blocks="full"`` (every attention block has a
  bank) or ``"midup"`` (mid + up blocks only).  The bank is *explicitly*
  returned here instead of being stored on module attributes via hooks — same
  semantics, easier to debug and to serialise.
* Training detail from the reproduction: ReferenceNet is trainable except
  ``up_blocks.3`` (the last decoder block), and features are only produced in
  the *conditional* branch (under CFG the unconditional half skips reference
  attention).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layers import SelfAttention2D

__all__ = ["ResBlock2D", "Downsample2D", "Upsample2D", "ReferenceAttentionBlock2D", "SimpleReferenceNet"]


class ResBlock2D(nn.Module):
    """Small 2D residual block (GroupNorm/SiLU/conv x2 + shortcut)."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels, eps=1e-5)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        if self.shortcut is not None:
            x = self.shortcut(x)
        return x + h


class Downsample2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ReferenceAttentionBlock2D(nn.Module):
    """ResBlock2D + spatial self-attention; emits the banked token sequence.

    The bank entry is ``attn.norm(tokens)`` — the normalized hidden states that
    the production writer appends to ``module.bank``.
    """

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.res = ResBlock2D(dim, dim)
        self.attn = SelfAttention2D(dim, heads)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: ``[B, C, h, w]`` -> (features ``[B, C, h, w]``, bank ``[B, h*w, C]``)."""
        h = self.res(x)
        tokens = rearrange(h, "b c h w -> b (h w) c")
        bank = self.attn.norm(tokens)  # matches production: store norm1(hidden_states)
        return self.attn(h), bank


class SimpleReferenceNet(nn.Module):
    """Shallow 2D UNet trunk whose channel ladder mirrors :class:`TinyVideoUNet`.

    The instance returns an *ordered list of bank tensors*, one per attention
    block, in depth order::

        [ level0 (h, w), level1 (h/2, w/2), mid (h/4, w/4), up0 (h/2, w/2), up1 (h, w) ]

    The denoising UNet has exactly the same five attention stacks in the same
    order, so ``bank[i]`` is fed to stack ``i``.  (Production pairs writer and
    reader modules by sorting normalized channel widths descending — see
    ``ReferenceAttentionControl.update``; with identical ladders both schemes
    coincide.)
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: Tuple[int, ...] = (32, 64, 64),
        heads: int = 4,
        store_all_blocks: bool = True,
    ) -> None:
        super().__init__()
        if len(base_channels) != 3:
            raise ValueError("SimpleReferenceNet currently expects exactly 3 channel levels (see docs).")
        ch0, ch1, ch2 = base_channels
        self.store_all_blocks = store_all_blocks
        self.conv_in = nn.Conv2d(in_channels, ch0, 3, padding=1)

        # down path
        self.block0 = ReferenceAttentionBlock2D(ch0, heads)
        self.down0 = Downsample2D(ch0, ch1)
        self.block1 = ReferenceAttentionBlock2D(ch1, heads)
        self.down1 = Downsample2D(ch1, ch2)

        # mid path
        self.mid_res = ResBlock2D(ch2, ch2)
        self.mid_block = ReferenceAttentionBlock2D(ch2, heads)

        # up path (skip connections mirror the denoising UNet)
        self.up0 = Upsample2D(ch2, ch2)
        self.reduce0 = ResBlock2D(ch2 + ch1, ch1)
        self.block_up0 = ReferenceAttentionBlock2D(ch1, heads)
        self.up1 = Upsample2D(ch1, ch1)
        self.reduce1 = ResBlock2D(ch1 + ch0, ch0)
        self.block_up1 = ReferenceAttentionBlock2D(ch0, heads)

    def forward(self, ref_latent: torch.Tensor) -> List[Optional[torch.Tensor]]:
        """``[B, 4, h, w]`` scaled reference latent -> list of bank tensors.

        With ``store_all_blocks=False`` only mid+up banks are returned (the
        others are ``None``), which mirrors ``fusion_blocks="midup"``.
        """
        assert ref_latent.dim() == 4, f"ReferenceNet expects [B,4,h,w]; got {tuple(ref_latent.shape)}"
        h0 = self.conv_in(ref_latent)
        h0, bank0 = self.block0(h0)
        h1 = self.down0(h0)
        h1, bank1 = self.block1(h1)
        m = self.down1(h1)
        m = self.mid_res(m)
        m, bank2 = self.mid_block(m)
        u = self.up0(m)
        u = self.reduce0(torch.cat([u, h1], dim=1))
        u, bank3 = self.block_up0(u)
        u = self.up1(u)
        u = self.reduce1(torch.cat([u, h0], dim=1))
        u, bank4 = self.block_up1(u)
        if self.store_all_blocks:
            return [bank0, bank1, bank2, bank3, bank4]
        return [None, None, bank2, bank3, bank4]

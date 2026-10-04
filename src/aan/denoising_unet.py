"""Denoising UNet: the main generative model (latent video denoiser).

Production facts encoded here
-----------------------------
* It is a Stable Diffusion UNet inflated to video: every 2D conv is applied
  frame-wise (``InflatedConv3d``), all spatially-varying math is per-frame, and
  the model predicts the diffusion target (epsilon or v) for a ``[B, 4, F, h,
  w]`` latent video.
* Timestep conditioning: sinusoidal embedding + MLP added inside every ResBlock
  (SD's ``time_embedding_norm="default"``).
* Pose enters inside the UNet, **added after conv_in**::

      sample = self.conv_in(sample)
      if pose_cond_fea is not None:
          sample = sample + pose_cond_fea          # [B, C_inner, F, h, w]

  ``C_inner`` = 320 in production (``block_out_channels[0]``); our tiny model
  uses ``unet.base_channels[0]``.  ``runtime.pose_fusion="concat"`` implements
  the alternative (concatenate pose channels with the noisy latent *before*
  conv_in, the SD-image-variation style used to initialise this project's base
  checkpoints).
* ReferenceNet features enter the **spatial attention** of every block as extra
  K/V tokens (see :class:`~aan.attention.ReferenceSpatialAttention`).
* CLIP features enter the **cross-attention** (``attn2``) of every block.
* Temporal attention (AnimateDiff "motion modules") is inserted after spatial
  blocks at the configured resolutions; in production that is every
  down/up/mid block at scale factors [1, 2, 4, 8] (``mm_sd_v15_v2`` weights,
  zero-initialised output projection).
* Skip connections connect the encoder feature of each scale to its decoder
  counterpart with channel concatenation — the standard U-Net topology that
  carries fine spatial detail (but *not* the character appearance; that is
  ReferenceNet's job).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from einops import rearrange

from .attention import CrossAttention, ReferenceSpatialAttention, TemporalAttention
from .config import PrototypeConfig
from .layers import Downsample3D, InflatedConv3d, ResBlock3D, TimeEmbedding, Upsample3D, zero_module

__all__ = ["AttentionStack", "TinyVideoUNet"]


class AttentionStack(nn.Module):
    """One transformer block: spatial(reference) attention → cross-attn → temporal.

    Mirrors the ordering in the production ``BasicTransformerBlock``:
    ``attn1`` (self-attention, augmented with the reference bank) → ``attn2``
    (cross-attention with CLIP) → feed-forward; the motion module runs after the
    spatial blocks.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        context_dim: int,
        temporal: Optional[TemporalAttention] = None,
    ) -> None:
        super().__init__()
        self.spatial = ReferenceSpatialAttention(dim, heads)
        self.cross = CrossAttention(dim, context_dim, heads)
        self.temporal = temporal

    def forward(
        self,
        x: torch.Tensor,
        bank: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        ref_strength: float = 1.0,
        use_temporal: bool = True,
    ) -> torch.Tensor:
        """x: ``[B, C, F, h, w]`` -> same shape."""
        b, c, f, h, w = x.shape
        tokens = rearrange(x, "b c f h w -> b f (h w) c")
        tokens = self.spatial(tokens, ref_tokens=bank, strength=ref_strength)
        tokens = self.cross(tokens, context=context)
        x = rearrange(tokens, "b f (h w) c -> b c f h w", h=h, w=w)
        if self.temporal is not None and use_temporal:
            x = self.temporal(x)
        return x



class TinyVideoUNet(nn.Module):
    """Small UNet with the exact interface of the production denoiser.

    Layout (h, w = latent grid, F = frames)::

        conv_in 4->ch0 (add pose here)                [B, ch0, F, h,   w  ]
        down0: Res + attn stack                       [B, ch0, F, h,   w  ]  -> skip0
               Downsample ch0->ch1                    [B, ch1, F, h/2, w/2]
        down1: Res + attn stack                       [B, ch1, F, h/2, w/2]  -> skip1
               Downsample ch1->ch2                    [B, ch2, F, h/4, w/4]
        mid:    Res + attn stack + Res                [B, ch2, F, h/4, w/4]
        up0:    Upsample, concat skip1, Res(ch2+ch1->ch1) + attn stack  -> [B, ch1, F, h/2, w/2]
        up1:    Upsample, concat skip0, Res(ch1+ch0->ch0) + attn stack  -> [B, ch0, F, h,   w  ]
        conv_out ch0->4 (zero-initialised, SD convention) -> predicted noise
    """

    def __init__(self, cfg: PrototypeConfig) -> None:
        super().__init__()
        if len(cfg.unet.base_channels) != 3:
            raise ValueError("TinyVideoUNet expects exactly 3 channel levels, e.g. (32, 64, 64).")
        ch0, ch1, ch2 = cfg.unet.base_channels
        heads = cfg.unet.heads
        ctx = cfg.unet.context_dim
        ted = cfg.unet.time_embed_dim
        in_ch = cfg.unet.in_channels
        self.pose_fusion = cfg.runtime.pose_fusion
        self.level0_temporal = 0 in cfg.motion.levels
        self.level1_temporal = 1 in cfg.motion.levels
        self.mid_temporal = cfg.motion.mid_block

        def temporal(dim: int, enabled: bool) -> Optional[TemporalAttention]:
            if not enabled:
                return None
            return TemporalAttention(dim, cfg.motion.heads, cfg.motion.max_frames, cfg.motion.zero_init_out)

        self.conv_in = InflatedConv3d(in_ch, ch0, kernel_size=3, padding=1)
        self.time_embed = TimeEmbedding(ch0, ted)

        # down path
        self.down0_res = ResBlock3D(ch0, ch0, ted)
        self.down0_attn = AttentionStack(ch0, heads, ctx, temporal(ch0, self.level0_temporal))
        self.down0_down = Downsample3D(ch0, ch1)
        self.down1_res = ResBlock3D(ch1, ch1, ted)
        self.down1_attn = AttentionStack(ch1, heads, ctx, temporal(ch1, self.level1_temporal))
        self.down1_down = Downsample3D(ch1, ch2)

        # mid path
        self.mid_res1 = ResBlock3D(ch2, ch2, ted)
        self.mid_attn = AttentionStack(ch2, heads, ctx, temporal(ch2, self.mid_temporal))
        self.mid_res2 = ResBlock3D(ch2, ch2, ted)

        # up path
        self.up0_up = Upsample3D(ch2)
        self.up0_res = ResBlock3D(ch2 + ch1, ch1, ted)
        self.up0_attn = AttentionStack(ch1, heads, ctx, temporal(ch1, self.level1_temporal))
        self.up1_up = Upsample3D(ch1)
        self.up1_res = ResBlock3D(ch1 + ch0, ch0, ted)
        self.up1_attn = AttentionStack(ch0, heads, ctx, temporal(ch0, self.level0_temporal))

        self.norm_out = nn.GroupNorm(8, ch0, eps=1e-5)
        self.act = nn.SiLU()
        conv_out = InflatedConv3d(ch0, 4, kernel_size=3, padding=1)
        # SD zero-inits the output conv (safe for a pretrained model). With random weights the
        # sampler would never move, so the educational configs disable it.
        self.conv_out = zero_module(conv_out) if cfg.unet.zero_init_out else conv_out

    # ------------------------------------------------------------------- API
    @property
    def num_attention_stacks(self) -> int:
        """Must equal ``len(reference_bank)`` from the ReferenceNet."""
        return 5

    def _timestep_embedding(self, timestep: torch.Tensor, batch: int, dtype: torch.dtype) -> torch.Tensor:
        if timestep.dim() == 0:
            timestep = timestep[None]
        if timestep.shape[0] != batch:
            timestep = timestep.expand(batch)
        return self.time_embed(timestep).to(dtype)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        pose_features: Optional[torch.Tensor] = None,
        reference_bank: Optional[Sequence[Optional[torch.Tensor]]] = None,
        ref_strength: float = 1.0,
        use_temporal: bool = True,
    ) -> torch.Tensor:
        """Predict the diffusion target for ``sample``.

        Args:
            sample: ``[B, 4, F, h, w]`` scaled noisy latent (``[B, 4+C_pose, ...]``
                when ``pose_fusion="concat"``).
            timestep: scalar or ``[B]`` diffusion timestep.
            context: ``[B, 1, D]`` CLIP embeddings or None (ablation E02).
            pose_features: ``[B, C_pose, F, h, w]`` pose guider output or None
                (ablation E03).  For ``"add"`` fusion ``C_pose == base_channels[0]``
                and the tensor is added after ``conv_in``.
            reference_bank: list of bank tensors / None entries from
                :class:`~aan.reference_net.SimpleReferenceNet` (ablation E01/E05).
            ref_strength: blend for reference attention (production: 1.0).
            use_temporal: disable the motion modules (ablation E04).
        Returns:
            ``[B, 4, F, h, w]`` predicted epsilon / velocity.
        """
        b, c, f, h, w = sample.shape
        temb = self._timestep_embedding(timestep, b, sample.dtype)

        x = self.conv_in(sample)
        if self.pose_fusion == "add" and pose_features is not None:
            assert pose_features.shape[0] == b and pose_features.shape[2] == f, (
                f"pose features {tuple(pose_features.shape)} must match batch/frames of sample {tuple(sample.shape)}"
            )
            assert pose_features.shape[-2:] == (h, w), (
                f"pose feature grid {tuple(pose_features.shape[-2:])} != latent grid {(h, w)}; "
                "check pose guider down factor vs VAE down factor."
            )
            x = x + pose_features.to(x.dtype)

        def bank_at(i: int) -> Optional[torch.Tensor]:
            if reference_bank is None or i >= len(reference_bank):
                return None
            return reference_bank[i]

        # ------------------------------------------------------------ down
        x = self.down0_res(x, temb)
        x = self.down0_attn(x, bank_at(0), context, ref_strength, use_temporal)
        skip0 = x
        x = self.down0_down(x)

        x = self.down1_res(x, temb)
        x = self.down1_attn(x, bank_at(1), context, ref_strength, use_temporal)
        skip1 = x
        x = self.down1_down(x)

        # ------------------------------------------------------------- mid
        x = self.mid_res1(x, temb)
        x = self.mid_attn(x, bank_at(2), context, ref_strength, use_temporal)
        x = self.mid_res2(x, temb)

        # -------------------------------------------------------------- up
        x = self.up0_up(x)
        x = torch.cat([x, skip1], dim=1)          # skip connection (r/2 level)
        x = self.up0_res(x, temb)
        x = self.up0_attn(x, bank_at(3), context, ref_strength, use_temporal)

        x = self.up1_up(x)
        x = torch.cat([x, skip0], dim=1)          # skip connection (full latent level)
        x = self.up1_res(x, temb)
        x = self.up1_attn(x, bank_at(4), context, ref_strength, use_temporal)

        return self.conv_out(self.act(self.norm_out(x)))


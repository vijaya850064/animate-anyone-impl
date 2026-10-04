"""Pose Guider: the motion-control branch (lightweight, pose -> UNet features).

Production facts encoded here (verified against the reference implementation)
-----------------------------------------------------------------------------
* ``PoseGuider(conditioning_embedding_channels=320, block_out_channels=(16,
  32, 96, 256))`` — a *tiny* network compared to the UNet.
* Architecture: ``conv_in`` → for each level pair ``(conv3x3 stride 1, conv3x3
  stride 2)`` → ``conv_out`` with **zero-initialised** parameters
  (``zero_module``), so at initialisation the pose branch injects nothing.
* All convs are ``InflatedConv3d`` (frame-wise 2D convs over ``[B,C,F,H,W]``).
* The paper describes a lightweight guider of ~4 convolutional layers with
  stride-2 downsampling to align the pose map with the latent grid (H/8).  With
  4 channel levels there are 3 stride-2 stages → exactly H/8×W/8 output.
* In the reproduction the pose guider is optionally *pre-trained* from
  ControlNet-OpenPose's ``controlnet_cond_embedding`` weights (all layers except
  ``conv_out``), i.e. it inherits ControlNet's pose encoder.
* Fusion into the denoising UNet: the output is passed as ``pose_cond_fea`` and
  **added** inside the UNet right after ``conv_in``:
  ``sample = self.conv_in(sample); sample = sample + pose_cond_fea``
  (verified in ``unet_3d.py``).  Output width 320 = UNet inner width, and the
  pose map resolution H/8 = latent resolution.  An alternative "concat" fusion
  (channels concatenated at the latent level, SD-image-variation style) is
  provided for study as ``runtime.pose_fusion: concat``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .layers import InflatedConv3d, zero_module

__all__ = ["PoseGuider", "preprocess_pose_frames"]


class PoseGuider(nn.Module):
    """Pose-map encoder producing per-frame features at the latent resolution."""

    def __init__(
        self,
        out_channels: int,
        block_out_channels: Tuple[int, ...] = (16, 32, 64, 128),
        conditioning_channels: int = 3,
        zero_init_out: bool = True,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.block_out_channels = tuple(block_out_channels)
        self.conv_in = InflatedConv3d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList()
        for i in range(len(block_out_channels) - 1):
            ch_in = block_out_channels[i]
            ch_out = block_out_channels[i + 1]
            self.blocks.append(InflatedConv3d(ch_in, ch_in, kernel_size=3, padding=1))
            self.blocks.append(InflatedConv3d(ch_in, ch_out, kernel_size=3, padding=1, stride=2))

        conv_out = InflatedConv3d(block_out_channels[-1], out_channels, kernel_size=3, padding=1)
        self.conv_out = zero_module(conv_out) if zero_init_out else conv_out
        self.act = nn.SiLU()

    @property
    def down_factor(self) -> int:
        """Spatial downsampling implied by the stride-2 stages (8 for 4 levels)."""
        return 2 ** (len(self.block_out_channels) - 1)

    def forward(self, pose_maps: torch.Tensor) -> torch.Tensor:
        """``[B, 3, F, H, W]`` pose maps in [0, 1] -> ``[B, out_channels, F, H/8, W/8]``.

        Mirrors the reference forward: conv_in → SiLU → blocks(SiLU after each)
        → conv_out (no activation).
        """
        assert pose_maps.dim() == 5, f"PoseGuider expects [B,3,F,H,W]; got {tuple(pose_maps.shape)}"
        b, c, f, h, w = pose_maps.shape
        assert h % self.down_factor == 0 and w % self.down_factor == 0, (
            f"pose map size {(h, w)} not divisible by pose guider down factor {self.down_factor}"
        )
        embedding = self.act(self.conv_in(pose_maps))
        for block in self.blocks:
            embedding = self.act(block(embedding))
        return self.conv_out(embedding)


def preprocess_pose_frames(images: Sequence[Image.Image], width: int, height: int) -> torch.Tensor:
    """Pose frames -> ``[1, 3, F, H, W]`` float tensor in [0, 1].

    Reference behaviour (``pipeline_pose2vid.py``): each pose image is resized
    to ``(width, height)``, converted to a float tensor and divided by 255 —
    **no** normalization (unlike the VAE path, which uses mean/std 0.5).  The
    pose map stays an RGB OpenPose rendering; no skeleton decoding is done by
    the model itself.
    """
    tensors = []
    for im in images:
        arr = np.asarray(im.convert("RGB").resize((width, height)), dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(np.transpose(arr, (2, 0, 1))))
    if not tensors:
        raise ValueError("preprocess_pose_frames received an empty pose sequence")
    stacked = torch.stack(tensors, dim=1)  # [C, F, H, W]
    return stacked.unsqueeze(0)            # [1, C, F, H, W]

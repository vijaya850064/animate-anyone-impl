"""VAE wrapper: pixel space <-> latent space, mirroring Animate Anyone's use.

Production facts encoded here
-----------------------------
* The paper's model is a Latent Diffusion Model: all diffusion happens at
  ``h = H/8, w = W/8`` with 4 latent channels (SD VAE, ``vae_scale_factor = 2 **
  (len(block_out_channels) - 1) = 8``).
* Latents are scaled by ``0.18215`` on encode and divided by the same constant
  before decode.  The reference pipeline does exactly
  ``ref_image_latents = vae.encode(x).latent_dist.mean * 0.18215`` and
  ``latents = 1 / 0.18215 * latents`` before decoding.
* The reference implementation decodes **frame by frame** in a loop to bound
  peak memory (``decode_latents`` in ``pipeline_pose2vid.py``); we mirror that
  in :meth:`decode_video_framewise` but also allow batched decode for speed.

``mode: tiny`` builds a small randomly-initialised ``AutoencoderKL`` so the
prototype runs on CPU with no downloads.  ``mode: pretrained`` loads a real
checkpoint (e.g. ``stabilityai/sd-vae-ft-mse``, the SD1.5 VAE used by the
reproduction) — license: CreativeML Open RAIL-M / MIT depending on checkpoint,
see docs/00_setup_guide.md.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from einops import rearrange

from .config import VaeConfig

__all__ = ["VaeWrapper"]

#: The Stable Diffusion latent scaling constant (identical in the reference
#: inference and training code: ``1 / 0.18215`` appears in ``decode_latents``).
SD_SCALING_FACTOR = 0.18215


class VaeWrapper(nn.Module):
    """Thin, dtype-safe wrapper around ``diffusers.AutoencoderKL``."""

    def __init__(self, cfg: VaeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.mode == "pretrained":
            if not cfg.pretrained_id:
                raise ValueError("vae.mode='pretrained' requires vae.pretrained_id")
            self.vae = AutoencoderKL.from_pretrained(cfg.pretrained_id)
        elif cfg.mode == "tiny":
            n_blocks = len(cfg.block_out_channels)
            self.vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=tuple(["DownEncoderBlock2D"] * n_blocks),
                up_block_types=tuple(["UpDecoderBlock2D"] * n_blocks),
                block_out_channels=tuple(cfg.block_out_channels),
                layers_per_block=cfg.layers_per_block,
                latent_channels=cfg.latent_channels,
                norm_num_groups=cfg.norm_num_groups,
                sample_size=cfg.sample_size,
            )
        else:
            raise ValueError(f"Unknown vae.mode '{cfg.mode}' (expected 'tiny' or 'pretrained')")
        self.vae.eval()
        self.vae.requires_grad_(False)  # frozen in the production architecture

    # ------------------------------------------------------------- geometry
    @property
    def down_factor(self) -> int:
        """Spatial compression factor (SD VAE: 8)."""
        return 2 ** (len(self.cfg.block_out_channels) - 1)

    @property
    def latent_channels(self) -> int:
        return self.cfg.latent_channels

    @property
    def scale_factor(self) -> float:
        return float(self.cfg.scaling_factor)

    # ------------------------------------------------------------ forward API
    @torch.no_grad()
    def encode(self, image: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """RGB ``[B, 3, H, W]`` in [-1, 1] -> scaled latent ``[B, 4, H/8, W/8]``.

        ``sample=False`` uses the posterior mean (deterministic; the reference
        pipeline uses ``latent_dist.mean`` for the reference image), ``True``
        draws from the posterior (used during training).
        """
        assert image.dim() == 4, f"VaeWrapper.encode expects [B,3,H,W]; got {tuple(image.shape)}"
        assert image.shape[1] == 3, f"expected 3 RGB channels, got {image.shape[1]}"
        h, w = image.shape[-2:]
        assert h % self.down_factor == 0 and w % self.down_factor == 0, (
            f"image size {(h, w)} not divisible by VAE down factor {self.down_factor}"
        )
        posterior = self.vae.encode(image.to(self.vae.dtype))
        latent = posterior.latent_dist.sample() if sample else posterior.latent_dist.mean
        return latent * self.scale_factor

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Scaled latent ``[B, 4, h, w]`` -> RGB ``[B, 3, 8h, 8w]`` in [-1, 1]."""
        assert latent.dim() == 4, f"VaeWrapper.decode expects [B,4,h,w]; got {tuple(latent.shape)}"
        return self.vae.decode((latent / self.scale_factor).to(self.vae.dtype)).sample

    @torch.no_grad()
    def decode_video_framewise(self, latent_video: torch.Tensor) -> torch.Tensor:
        """``[B, 4, F, h, w]`` -> ``[B, 3, F, H, W]`` decoding one frame at a time.

        Matches the reference ``decode_latents`` loop (memory saver).  For long
        videos on small GPUs this is the difference between OOM and success,
        because the decoder's activations double at 2x resolution per up block.
        """
        assert latent_video.dim() == 5, f"expected [B,4,F,h,w]; got {tuple(latent_video.shape)}"
        frames = [self.decode(latent_video[:, :, i]) for i in range(latent_video.shape[2])]
        return torch.stack(frames, dim=2)

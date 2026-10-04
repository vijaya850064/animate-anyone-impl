"""CLIP image encoder: semantic conditioning for cross-attention.

Production facts encoded here
-----------------------------
* Animate Anyone uses a **frozen CLIP image encoder**.  In the
  Moore-AnimateAnyone reproduction the module is
  ``transformers.CLIPVisionModelWithProjection`` loaded from the SD
  image-variations repo's ``image_encoder`` folder; training scripts call
  ``image_enc.requires_grad_(False)``.
* The reference pipeline takes the **image_embeds** (projected output) and
  unsqueezes a sequence dimension:
  ``encoder_hidden_states = clip_image_embeds.unsqueeze(1)`` → ``[B, 1, D]``.
  With SD image-variations the projection width ``D`` equals the UNet
  cross-attention width (1024 for the CLIP-ViT-H/14 variant used there); no
  extra trainable projection layer is inserted (``image_proj_model=None``).
* Under classifier-free guidance the unconditional branch uses
  ``torch.zeros_like(encoder_hidden_states)``.

Preprocessing (documented for the *pretrained* path)
----------------------------------------------------
Real CLIP preprocessing = resize shortest side to 224 (bicubic) → center crop
224 → rescale 1/255 → normalize with mean (0.48145466, 0.4578275, 0.40821073)
and std (0.26862954, 0.26130258, 0.27577711).  The tiny fallback below applies
the same recipe at ``image_size`` (32) so the *contract* is identical even
though the network is random.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

from .config import ClipConfig

__all__ = ["ClipImageEncoder", "CLIP_MEAN", "CLIP_STD"]

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipImageEncoder(nn.Module):
    """Frozen CLIP vision tower with its built-in projection head."""

    def __init__(self, cfg: ClipConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.mode == "pretrained":
            if not cfg.pretrained_id:
                raise ValueError("clip.mode='pretrained' requires clip.pretrained_id")
            self.model = CLIPVisionModelWithProjection.from_pretrained(cfg.pretrained_id)
            self.processor = _try_load_processor(cfg.pretrained_id)
        elif cfg.mode == "tiny":
            vision_config = CLIPVisionConfig(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                num_hidden_layers=cfg.num_hidden_layers,
                num_attention_heads=cfg.num_attention_heads,
                image_size=cfg.image_size,
                patch_size=cfg.patch_size,
                projection_dim=cfg.projection_dim,
            )
            self.model = CLIPVisionModelWithProjection(vision_config)
            self.processor = None
        else:
            raise ValueError(f"Unknown clip.mode '{cfg.mode}' (expected 'tiny' or 'pretrained')")
        self.model.eval()
        self.model.requires_grad_(False)  # frozen in the production architecture

    # ------------------------------------------------------------- geometry
    @property
    def embed_dim(self) -> int:
        """Projection width = UNet cross-attention context width."""
        return int(self.model.config.projection_dim)

    # ---------------------------------------------------------- preprocessing
    def preprocess(self, images: Union[Image.Image, Sequence[Image.Image]]) -> torch.Tensor:
        """PIL image(s) -> ``[B, 3, S, S]`` normalized pixel values.

        Uses the HF processor when available (pretrained mode) and an explicit
        CLIP-equivalent resize/center-crop/normalize otherwise.
        """
        if self.processor is not None:
            return self.processor(images=images, return_tensors="pt").pixel_values
        if isinstance(images, Image.Image):
            images = [images]
        size = self.cfg.image_size
        arrs = []
        for im in images:
            im = im.convert("RGB")
            # resize shortest side to `size` (bicubic), then center-crop `size`
            scale = size / min(im.size)
            new_size = (max(size, int(round(im.size[0] * scale))), max(size, int(round(im.size[1] * scale))))
            resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
            im = im.resize(new_size, resample=resample)
            left = (im.size[0] - size) // 2
            top = (im.size[1] - size) // 2
            im = im.crop((left, top, left + size, top + size))
            arr = np.asarray(im, dtype=np.float32) / 255.0
            arr = (arr - np.asarray(CLIP_MEAN, dtype=np.float32)) / np.asarray(CLIP_STD, dtype=np.float32)
            arrs.append(np.transpose(arr, (2, 0, 1)))
        return torch.from_numpy(np.stack(arrs, axis=0)).float()

    # ------------------------------------------------------------- forward
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[B, 3, S, S]`` -> image embeddings ``[B, projection_dim]``."""
        assert pixel_values.dim() == 4, f"CLIP expects [B,3,S,S]; got {tuple(pixel_values.shape)}"
        return self.model(pixel_values=pixel_values.to(self.model.dtype)).image_embeds

    def encode(self, images: Union[Image.Image, Sequence[Image.Image]]) -> torch.Tensor:
        """Convenience: PIL image(s) -> embeddings (device/dtype of the model)."""
        pixels = self.preprocess(images).to(next(self.model.parameters()).device)
        return self.forward(pixels)


def _try_load_processor(model_id: str):
    """Load a CLIPImageProcessor if available; fall back to None on failure."""
    try:
        from transformers import CLIPImageProcessor

        return CLIPImageProcessor.from_pretrained(model_id)
    except Exception:
        return None

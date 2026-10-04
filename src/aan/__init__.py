"""aan — Animate Anyone Study package (Team Member 3 deliverable).

This package is an *educational* re-implementation of the core components of
the Animate Anyone architecture (Hu et al., "Animate Anyone: Consistent and
Controllable Image-to-Video Synthesis for Character Animation", CVPR 2024,
arXiv:2311.17117) plus the mechanics used by the widely used
MooreThreads/Moore-AnimateAnyone reproduction.

Scope and honesty statement
---------------------------
* ``aan`` is NOT the production Animate Anyone model. It is a small, CPU
  friendly *structural* prototype whose modules mirror the *interfaces and
  data-flow* of the production architecture (ReferenceNet bank transfer,
  pose-feature addition inside the UNet, CLIP cross-attention, frame-wise
  VAE encode/decode, temporal attention across frames).
* Weights are tiny and randomly initialised by default so that everything is
  reproducible without downloads. Optional switches allow small pretrained
  components (see ``configs/``) but the prototype still does not reproduce
  paper-level video quality — that requires the full trained checkpoints and
  datasets, which are out of scope (and require consent/licensing).

Module map (mirrors the production pipeline)
--------------------------------------------
``layers``         building blocks (inflated convs, res blocks, time embedding)
``attention``      spatial / reference spatial / cross / temporal attention
``vae``            pixel <-> latent space (diffusers AutoencoderKL wrapper)
``clip_encoder``   semantic image features (CLIPVisionModelWithProjection)
``reference_net``  appearance feature extractor + feature "bank"
``pose_guider``    pose map -> inner UNet feature injection
``denoising_unet`` small video UNet that composes all attention types
``pipeline``       full image-to-video denoising + decode loop (with CFG)
``metrics``        evaluation metrics used by the ablation harness
``config``         dataclass configs loaded from YAML
``shape_log``      tensor-shape / timing bookkeeping
``utils``          seeding, device selection, assertions, RSS tracking
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

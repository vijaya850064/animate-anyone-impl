"""Evaluation metrics for the ablation harness (deliverable: metrics doc).

Honesty rules
-------------
The default prototype runs with **randomly initialised** weights, so semantic
metrics (identity, pose accuracy, CLIP similarity) are not meaningful on those
outputs.  The harness therefore reports:

* *Structural* metrics that are always computable: temporal delta (frame-to-frame
  change), flicker index (instability of that change), colour-histogram
  correlation to the reference image, and VAE round-trip PSNR/SSIM.
* *Semantic* metrics only when real pretrained encoders are enabled (CLIP
  similarity) — flagged in the report as such.

Every metric below documents what it measures and its known failure modes; none
of them alone is a valid evaluation of character animation quality.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch

__all__ = [
    "to_unit_range",
    "psnr",
    "ssim",
    "temporal_delta",
    "flicker_index",
    "histogram_correlation",
    "cosine_similarity",
    "summarize_generation",
]


def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    """Map tensors assumed in [-1, 1] to [0, 1] (no-op if already inside)."""
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Peak signal-to-noise ratio on [0,1] data; higher is better.

    Limitation: pure pixel error does not capture perceptual quality and is
    dominated by MSE behaviour; use only for VAE round-trip or controlled A/B.
    """
    a = to_unit_range(a.float())
    b = to_unit_range(b.float())
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def ssim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    """Simplified global SSIM (single window over the whole frame).

    Production-grade SSIM uses a Gaussian window and per-pixel statistics; this
    cheap version is enough to catch VAE blur/colour regressions in the
    ablation harness.  Documented limitation: insensitive to local artefacts.
    """
    a = to_unit_range(a.float())
    b = to_unit_range(b.float())
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(unbiased=False), b.var(unbiased=False)
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2) + eps))


def temporal_delta(frames: torch.Tensor) -> float:
    """Mean absolute difference between consecutive frames ``[F, C, H, W]``.

    A *sanity* measure: very low values indicate frozen motion, very high values
    indicate jitter.  Without trained weights it measures nothing semantic.
    """
    assert frames.dim() == 4, f"expected [F,C,H,W], got {tuple(frames.shape)}"
    if frames.shape[0] < 2:
        return 0.0
    d = (frames[1:].float() - frames[:-1].float()).abs().mean(dim=(1, 2, 3))
    return float(d.mean())


def flicker_index(frames: torch.Tensor) -> float:
    """Std-dev of consecutive-frame deltas — proxy for temporal instability.

    Values near 0 mean the model changes the video at a constant rate; spikes
    mean the frame-to-frame change itself is unstable (perceived as flicker).
    """
    assert frames.dim() == 4
    if frames.shape[0] < 3:
        return 0.0
    d = (frames[1:].float() - frames[:-1].float()).abs().mean(dim=(1, 2, 3))
    return float(d.std())


def histogram_correlation(ref_image: torch.Tensor, frame: torch.Tensor, bins: int = 64) -> float:
    """Correlation of grayscale histograms between reference and a frame.

    Cheap colour-distribution proxy for "does the output look like the
    reference at all".  Limitation: colour statistics are far weaker than
    identity; saturated outputs can score high with wrong identity.
    """
    def _hist(img: torch.Tensor) -> torch.Tensor:
        gray = img.float().mean(dim=0)
        hist = torch.histc(gray.clamp(0, 1), bins=bins, min=0, max=1)
        return hist / (hist.sum() + 1e-8)

    h1, h2 = _hist(to_unit_range(ref_image)), _hist(to_unit_range(frame))
    h1 = h1 - h1.mean()
    h2 = h2 - h2.mean()
    denom = (h1.norm() * h2.norm()) + 1e-8
    return float((h1 * h2).sum() / denom)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between flattened tensors (e.g. CLIP embeddings)."""
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


def summarize_generation(
    frames01: torch.Tensor,
    ref01: torch.Tensor,
    extra: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Bundle the always-computable structural metrics for one generation run.

    Args:
        frames01: ``[F, 3, H, W]`` generated frames in [0, 1] (CPU).
        ref01: ``[3, H, W]`` reference image in [0, 1] (CPU).
        extra: additional metrics (e.g. clip_ref_similarity when available).
    """
    metrics = {
        "temporal_delta": temporal_delta(frames01),
        "flicker_index": flicker_index(frames01),
        "hist_corr_first": histogram_correlation(ref01, frames01[0]),
        "hist_corr_mean": float(
            sum(histogram_correlation(ref01, frames01[i]) for i in range(frames01.shape[0]))
            / frames01.shape[0]
        ),
    }
    if extra:
        metrics.update(extra)
    return metrics

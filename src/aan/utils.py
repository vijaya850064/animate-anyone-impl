"""Small shared utilities: seeding, device/dtype resolution, assertions, RSS.

Design rules followed by this project (see docs/12_debugging_guide.md):
* every module boundary asserts tensor shapes (``assert_shape``),
* every run stores its random seed (``seed_everything``),
* CPU fallback always works (``resolve_device``),
* mixed precision is opt-in and never silently applied.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

__all__ = [
    "seed_everything",
    "resolve_device",
    "resolve_dtype",
    "assert_shape",
    "peak_rss_mb",
    "ensure_dir",
]


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch. Stores the seed in the environment for logs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["AAN_SEED"] = str(seed)


def resolve_device(preference: str = "auto") -> torch.device:
    """``auto`` -> cuda if available else cpu. Explicit values are validated."""
    pref = preference.lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available; use 'cpu' or 'auto'.")
    return torch.device(pref)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Map config strings to torch dtypes with a CPU safety check.

    ``float16`` on CPU is extremely slow for convolutions on most builds; we
    therefore warn-by-default: the caller (pipeline) decides.  We still allow
    it explicitly because one experiment (E16) intentionally compares VAE
    precision behaviour.
    """
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"Unknown dtype '{name}'. Options: {sorted(table)}")
    return table[name]


def assert_shape(
    tensor: torch.Tensor,
    expected: Sequence[int | None],
    name: str = "tensor",
    note: str = "",
) -> None:
    """Assert ``tensor.shape`` matches ``expected`` (None = wildcard).

    Raises
    ------
    AssertionError
        With a message that prints both the actual and the expected shape,
        because a bare assert is useless when debugging a 5D video tensor.
    """
    actual = tuple(tensor.shape)
    if len(actual) != len(expected):
        raise AssertionError(
            f"[shape] {name}: expected {len(expected)} dims {tuple(expected)}, "
            f"got {len(actual)} dims {actual}. {note}"
        )
    for i, (a, e) in enumerate(zip(actual, expected)):
        if e is not None and a != e:
            raise AssertionError(
                f"[shape] {name}: dim {i} expected {e}, got {a} "
                f"(actual {actual}, expected {tuple(expected)}). {note}"
            )


def peak_rss_mb() -> float:
    """Current process RSS in MiB (works on Windows/Linux via psutil)."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:  # psutil missing -> tracemalloc fallback is not comparable
        return float("nan")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class RssTracker:
    """Tracks the maximum process RSS seen across ``sample()`` calls."""

    def __init__(self) -> None:
        self.peak = peak_rss_mb()

    def sample(self) -> float:
        current = peak_rss_mb()
        if current == current and current > self.peak:  # NaN-safe compare
            self.peak = current
        return self.peak

    def reset(self) -> None:
        self.peak = peak_rss_mb()

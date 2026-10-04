"""Tensor-shape and timing bookkeeping for the educational prototype.

Every module boundary in this project records what it produced into a
:class:`ShapeLogger`.  The resulting JSON file is the *measured* half of the
tensor-shape reference table in ``docs/08_tensor_shapes.md``.

Why this exists
---------------
Shape bugs are the number-one failure mode when reproducing a diffusion
architecture (frame dimension in the wrong position, latent resolution not
matching the pose-feature resolution, CLIP context length mismatch, ...).
The logger makes the symbolic shapes in the documentation directly
verifiable against a real run.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn

__all__ = [
    "ShapeRecord",
    "ShapeLogger",
    "timer_ms",
    "count_parameters",
]


@dataclass
class ShapeRecord:
    """A single tensor snapshot: shape, dtype, device, range, grad flag."""

    name: str
    shape: List[int]
    dtype: str
    device: str
    requires_grad: bool
    min: Optional[float]
    max: Optional[float]
    numel: int
    note: str = ""

    @classmethod
    def from_tensor(cls, name: str, tensor: torch.Tensor, note: str = "") -> "ShapeRecord":
        t = tensor.detach()
        f = t.float()
        has_values = f.numel() > 0
        return cls(
            name=name,
            shape=list(t.shape),
            dtype=str(t.dtype),
            device=str(t.device),
            requires_grad=bool(tensor.requires_grad),
            min=float(f.min()) if has_values else None,
            max=float(f.max()) if has_values else None,
            numel=int(t.numel()),
            note=note,
        )


class ShapeLogger:
    """Collects :class:`ShapeRecord` entries and serialises them to JSON.

    Usage
    -----
    >>> log = ShapeLogger()                                       # doctest: +SKIP
    >>> log.record("reference_latent", z, "VAE encode x scaling") # doctest: +SKIP
    >>> log.record("pose_features", pose_fea, "[B, C_in, F, h, w]") # doctest: +SKIP
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.records: List[ShapeRecord] = []
        self._seen: set[str] = set()

    # ------------------------------------------------------------------ API
    def record(self, name: str, tensor: Optional[torch.Tensor], note: str = "") -> Optional[torch.Tensor]:
        """Record ``tensor`` (if not None) and return it unchanged.

        Returning the tensor lets call sites stay compact::

            latents = logger.record("noisy_latents", latents, "...")
        """
        if self.enabled and tensor is not None and isinstance(tensor, torch.Tensor):
            self.records.append(ShapeRecord.from_tensor(name, tensor, note))
        return tensor

    def record_once(self, name: str, tensor: Optional[torch.Tensor], note: str = "") -> Optional[torch.Tensor]:
        """Like :meth:`record` but stores only the first occurrence of ``name``."""
        if name not in self._seen:
            self._seen.add(name)
            return self.record(name, tensor, note)
        return tensor

    def log_module(self, prefix: str, module: nn.Module) -> None:
        """Record parameter counts of every direct child of ``module``."""
        if not self.enabled:
            return
        for child_name, child in module.named_children():
            total = count_parameters(child)
            if total > 0:
                self.records.append(
                    ShapeRecord(
                        name=f"params::{prefix}.{child_name}",
                        shape=[total],
                        dtype="params",
                        device="-",
                        requires_grad=any(p.requires_grad for p in child.parameters()),
                        min=None,
                        max=None,
                        numel=total,
                        note="total parameters (trainable flag aggregated over children)",
                    )
                )

    # ------------------------------------------------------------ reporting
    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "records": [asdict(r) for r in self.records],
            "count": len(self.records),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def markdown_table(self, only_shape_records: bool = True) -> str:
        """Render the collected records as a Markdown table."""
        rows = [
            "| Tensor / module | Shape | dtype | device | min | max | grad | Note |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in self.records:
            if only_shape_records and r.dtype == "params":
                continue
            mn = "" if r.min is None else f"{r.min:.3g}"
            mx = "" if r.max is None else f"{r.max:.3g}"
            shape = "x".join(str(s) for s in r.shape)
            rows.append(
                f"| {r.name} | ({shape}) | {r.dtype} | {r.device} | {mn} | {mx} | {r.requires_grad} | {r.note} |"
            )
        return "\n".join(rows)

    def summary(self) -> str:
        lines = []
        for r in self.records:
            if r.dtype == "params":
                lines.append(f"{r.name:<48} params={r.numel:,}")
            else:
                lines.append(f"{r.name:<48} {tuple(r.shape)} {r.dtype} grad={r.requires_grad}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------
@contextmanager
def timer_ms(name: str, store: Dict[str, float]) -> Iterator[None]:
    """Measure a block of code in milliseconds, writing into ``store[name]``.

    >>> timings: Dict[str, float] = {}                       # doctest: +SKIP
    >>> with timer_ms("vae_encode", timings):                # doctest: +SKIP
    ...     z = vae.encode(x)                                # doctest: +SKIP
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[name] = round((time.perf_counter() - t0) * 1000.0, 3)


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Number of (optionally trainable-only) parameters in ``module``."""
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


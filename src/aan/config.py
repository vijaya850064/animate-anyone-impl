"""Typed configuration objects loaded from YAML (config separated from code).

The production Animate Anyone training configs (see
``reference/Moore-AnimateAnyone/configs/``) centralise every choice the same
way: dataset, model paths, noise scheduler, solver.  We mirror that pattern
with nested dataclasses so an experiment is a diff of YAML files, never a
code edit — a hard requirement for the ablation report.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

__all__ = [
    "InputConfig",
    "VaeConfig",
    "ClipConfig",
    "UnetConfig",
    "PoseGuiderConfig",
    "ReferenceNetConfig",
    "MotionConfig",
    "DiffusionConfig",
    "RuntimeConfig",
    "PrototypeConfig",
    "load_config",
    "config_to_dict",
    "set_by_path",
    "validate_config",
]


@dataclass
class InputConfig:
    """Resolution/frame-count of the working space (pixels)."""

    ref_height: int = 96
    ref_width: int = 96
    frames: int = 8


@dataclass
class VaeConfig:
    """VAE settings. ``tiny`` builds a randomly-initialised AutoencoderKL
    from a small config (no downloads); ``pretrained`` loads e.g.
    ``stabilityai/sd-vae-ft-mse`` — the exact VAE family used by Stable
    Diffusion / Animate Anyone (scaling factor 0.18215)."""

    mode: str = "tiny"  # tiny | pretrained
    pretrained_id: Optional[str] = None
    block_out_channels: List[int] = field(default_factory=lambda: [16, 32, 32, 32])
    latent_channels: int = 4
    layers_per_block: int = 1
    norm_num_groups: int = 8
    sample_size: int = 96
    scaling_factor: float = 0.18215  # SD convention
    decode_framewise: bool = True  # production decodes frame-by-frame (memory)
    dtype: Optional[str] = None  # None -> runtime dtype (E16 experiments override)


@dataclass
class ClipConfig:
    """CLIP image encoder settings (semantic conditioning for cross-attention)."""

    mode: str = "tiny"  # tiny | pretrained
    pretrained_id: Optional[str] = None
    # --- tiny config only (mirrors the tiny test models used by libraries) ---
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 2
    image_size: int = 32
    patch_size: int = 16
    projection_dim: int = 64
    use_null_embedding: bool = True  # CFG: uncond context = zeros_like(embeds)


@dataclass
class UnetConfig:
    """Denoising UNet geometry (very small vs SD's (320,640,1280,1280))."""

    in_channels: int = 4  # latent channels ("add" pose fusion) or latent+pose ("concat")
    base_channels: List[int] = field(default_factory=lambda: [32, 64, 64])
    heads: int = 4
    context_dim: int = 64  # must equal clip.projection_dim (cross-attention width)
    time_embed_dim: int = 128
    # SD zero-initialises the output convolution. That convention only makes sense for a
    # *trained* UNet; with random weights it makes the denoiser output exactly zero and the
    # sampler inert, so the educational configs set this to False (production: True).
    zero_init_out: bool = True


@dataclass
class PoseGuiderConfig:
    """Mirrors production PoseGuider(320, block_out_channels=(16, 32, 96, 256)):
    conv_in -> (conv, conv-stride2) x (levels-1) -> zero-initialised conv_out.
    Output resolution is H/2^(levels-1) = H/8 with 4 levels, i.e. latent grid."""

    block_out_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    out_channels: Optional[int] = None  # None -> resolved by validate/pipeline
    zero_init_out: bool = True


@dataclass
class ReferenceNetConfig:
    """ReferenceNet is structurally the denoising UNet trunk *minus temporal
    layers*; channels must line up 1:1 with the denoising UNet so the stored
    ('banked') features can be concatenated as keys/values in its attention."""

    base_channels: List[int] = field(default_factory=lambda: [32, 64, 64])
    heads: int = 4
    store_all_blocks: bool = True  # production uses fusion_blocks="full"


@dataclass
class MotionConfig:
    """Temporal (motion) module placement. Production reference: AnimateDiff
    ``VanillaTemporalModule`` at resolutions [1, 2, 4, 8], mid-block off (v1)
    / on (v2), 8 heads, sinusoidal temporal position encoding."""

    enabled: bool = True
    levels: List[int] = field(default_factory=lambda: [0, 1])  # encoder depth indices
    mid_block: bool = False
    heads: int = 4
    max_frames: int = 32
    zero_init_out: bool = True  # mirrors zero_module(proj_out) in AnimateDiff


@dataclass
class DiffusionConfig:
    """Noise schedule identical to SD 1.5 / Animate Anyone inference configs
    (beta 0.00085..0.012 linear, steps_offset 1). Sampling defaults follow
    scripts/pose2vid.py: DDIM, 30 steps, CFG 3.5 (shrunk here for CPU)."""

    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "linear"
    steps_offset: int = 1
    prediction_type: str = "epsilon"
    num_inference_steps: int = 8
    guidance_scale: float = 3.5
    use_cfg: bool = True
    eta: float = 0.0


@dataclass
class RuntimeConfig:
    """Run-time switches; the ablation harness only touches this block."""

    device: str = "auto"
    dtype: str = "float32"
    seed: int = 1234
    use_reference: bool = True
    use_clip: bool = True
    use_pose: bool = True
    use_temporal: bool = True
    pose_fusion: str = "add"  # add (production) | concat (SD-image-variation style)
    reference_attention_strength: float = 1.0  # 1.0 full, 0.0 plain self-attention
    log_shapes: bool = True



# ==========================================================================
# Loading / overriding / validation
# ==========================================================================
def _build_dataclass(cls: type, data: Dict[str, Any]) -> Any:
    """Recursively instantiate a dataclass from nested dict data.

    ``get_type_hints`` is required because ``from __future__ import annotations``
    turns ``field.type`` into strings, which would break ``is_dataclass``.
    """
    from typing import get_type_hints

    kwargs: Dict[str, Any] = {}
    valid = {f.name: f for f in fields(cls)}
    hints = get_type_hints(cls)
    for key, value in (data or {}).items():
        if key not in valid:
            raise ValueError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = hints[key]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build_dataclass(ftype, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> PrototypeConfig:
    """Load a YAML config into :class:`PrototypeConfig` and validate it."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = _build_dataclass(PrototypeConfig, raw)
    warnings = validate_config(cfg)
    for w in warnings:
        print(f"[config][warn] {w}")
    return cfg


def config_to_dict(cfg: PrototypeConfig) -> Dict[str, Any]:
    return copy.deepcopy(asdict(cfg))


def set_by_path(cfg: Any, dotted: str, value: Any) -> None:
    """Set ``cfg.a.b.c = value`` with type coercion against the current value.

    Used by the ablation harness so an experiment stays a list of
    ``{"runtime.use_reference": false}`` style overrides instead of code.
    """
    parts = dotted.split(".")
    obj = cfg
    for p in parts[:-1]:
        if not hasattr(obj, p):
            raise AttributeError(f"Config has no attribute '{p}' in path '{dotted}'")
        obj = getattr(obj, p)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise AttributeError(f"Config has no attribute '{leaf}' in path '{dotted}'")
    current = getattr(obj, leaf)
    if isinstance(current, bool) and not isinstance(value, bool):
        if isinstance(value, str):
            value = value.strip().lower() in {"1", "true", "yes", "on"}
        else:
            value = bool(value)
    elif isinstance(current, int) and not isinstance(current, bool) and not isinstance(value, int):
        value = int(value)
    elif isinstance(current, float) and not isinstance(value, (int, float)):
        value = float(value)
    elif isinstance(current, list) and isinstance(value, (list, tuple)):
        value = list(value)
    elif current is None and isinstance(value, str) and value.lower() in {"none", "null"}:
        value = None
    setattr(obj, leaf, value)


def resolve_out_channels(cfg: PrototypeConfig) -> int:
    """PoseGuider out_channels default depends on the fusion mode:
    * add    -> inner UNet width  (production: 320) so it can be added after conv_in
    * concat -> latent channels   (SD image-variation style channel concat)
    """
    if cfg.pose_guider.out_channels is not None:
        return cfg.pose_guider.out_channels
    if cfg.runtime.pose_fusion == "concat":
        return cfg.vae.latent_channels
    return cfg.unet.base_channels[0]


def validate_config(cfg: PrototypeConfig) -> List[str]:
    """Hard-fails on structural mismatches; returns soft warnings."""
    warnings: List[str] = []

    # --- cross-module contracts -------------------------------------------
    if cfg.clip.projection_dim != cfg.unet.context_dim:
        raise ValueError(
            f"clip.projection_dim ({cfg.clip.projection_dim}) must equal "
            f"unet.context_dim ({cfg.unet.context_dim}); cross-attention K/V width mismatch."
        )
    if cfg.reference_net.base_channels != cfg.unet.base_channels:
        raise ValueError(
            "reference_net.base_channels must equal unet.base_channels so that "
            "banked reference features match the denoising UNet attention widths "
            f"(got {cfg.reference_net.base_channels} vs {cfg.unet.base_channels})."
        )
    if cfg.vae.latent_channels != cfg.unet.in_channels and cfg.runtime.pose_fusion == "add":
        raise ValueError(
            "With pose_fusion='add' the UNet input is the plain latent, so "
            "vae.latent_channels must equal unet.in_channels."
        )
    if (
        cfg.runtime.pose_fusion == "concat"
        and cfg.vae.latent_channels + resolve_out_channels(cfg) != cfg.unet.in_channels
    ):
        raise ValueError(
            "With pose_fusion='concat' unet.in_channels must equal "
            "vae.latent_channels + pose_guider.out_channels."
        )

    # --- geometry ----------------------------------------------------------
    down_factor = 2 ** (len(cfg.vae.block_out_channels) - 1)
    if down_factor != 8:
        warnings.append(
            f"VAE downsample factor is {down_factor} (block_out_channels={cfg.vae.block_out_channels}); "
            "the production SD VAE uses 8. Pose guider also downsamples by 8 — keep these aligned."
        )
    for name, size in (("ref_height", cfg.input.ref_height), ("ref_width", cfg.input.ref_width)):
        if size % down_factor != 0:
            raise ValueError(f"input.{name}={size} is not divisible by VAE down factor {down_factor}.")
    pose_down = 2 ** (len(cfg.pose_guider.block_out_channels) - 1)
    if pose_down != down_factor:
        warnings.append(
            f"PoseGuider downsamples by {pose_down} but the VAE downsamples by {down_factor}; "
            "pose features and video latents will not share a spatial grid."
        )
    if cfg.diffusion.num_train_timesteps < 10:
        raise ValueError("diffusion.num_train_timesteps too small.")
    if cfg.diffusion.num_inference_steps < 1:
        raise ValueError("diffusion.num_inference_steps must be >= 1.")
    if not (0.0 <= cfg.runtime.reference_attention_strength <= 1.0):
        raise ValueError("runtime.reference_attention_strength must be in [0, 1].")
    if cfg.runtime.pose_fusion not in {"add", "concat"}:
        raise ValueError("runtime.pose_fusion must be 'add' or 'concat'.")
    if cfg.motion.levels and max(cfg.motion.levels) >= len(cfg.unet.base_channels):
        raise ValueError(
            f"motion.levels={cfg.motion.levels} exceeds UNet depth {len(cfg.unet.base_channels)}."
        )
    if cfg.vae.mode == "pretrained" and not cfg.vae.pretrained_id:
        raise ValueError("vae.mode='pretrained' requires vae.pretrained_id.")
    if cfg.clip.mode == "pretrained" and not cfg.clip.pretrained_id:
        raise ValueError("clip.mode='pretrained' requires clip.pretrained_id.")
    if cfg.vae.mode == "pretrained":
        warnings.append(
            "Pretrained VAE/CLIP require downloads and, with the tiny random UNet, still "
            "cannot produce meaningful videos. Educational use only."
        )
    return warnings


@dataclass
class PrototypeConfig:
    input: InputConfig = field(default_factory=InputConfig)
    vae: VaeConfig = field(default_factory=VaeConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    unet: UnetConfig = field(default_factory=UnetConfig)
    pose_guider: PoseGuiderConfig = field(default_factory=PoseGuiderConfig)
    reference_net: ReferenceNetConfig = field(default_factory=ReferenceNetConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output_dir: str = "outputs/run_small"

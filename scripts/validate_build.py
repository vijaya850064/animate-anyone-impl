"""Self-checks for every module boundary — the project's test suite.

Runs without pytest and without any downloads; CPU only.  It validates the
*contract* of each module (shapes, dtypes, no-op-at-init behaviour, attention
math) plus a full end-to-end generation, so regressions in the architecture are
caught before any experiment is run.

Usage::

    python scripts/validate_build.py

Exit code 0 = all checks passed (the ablation harness relies on this).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))            # repo root
sys.path.insert(0, str(ROOT / "src"))     # the `aan` package lives in src/
sys.path.insert(0, str(ROOT / "scripts"))  # sibling helper scripts (make_demo_inputs)

import torch
import torch.nn.functional as F

from aan.attention import (
    CrossAttention,
    ReferenceSpatialAttention,
    TemporalAttention,
    scaled_dot_product_attention,
)
from aan.clip_encoder import ClipImageEncoder
from aan.config import PrototypeConfig, load_config, resolve_out_channels, set_by_path
from aan.denoising_unet import TinyVideoUNet
from aan.layers import InflatedConv3d
from aan.metrics import flicker_index, temporal_delta
from aan.pose_guider import PoseGuider, preprocess_pose_frames
from aan.reference_net import SimpleReferenceNet
from aan.shape_log import ShapeLogger
from aan.utils import seed_everything
from aan.vae import VaeWrapper
from make_demo_inputs import _fk_keypoints, render_pose_frame, render_reference

_CHECKS: List[Tuple[str, Callable[[], str]]] = []


def check(name: str) -> Callable[[Callable[[], str]], Callable[[], str]]:
    """Register a check function; it returns a detail string on success."""

    def deco(fn: Callable[[], str]) -> Callable[[], str]:
        _CHECKS.append((name, fn))
        return fn

    return deco


@check("config: YAML load + dotted-path override with type coercion")
def _cfg() -> str:
    cfg = load_config(ROOT / "configs" / "prototype_cpu_quick.yaml")
    assert cfg.input.ref_height == 64 and cfg.diffusion.num_inference_steps == 4, cfg
    set_by_path(cfg, "runtime.use_reference", "false")     # string -> bool
    set_by_path(cfg, "diffusion.guidance_scale", "7.5")     # string -> float
    set_by_path(cfg, "motion.levels", [0, 1, 2])
    assert cfg.runtime.use_reference is False
    assert abs(cfg.diffusion.guidance_scale - 7.5) < 1e-9
    assert cfg.motion.levels == [0, 1, 2]
    return f"frames={cfg.input.frames} latent={cfg.input.ref_height // 8}x{cfg.input.ref_width // 8}"


@check("InflatedConv3d equals per-frame 2D convolution")
def _inflated() -> str:
    conv = InflatedConv3d(3, 5, 3, padding=1)
    x = torch.randn(2, 3, 4, 8, 8)
    y = conv(x)
    per_frame = torch.stack(
        [F.conv2d(x[:, :, i], conv.weight, conv.bias, padding=1) for i in range(x.shape[2])], dim=2
    )
    assert y.shape == (2, 5, 4, 8, 8), y.shape
    assert torch.allclose(y, per_frame, atol=1e-5), (y - per_frame).abs().max()
    return "shape (2,5,4,8,8) matches frame-wise reference"


@check("attention core == explicit softmax(QK^T/sqrt(dk))V")
def _attn_core() -> str:
    q, k, v = (torch.randn(2, 4, 16, 8) for _ in range(3))
    out = scaled_dot_product_attention(q, k, v)
    manual = torch.softmax((q @ k.transpose(-1, -2)) * (8 ** -0.5), dim=-1) @ v
    assert out.shape == q.shape
    assert torch.allclose(out, manual, atol=1e-5), (out - manual).abs().max()
    return "max abs diff %.2e" % (out - manual).abs().max()


@check("reference spatial attention: bank changes output, mismatch raises")
def _ref_attn() -> str:
    attn = ReferenceSpatialAttention(dim=16, heads=4)
    x = torch.randn(1, 3, 25, 16)                 # B=1, F=3, N=25 tokens
    bank = torch.randn(1, 25, 16)
    out_no = attn(x, ref_tokens=None)
    out_yes = attn(x, ref_tokens=bank, strength=1.0)
    out_half = attn(x, ref_tokens=bank, strength=0.5)
    assert out_no.shape == x.shape == out_yes.shape
    assert not torch.allclose(out_no, out_yes, atol=1e-6), "reference tokens had no effect"
    assert not torch.allclose(out_yes, out_half, atol=1e-6), "strength had no effect"
    # zero strength == plain self-attention path
    assert torch.allclose(attn(x, ref_tokens=bank, strength=0.0), out_no, atol=1e-6)
    # classifier-free guidance doubles the batch: a single bank entry must broadcast
    x_cfg = torch.randn(2, 3, 25, 16)
    out_cfg = attn(x_cfg, ref_tokens=bank, strength=1.0)
    assert out_cfg.shape == x_cfg.shape and torch.isfinite(out_cfg).all()
    # channel mismatch must raise a helpful error
    try:
        attn(x, ref_tokens=torch.randn(1, 25, 8))
    except AssertionError as exc:
        assert "channel width" in str(exc), str(exc)
    else:
        raise AssertionError("expected AssertionError for channel mismatch")
    return "bank changes output; strength=0 == plain self-attn; mismatch raises"


@check("cross-attention: context=None is an exact identity (ablation E02)")
def _cross_attn() -> str:
    attn = CrossAttention(dim=16, context_dim=8, heads=4)
    x = torch.randn(2, 3, 25, 16)
    ctx = torch.randn(2, 1, 8)
    assert torch.equal(attn(x, context=None), x)
    out = attn(x, context=ctx)
    assert out.shape == x.shape and torch.isfinite(out).all()
    return "identity without context; [B,F,N,C] preserved with context"


@check("temporal attention: zero-init output projection => identity at init")
def _temporal() -> str:
    attn = TemporalAttention(dim=16, heads=4, max_frames=8, zero_init_out=True)
    x = torch.randn(2, 16, 5, 4, 4)
    out = attn(x)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-6), "zero-init must make the module an identity"
    attn_nz = TemporalAttention(dim=16, heads=4, max_frames=8, zero_init_out=False)
    assert not torch.allclose(attn_nz(x), x, atol=1e-6)
    try:
        TemporalAttention(dim=16, heads=4, max_frames=3)(x)
    except ValueError as exc:
        assert "max_frames" in str(exc)
    else:
        raise AssertionError("expected ValueError for too many frames")
    return "identity at init; frame guard raises for F > max_frames"


@check("pose guider: H/8 output grid, zero-init conv_out => zeros at init")
def _pose_guider() -> str:
    guider = PoseGuider(out_channels=32, block_out_channels=(16, 32, 64, 128), zero_init_out=True)
    pose = torch.rand(1, 3, 4, 64, 64)
    out = guider(pose)
    assert out.shape == (1, 32, 4, 8, 8), out.shape
    assert torch.count_nonzero(out) == 0, "zero-init conv_out must output zeros at init"
    nonzero = PoseGuider(out_channels=32, zero_init_out=False)(pose)
    assert torch.isfinite(nonzero).all()
    assert torch.count_nonzero(nonzero) > 0, "non-zero-init conv_out must produce a non-trivial signal"
    assert guider.down_factor == 8
    return f"out {tuple(out.shape)} (H/8 grid, zeros at init, signal when zero_init_out=False)"


@check("pose preprocessing: [1,3,F,H,W] in [0,1], resize honoured")
def _pose_pre() -> str:
    import math

    frames = [render_pose_frame(_fk_keypoints(2 * math.pi * i / 4, 128, 128), 128) for i in range(4)]
    t = preprocess_pose_frames(frames, 64, 64)
    assert t.shape == (1, 3, 4, 64, 64), t.shape
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0
    return f"{tuple(t.shape)} range=[{float(t.min()):.2f},{float(t.max()):.2f}]"


@check("reference net: 5 bank tensors with matching ladder widths/tokens")
def _refnet() -> str:
    net = SimpleReferenceNet(in_channels=4, base_channels=(32, 64, 64), heads=4)
    z = torch.randn(1, 4, 8, 8)
    banks = net(z)
    assert len(banks) == 5, len(banks)
    widths = [b.shape[-1] for b in banks]
    tokens = [b.shape[1] for b in banks]
    assert widths == [32, 64, 64, 64, 32], widths
    assert tokens == [64, 16, 4, 16, 64], tokens
    midup = SimpleReferenceNet(base_channels=(32, 64, 64), store_all_blocks=False)(z)
    assert midup[0] is None and midup[1] is None and midup[2] is not None
    return f"widths={widths} tokens={tokens}"


@check("denoising UNet: forward shapes for add-fusion and concat-fusion")
def _unet() -> str:
    cfg: PrototypeConfig = load_config(ROOT / "configs" / "prototype_cpu_quick.yaml")
    unet = TinyVideoUNet(cfg).eval()
    B, F, h = 2, 4, 8                                   # batch 2 == CFG-style doubled batch
    sample = torch.randn(B, 4, F, h, h)
    pose = torch.randn(B, cfg.unet.base_channels[0], F, h, h)
    ctx = torch.randn(B, 1, cfg.unet.context_dim)
    banks = [torch.randn(B, n, c) for n, c in ((64, 32), (16, 64), (4, 64), (16, 64), (64, 32))]
    with torch.no_grad():
        out = unet(sample, torch.tensor(700.0), context=ctx, pose_features=pose, reference_bank=banks)
    assert out.shape == sample.shape, out.shape
    assert torch.isfinite(out).all()
    assert unet.num_attention_stacks == len(banks)

    # concat fusion: pose channels appended to the latent channels before conv_in
    cfg_c = load_config(ROOT / "configs" / "prototype_cpu_quick.yaml")
    set_by_path(cfg_c, "runtime.pose_fusion", "concat")
    set_by_path(cfg_c, "unet.in_channels", 8)
    unet_c = TinyVideoUNet(cfg_c).eval()
    pose_c = torch.randn(B, resolve_out_channels(cfg_c), F, h, h)
    with torch.no_grad():
        out_c = unet_c(torch.randn(B, 8, F, h, h), torch.tensor(700.0), context=ctx, pose_features=None)
    assert out_c.shape == (B, 4, F, h, h)
    # bank/pose frame mismatch must be caught
    try:
        unet(sample, torch.tensor(700.0), context=ctx, pose_features=torch.randn(B, 32, F + 1, h, h))
    except AssertionError as exc:
        assert "batch/frames" in str(exc)
    else:
        raise AssertionError("expected AssertionError for frame-count mismatch")
    return "add-fusion (2,4,4,8,8) ok; concat-fusion 8->4 channels ok; mismatch raises"


@check("VAE: encode/decode round trip, frame-wise video decode, 1/8 grid")
def _vae() -> str:
    vae = VaeWrapper(load_config(ROOT / "configs" / "prototype_cpu_quick.yaml").vae).eval()
    img = torch.rand(1, 3, 64, 64) * 2 - 1
    z = vae.encode(img)
    assert z.shape == (1, 4, 8, 8), z.shape
    back = vae.decode(z)
    assert back.shape == (1, 3, 64, 64), back.shape
    vid = vae.decode_video_framewise(torch.randn(1, 4, 3, 8, 8))
    assert vid.shape == (1, 3, 3, 64, 64), vid.shape
    assert vae.down_factor == 8
    return "latent (1,4,8,8); decode (1,3,64,64); video decode (1,3,3,64,64)"


@check("end-to-end: pipeline generates a video, saves mp4, logs shapes")
def _e2e() -> str:
    import math
    import time

    from aan.pipeline import AnimateAnyoneLite

    cfg = load_config(ROOT / "configs" / "prototype_cpu_quick.yaml")
    cfg.diffusion.num_inference_steps = 2
    seed_everything(0)
    logger = ShapeLogger()
    pipe = AnimateAnyoneLite(cfg, logger=logger).eval()
    ref = render_reference(64, seed=7)
    poses = [render_pose_frame(_fk_keypoints(2 * math.pi * i / 4, 64, 64), 64) for i in range(cfg.input.frames)]
    t0 = time.perf_counter()
    out = pipe.generate(ref, poses)
    elapsed = (time.perf_counter() - t0) * 1000.0
    video, ref01 = out["video01"], out["ref01"]
    assert video.shape == (cfg.input.frames, 3, 64, 64), video.shape
    assert ref01.shape == (3, 64, 64)
    assert torch.isfinite(video).all() and 0.0 <= float(video.min()) and float(video.max()) <= 1.0
    assert "total" in out["timings"]
    mp4 = ROOT / "outputs" / "validate" / "smoke.mp4"
    pipe.save_video(video, mp4, fps=8)
    assert mp4.exists() and mp4.stat().st_size > 0
    logger.to_json(ROOT / "outputs" / "validate" / "smoke_shape_log.json")
    return f"{tuple(video.shape)} in {elapsed:.0f} ms, mp4={mp4.stat().st_size} B, params={pipe.parameter_report()}"


@check("metrics: constant video -> zero delta/flicker, changing video -> positive")
def _metrics() -> str:
    const = torch.ones(4, 3, 16, 16) * 0.5
    moving = torch.rand(4, 3, 16, 16)
    assert temporal_delta(const) == 0.0 and flicker_index(const) == 0.0
    assert temporal_delta(moving) > 0.0 and flicker_index(moving) >= 0.0
    return f"delta(const)={temporal_delta(const):.4f} delta(random)={temporal_delta(moving):.4f}"


def main() -> int:
    passed, failed = 0, []
    for name, fn in _CHECKS:
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - report everything, keep going
            failed.append((name, exc))
            print(f"[FAIL] {name}\n       {type(exc).__name__}: {exc}")
            if "--trace" in sys.argv:
                traceback.print_exc()
        else:
            passed += 1
            print(f"[ ok ] {name}\n       {detail}")
    print(f"\n{passed}/{len(_CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
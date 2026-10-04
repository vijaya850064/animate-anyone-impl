"""Run the educational prototype once and save every artifact.

Produces (default ``outputs/run_small``):

==========================  ====================================================
``video.mp4``               generated clip (H.264, fps configurable)
``contact_sheet.png``       reference + evenly spaced generated frames
``shape_log.json``          every module-boundary tensor (shape/dtype/device/grad)
``run_log.json``            settings, seeds, per-stage timings, params, metrics
``metrics.json``            structural evaluation metrics
``config_used.yaml``        the exact config used (reproducibility)
==========================  ====================================================

Usage::

    python scripts/run_prototype.py --config configs/prototype_cpu_small.yaml
    python scripts/run_prototype.py --frames 16 --steps 12 --guidance 3.5
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import yaml
from PIL import Image

from aan.config import config_to_dict, load_config, set_by_path
from aan.metrics import summarize_generation
from aan.pipeline import AnimateAnyoneLite
from aan.shape_log import ShapeLogger
from aan.utils import RssTracker, seed_everything
from make_demo_inputs import synthesize_demo


def load_inputs(inputs_dir: Path, frames: int) -> Tuple[Image.Image, List[Image.Image]]:
    """Load the reference image and the first ``frames`` pose frames."""
    ref = Image.open(inputs_dir / "ref.png").convert("RGB")
    pose_files = sorted((inputs_dir / "poses").glob("frame_*.png"))
    if len(pose_files) < frames:
        raise SystemExit(f"need {frames} pose frames in {inputs_dir}, found {len(pose_files)} (regenerate inputs)")
    return ref, [Image.open(p).convert("RGB") for p in pose_files[:frames]]


def save_contact_sheet(
    frames01: torch.Tensor,
    ref01: torch.Tensor,
    path: Path,
    max_frames: int = 6,
) -> Path:
    """Write one row: [reference | evenly spaced generated frames]."""
    idx = [int(round(i * (frames01.shape[0] - 1) / max(1, min(max_frames, frames01.shape[0]) - 1)))
           for i in range(min(max_frames, frames01.shape[0]))]
    imgs = [ref01] + [frames01[i] for i in idx]
    h, w = frames01.shape[-2], frames01.shape[-1]
    gap = 4
    sheet = Image.new("RGB", (w * len(imgs) + gap * (len(imgs) - 1), h), (20, 20, 20))
    for i, t in enumerate(imgs):
        arr = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        sheet.paste(Image.fromarray(arr), (i * (w + gap), 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Animate Anyone educational prototype end-to-end.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "prototype_cpu_small.yaml"))
    parser.add_argument("--inputs", type=str, default=str(ROOT / "inputs" / "demo"))
    parser.add_argument("--outdir", type=str, default=None, help="defaults to config.output_dir")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="auto | cpu | cuda")
    parser.add_argument("--dtype", type=str, default=None, help="float32 | float16")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--no-video", action="store_true", help="skip mp4 encoding")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.frames is not None:
        set_by_path(cfg, "input.frames", args.frames)
    if args.steps is not None:
        set_by_path(cfg, "diffusion.num_inference_steps", args.steps)
    if args.guidance is not None:
        set_by_path(cfg, "diffusion.guidance_scale", args.guidance)
    if args.seed is not None:
        set_by_path(cfg, "runtime.seed", args.seed)
    if args.device is not None:
        set_by_path(cfg, "runtime.device", args.device)
    if args.dtype is not None:
        set_by_path(cfg, "runtime.dtype", args.dtype)

    # ---- inputs (procedural, generated on demand) ------------------------
    inputs_dir = Path(args.inputs)
    if not (inputs_dir / "ref.png").exists():
        with tempfile.TemporaryDirectory() as tmp:
            synthesize_demo(inputs_dir, frames=max(16, cfg.input.frames), size=256, seed=7)
    ref, poses = load_inputs(inputs_dir, cfg.input.frames)

    # ---- run --------------------------------------------------------------
    seed_everything(cfg.runtime.seed)
    logger = ShapeLogger(enabled=cfg.runtime.log_shapes)
    pipe = AnimateAnyoneLite(cfg, logger=logger).eval()
    rss = RssTracker()
    rss.sample()
    t0 = time.perf_counter()
    result = pipe.generate(ref, poses)
    wall_s = time.perf_counter() - t0
    rss.sample()

    outdir = Path(args.outdir or cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    video, ref01 = result["video01"], result["ref01"]

    artifacts: Dict[str, str] = {}
    if not args.no_video:
        artifacts["video"] = str(pipe.save_video(video, outdir / "video.mp4", fps=args.fps))
    artifacts["contact_sheet"] = str(save_contact_sheet(video, ref01, outdir / "contact_sheet.png"))
    artifacts["shape_log"] = str(logger.to_json(outdir / "shape_log.json"))

    metrics = summarize_generation(video, ref01)
    run_log = {
        "meta": result["meta"],
        "timings_ms": result["timings"],
        "wall_clock_s": round(wall_s, 3),
        "peak_rss_mb": round(rss.peak, 1),
        "parameters": pipe.parameter_report(),
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(pipe.device),
            "dtype": str(pipe.dtype),
            "weights": "randomly initialised (educational prototype, NOT production checkpoints)",
        },
        "artifacts": artifacts,
        "disclaimer": "Synthetic procedural inputs + random weights: output is structural test data, "
                      "not a faithful character animation. Do not present generated media as real footage.",
    }
    (outdir / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (outdir / "config_used.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")

    # ---- console summary ---------------------------------------------------
    print("\n=== Animate Anyone educational prototype ===")
    print(f"config      : {args.config}")
    print(f"resolution  : {cfg.input.ref_height}x{cfg.input.ref_width}, frames={cfg.input.frames}, "
          f"latent={pipe.latent_hw[0]}x{pipe.latent_hw[1]}, steps={result['meta']['steps']}, "
          f"CFG={result['meta']['guidance_scale']} (active={result['meta']['use_cfg']})")
    print(f"device      : {result['meta']['device']} / {result['meta']['dtype']}")
    print(f"parameters  : {pipe.parameter_report()}")
    print("timings (ms):")
    for k, v in result["timings"].items():
        print(f"    {k:<28} {v:9.1f}")
    print(f"metrics     : {json.dumps(metrics, indent=4)}")
    print(f"peak RSS    : {run_log['peak_rss_mb']} MB")
    print(f"artifacts   : {json.dumps(artifacts, indent=4)}")
    print("NOTE: random weights -> the video is noise; shapes/timings/metrics are the real output.")


if __name__ == "__main__":
    main()
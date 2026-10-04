"""Run the ablation matrix and write a reproducible experiment report.

For every experiment in ``configs/ablation_matrix.yaml`` this script:
1. loads the base config and applies the experiment's ``overrides`` (dotted keys)
   plus harness ``options`` (frames / resolution / steps / pose degradation);
2. synthesises procedural inputs (motion speed from ``options``);
3. runs the full pipeline with a fixed seed, measuring wall-clock time and peak
   process RSS (CPU) — on GPU, ``torch.cuda.max_memory_allocated`` would be the
   analogue (documented in docs/12_debugging_guide.md);
4. computes structural metrics + a VAE round-trip fidelity measurement;
5. writes ``ablation_report.csv``, ``ablation_report.md`` and ``ablation_runs.json``.

Usage::

    python scripts/run_ablations.py
    python scripts/run_ablations.py --only E01_no_reference E04_no_temporal
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import yaml
from PIL import Image, ImageFilter

from aan.config import load_config, set_by_path
from aan.metrics import psnr, ssim, summarize_generation
from aan.pipeline import AnimateAnyoneLite
from aan.shape_log import ShapeLogger
from aan.utils import RssTracker, seed_everything
from make_demo_inputs import synthesize_demo

# Honest, experiment-family specific interpretation shown in the report.
INTERPRETATIONS: Dict[str, str] = {
    "E00": "Reference configuration. All structural numbers are baselines; identity/pose quality needs trained weights.",
    "E01": "Removing ReferenceNet removes appearance injection. Trained models show identity drift/clothing distortion; "
           "here we can only verify the code path (bank=None) and the runtime/memory delta.",
    "E02": "Cross-attention is skipped entirely (context=None is an exact identity in our CrossAttention). "
           "Trained models lose semantic/global appearance consistency (e.g. character category, palette).",
    "E03": "No pose conditioning: a trained model ignores the driving signal and produces pose-unrelated motion. "
           "Here the pose tensor is simply not computed (timing/param delta is the measurable effect).",
    "E04": "Temporal (motion) modules disabled. In production they are zero-initialised, so an untrained model shows no "
           "flicker difference; with trained weights this experiment is the primary flicker measurement.",
    "E05": "Reference attention strength 0.5 / 0.0 linearly blends reference-augmented and plain self-attention outputs. "
           "0.0 must be numerically equivalent to removing the bank (property verified by validate_build.py).",
    "E06": "Fewer DDIM steps trade quality for speed (compute scales ~linearly with steps).",
    "E07": "Guidance 1.0 disables CFG (single forward pass -> ~2x faster); high guidance increases conditioning strength "
           "and can over-saturate colours in trained models.",
    "E08": "Degraded pose maps (blur/jitter/occlusion) mimic detector failures: trained models show pose lag, distorted "
           "limbs or ignored poses.",
    "E09": "Frame-count sweep: cost is ~linear in F (temporal attention and VAE decode dominate); production v1/v2 use "
           "24-frame windows with positional encodings sized for that window.",
    "E10": "Resolution sweep: latent grid scales with H/8; attention cost grows quadratically in the token count, so "
           "cost grows ~4x per doubling.",
    "E11": "VAE precision: float16 changes decode speed and introduces rounding; with an untrained VAE this only measures "
           "numerics, not the colour/detail artefacts seen with trained VAEs.",
    "E12": "Motion-speed sweep on a fixed frame budget: slower motion means smaller inter-frame deltas (metrics scale "
           "with true motion, which is why temporal metrics must be read together with the pose sequence).",
    "E13": "Pose fusion by channel concatenation (SD-image-variation style, 8-channel UNet input) instead of the "
           "production addition after conv_in; conv_in parameter count grows.",
}


def degrade_poses(
    poses: List[Image.Image],
    blur: float = 0.0,
    jitter: int = 0,
    occlude: float = 0.0,
    seed: int = 0,
) -> List[Image.Image]:
    """Blur / shift / black-out pose frames to emulate noisy detector output."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for p in poses:
        im = p.convert("RGB")
        if blur > 0:
            im = im.filter(ImageFilter.GaussianBlur(radius=blur))
        if jitter > 0:
            dx, dy = rng.integers(-jitter, jitter + 1, size=2)
            arr = np.roll(np.asarray(im), shift=(int(dy), int(dx)), axis=(0, 1))
            im = Image.fromarray(arr)
        if occlude > 0:
            arr = np.asarray(im).copy()
            arr[-int(arr.shape[0] * occlude):, :, :] = 0
            im = Image.fromarray(arr)
        out.append(im)
    return out


def vae_roundtrip_metrics(pipe: AnimateAnyoneLite, video01: torch.Tensor) -> Dict[str, float]:
    """Re-encode + re-decode the generated frames to measure VAE fidelity."""
    x = (video01 * 2 - 1).permute(1, 0, 2, 3)[None]  # [1,3,F,H,W] in [-1,1]
    with torch.no_grad():
        # per-frame latents are [B,4,h,w]; add a frame axis before stacking -> [B,4,F,h,w]
        z = torch.cat([pipe.vae.encode(x[:, :, i]).unsqueeze(2) for i in range(x.shape[2])], dim=2)
        rec = pipe.vae.decode_video_framewise(z)
    rec01 = ((rec[0].float() / 2 + 0.5).clamp(0, 1)).permute(1, 0, 2, 3)
    return {"vae_psnr": psnr(rec01, video01), "vae_ssim": ssim(rec01, video01)}


def interpretation(exp_id: str) -> str:
    for prefix, text in INTERPRETATIONS.items():
        if exp_id.startswith(prefix):
            return text
    return "See docs/10_experiments_ablation.md."


def run_one(
    exp: Dict[str, Any],
    base_config: Path,
    seed: int,
    outdir: Path,
    save_video: bool,
) -> Dict[str, Any]:
    cfg = load_config(base_config)
    for key, value in (exp.get("overrides") or {}).items():
        set_by_path(cfg, key, value)
    opts: Dict[str, Any] = exp.get("options") or {}
    if "frames" in opts:
        set_by_path(cfg, "input.frames", opts["frames"])
    if "resolution" in opts:
        set_by_path(cfg, "input.ref_height", opts["resolution"])
        set_by_path(cfg, "input.ref_width", opts["resolution"])
    if "steps" in opts:
        set_by_path(cfg, "diffusion.num_inference_steps", opts["steps"])

    motion_speed = float(opts.get("motion_speed", 1.0))
    n_frames = cfg.input.frames
    with tempfile.TemporaryDirectory() as tmp:
        inputs = synthesize_demo(Path(tmp) / "demo", frames=max(n_frames, 16), size=256, seed=7,
                                 motion_speed=motion_speed)
        ref = Image.open(inputs / "ref.png").convert("RGB")
        pose_files = sorted((inputs / "poses").glob("frame_*.png"))[:n_frames]
        poses = [Image.open(p).convert("RGB") for p in pose_files]
        if opts:
            poses = degrade_poses(
                poses,
                blur=float(opts.get("pose_blur", 0.0)),
                jitter=int(opts.get("pose_jitter", 0)),
                occlude=float(opts.get("pose_occlude", 0.0)),
                seed=seed,
            )

        seed_everything(seed)
        pipe = AnimateAnyoneLite(cfg, logger=ShapeLogger(enabled=False)).eval()
        rss = RssTracker()
        rss.sample()
        t0 = time.perf_counter()
        result = pipe.generate(ref, poses)
        wall_s = time.perf_counter() - t0
        rss.sample()

    video, ref01 = result["video01"], result["ref01"]
    metrics = summarize_generation(video, ref01)
    metrics.update(vae_roundtrip_metrics(pipe, video))

    mp4 = ""
    if save_video:
        mp4 = str(pipe.save_video(video, outdir / f"{exp['id']}.mp4", fps=8))

    params = pipe.parameter_report()
    return {
        "id": exp["id"],
        "description": exp.get("description", ""),
        "overrides": json.dumps(exp.get("overrides") or {}, sort_keys=True),
        "options": json.dumps(opts, sort_keys=True),
        "seed": seed,
        "frames": n_frames,
        "resolution": f"{cfg.input.ref_height}x{cfg.input.ref_width}",
        "latent_hw": f"{pipe.latent_hw[0]}x{pipe.latent_hw[1]}",
        "steps": cfg.diffusion.num_inference_steps,
        "guidance_scale": cfg.diffusion.guidance_scale,
        "use_cfg": result["meta"]["use_cfg"],
        "runtime_s": round(wall_s, 3),
        "peak_rss_mb": round(rss.peak, 1),
        "params_total": int(sum(params.values())),
        "temporal_delta": round(metrics["temporal_delta"], 5),
        "flicker_index": round(metrics["flicker_index"], 5),
        "hist_corr_mean": round(metrics["hist_corr_mean"], 4),
        "vae_psnr": round(metrics["vae_psnr"], 3),
        "vae_ssim": round(metrics["vae_ssim"], 4),
        "identity_quality": "n/a (random weights - needs trained model + real reference)",
        "pose_accuracy": "n/a (random weights - needs trained model + pose detector)",
        "temporal_smoothness": "structural proxy only (see flicker_index)",
        "artifacts": mp4,
        "interpretation": interpretation(exp["id"]),
    }


def write_reports(rows: List[Dict[str, Any]], outdir: Path, matrix_path: Path, seed: int) -> None:
    """Write CSV + Markdown report, including deltas against the baseline run."""
    csv_path = outdir / "ablation_report.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    baseline = next((r for r in rows if r["id"].startswith("E00")), rows[0])
    for r in rows:
        if r is baseline:
            r["runtime_delta_pct"] = 0.0
        else:
            r["runtime_delta_pct"] = round(
                100.0 * (r["runtime_s"] - baseline["runtime_s"]) / max(baseline["runtime_s"], 1e-9), 1
            )

    md = [
        "# Animate Anyone study — ablation report (automatically generated)",
        "",
        f"* matrix: `{matrix_path}`",
        f"* base config: `{baseline['resolution']}`, seed {seed}, CPU-only run",
        f"* baseline run: `{baseline['id']}` ({baseline['runtime_s']} s)",
        "",
        "> **Reading guide.** All numbers below come from an *educational prototype with randomly initialised",
        "> weights*: they measure the data path, shapes, runtime and memory of each configuration — **not**",
        "> animation quality. Identity/pose/flicker quality requires the trained production checkpoints.",
        "> `temporal_delta` and `flicker_index` are structural proxies computed on the decoded frames;",
        "> `vae_psnr`/`vae_ssim` measure VAE round-trip fidelity (the one quality metric that is meaningful",
        "> even with random weights, because it compares the decoder against its own encoder).",
        "",
        "| id | frames | res | steps | CFG | runtime (s) | Δruntime | peak RSS (MB) | params | temporal_delta | flicker | hist_corr | VAE PSNR | VAE SSIM |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['id']} | {r['frames']} | {r['resolution']} | {r['steps']} | {r['guidance_scale']} | "
            f"{r['runtime_s']:.2f} | {r['runtime_delta_pct']:+.1f}% | {r['peak_rss_mb']:.0f} | {r['params_total']:,} | "
            f"{r['temporal_delta']:.4f} | {r['flicker_index']:.4f} | {r['hist_corr_mean']:.3f} | "
            f"{r['vae_psnr']:.2f} | {r['vae_ssim']:.3f} |"
        )

    md += ["", "## Per-experiment interpretation", ""]
    for r in rows:
        md += [
            f"### {r['id']} — {r['description']}",
            "",
            f"* config changes: `{r['overrides']}` | options: `{r['options']}`",
            f"* latent grid {r['latent_hw']}, {r['steps']} steps, CFG {r['guidance_scale']} (active={r['use_cfg']}), seed {r['seed']}",
            f"* runtime {r['runtime_s']:.2f} s ({r['runtime_delta_pct']:+.1f}% vs baseline), peak RSS {r['peak_rss_mb']:.0f} MB",
            f"* identity quality: {r['identity_quality']}; pose accuracy: {r['pose_accuracy']}; "
            f"temporal smoothness: {r['temporal_smoothness']}",
            f"* interpretation: {r['interpretation']}",
            "",
        ]

    (outdir / "ablation_report.md").write_text("\n".join(md), encoding="utf-8")
    (outdir / "ablation_runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Animate Anyone ablation matrix.")
    parser.add_argument("--matrix", type=str, default=str(ROOT / "configs" / "ablation_matrix.yaml"))
    parser.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "ablations"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--only", type=str, nargs="*", default=None, help="substring filter on experiment ids")
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args()

    matrix = yaml.safe_load(Path(args.matrix).read_text(encoding="utf-8"))
    base_config = Path(args.matrix).parent.parent / matrix["base_config"]
    if not base_config.exists():
        base_config = ROOT / matrix["base_config"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    experiments = matrix["experiments"]
    if args.only:
        experiments = [e for e in experiments if any(f in e["id"] for f in args.only)]

    rows: List[Dict[str, Any]] = []
    for i, exp in enumerate(experiments, start=1):
        print(f"[{i}/{len(experiments)}] {exp['id']} ...", flush=True)
        row = run_one(exp, base_config, args.seed, outdir, save_video=not args.no_videos)
        rows.append(row)
        print(
            f"      runtime={row['runtime_s']:.2f}s  dT={row['temporal_delta']:.4f}  "
            f"flicker={row['flicker_index']:.4f}  vae_psnr={row['vae_psnr']:.2f}",
            flush=True,
        )

    write_reports(rows, outdir, Path(args.matrix), args.seed)
    print(f"\nWrote {outdir / 'ablation_report.csv'}")
    print(f"Wrote {outdir / 'ablation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
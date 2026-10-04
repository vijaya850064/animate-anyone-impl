"""Build a one-page visual showcase for presenting the work.

Creates ``<outdir>/showcase.png``: the inputs the model receives (reference
image + pose maps), the frames it produces, and a caption strip stating the
measured run statistics — i.e. everything a reviewer needs in one image,
without opening any file.

Usage::

    python scripts/make_showcase.py                     # uses outputs/run_small
    python scripts/make_showcase.py --outdir outputs/run_small --title "Animate Anyone study"

Honesty note printed at the end: with untrained weights the frames are noise;
the showcase demonstrates the *pipeline*, not image quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from PIL import Image, ImageDraw

CELL = 150          # thumbnail size
PAD = 8
CAPTION = 16
HEADER = 46


def _read_video_frames(mp4: Path) -> list[Image.Image]:
    import imageio.v3 as iio

    return [Image.fromarray(f) for f in iio.imread(mp4)]


def build_showcase(outdir: Path, inputs_dir: Path, title: str) -> Path:
    """Compose inputs (row 1) and generated frames (row 2) into one PNG."""
    run_log = json.loads((outdir / "run_log.json").read_text(encoding="utf-8"))
    meta, timings = run_log["meta"], run_log["timings_ms"]

    ref = Image.open(inputs_dir / "ref.png").convert("RGB").resize((CELL, CELL))
    pose_files = sorted((inputs_dir / "poses").glob("frame_*.png"))
    pose_idx = np.linspace(0, len(pose_files) - 1, 3).round().astype(int)
    poses = [Image.open(pose_files[i]).convert("RGB").resize((CELL, CELL)) for i in pose_idx]

    frames = _read_video_frames(outdir / "video.mp4")
    f_idx = np.linspace(0, len(frames) - 1, 4).round().astype(int)
    gen = [frames[i].resize((CELL, CELL)) for i in f_idx]

    cols = 4
    width = PAD + cols * (CELL + PAD)
    height = HEADER + 2 * (CAPTION + CELL + PAD)
    sheet = Image.new("RGB", (width, height), (16, 16, 20))
    d = ImageDraw.Draw(sheet)

    header = (f"{title}  |  {meta['frames']} frames @ {meta['height']}x{meta['width']} "
              f"(latent {meta['latent_hw'][0]}x{meta['latent_hw'][1]})  |  {meta['steps']} DDIM steps, "
              f"CFG {meta['guidance_scale']} ({'on' if meta['use_cfg'] else 'off'})  |  seed {meta['seed']}")
    d.text((PAD, 10), header, fill=(240, 240, 245))
    d.text((PAD, 26), "ReferenceNet + Pose Guider + CLIP + temporal attention + DDIM sampling, "
                      f"{run_log['environment']['device']}, {timings['total']:.0f} ms, "
                      f"peak RSS {run_log['peak_rss_mb']:.0f} MB", fill=(170, 200, 240))

    labels_imgs = [
        ("INPUT  reference image", ref),
        ("INPUT  pose map (frame 0)", poses[0]),
        ("INPUT  pose map (mid)", poses[1]),
        ("INPUT  pose map (last)", poses[2]),
        ("OUTPUT generated frame 0", gen[0]),
        ("OUTPUT generated frame (mid)", gen[1]),
        ("OUTPUT generated frame (mid)", gen[2]),
        ("OUTPUT generated frame last", gen[3]),
    ]
    for i, (label, im) in enumerate(labels_imgs):
        r, c = divmod(i, cols)
        x = PAD + c * (CELL + PAD)
        y = HEADER + r * (CAPTION + CELL + PAD)
        d.text((x + 2, y + 3), label, fill=(235, 235, 240))
        sheet.paste(im, (x, y + CAPTION))
    d.text((PAD, height - 12),
           "randomly initialised weights -> frames are noise; the showcase demonstrates the pipeline, "
           "not image quality", fill=(255, 200, 120))

    path = outdir / "showcase.png"
    sheet.save(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the visual showcase PNG.")
    parser.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "run_small"))
    parser.add_argument("--inputs", type=str, default=str(ROOT / "inputs" / "demo"))
    parser.add_argument("--title", type=str, default="Animate Anyone core model - educational prototype")
    args = parser.parse_args()

    outdir, inputs_dir = Path(args.outdir), Path(args.inputs)
    if not (outdir / "run_log.json").exists():
        print(f"No run_log.json in {outdir} - run scripts/run_prototype.py first.", file=sys.stderr)
        return 1
    if not (inputs_dir / "ref.png").exists():
        print(f"No inputs in {inputs_dir} - run scripts/make_demo_inputs.py first.", file=sys.stderr)
        return 1

    path = build_showcase(outdir, inputs_dir, args.title)
    log = json.loads((outdir / "run_log.json").read_text(encoding="utf-8"))
    print(f"showcase : {path}")
    print(f"video    : {outdir / 'video.mp4'}")
    print(f"contact  : {outdir / 'contact_sheet.png'}")
    print(f"metrics  : {json.dumps(log['metrics'])}")
    print(f"timings  : total {log['timings_ms']['total']:.0f} ms, peak RSS {log['peak_rss_mb']:.0f} MB")
    print(f"params   : {log['parameters']}")
    print("NOTE: untrained weights -> frames are noise; present shapes/timings/ablation deltas as the result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
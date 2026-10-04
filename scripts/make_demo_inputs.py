"""Synthesise demo inputs: one reference 'character' + a pose sequence.

No external or user data is used — everything is drawn procedurally so the
repository stays free of dataset/licensing/privacy issues (project requirement).

* **Reference image**: a blocky character (head, hair, face marks, shirt with a
  contrasting stripe, belt, trousers, shoes) on a vertical gradient background.
  The point is structured, colour-rich appearance content for the appearance
  modules to carry.
* **Pose sequence**: a COCO-18/OpenPose-style stick figure performing an arm
  swing + stepping cycle, drawn with OpenPose-like limb colours on black.  This
  mimics what a pose detector (DWPose/OpenPose) produces in the real pipeline.

Outputs (default ``inputs/demo``)::

    ref.png                 reference image (256x256)
    poses/frame_000.png ... pose frames (F)
    poses_keypoints.npz     [F, 18, 2] float32 pixel keypoints (for experiments)
    meta.json               seed / size / frames / fps

Usage::

    python scripts/make_demo_inputs.py --out inputs/demo --frames 16 --size 256 --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

# COCO-18 keypoint names (OpenPose order)
KP_NAMES = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear",
]

# OpenPose-like bone colours (approximate palette: "like" the real palette)
BONES: List[Tuple[int, int, str]] = [
    (1, 2, "#ff9d00"), (2, 3, "#ff7d00"), (3, 4, "#ff6000"),        # right arm
    (1, 5, "#00a3ff"), (5, 6, "#0077ff"), (6, 7, "#004dff"),        # left arm
    (1, 8, "#ffc400"), (8, 9, "#ffa000"), (9, 10, "#ff8000"),       # right leg
    (1, 11, "#00ffc8"), (11, 12, "#00d8a8"), (12, 13, "#00b58c"),   # left leg
    (1, 0, "#ffd28c"),                                              # neck -> nose
    (0, 14, "#ffd28c"), (0, 15, "#ffd28c"),                         # eyes
    (14, 16, "#ffd28c"), (15, 17, "#ffd28c"),                       # ears
]
JOINT_FILL = "#ffffff"


def _fk_keypoints(t: float, w: int, h: int) -> np.ndarray:
    """Forward-kinematic stick figure at phase ``t`` (radians) -> ``[18, 2]``."""
    hip_y = 0.60 * h
    cx = 0.5 * w
    torso = 0.17 * h
    shoulder_y = hip_y - torso
    head_r = 0.065 * h
    upper_arm = lower_arm = 0.10 * h
    upper_leg = lower_leg = 0.155 * h

    swing = math.sin(t)
    swing_l = math.sin(t + math.pi)

    kp = np.zeros((18, 2), dtype=np.float32)
    kp[1] = (cx, shoulder_y)                                           # neck
    kp[0] = (cx, shoulder_y - 0.45 * torso - head_r * 0.8)             # nose
    kp[14] = (kp[0][0] - 0.15 * head_r, kp[0][1] - 0.15 * head_r)      # r_eye
    kp[15] = (kp[0][0] + 0.15 * head_r, kp[0][1] - 0.15 * head_r)      # l_eye
    kp[16] = (kp[0][0] - 0.45 * head_r, kp[0][1])                      # r_ear
    kp[17] = (kp[0][0] + 0.45 * head_r, kp[0][1])                      # l_ear

    shoulder_dx = 0.085 * w
    kp[2] = (cx - shoulder_dx, shoulder_y)                             # r_shoulder
    kp[5] = (cx + shoulder_dx, shoulder_y)                             # l_shoulder

    for key_shoulder, key_elbow, key_wrist, phase in ((2, 3, 4, swing), (5, 6, 7, swing_l)):
        a = 0.55 * phase
        sx, sy = kp[key_shoulder]
        ex, ey = sx + math.sin(a) * upper_arm, sy + math.cos(a) * upper_arm
        kp[key_elbow] = (ex, ey)
        kp[key_wrist] = (ex + math.sin(a * 0.5) * lower_arm, ey + math.cos(a * 0.5) * lower_arm)

    hip_dx = 0.05 * w
    kp[8] = (cx - hip_dx, hip_y)                                       # r_hip
    kp[11] = (cx + hip_dx, hip_y)                                      # l_hip
    for key_hip, key_knee, key_ankle, phase in ((8, 9, 10, swing_l), (11, 12, 13, swing)):
        b = 0.42 * phase
        hx, hy = kp[key_hip]
        kx, ky = hx + math.sin(b) * upper_leg, hy + math.cos(b) * upper_leg
        kp[key_knee] = (kx, ky)
        kp[key_ankle] = (kx + math.sin(b * 0.35) * lower_leg, ky + math.cos(b * 0.35) * lower_leg)

    return kp


def render_pose_frame(kp: np.ndarray, size: int, line_width: int = 4) -> Image.Image:
    """Draw the skeleton with OpenPose-like colours on a black background."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(img)
    for a, b, color in BONES:
        d.line([tuple(kp[a]), tuple(kp[b])], fill=color, width=line_width)
    for i in range(18):
        x, y = kp[i]
        r = line_width + 1
        d.ellipse([x - r, y - r, x + r, y + r], fill=JOINT_FILL)
    return img


def render_reference(size: int, seed: int = 7) -> Image.Image:
    """Draw the demo character (deterministic given ``seed``).

    The seed drives a small palette/geometry jitter so that two different seeds
    give genuinely different references (useful for conditioning-sensitivity
    tests); everything is deterministic for a fixed seed.
    """
    rng = np.random.default_rng(seed)

    def jitter(hex_color: str, amount: float = 0.18) -> str:
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        arr = np.array([r, g, b], dtype=np.float32) * (1.0 + float(rng.uniform(-amount, amount)))
        return "#%02x%02x%02x" % tuple(int(np.clip(v, 0, 255)) for v in arr)

    top = np.array([35, 40, 60], dtype=np.float32) + rng.integers(-10, 10, size=3)
    bottom = np.array([120, 130, 150], dtype=np.float32) + rng.integers(-10, 10, size=3)
    grad = np.linspace(0, 1, size, dtype=np.float32)[:, None, None]
    bg = (top[None, None, :] * (1 - grad) + bottom[None, None, :] * grad)
    bg = np.repeat(bg, size, axis=1).astype(np.uint8)
    img = Image.fromarray(bg, "RGB")
    d = ImageDraw.Draw(img)

    cx = size // 2 + int(rng.integers(-size // 24, size // 24 + 1))
    skin, hair = jitter("#e6bea0", 0.08), jitter("#2a211c", 0.15)
    shirt, stripe = jitter("#d94f3a"), jitter("#f2e6c9", 0.10)   # red shirt + cream stripe
    pants, belt, shoes = jitter("#33517a"), jitter("#c8a437", 0.10), jitter("#1a1a1a", 0.2)

    # legs / trousers / shoes
    d.rectangle([cx - size * 0.13, size * 0.60, cx - size * 0.02, size * 0.88], fill=pants)
    d.rectangle([cx + size * 0.02, size * 0.60, cx + size * 0.13, size * 0.88], fill=pants)
    d.rectangle([cx - size * 0.15, size * 0.88, cx - size * 0.01, size * 0.93], fill=shoes)
    d.rectangle([cx + size * 0.01, size * 0.88, cx + size * 0.15, size * 0.93], fill=shoes)
    # belt + shirt with stripe
    d.rectangle([cx - size * 0.16, size * 0.34, cx + size * 0.16, size * 0.44], fill=belt)
    d.rectangle([cx - size * 0.17, size * 0.40, cx + size * 0.17, size * 0.62], fill=shirt)
    d.rectangle([cx - size * 0.17, size * 0.50, cx + size * 0.17, size * 0.54], fill=stripe)
    # sleeves + hands
    d.rectangle([cx - size * 0.24, size * 0.40, cx - size * 0.15, size * 0.60], fill=shirt)
    d.rectangle([cx + size * 0.15, size * 0.40, cx + size * 0.24, size * 0.60], fill=shirt)
    d.ellipse([cx - size * 0.25, size * 0.59, cx - size * 0.16, size * 0.66], fill=skin)
    d.ellipse([cx + size * 0.16, size * 0.59, cx + size * 0.25, size * 0.66], fill=skin)
    # head / hair / face
    d.ellipse([cx - size * 0.10, size * 0.16, cx + size * 0.10, size * 0.38], fill=skin)
    d.pieslice([cx - size * 0.11, size * 0.13, cx + size * 0.11, size * 0.34], 180, 360, fill=hair)
    d.rectangle([cx - size * 0.115, size * 0.24, cx - size * 0.085, size * 0.30], fill=hair)
    d.rectangle([cx + size * 0.085, size * 0.24, cx + size * 0.115, size * 0.30], fill=hair)
    d.ellipse([cx - size * 0.05, size * 0.26, cx - size * 0.02, size * 0.29], fill="#1a1a1a")
    d.ellipse([cx + size * 0.02, size * 0.26, cx + size * 0.05, size * 0.29], fill="#1a1a1a")
    d.line([cx - size * 0.03, size * 0.33, cx + size * 0.03, size * 0.33], fill="#8a4b3a", width=2)
    return img


def synthesize_demo(
    out_dir: str | Path,
    frames: int = 16,
    size: int = 256,
    seed: int = 7,
    motion_speed: float = 1.0,
) -> Path:
    """Generate the full demo input set and return the output directory."""
    out_dir = Path(out_dir)
    (out_dir / "poses").mkdir(parents=True, exist_ok=True)

    render_reference(size, seed).save(out_dir / "ref.png")

    all_kp = []
    for i in range(frames):
        phase = 2.0 * math.pi * motion_speed * (i / frames)
        kp = _fk_keypoints(phase, size, size)
        render_pose_frame(kp, size).save(out_dir / "poses" / f"frame_{i:03d}.png")
        all_kp.append(kp)
    np.savez_compressed(out_dir / "poses_keypoints.npz", keypoints=np.stack(all_kp), names=np.array(KP_NAMES))

    meta = {
        "seed": seed,
        "frames": frames,
        "size": size,
        "fps": 8,
        "motion_speed": motion_speed,
        "note": "procedural demo inputs - no external data, free to regenerate",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate procedural demo inputs (reference + pose sequence).")
    parser.add_argument("--out", type=str, default="inputs/demo")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    out = synthesize_demo(args.out, args.frames, args.size, args.seed)
    print(f"Wrote demo inputs to {out}  (ref.png + {args.frames} pose frames)")


if __name__ == "__main__":
    main()

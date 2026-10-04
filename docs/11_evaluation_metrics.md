# 11 — Evaluation metrics: what to measure, and what each one misses

**Governing rule for this project:** no single metric decides whether a change
is good. Every number is reported together with its limitations and with a
visual inspection of the frames.

---

## 1. Metric families

| metric | what it measures | range / direction | implemented here | limitations |
|---|---|---|---|---|
| **CLIP similarity (ref ↔ generated frames)** | semantic/appearance agreement in CLIP space | ↑ better | `aan/metrics.py` ready; meaningful only with a real CLIP (`clip.mode: pretrained`) | insensitive to identity details; a wrong person of the same class can score high; rewards global colour matching |
| **Face/identity embedding similarity** | biometric identity match | ↑ better | not shipped (legal/ethical) — use ArcFace only with explicit consent and never on real people's likenesses | biometric data handling; embedding-space bias; may be unreliable on stylised/occluded faces |
| **Pose similarity (target vs generated skeleton)** | pose adherence (PCK, MPJPE, keypoint-projection error) | ↑ better | proxy: `temporal_delta` + visual inspection; full PCK needs a pose estimator on outputs | detector failures on generated images; 2D ambiguity (depth); requires DWPose on generated frames |
| **Temporal consistency** | frame-to-frame stability | ↑ better (lower variance) | `temporal_delta`, `flicker_index` | "low" can mean *frozen motion* as easily as smoothness — must be read with the pose sequence |
| **Optical-flow stability** | whether motion trajectories are smooth | ↑ better | not shipped (OpenCV available as dependency) | flow failures on fast motion/occlusion |
| **Background consistency** | static regions stay static | ↑ better | partial: `histogram_correlation` per frame | histogram is colour-only; use with a mask for a real measurement |
| **VAE reconstruction quality** | decoder fidelity (PSNR/SSIM round-trip) | ↑ better | `psnr`, `ssim` on encode→decode of the frames | pixel metrics miss perceptual change; meaningful even with random weights (our only real quality signal) |
| **Human evaluation** | everything above, holistic | rated | recommended protocol in §4 | expensive, subjective, needs inter-rater agreement |
| **Artefact frequency** | rate of visible defects | ↓ better | manual pass over contact sheets | definition-dependent (what counts as an artefact) |
| **Inference speed** | wall-clock per clip | ↓ better | `timings` in `run_log.json` | machine-dependent; must state hardware |
| **Peak memory** | RAM/VRAM | ↓ better | `RssTracker` (CPU RSS via psutil); on GPU use `torch.cuda.max_memory_allocated()` | RSS includes the interpreter and the allocator cache |

---

## 2. Metrics as implemented (`src/aan/metrics.py`)

```python
def psnr(a, b)                    # [0,1] data; 10·log10(1/MSE); inf if identical
def ssim(a, b)                   # global (single-window) SSIM — cheap, catches VAE regressions
def temporal_delta(frames)       # [F,C,H,W] -> mean |x_t − x_{t−1}|  (frozen ⇔ 0)
def flicker_index(frames)        # std of those deltas (unstable motion ⇔ high)
def histogram_correlation(ref, frame)   # 64-bin grey histogram Pearson correlation
def cosine_similarity(a, b)      # for CLIP embeddings (needs a real CLIP)
def summarize_generation(frames01, ref01, extra=None) -> dict
```

Design choices worth stating:

* **global SSIM, not windowed** — a deliberate simplification for a CPU harness;
  it cannot see local artefacts, so it is only used as a coarse regression alarm;
* **no single "identity score"** is shipped, deliberately: with random weights it
  would be meaningless, and shipping a face-embedding metric invites misuse;
* every metric is computed on CPU `float32` in `[0,1]` — the module never assumes
  a layout, but callers must pass `[F,C,H,W]` (frames first).

---

## 3. Which metrics are valid in this repository?

| situation | valid metrics | invalid |
|---|---|---|
| prototype with random weights | shapes, timings, peak RSS, parameter counts, `vae_psnr/ssim` round-trip | everything semantic (identity, pose accuracy, CLIP similarity, flicker-as-quality) |
| trained model, synthetic inputs | + CLIP similarity, pose similarity (DWPose on outputs), temporal metrics | biometric identity (unless consented) |
| trained model, consented real footage | all of the above + human study | — |

This distinction is printed at the top of every generated report
(`outputs/ablations/ablation_report.md`) and in each CSV row
(`identity_quality`, `pose_accuracy`, `temporal_smoothness` columns).

---

## 4. Recommended human-evaluation protocol (for the trained model)

1. **Blinded side-by-side**: reference | target pose strip | generated clip, with
   the model identity anonymised;
2. **3 questions, 1–5 scale**: (a) is this the same person? (b) does the motion
   match the pose sequence? (c) is it temporally stable?
3. **≥ 5 raters**, report mean ± std and inter-rater agreement;
4. **artefact checklist**: face/hand deformation, extra limbs, texture swimming,
   background leakage, colour pops;
5. **pairwise A/B** for ablations (e.g. E00 vs E01) is far more sensitive than
   absolute scores — always present the baseline next to the variant.

---

## 5. Metric pitfalls specific to this architecture

| pitfall | why it happens here | mitigation |
|---|---|---|
| CLIP similarity *rises* when identity is wrong | CLIP rewards "same class + similar palette" | pair CLIP with a pose metric and human review; never alone |
| temporal delta *falls* when the model freezes | a static video has zero delta | always compare against the driving pose's own inter-frame motion |
| PSNR/SSIM on generated frames is meaningless | there is no ground truth for a generative sample | use PSNR/SSIM only for the **VAE round-trip** (encode→decode the same frames) |
| flicker index conflated with VAE noise | the decoder adds high-frequency noise per frame | compute flicker on decoded frames *and* compare to the round-trip baseline |
| histogram correlation rewards saturated output | clipped colours still correlate | pair with a saturation/clipping check on the raw frames |
| comparing runs with different seeds | the seed changes *which* variant you get | fix the seed for A/B (`run_prototype.py --seed 1234`) and run ≥3 seeds for quality claims |
| metrics computed at different resolutions | the latent grid changes | normalise to a fixed evaluation resolution and document it |

---

## 6. Reproducing the numbers in the reports

```bash
python scripts/run_ablations.py                 # writes CSV + Markdown + JSON
python scripts/run_prototype.py --seed 1234     # single run with full timing/metric log
```

Each run stores: seed, config fields, frames, resolution, latent grid, steps,
guidance, wall-clock, per-stage timings, peak RSS, parameter counts,
`vae_psnr`/`vae_ssim`, `temporal_delta`, `flicker_index`, `hist_corr_mean` and an
interpretation string. Anything not measurable is written as `n/a` with the
reason, rather than as a plausible-looking number.
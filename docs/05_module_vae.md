# 05 — VAE: pixel space ↔ latent space

**Question it answers:** *"How do we move between compressed latent space and
visible video?"*

---

## 1. Why not diffuse RGB pixels directly?

A 784×512 RGB frame has 1.2M values; its 8×-downscaled latent has
4·64·98 = 25k values — **48× fewer**. Diffusion cost scales with the number of
values the UNet processes, so latent diffusion (Rombach et al. 2020) is what
makes 24-frame generation feasible:

| | pixel-space diffusion | latent diffusion (Animate Anyone) |
|---|---|---|
| sequence modelled | 3×H×W per frame | 4×(H/8)×(W/8) per frame |
| UNet cost (784×512) | ~1.2M values/frame | ~25k values/frame (48× cheaper) |
| memory (24 frames, batch 1) | tens of GB of activations | a few GB |
| detail ceiling | none | limited by the decoder (8× reconstruction) |

Animate Anyone inherits the **Stable Diffusion VAE unchanged and frozen**:
`AutoencoderKL.from_pretrained(cfg.vae_model_path)` with
`vae.requires_grad_(False)` (`train_stage_1.py:262, 303`). The reproduction uses
the *MSE-tuned* VAE (`sd-vae-ft-mse`), which decodes sharper than the KL-tuned one.

---

## 2. Encode / decode mechanics and the scaling factor

```python
# production encode (pipeline_pose2vid.py:362-369)
ref_image_tensor  = self.ref_image_processor.preprocess(ref_image, height=height, width=width)
ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean     # posterior MEAN
ref_image_latents = ref_image_latents * 0.18215                             # (b, 4, h, w)

# production decode (pipeline_pose2vid.py:decode_latents; train_stage_2.py)
latents = 1 / 0.18215 * latents
latents = rearrange(latents, "b c f h w -> (b f) c h w")
video = [vae.decode(latents[i:i+1]).sample for i in range(latents.shape[0])]   # FRAME BY FRAME
video = torch.cat(video)
video = (video / 2 + 0.5).clamp(0, 1)
```

Four facts to remember:

1. **range convention**: the VAE works in `[-1, 1]`; `video/2 + 0.5` converts to
   `[0, 1]` for saving.
2. **scaling factor `0.18215`** applied on encode, divided before decode. It
   matches latent magnitude to the diffusion prior `N(0, I)`: SD latents have
   std ≈ 4–6 before scaling, so multiplying by 0.18215 brings them near unit
   scale. Forgetting to divide before decoding washes out contrast and shifts
   colours — the classic "why is my VAE output grey" bug.
3. **posterior mean vs sample**: inference uses `.mean` (deterministic),
   `.sample()` during training. Prototype supports both.
4. **frame-by-frame decode**: there is **no temporal modelling inside the VAE**.
   Frame-to-frame consistency is entirely the diffusion side's job (motion
   modules). Decoding frame-wise is a memory strategy — decoder activations
   double at each 2× up-resolution, so a 24-frame 784×512 batch is exactly where
   OOMs happen.

Prototype conventions (`src/aan/vae.py`), verified by `validate_build.py`:

```python
reference_latent = vae.encode(reference_image)      # [1,3,96,96] -> [1,4,12,12] (×0.18215 inside)
decoded_frames   = vae.decode(denoised_latents)     # [1,4,8,12,12] -> [1,3,96,96] (÷0.18215)
video            = vae.decode_video_framewise(z)    # [1,4,8,12,12] -> [1,3,8,96,96] (production loop)
```

Measured (96×96, 8 frames, fp32 CPU): 33 ms reference encode, 187 ms for
8-frame decode (≈ 23 ms/frame).

---

## 3. Pixel space vs latent space

| | pixel space | latent space (used here) |
|---|---|---|
| representation | `[0,1]` or `[-1,1]` RGB | scaled 4-channel values (std ≈ 1 after scaling) |
| semantics | directly interpretable, no decoder needed | needs the decoder for *any* visualisation |
| diffusion prior | pixel noise is not isotropic in a useful sense | latents are near-Gaussian → schedules work |
| sharpness ceiling | — | decoder's 8× reconstruction (§5) |
| debugging | easy to eyeball | always decode to inspect (`outputs/.../contact_sheet.png`) |

---

## 4. Memory strategies: slicing and tiling

* **slicing** — process batch elements (here: frames) one at a time. The
  reference pipeline's per-frame decode loop *is* this technique applied
  manually; `enable_vae_slicing()` exists in the diffusers pipeline wrapper
  (`pipeline_pose2vid.py`).
* **tiling** — split a large image into overlapping tiles, decode each, blend
  the seams; needed above ~1024 px per side on consumer GPUs, at the cost of
  slight seam artefacts.

Not needed in the prototype (96×96, 8 frames, 385 MB peak RSS) but expected at
production resolutions.

---

## 5. Precision, artefacts, and the ceiling they impose

| precision | effect |
|---|---|
| fp32 | reference-quality decode; ~2× memory, slower |
| fp16 (the repo's inference default) | standard for SD pipelines; small rounding differences |
| bf16 | like fp16 with more mantissa; good on modern GPUs |
| **CPU fp16** | slow and numerically fragile for convolutions — measured in E11 |

Typical VAE artefacts (independent of diffusion quality):

* **blur / loss of fine detail** — the dominant one: the decoder reconstructs an
  8× compressed code, so hair strands, fabric weave and text soften;
* **colour shift / desaturation** — MSE-tuned VAEs trade colour accuracy for
  perceptual sharpness;
* **blockiness** at high-frequency edges, especially at small latent grids
  (our 12×12 latents make this very visible);
* **texture leakage** between conditioning and output when a texture appears in
  both reference and target.

Consequence for evaluation: the VAE is a hard ceiling on sharpness, so any
"blurry output" complaint should first be tested with a VAE round-trip of the
*ground-truth* video. The ablation harness reports exactly that —
`vae_psnr`/`vae_ssim` (`scripts/run_ablations.py:vae_roundtrip_metrics`) — which
is the one quality metric that stays meaningful with random weights.

---

## 6. Task-brief pseudocode vs implemented conventions

```python
# brief
reference_latent = vae.encode(reference_image).latent_dist.sample()
reference_latent = reference_latent * vae_scaling_factor
decoded_frames   = vae.decode(denoised_latents / vae_scaling_factor)
```

Our exact conventions (`src/aan/vae.py`), matching production:

```python
# encode: default to the posterior MEAN for determinism (production does this)
z = vae.encode(ref_image, sample=False)          # internally: mean * 0.18215
assert z.shape == (1, 4, H // 8, W // 8)

# decode: the wrapper divides by the factor internally
rgb = vae.decode(z)                              # [1,3,H,W] in [-1,1]
video = vae.decode_video_framewise(z_video)      # [1,3,F,H,W] in [-1,1]
frames01 = (video[0] / 2 + 0.5).clamp(0, 1)      # [F,3,H,W] in [0,1]
```

Differences from the brief worth flagging: production uses `latent_dist.mean`
(not `sample`) for the reference image, and the frame axis is explicit
(`b c f h w`), not squeezed into the batch dimension.

---

## 7. Shape/dtype/device/grad contract

| property | value (prototype run) |
|---|---|
| input dtype | `float32` (`float16` optional via `runtime.dtype`; VAE may differ via `vae.dtype`, E11) |
| device | CPU or CUDA (`runtime.device: auto`) |
| gradient | **disabled** — `@torch.no_grad()` on `encode`/`decode`/`decode_video_framewise`; the VAE is frozen in production too |
| output range | latent: std ≈ 1 after scaling; RGB: `[-1,1]` → `[0,1]` after `video/2+0.5` |
| memory | reference encode is one frame (negligible); decode is the peak-memory stage (frame-wise loop keeps it to one frame's activations) |

Everything above is asserted at the module boundary: 4-D input for `encode`, 4-D
for `decode`, 5-D for `decode_video_framewise`, and `H % 8 == 0`
(`utils.assert_shape` / inline asserts, see `docs/12_debugging_guide.md`).
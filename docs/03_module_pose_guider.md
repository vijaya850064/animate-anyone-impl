# 03 — Pose Guider: the motion-control module

**Question it answers:** *"How should the character move?"*

---

## 1. What pose information is extracted from the driving video

The pipeline does **not** consume keypoints directly: a pose detector (the
reference repo uses **DWPose**, `tools/extract_dwpose_from_vid.py`) runs on the
driving video and produces a per-frame **RGB pose map** — the format ControlNet
consumes. The dataset confirms the pairing: every sample has `video_path` *and*
`kps_path` with equal frame counts (`src/dataset/dance_video.py:87-89`).

| representation | used? | notes |
|---|---|---|
| raw COCO-18 keypoint coordinates | ✗ (only by our demo generator) | detector output, converted to a drawing |
| **OpenPose / DWPose RGB map** | ✓ | 3-channel rendering, the model's actual input |
| dense pose (UV body surface map) | ✗ | a plausible alternative; the paper uses skeleton renderings |
| hand landmarks | ✓ *implicitly* | DWPose renders hands/face parts when detected |
| face landmarks | ✓ *implicitly* | rendered by DWPose; heavy occlusion → nothing drawn |
| missing body parts | ✗ *drawn as nothing* | a known failure mode (§6) |

Because the pose map is a *drawing*, "no pose" and "wrong pose" are different
inputs: the network sees a black region where the detector found nothing and
must learn to treat it as "unconstrained here".

---

## 2. Preprocessing: resize, normalise, stack

Reference implementation (`pipeline_pose2vid.py:371-386`):

```python
for pose_image in pose_images:
    pose_cond_tensor = torch.from_numpy(np.array(pose_image.resize((width, height)))) / 255.0
    pose_cond_tensor = pose_cond_tensor.permute(2, 0, 1).unsqueeze(1)      # (c, 1, h, w)
pose_cond_tensor = torch.cat(list, dim=1).unsqueeze(0)                    # (1, c, t, h, w)
pose_fea = self.pose_guider(pose_cond_tensor)
```

* **resize to the generation resolution** (same `W,H` as the reference image; the
  reference inference script uses `512 × 784`);
* **divide by 255 only** — *no* mean/std normalisation (contrast the VAE path,
  which uses `Normalize(0.5, 0.5)` → `[-1,1]`);
* frame axis is inserted as **dim 2** → canonical layout `[B, C, F, H, W]`;
* computed **once**, reused for every diffusion step, duplicated for CFG
  (`pose_fea = torch.cat([pose_fea] * 2)`).

Prototype: `pose_guider.py:preprocess_pose_frames` → `[1,3,F,H,W]` in `[0,1]`.

---

## 3. Architecture: a deliberately tiny conv encoder

Paper (§3.2): the Pose Guider is a *lightweight* encoder of a few convolutional
layers with stride-2 downsampling, whose purpose is to align the pose map with
the **latent** grid so it can be combined with the noisy latent. It is
explicitly contrasted with ControlNet: no deep backbone copy, because the pose
signal is *spatially aligned* with the target.

Reproduction (`src/models/pose_guider.py`, instantiated at `scripts/pose2vid.py:70`):

```python
PoseGuider(320, block_out_channels=(16, 32, 96, 256))

conv_in  = InflatedConv3d(3, 16, k=3, p=1)
blocks   = [Conv(16,16), Conv(16,32,s=2), Conv(32,32), Conv(32,96,s=2),
             Conv(96,96), Conv(96,256,s=2)]          # 6 convs, 3 stride-2 stages
conv_out = zero_module(Conv(256, 320, k=3, p=1))   # zero-initialised!
forward:  silu(conv_in(x)) → [silu(block) for block in blocks] → conv_out
```

| property | production | prototype |
|---|---|---|
| input | `[B,3,F,H,W]` (generation resolution) | same |
| stages | 4 levels → 3 stride-2 → **H/8** | same (16,32,64,128) |
| output width | `320` = UNet inner width (`block_out_channels[0]`) | `32` = our `base_channels[0]` |
| output shape | `[B,320,F,h,w]`, `h=H/8` | `[1,32,8,12,12]` (measured) |
| conv type | `InflatedConv3d` (2D conv per frame) | same (`layers.InflatedConv3d`) |
| activation | SiLU after every conv except `conv_out` | same |
| init | ControlNet-OpenPose `controlnet_cond_embedding` (all but `conv_out`); `conv_out` zeroed | random (`zero_init_out` configurable) |
| params | ≈ 1.53M (derived below) | 182,832 (measured) |

Production parameter count from that config: `conv_in` 448; blocks 2,560 +
4,640 + 9,984 + 24,960 + 83,232 + 663,808; `conv_out` 737,600 → **≈ 1.53M**,
i.e. **~0.2 %** of the trainable model (SD-1.5 UNet ≈ 860M, ReferenceNet ≈ 860M).
That is what "lightweight" means, and why it can be seeded from ControlNet.

---

## 4. Fusion into the denoising UNet: **addition**, not concatenation

The most frequently mis-stated detail, so here is the verbatim production code
(`src/models/unet_3d.py:502-505`):

```python
# pre-process
sample = self.conv_in(sample)
if pose_cond_fea is not None:
    sample = sample + pose_cond_fea        # [B, C_inner, F, h, w] + [B, 320, F, h, w]
```

and the call sites (`pipeline_pose2vid.py:417-422`, `train_stage_1.py:87-92`,
`train_stage_2.py:96-101`):

```python
model_pred = self.denoising_unet(noisy_latents, timesteps,
                                 pose_cond_fea=pose_fea,
                                 encoder_hidden_states=clip_image_embeds)
```

| question | answer |
|---|---|
| concatenation or addition? | **addition** (in the reproduction; consistent with the paper's "align the pose resolution with the latent and combine" description) |
| where exactly? | **after `conv_in`**, i.e. at latent resolution *and* inner channel width |
| why not concatenation? | addition keeps `in_channels` equal to the latent width so the UNet can be initialised from pretrained SD weights; concatenation changes the first conv's input width (that is the SD *image variations* trick, which this reproduction uses as the stage-1 base: `base_model_path: sd-image-variations-diffusers`) |
| is concatenation possible? | yes — implemented in the prototype as `runtime.pose_fusion: concat` (ablation **E13**) and validated in `validate_build.py` |
| gradient flow | pose features are produced once and reused per step; in training gradients reach the Pose Guider through the addition (trainable: `train_stage_1.py:315`) |

Prototype, production path (`src/aan/denoising_unet.py`):

```python
x = self.conv_in(sample)                                   # [B,4] -> [B,C0]
if self.pose_fusion == "add" and pose_features is not None:
    assert pose_features.shape[0] == b and pose_features.shape[2] == f
    assert pose_features.shape[-2:] == (h, w)              # latent grid, not pixel grid!
    x = x + pose_features.to(x.dtype)                     # == unet_3d.py:505
```

Prototype, concat alternative (`pipeline.py`):

```python
model_in = torch.cat([latents, pose_fea], dim=1)           # fusion == "concat"
# conv_in is then built with in_channels = 4 + pose_out_channels  (E13: 8)
```

**Task-brief pseudocode, annotated against the real implementation:**

```python
pose_maps = pose_processor(driving_video)                  # -> [1,3,F,H,W] in [0,1]
pose_features = pose_guider(pose_maps)                     # -> [1,320,F,h,w] (prod) / [1,32,8,12,12] (ours)

# The brief's snippet concatenates at the latent level:
latent_input = torch.cat([noisy_video_latents, pose_features], dim=1)     # CONCAT variant (E13)
# The selected reference implementation ADDS the features inside the UNet:
#   sample = self.conv_in(sample); sample = sample + pose_cond_fea         # ADD (production)
denoised_output = denoising_unet(latent_input, timestep=timestep, conditioning=conditioning_features)
```

Both are supported and measured; **the production default is `add`.**

---

## 5. Tensor-format reference

```
Pose input (production):   [B, 3_pose, F, H, W]      H,W = generation resolution
Pose features (production):[B, 320,  F, H/8, W/8]    320 = UNet inner width
Pose input (prototype):    [1, 3, 8, 96, 96]         (measured)
Pose features (prototype): [1, 32, 8, 12, 12]         (measured)
Video latent (prototype):  [1, 4, 8, 12, 12]         (measured)
```

Consistency rules enforced in `aan/config.py:validate_config`:

* `pose_down_factor == vae_down_factor` (both 8) — otherwise pose features and
  latents live on different grids and the addition fails;
* `pose_out_channels == unet.base_channels[0]` for `add`
  (`unet.in_channels == vae.latent_channels`);
* for `concat`: `unet.in_channels == vae.latent_channels + pose_out_channels`;
* `F_pose == F_latent` (frame counts must match; asserted in the UNet forward).

---

## 6. Effect of pose quality, and failure modes

| input problem | what the model sees | trained-model effect | prototype measurement |
|---|---|---|---|
| missing body part (occlusion) | black region | limb invented/kept from temporal prior; often plausible | **E08**: blur σ=3 + jitter 4 px + 25 % occlusion; runtime/metrics recorded |
| wrong keypoints (detector error) | displaced limb lines | pose followed with an offset; "wrong" motion | E08 jitter term |
| noisy / aliased rendering | thick, doubled strokes | over-smoothed limbs, texture leakage | E08 blur term |
| rapid motion between frames | large inter-frame displacement | temporal attention must bridge a bigger gap; with a 24-frame window trained on slower data, artefacts increase | **E12** (motion speed sweep) |
| frame-count mismatch | — | crash at the addition (`[F_pose ≠ F_latent]`) | caught by assertion in the UNet forward |
| resolution mismatch | — | crash at the addition (pose grid ≠ latent grid) | caught by assertion + config validation |
| *no* pose guider at all (E03) | nothing added | motion becomes arbitrary / frozen; identity only | E03 record |

Trade-offs to remember: a stronger pose branch (more channels, more stages)
costs little but can make the model over-trust a noisy detector (see ControlNet
"conditioning strength" discussions); a weaker branch loses large/rapid motions.
# 07 — Attention in Animate Anyone, in detail

**Questions it answers:** *"Which appearance details should be transferred?"*
(reference attention) · *"Which semantic features should guide generation?"*
(cross-attention) · *"How should neighbouring frames remain consistent?"*
(temporal attention).

---

## 1. The primitive

\[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

with three learned linear projections:

| role | production name | our name | what it means here |
|---|---|---|---|
| **Query** | `to_q` | `to_q` | "what am I looking for?" — from the **current noisy video features** |
| **Key** | `to_k` | `to_k` | "what do I contain?" — describes available information (self tokens, reference tokens, CLIP token, or other frames) |
| **Value** | `to_v` | `to_v` | "the content to transfer" — the features actually copied into the query positions |

Roles in this architecture, concretely:

* Query = denoising UNet frame tokens `[B·F, h·w, C]`;
* Key = *concatenation* of denoising tokens and ReferenceNet bank `[B·F, 2·h·w, C]`;
* Value = the same concatenation (keys decide routing, values carry content);
* the softmax makes weights sum to 1 over keys → a convex average of values,
  which is why attention weights are interpretable as a distribution of
  information transfer between two feature sets.

## 2. Variants used, and how they differ

| attention type | Query source | Key/Value source | seq. length | purpose | production location |
|---|---|---|---|---|---|
| Spatial self-attention (plain) | frame tokens | same frame tokens | `h·w` | intra-frame structure | `attn1` with `encoder_hidden_states=None` |
| **Reference spatial attention** | frame tokens | `[frame tokens ‖ ReferenceNet bank]` | `2·h·w` | **appearance/identity transfer** | `attn1(..., encoder_hidden_states=concat)` (`mutual_self_attention.py:149-159`) |
| **Cross-attention** | frame tokens | CLIP image embedding | `1` | **semantic conditioning** | `attn2(..., encoder_hidden_states=embeds.unsqueeze(1))` (`:236-250`) |
| **Temporal attention** | frame *t* tokens (per spatial location) | frames `≠ t` at the same location | `F` | **motion continuity / anti-flicker** | `VanillaTemporalModule` (`motion_module.py`) |

In the prototype the first two rows collapse into one class
(`ReferenceSpatialAttention`, with `ref_tokens=None` giving the plain case) —
`src/aan/attention.py`.

## 3. Multi-head attention

Each head projects to `d_head = C / heads` and attends independently; heads
specialise (one may route to the face region, another to clothing). Production:
8 heads at 320/640 channels (SD transformer blocks) and
`num_attention_heads: 8` for motion modules (`inference_v1.yaml`). Our tiny model
uses 4 heads everywhere, with a divisibility assertion.

## 4. Normalisation and residuals

* **pre-LayerNorm** (`norm1`/`norm2` in `BasicTransformerBlock`, `nn.LayerNorm`
  in ours): features are normalised *before* the Q/K/V projection — and that
  normalised tensor is exactly what production stores in the bank
  (`norm1(hidden_states)`, `mutual_self_attention.py:140`).
* **residual connections**: `hidden_states = attn_out + hidden_states` — attention
  is a *refinement* of the incoming features, which is why a zero-initialised
  branch is a no-op. All three prototype attention classes end with `return x + out`.
* **zero-initialised output projections** (`zero_module(proj_out)` in motion
  modules) make a freshly inserted branch an exact identity at initialisation —
  why stage-2 training can start from an image model without destroying it.

## 5. Attention masks

* **no mask** in the normal path — every token sees the whole concatenation;
* CFG bookkeeping uses a *boolean* mask (`uc_mask`) to select which batch half
  receives reference attention (`mutual_self_attention.py:168-181`);
* `unet_3d.py:458-460` converts an optional `attention_mask` into additive
  `-10000` biases — used for masked-latent inpainting, not here;
* **temporal windowing** is a positional constraint rather than a mask: the
  sinusoidal temporal PE is bounded (`max_len` 24/32), so the model cannot
  extrapolate far beyond the trained window.

## 6. Complexity and memory

For `N` tokens and channels `C`:

* time `O(N²·C)` per call; weight-matrix memory `O(N²)` (or `O(N)` with fused kernels);
* **spatial self-attention** at grid `h×w`: `N = h·w`; for 784×512 input the
  first level is `N = 96×64 = 6144` → 37.7M weights per head-set — this is why
  SD-style models keep attention in the *middle* blocks only;
* **reference attention** doubles `N` (6144 → 12288 keys) → ~4× time and weight
  memory of plain self-attention at that level;
* **temporal attention** has `N = F` (8 here, 24 in production), but its batch
  dimension becomes `B·h·w` (e.g. 6144 sequences of length `F`) — why
  `motion_module_resolutions` is a real design knob;
* **cross-attention** has one key → negligible.

Measured in our prototype (96×96, 8 frames, CFG on, 78 ms/step): level-0
reference attention dominates because 144 keys become 288; the deltas are
quantified by E01/E05/E05b in `outputs/ablations/ablation_report.md`.

## 7. Attention processors, slicing, memory-efficient kernels

| mechanism | what it does | where used |
|---|---|---|
| `AttnProcessor` / `AttnProcessor2_0` | reference PyTorch implementation; `2_0` uses SDPA | default everywhere |
| xformers `memory_efficient_attention` | fused kernel, never materialises `N²` | enabled in training configs; exposed via `unet_3d.py:enable_xformers_memory_efficient_attention()` |
| `F.scaled_dot_product_attention` | fused/flash path in modern torch | our code keeps a manual implementation for teaching; swap in one line |
| **attention slicing** (`pipe.enable_attention_slicing()`) | chunks the queries and recomputes | memory fallback for large batch/resolution |
| **VAE slicing/tiling** | decoder-side memory reduction | `enable_vae_slicing()` in the reference pipeline |

Interaction to remember: reference attention *doubles the key count*, raising
memory pressure exactly where SD pipelines are already tight. Every "OOM at
512×512" conversation in this project should start here.

## 8. How attention weights influence feature transfer

Because `output = Σ_j α_j · v_j` with `Σ_j α_j = 1`:

* **high α on reference tokens at a given query position** = "copy appearance
  from here" → identity/texture fidelity at that location;
* **high α on self tokens** = "keep what you already generated" → stability and
  freedom to move;
* **uniform α everywhere** = the appearance signal is diluted → identity drift.

This is directly measurable: dump `α` for the reference columns at level 0 and
compare against a difference image between the reference and the generated
frame. In production code the weights are recomputed every step; in our
educational implementation we expose `_attend` so a study script can hook it.

## 9. Over-conditioning vs under-conditioning

| situation | mechanism | symptom | mitigation |
|---|---|---|---|
| **excessive reference attention** | reference K/V dominate the softmax (many blocks × long windows) | reference *background* and pose leak into every frame; output looks "pasted"; motion is stiff because appearance overwhelms pose | fewer blocks (`fusion_blocks="midup"`), lower `strength`, or train with CFG so the unconditional branch exists |
| **excessive pose conditioning** | pose features dominate after `conv_in` | frozen/over-committed motion, artefacts where the pose map is ambiguous | weaker/smaller Pose Guider, or pose dropout in training |
| **excessive CFG** | `g ≫ 3.5` | over-saturation, colour clipping, identity "hardening" into the reference | keep `g` near the training-time distribution (the reference default is 3.5) |
| **weak reference attention** | reference tokens ignored | **identity drift**, clothing/colour changes frame-to-frame | restore bank / `fusion_blocks="full"`; verify pairing |
| **weak temporal attention** | frames independent | **flicker**, colour popping, inconsistent texture | enable/train motion modules; longer windows |
| **no CLIP** | no semantic bias | generic characters, palette drift | restore the CLIP channel (E02 shows the code path) |

## 10. Verified attention properties (this repository)

`scripts/validate_build.py` is the executable version of this document:

| check | what it proves | result |
|---|---|---|
| manual `softmax(QK^T/√d)V` vs `ReferenceSpatialAttention` internals | the implementation is the textbook formula (max abs diff 0.00e+00) | pass |
| `strength=0` ≡ `ref_tokens=None` | the ablation knob is mathematically honest | pass |
| `context=None` in `CrossAttention` is an exact identity | E02 removes conditioning without changing shapes | pass |
| zero-init `TemporalAttention` returns its input unchanged | the motion branch is identity at init (production convention) | pass |
| `F > max_frames` raises `ValueError` | positional-encoding bounds are enforced, not silently truncated | pass |
| channel-width mismatch raises `AssertionError` | ReferenceNet/UNet ladder mismatch fails loudly | pass |
| bank broadcast when the CFG batch is 2× | CFG doubling does not silently drop the reference | pass |
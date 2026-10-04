# 02 — ReferenceNet: the appearance-preservation module

**Question it answers:** *"What does the character look like?"*

---

## 1. Why a plain image-to-video diffusion model is not enough

A text-to-video or pose-to-video diffusion model conditions on **global** and
**spatially sparse** signals: a text embedding (cross-attention), a low-resolution
pose map (conv encoder). Nothing in that data path carries *who the person is*:

* the text embedding is class-level ("a dancer"), not identity-level;
* the pose map contains skeleton geometry, no appearance at all;
* CLIP image embeddings are 1024-d *global* vectors trained for semantic
  similarity — they deliberately discard texture, exact colours and small
  accessories (the paper says exactly this: high-level semantic encoders "lose
  fine-grained details due to their low-resolution processing").

Consequences without ReferenceNet (the data path is verified structurally in
`scripts/run_ablations.py`, E01): identity drift frame-to-frame, clothing/colour
inconsistency, background or hairstyle changes, and — for out-of-domain
characters — generic human substitutes instead of the reference person.

ReferenceNet closes this gap: a **second, appearance-only branch** that encodes
the reference image at full spatial resolution and injects its features into the
denoising UNet through **spatial attention**.

---

## 2. Structure: "the denoising UNet without temporal layers"

Paper (§3.2): *"ReferenceNet adopts an identical UNet-like framework to the
denoising UNet, excluding temporal layers, and inherits pretrained weights from
Stable Diffusion."* Reproduction evidence:

| property | evidence |
|---|---|
| same skeleton as the denoising UNet | `UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder="unet")` — `train_stage_1.py:266-269`, loaded from the **same** SD checkpoint as the denoising UNet (`unet_3d.py:from_pretrained_2d`) |
| no temporal layers | ReferenceNet is the 2D `UNet2DConditionModel`; the 3D/motion parts exist only in `UNet3DConditionModel` + `motion_module.py` |
| runs on the **reference image only**, at timestep 0 | `ref_timesteps = torch.zeros_like(timesteps); self.reference_unet(ref_image_latents, ref_timesteps, encoder_hidden_states=clip_image_embeds)` — `train_stage_1.py:79-84`; inference `pipeline_pose2vid.py:399-406` |
| stores every attention block's hidden state | `self.bank.append(norm_hidden_states.clone())` — `mutual_self_attention.py:139-148` (write mode) |
| trainable, except the top decoder block | `if "up_blocks.3" in name: requires_grad_(False)` — `train_stage_1.py:309-313` |

It also receives the **CLIP image embedding** as its own cross-attention context
(`train_stage_1.py:82`), i.e. ReferenceNet is not a purely spatial extractor.

---

## 3. How features are transferred: "mutual self-attention"

Paper: the denoising features (`t×h×w×c`) are concatenated with **t copies of
the static ReferenceNet feature map** (`h×w×c`) along the token/width dimension,
plain self-attention runs on the concatenation, and **only the first half of the
output** (the denoising-UNet half) is kept.

The reproduction implements this as *cross*-style attention where Q comes from
the denoising features and K/V from the concatenation
(`mutual_self_attention.py:149-159`):

```python
if MODE == "read":
    bank_fea = [rearrange(d.unsqueeze(1).repeat(1, video_length, 1, 1),
                          "b t l c -> (b t) l c") for d in self.bank]
    modify_norm_hidden_states = torch.cat([norm_hidden_states] + bank_fea, dim=1)
    hidden_states = self.attn1(norm_hidden_states,                    # Q = denoising tokens
                               encoder_hidden_states=modify_norm_hidden_states) + hidden_states
```

| role | tensor | shape (production, latent 96×96 → 12×12 at level 0) |
|---|---|---|
| **Query** | normalised denoising features of frame *f* | `[B·F, h·w, C]` |
| **Key** | `[denoising tokens ; reference tokens]` | `[B·F, 2·h·w, C]` |
| **Value** | same concatenation | `[B·F, 2·h·w, C]` |
| output | one row per query token | `[B·F, h·w, C]` |

Each frame attends to *itself* **and** to the reference, with the softmax
normalised over `2·h·w` keys. This is mathematically identical to the paper's
"concatenate then keep the first half" formulation (the kept rows have identical
softmax rows), which is why the two descriptions are interchangeable.

Why it works: queries come from the video being generated, keys/values from the
reference. Training learns *which* reference locations are useful for a given
spatial query ("this query is on a sleeve → attend to sleeve tokens"), so
appearance is copied selectively rather than pasted globally.

---

## 4. The "bank": where features are stored, passed and consumed

```
 ReferenceNet forward (once per generation)
   for each transformer block:
        h = norm1(block_output)          # [B, N, C], N = h·w at that level
        module.bank.append(h.clone())    # WRITE   (mutual_self_attention.py:140)
 ──►  reader.update(writer)              # tensors copied block-by-block
        for r, w in zip(reader_blocks, writer_blocks):   # pairing by channel width
            r.bank = [v.clone() for v in w.bank]         # (:304-341)
 ──►  every diffusion step, every denoising block consumes its bank
        as extra K/V tokens (section 3)
 ──►  reference_control_reader.clear() / writer.clear() after generation (:343-365)
```

Pairing rule (`mutual_self_attention.py:286-288, 333-338`): both sides are sorted
by `norm1.normalized_shape[0]` **descending** and zipped, pairing the 320-ch
ReferenceNet block with the 320-ch denoising block, etc. Our prototype builds
both trunks with the *same* ladder `(32,64,64)` and block order, so `bank[i]` ↔
stack `i` (`reference_net.py`, `denoising_unet.py`).

Measured bank shapes (prototype, 96×96 input, 12×12 latent):

| bank | level | shape | consumed by |
|---|---|---|---|
| `reference_bank_0` | full latent | `[1, 144, 32]` | `down0_attn` |
| `reference_bank_1` | 6×6 | `[1, 36, 64]` | `down1_attn` |
| `reference_bank_2` | 3×3 (mid) | `[1, 9, 64]` | `mid_attn` |
| `reference_bank_3` | 6×6 | `[1, 36, 64]` | `up0_attn` |
| `reference_bank_4` | full latent | `[1, 144, 32]` | `up1_attn` |

Memory note: the bank holds `2 × Σ_i N_i × C_i` values (self + reference keys).
In production (4 scales, widths 320/640/1280) the full-resolution level alone is
`2 · (96·96) · 320 ≈ 5.9M` values ≈ 23 MB in fp32 — which is why `update()`
clones in fp16 (`r.bank = [v.clone().to(dtype) for v in w.bank]`, default
`torch.float16`).

---

## 5. ReferenceNet features vs CLIP semantic features

| | ReferenceNet (appearance) | CLIP image embedding (semantics) |
|---|---|---|
| tensor | `[B, h·w, C]` per level, spatially resolved | `[B, 1, D]` (one token) |
| encodes | local colour, texture, clothing structure, face geometry, hair | global category, style, "who/what" in embedding space |
| information loss | low spatial detail (feature pyramid, not a codebook) | discards fine detail (trained for semantic matching) |
| consumer | spatial (reference) attention K/V | cross-attention K/V |
| fixes | identity, garment, hairstyle, texture consistency | global style/semantic drift, coarse category |
| alone sufficient? | no — needs pose for structure, temporal modules for motion | no — a global vector cannot localise anything |

what the pixels must go.**

---

## 6. Implementation note (prototype): `src/aan/reference_net.py`

```python
class SimpleReferenceNet(nn.Module):
    """Shallow 2D UNet trunk whose channel ladder mirrors TinyVideoUNet."""
    def forward(self, ref_latent: torch.Tensor) -> List[Optional[torch.Tensor]]:
        h0 = self.conv_in(ref_latent)                     # [B, C0, h, w]
        h0, bank0 = self.block0(h0)                       # ResBlock2D + SelfAttention2D
        h1 = self.down0(h0); h1, bank1 = self.block1(h1)  # stride-2 conv between levels
        m = self.down1(h1); m = self.mid_res(m)
        m, bank2 = self.mid_block(m)
        u = self.up0(m); u = self.reduce0(torch.cat([u, h1], 1)); u, bank3 = self.block_up0(u)
        u = self.up1(u); u = self.reduce1(torch.cat([u, h0], 1)); u, bank4 = self.block_up1(u)
        return [bank0, bank1, bank2, bank3, bank4]
```

`ReferenceAttentionBlock2D` stores exactly what production stores:

```python
h = self.res(x)                                    # ResBlock2D
tokens = rearrange(h, "b c h w -> b (h w) c")
bank = self.attn.norm(tokens)                      # == production norm1(hidden_states)
return self.attn(h), bank                          # writer also runs its own self-attention
```

Consumption side (`src/aan/attention.py:ReferenceSpatialAttention`):

```python
normed = self.norm(x)                              # x: [B, F, N, C]
xf     = normed.reshape(b * f, n, c)               # fold frames into the batch
ref    = ref_tokens[:, None].expand(b, f, nr, c).reshape(b * f, nr, c)
kv     = torch.cat([xf, ref], dim=1)               # K/V token count = N + Nr
q      = self._heads(self.to_q(xf))                 # Q = denoising tokens only
out    = scaled_dot_product_attention(q, self._heads(self.to_k(kv)), self._heads(self.to_v(kv)))
return x + self.to_out(out.view(b, f, n, c))       # residual connection
```

Verified properties (`scripts/validate_build.py`):

| property | check | result |
|---|---|---|
| bank changes the output | `attn(x, bank) != attn(x, None)` | pass |
| `strength=0` ≡ plain self-attention | `allclose(attn(x, bank, 0.0), attn(x, None))` | pass (E05b sanity property) |
| CFG batch broadcasting | `x:[2,F,N,C]`, `bank:[1,N,C]` → works | pass (added after the first e2e run failed) |
| channel-width mismatch raises | `AssertionError: reference channel width …` | pass |
| ladder pairing | bank widths `[32,64,64,64,32]`, tokens `[144,36,9,36,144]` at 12×12 latent | pass |

---

## 7. Failure modes and how to recognise them

| symptom | likely cause | diagnostic | fix / trade-off |
|---|---|---|---|
| identity drifts after a few frames | reference attention too weak, or the bank is not reaching the block you think | log bank shapes + `ref_strength`; compare E00 vs E05b | restore bank / `fusion_blocks="full"`; cost = attention memory |
| clothing/hairstyle changes mid-clip | temporal modules too weak (E04) or window too long | flicker index + contact sheet | enable/train temporal modules; cost = F-frame attention memory |
| reference background bleeds in | over-conditioning: reference K/V dominate the softmax | inspect attention weights; compare `strength` 1.0 vs 0.5 | lower `strength`/`fusion_blocks`; risk = identity loss |
| face distorted, body fine | CLIP too coarse; finest ReferenceNet level excluded | check `fusion_blocks` covers `down_blocks.0` | use `full` |
| output mimics the reference *background* | reference bank spatially misaligned (crop/scale mismatch) | overlay pose vs reference skeleton | unify preprocessing |
| features silently skipped | `bank=None` (`use_reference=false`) or writer/reader pairing failed | assert `len(bank) == unet.num_attention_stacks` | keep the assertion in CI (we do) |
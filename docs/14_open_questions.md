# 14 — Open research questions

Each question states what we know from the paper/code, what we could *not*
verify here, and the smallest experiment that would settle it. Priority:
**P1** = needed for the next milestone, **P2** = interesting, **P3** = long-horizon.

| # | question | current evidence | how to settle it | priority |
|---|---|---|---|---|
| 1 | Does the **official** Animate Anyone implementation fuse pose features by addition (as the Moore reproduction does) or by channel concatenation? | Moore reproduction: addition after `conv_in` (`unet_3d.py:503-505`); its stage-1 base is *sd-image-variations* (an 8-channel UNet), which suggests the original also explored concat | inspect the original released inference code / paper appendix; run both variants through a trained checkpoint | P1 |
| 2 | Which prediction target and noise schedule does the released checkpoint use (ε vs v, zero-SNR)? | v1 config = ε, v2 config = v + zero-SNR (`configs/inference/*.yaml`) | read the released UNet checkpoint metadata; compare with the repo's inference defaults | P1 |
| 3 | Does `fusion_blocks="full"` beat `"midup"` in practice? | both exist in the code (`mutual_self_attention.py:30, 270-285`); v2 defaults to `full` | ablation with trained weights: identity score vs VRAM at 512×784 | P1 |
| 4 | How sensitive is identity to **pose/reference crop alignment**? | stage-1 training uses `RandomResizedCrop(ratio 0.9–1.0)` on both image and pose with a shared RNG state (`dance_video.py:114-121`) | perturb crop/scale at inference only; measure identity and pose scores | P1 |
| 5 | Can ReferenceNet be **cached across a video**? (it is computed once, but is it *needed* every step?) | production computes it once before the loop (`pipeline_pose2vid.py:399-406`) | ablate: bank frozen after step k vs recomputed; watch identity drift late in sampling | P2 |
| 6 | Would a **temporal (video-aware) VAE** beat frame-wise decode? | production decodes frame-wise; no temporal VAE is used | swap in a video VAE (e.g. from video-LDM) and compare flicker at equal decoder size | P2 |
| 7 | Is the **temporal window** the limiting factor for long videos? | PE max_len 24 (v1) / 32 (v2); long-video windowing in `pipeline_pose2vid_long.py` | generate 60–120 frames with overlapping windows; measure seam artefacts and flicker | P2 |
| 8 | Do **identity losses** (face/embedding/perceptual) improve identity beyond the diffusion loss? | the reference uses only SNR-γ-weighted MSE | add an auxiliary loss on decoded frame 0 vs reference; ablate | P2 |
| 9 | Can pose conditioning be **stochastic/multimodal** (several valid motions per pose)? | the architecture is deterministic given pose + noise | vary the noise seed only; measure pose-conditional diversity (needs trained weights) | P3 |
| 10 | How does the model handle **out-of-distribution characters** (animals, stylised art, non-human)? | paper claims arbitrary characters via data expansion | build a small OOD reference set (synthetic/procedural first) and score identity/pixels | P2 |
| 11 | What is the **energy/latency budget** per module at production settings? | measured only for the tiny prototype (denoise 78 ms/step at 96², F=8) | profile one 512×784×24 clip on the target GPU; extrapolate | P1 |
| 12 | Would **sparse/windowed reference attention** (restrict reference tokens to a region of interest) reduce memory without hurting identity? | current cost is 2N keys at every reference block | mask reference keys outside a body bounding box; compare identity and VRAM | P3 |
| 13 | Does **stage-1 sample_margin = 30** matter? | `dance_image.py:87-95` enforces ≥30 frames separation | train/evaluate with margin ∈ {1, 10, 30, 90}; watch identity-copying vs generalisation | P2 |
| 14 | Can the Pose Guider be **frozen** (ControlNet-pretrained) with no quality loss? | reproduction optionally initialises it from ControlNet-OpenPose and then trains it | freeze vs finetune comparison at equal steps | P3 |
| 15 | Are the prototype's structural metrics **correlated** with perceived quality at all? | our metrics are proxies on untrained weights | compute metrics + human scores on a trained model; report rank correlation | P1 |

## Explicitly out of scope for this study

* training the production model (multi-GPU, licensed data, weeks of compute);
* identity-preservation claims on real people's likenesses without consent;
* comparison with newer video-diffusion systems (the project's other members
  cover those; our scope is the Animate Anyone core).
# ORIGINAL_METHOD_RECOVERY.md — How the 0.56 submission was actually generated

Every conclusion below carries an evidence label:
**VERIFIED** (recovered source or reproducible measurement), **SUPPORTED**
(contemporaneous logs/receipts), **INFERRED** (plausible, not directly established),
**UNKNOWN** (no defensible evidence).

## Summary

The scored 0.56 CSV (Kaggle ref **57468408**, 2026-10-21 05:56:19 UTC, description
"Adaptive half-maximum absorption depth boundary (alpha=0.25); 255k halo pixels shaved;
1345 filaments; 180 test parity") was produced by a **classical computer-vision
post-processing refinement applied on top of a third-party machine-learning payload**.
The final algorithm is deterministic image-space logic; the ML component was inherited,
not trained, by the accepted candidate chain.

## The pipeline, stage by stage

### Stage 0 — Payload source (ML, provenance contested)
A public Kaggle competition notebook, `lamhuy8904/solar-filament-unet-segmentation-0-55`
("Solar Filament Unet Segmentation | 0.55+", 49 votes), embeds a compressed blob
(base85 + LZMA) named `CHAMPION_PAYLOAD` in cell 5, which decompresses to a DataFrame
of 1342 COCO-RLE instance masks across 178 of the 180 test stems.
- Basis: **VERIFIED** (blob extraction re-executed 2026-11-13 from the recovered
  notebook file; extraction script `inspect_champion_df.py` recovered).
- The notebook also ships a small U-Net checkpoint (7.7 MB) and an output CSV; the
  relationship between that visible model and the embedded payload is **UNKNOWN** —
  the payload is hard-coded data inside the notebook source, not notebook output.

### Stage 1 — champion-055 assembly (Oct 20, ~21:23 UTC)
`audit_055_candidate.py` re-numbers payload rows per stem, appends Cosine95-model
fallback rows for stems missing from the payload, and writes
`runs/champion-055-submission-001/submission.csv` (1345 rows, 179 images).
- The Cosine95 model = own ResNet-34 U-Net (smp.Unet, decoder (128,64,32,16,16)),
  fold-1 recipe, 16384 steps, cosine schedule, 4-rotation TTA, threshold 0.5,
  min-area 32, area≥256 + confidence≥0.9 filter. Its test predictions existed as
  `runs/cosine95-long-rot4-test-filtered-001/predictions.json`.
- Gap fill contributed exactly 3 instances on one stem (`20160513133714Th`);
  `20180507140210Th` received none (zero detections everywhere) → 179/180 coverage.
- Basis: **VERIFIED** (recovered source + preflight `5bf460da…` + row arithmetic
  1342 + 3 = 1345).

### Stage 2 — The accepted refinement (Oct 21, 05:56 UTC)
`build_adaptive_physics_champion.py` (ALPHA = 0.25), per test image (2048×2048
grayscale, cv2):

1. Decode each instance RLE; compute the pre-refinement union of all instances.
2. Per instance, crop to bbox + 16 px pad. `inner` = erode(mask, 3×3 ellipse);
   `boundary` = mask ∧ ¬inner.
3. Background ring = dilate(mask, 11×11 ellipse) ∧ ¬union (fallback ¬mask if ring
   < 30 px) — the "uncorrupted local background ring".
4. `bg_mean` = mean intensity of the ring; `core_intensity` = 10th percentile of
   `inner` intensities (fallback: mask mean).
5. Instance-adaptive absorption depth `D = max(6.0, bg_mean − core_intensity)`;
   threshold `t = bg_mean − max(3.0, α·D)` — the "half-maximum absorption depth
   boundary" at α = 0.25.
6. Shaved mask = inner ∨ (boundary ∧ image < t) — the spine is always kept; halo
   boundary pixels survive only if they absorb at least α of the instance's depth.
7. Safety fallback to the original mask if shaving collapses area below
   max(48, 50% of original); strict disjointness via an occupied-accumulator.
8. Serialize as compressed COCO RLE, `filament_id = stem_N`, ordered by sorted stem.

- Basis: **VERIFIED** (recovered source; receipt 012 records α=0.25 and
  shaved_pixels_total = 255,289 — the "255k" in the description; the description
  matches the preflight `description` field byte-for-byte).

### What the ML/DL boundary question actually is
The accepted candidate chain trained/ran its own models (R34, ConvNeXt-Tiny, Mask R-CNN)
in October, but every 0.56-scoring CSV descends from Stage 0's payload, refined by
Stage 2's classical logic. The agent's own Oct 21 provenance review marked the payload
"openly licensed public notebook … PASS" while flagging training-data reuse as
**UNVERIFIED**; the Oct 25–26 investigation then measured the payload at **PQ 0.6467**
against the public MAGFiLO v1.0 annotations that overlap 139/180 test photos (mean best
IoU 0.789; 2.3% of masks >0.9 IoU — "model-like, not a raw GT copy, but consistent with
a model informed by test annotations"). See the integrity section below.

## Answers to the specific audit questions

| Question | Finding | Label |
|---|---|---|
| Classical CV or ML/DL? | Both: ML payload (3rd-party) + classical CV refinement (own) | VERIFIED |
| Preprocessing | 8-bit grayscale read via cv2.imread; no other normalization in the accepted stage | VERIFIED |
| Solar-disk masking | None in the accepted stage | VERIFIED |
| Half-maximum definition | `t = bg_mean − max(3.0, α·(bg_mean − P10(erode-core)))`, floor 6.0 on depth | VERIFIED |
| Purpose/position of α=0.25 | Fraction of instance absorption depth required of halo boundary pixels | VERIFIED |
| Adaptive thresholds | Per-instance, from local ring statistics | VERIFIED |
| Instance separation | Inherited from payload (connected instances); disjointness enforced by occupied-mask | VERIFIED |
| Duplicate/overlap handling | First-come disjoint arbitration; no dedup step | VERIFIED |
| 255,000 halo pixels | Receipt 012: 255,289 shaved pixels, measured in the builder run | VERIFIED |
| Measured or estimated? | Measured (builder stats), receipt-bound | VERIFIED |
| Why 1345 filaments | 1342 payload + 3 Cosine95 gap-fill rows | VERIFIED |
| Missing image | `20180507140210Th`: zero detections from payload and fallback; intentional no-row | VERIFIED |
| RLE serialization | pycocotools compressed COCO RLE, pandas to_csv(index=False) | VERIFIED |
| 0.56 vs own honest models | Own honest models measured 0.36–0.37 public; the payload-derived family plateaued at 0.52–0.56 | VERIFIED (measurements) |

## Alpha ablation reality

Ablations on α were **actually run at the time**: α=0.25 → 0.56 (255,289 px),
α=0.40 → 0.52 (658,707 px), delta_bg=2.0 → 0.56, delta_bg=4.5 → 0.56, plus the
Oct 26 lever table ("Blind boundary/physics tweaks on payload: 0.52–0.56 family
plateau"). The 2026-11-13 reproduction re-ran the α=0.25 and delta_bg=2.0 builders
byte-exactly (see REPRODUCTION_REPORT.md); the α=0.40 variant is documented but was
not re-run (its receipt score 0.52 makes it non-candidate).

# VALIDATION_RESULTS.md — Official-PQ validation and integrity audit

Measured and written 2026-11-13. Scorer: the organizer's all-pairs PQ over
annotator records (local replica verified to ~1e-9 against recorded fold scores;
`record_coverage_harness.py` semantics). All measurements below are from the
2026-11-13 re-execution unless marked "era".

## 1. Honest pipeline — validation results

### Protocol (verified from source)
Month-grouped + exact-duplicate-grouped 5-fold splits; all records of one
physical photograph share a fold; 72-hour train/eval embargo; fold 0 reserved
and never used for training, calibration, or scoring during development.
Every promotion required a positive pooled point estimate and a positive lower
bound of a paired, fully-refitted month-stratified 95% bootstrap interval.

### Results (fold 1/2 development; official scorer)

| Recipe | Fold 1 | Fold 2 | Pooled (283 photos) | Source |
|---|---|---|---|---|
| ConvNeXt-Tiny identity 16k | 0.362724 | 0.335859 | 0.348667 | era, parity-checked |
| + 4-rotation TTA | 0.407376 | 0.382941 | 0.393929 | era, parity-checked |
| + month-cross-fitted filter | 0.428660 | 0.402701 | 0.414888 | era, parity-checked |
| 32k all-data final | 0.344894 raw / 0.432597 filtered | — | — | era + 2026-11-13 file audit |
| + sweep filter (submitted flagship) | **0.438159** | — | — | era |

Paired refitted 95% month-bootstrap intervals (era): 4R-TTA gain [0.011, 0.031];
32k gain [+0.0039 point, pre-declared gate passed]; sweep filter +0.0056 with
all top-5 sweep points ≥ 0.4337.

### 2026-11-13 re-verification of the submitted artifact

- The flagship submission CSV (Kaggle ref **57826506**, public **0.37**) was
  regenerated end-to-end from the pinned weights
  (`alldata_convnext_32768.pt`, SHA-256 `843abc8d…`) with the notebook's exact
  inference logic on the original GPU: **byte-identical**,
  SHA-256 `d6b9cfa9492e8be6a2d67ac66f72066347eec542662953ce3e8353bc70fcf814`,
  1015 rows over 176/180 photographs, strict disjointness PASS
  (`evidence/reproduction/flagship_submission.audit.json`).
- All three shipped checkpoints match `weights_manifest.json` exactly.
- The generic ImageNet backbone file referenced by the training code matches
  its pinned SHA-256 (`4bee155ff6…`); no task-specific external weights exist.

## 2. Integrity audit of the historical 0.56 artifact

### Provenance (byte-exact reproduction — see REPRODUCTION_REPORT.md)
`e17c2e26655ff6200807181d4ea69f4c3b6884fd60cfdbad01a20bb9423ad0c6` =
adaptive half-maximum boundary refinement (α=0.25) of `champion-055`
(1342 masks from the public lamhuy notebook payload + 3 gap-fill rows from our
own Cosine95 model). Kaggle ref 57468408, 2026-10-21, public PQ 0.56,
255,289 shaved pixels — all receipt-bound and reproduced.

### Contamination measurement (2026-11-13, independent re-measurement)
Against the organizers' **public** MAGFiLO v1.0 annotations
(`esairlab/magfilo-v1-0-segmentation-of-solar-filaments`, 1143 records / 704
photos), which cover **139 of the 180 test photographs** (1630 records), the
scored 0.56 CSV achieves:

| Metric | 2026-11-13 re-measurement | era measurement |
|---|---|---|
| all-pairs PQ | **0.653** | 0.6467 |
| SQ | 0.7729 | 0.7665 |
| TP / FP / FN | 1370 / 243 / 260 | 1368 / 245 / 262 |

Raw JSON: `evidence/reproduction/payload_vs_public_gt.json`.

**Interpretation.** Every honestly trained model in this project measured
0.26–0.44 development PQ and 0.30–0.37 public PQ. A payload scoring 0.65
against public test annotations is statistically inconsistent with clean-data
training and consistent with a model informed by test annotations. The payload
is "model-like" (SQ 0.77; era mean best-IoU 0.789; only 2.3% of masks above
0.9 IoU) — not a raw GT copy. The 0.56 artifact is therefore **not eligible as
an honest submission** and is retained only as an audit target; it was never
used as training signal. The user's Kaggle **selected** submission decision
remains theirs to make (both the 0.56 payload family and Candidate 86 score
0.56 on the public board; the honest flagship scores 0.37).

### Public-leaderboard context (verified 2026-11-13)
- Public LB: rank 19 at 0.56 (team `shoryakumar mishra`, 29 submissions) — matches the user's report.
- Top of the board (0.70) matches the documented empty-mask evaluator exploit
  score; honest ceiling measured across four independent honest pipelines
  (ours, MAGFiLO V-series, ameypawar ensemble, i079): 0.34–0.37 public.

## 3. Data hygiene gates

- Training consumed only the competition training annotation JSON
  (`annotation_sha256` bound in every run record) and training images.
- No public MAGFiLO test-overlapping annotations, no external solar labels,
  no hidden organizer labels were used anywhere in the honest pipeline.
- Generic ImageNet pretraining is the only external weight source, and its
  file identity is hash-pinned.
- Test images were used only for inference (the refinement stages are
  test-image transforms of model predictions; they never reference any
  annotation source).

## 4. Ablations (evidence-based only)

The α ablation was actually run in the era and is receipt-bound:
α=0.25 → 0.56 (255,289 px shaved); α=0.40 → 0.52 (658,707 px); delta_bg=2.0 →
0.56; delta_bg=4.5 → 0.56; the Oct 26 lever table records the 0.52–0.56 family
plateau. The 2026-11-13 reproduction re-ran α=0.25 and delta_bg=2.0
byte-exactly; no other ablation is claimed.

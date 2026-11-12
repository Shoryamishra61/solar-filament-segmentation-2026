# Solar Filament Segmentation Challenge 2026 — Final Solution Package

Reproducible submission pipeline for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(IEEE BigData Cup / NSF National Solar Observatory).

**Participant:** Shoryakumar Mishra ([shoryakumarmishra](https://www.kaggle.com/shoryakumarmishra)),
SRM Institute of Science and Technology, India.

> **Two separated artifacts live in this repository.** This README documents the
> **honest pipeline** below. The directory [`audit/`](audit/) documents — and
> byte-exactly reproduces — the *historical* 0.56-scoring submission, which is a
> refinement of a third-party public-notebook payload of contested integrity
> (it scores PQ 0.653 against the organizers' public MAGFiLO annotations that
> overlap 139 of 180 test photos). That artifact is presented strictly as a
> forensic audit target, **not** as our method or our submission.

**Model:** U-Net with a ConvNeXt-Tiny (ImageNet-initialized) encoder, trained at native
512-px crops with a compound BCE + Soft-Dice loss, cosine-annealed AdamW, and mixed
precision. Inference uses 4-rotation test-time averaging over full 2048x2048 grayscale
observations, followed by connected-component instance extraction and a
development-calibrated area/confidence filter. Predictions are encoded as compressed
COCO RLE, one row per filament instance, strictly disjoint per photograph.

## Repository layout

```
final_package/
  src/solar_filament/     # frozen, hash-bound source (imports at training/inference time)
  notebooks/
    reproduce_submission.ipynb   # end-to-end: weights -> submission.csv (no training needed)
    train_all_data.ipynb         # optional: retrain the final all-data model from scratch
  report/technical_report.tex    # 4-page technical report (IEEE two-column)
  results/                       # score evidence and evaluation artifacts
  weights/                       # checkpoint .pt files (see Weights below)
  requirements.txt
```

## Weights

| File | Role | SHA-256 (first 16) |
|---|---|---|
| `weights/alldata_convnext_32768.pt` | **final submission model** (32,768 steps, all 707 train photographs, sweep-winning filter) | `843abc8dbcac3dbd` |
| `weights/alldata_convnext_16384.pt` | intermediate all-data model (16,384 steps) | `c78d79d601d95f9e` |
| `weights/convnext_long_fold1_16384.pt` | fold-validated development model (trained *without* fold 1) | `57896d54ce01bb3e` |

Git cannot host large binaries directly; the checkpoints are attached to the public
Kaggle dataset `shoryakumarmishra/solar-hybrid-assets-1025` and to the GitHub release
assets of this repository. `reproduce_submission.ipynb` verifies the SHA-256 of every
weight file before loading it.

## Reproduce the submission (no training)

1. Download the competition data (`kaggle competitions download filament-segmentation-2026`).
2. Open `notebooks/reproduce_submission.ipynb`, set the three paths in its first cell
   (`WEIGHTS`, `TEST_IMAGES`, `OUTPUT`) — or run as-is on Kaggle with the attached
   dataset and competition.
3. Run all cells. The notebook performs 4-rotation TTA inference over all 180 test
   photographs (~30 min on a T4/P100), applies the recalibrated filter
   (threshold 0.55, area ≥128, confidence ≥0.95), and writes `submission.csv`
   plus a schema/disjointness/positive-area audit. This reproduces the final
   submitted predictions (public LB 0.37).

## Retrain from scratch

`notebooks/train_all_data.ipynb` repeats the exact training recipe (config in
`src/solar_filament/config.py`, seed 2026, 16,384 optimizer updates, batch 2, 512-px
crops, `positive_crop_probability=0.95`, AdamW lr 2e-4 / wd 1e-4, cosine annealing to
5% of the peak learning rate, AMP with a bounded overflow budget, deterministic
algorithms). It consumes only the competition training annotations (segmentation
polygons); no external solar data, labels, or checkpoints are used.

## Validation evidence (development, held-out)

Evaluated with the organizer self-evaluation metric (Panoptic Quality over annotator
records), month-grouped folds with a 72-hour embargo; fold 0 was never used for
training, calibration, or scoring:

| Recipe | Fold 1 PQ | Fold 2 PQ | Pooled (283 photos) |
|---|---|---|---|
| identity inference, 16,384 steps | 0.362724 | 0.335859 | 0.348667* |
| **4-rotation TTA (submitted)** | **0.407376** | **0.382941** | **0.393929** |
| + month-cross-fitted filter | 0.428660 | 0.402701 | 0.414888 |
| 32,768 steps (final model) | 0.344894 raw / **0.432597** filtered | — | — |
| + sweep filter (submitted flagship) | **0.438159** | — | — |

\* pooled identity over 283 photos; see `results/` for the paired month-bootstrap
comparisons. Every calibrated score reproduces the official scorer exactly
(zero-difference parity checks).

## Evidence index

- `evidence/RECOVERY_INVENTORY.md` — recovered artifacts, paths, hashes, confidence.
- `evidence/REPRODUCTION_REPORT.md` — byte-exact regeneration of the 0.56 artifact
  (`e17c2e26…`), its input chain, an independent control, and a determinism re-run.
- `evidence/VALIDATION_RESULTS.md` — official-PQ validation, integrity measurements,
  and the public-ground-truth contamination analysis of the 0.56 artifact.
- `results/` — recorded run metadata, score evidence, submission records.

## Metric observations

The official evaluation pools Panoptic Quality over *annotator records*: the same
prediction earns true-positive credit against every annotator's version of a filament
it overlaps with IoU > 0.5. Kaggle additionally rejects submissions whose predicted
masks overlap. Under these two rules the effective strategy space collapses to
one well-calibrated mask per physical filament; we document in the technical report
which alternative mechanisms (mask variants, duplicate coverage, gap recovery) are
therefore unavailable, with measurements.

## License and attribution

Competition data: CC BY-NC 4.0 (AURA/NSO). Code in this package: MIT.
GONG H-alpha observations courtesy of NSO/NISP.

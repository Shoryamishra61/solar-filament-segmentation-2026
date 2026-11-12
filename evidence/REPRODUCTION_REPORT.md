# REPRODUCTION_REPORT.md — Byte-exact regeneration of the 0.56 submission

Date: 2026-11-13. Environment: fresh Python 3.12.3 venv, pinned era stack
(torch 2.6.0+cu124, torchvision 0.21.0+cu124, numpy 1.26.4, pandas 2.2.3,
pycocotools 2.0.10, scipy 1.15.2, opencv-python 4.11.0.86,
segmentation-models-pytorch 0.5.0, timm 1.0.30), original RTX 3050 Laptop GPU,
Windows 11. Competition images re-downloaded from the organizer mount on
2026-11-13 (180/180 stems match the recovered per-image SHA-256 manifest).

## Result

**The scored 0.56 CSV was reproduced BYTE-EXACTLY, three ways.**

| # | Artifact | Era binding | Regenerated SHA-256 | Verdict |
|---|---|---|---|---|
| 1 | `champion-055-submission-001/submission.csv` (input to the accepted stage) | preflight SHA `5bf460da…` | `5bf460da16419888bae2676d80664fd5a7104ba489efa6d7121194df4ba2bce0` | **BYTE MATCH** |
| 2 | `champion-064-adaptive-physics-001/submission.csv` = **the scored 0.56 CSV** (Kaggle ref 57468408) | receipt 012 + preflight SHA `e17c2e26…` | `e17c2e26655ff6200807181d4ea69f4c3b6884fd60cfdbad01a20bb9423ad0c6` | **BYTE MATCH** |
| 3 | `champion-060-physics-refined-001/submission.csv` (independent control, delta_bg=2.0) | receipt 009 SHA `fbf48797…`, byte copy survives at `state/kaggle-upload-hybrid-1025/champion_056_payload.csv` | `fbf487976803b0895b4e8d0382242395aace65aeb7625ecce2f3253a16b5c57e` | **BYTE MATCH** (+ byte-identical to the surviving copy) |

Physical-statistics cross-check for the scored CSV: shaved_pixels_total
**255,289** (reproduced) = 255,289 (era receipt 012); 1345 rows; 179 images
with predictions; `20180507140210Th` has no row (zero detections).

## How each deleted input was regenerated

1. **Payload (1342 rows / 178 stems):** re-extracted from the recovered lamhuy
   notebook cell 5 (base85 → LZMA → DataFrame). Deterministic: the extracted
   `champion_055_database.csv` has SHA-256 `7e657038…` under both pandas 2.3.3
   and the pinned pandas 2.2.3.
2. **Cosine95 gap-fill rows (3 instances on `20160513133714Th`):** re-run of the
   era model — checkpoint SHA-verified against the era receipt
   (`671ebe63…`), smp.Unet(resnet34) rot4-TTA tiled inference on GPU, threshold
   0.5 / min-area 32 extraction, then the frozen area≥256 / confidence≥0.9
   filter. Exactly 3 instances survive; 6 fall below confidence 0.9. GPU
   inference is bit-reproducible on this machine (identity probability SHA
   identical across repeated runs).
3. **champion-055 assembly:** recovered `audit_055_candidate.py` logic, verbatim.
4. **The accepted refinement:** recovered `build_adaptive_physics_champion.py`,
   verbatim algorithm, ALPHA=0.25.

## Determinism

A full repeat of the adaptive run was executed into a separate output path;
the CSV bytes are identical (`e17c2e26…`). All three stages are deterministic
image-space logic plus bit-stable GPU inference; no sampling or augmentation
is involved in the submission path.

## Byte identity vs decoded-mask identity

Byte identity was achieved, so decoded-mask identity follows trivially. The
distinction the mission asks for still matters for one stage: the gap-fill
model inference was re-run rather than replayed from cache — its bit-exactness
is an empirical finding on the original hardware/software stack, not a
guarantee on other machines.

## What this does and does not claim

- **Claimed:** the exact artifact that Kaggle scored 0.56 can be regenerated
  end-to-end from the recovered chain without access to the original CSV.
- **Not claimed:** the payload input (Stage 0) is a legitimate, contamination-
  free prediction source. See VALIDATION_RESULTS.md for the integrity audit;
  the payload is treated as an audit target, never as training signal.
- No test-image-specific rules were added during reproduction; the only
  inference performed was the era algorithm's own, on one image the era
  pipeline itself had processed.

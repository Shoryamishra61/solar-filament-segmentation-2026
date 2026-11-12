# RECOVERY_INVENTORY.md — Forensic recovery of the Solar Filament Segmentation 2026 submission

Recovered 2026-11-13 by the follow-up agent session. All artifacts below were found on
the local machine and verified by direct inspection. Nothing was fabricated; where
evidence ends, that is stated.

## 1. Primary project (fully recovered)

**Location:** `D:\zcode\workspace\default\kaggle-work\solar_filament\` (8.5 GB, git repo,
initial commit 2026-10-29 "Final solution: consensus-aware ConvNeXt U-Net + evaluation
investigation", HEAD `ccf0a5f` 2026-11-07).

| Artifact | Path | Date | Confidence |
|---|---|---|---|
| Champion builder (the 0.56 algorithm) | `state/experiments/build_adaptive_physics_champion.py` | 2026-10-21 06:15 | VERIFIED (source) |
| Alpha=0.40 sibling builder | `state/experiments/build_adaptive_physics_champion_v2.py` | 2026-10-21 11:47 | VERIFIED (source) |
| delta_bg=2.0 refinement builder | `state/experiments/build_physics_refined_champion.py` | 2026-10-21 03:44 | VERIFIED (source) |
| champion-055 assembler | `state/experiments/audit_055_candidate.py` | 2026-10-21 02:49 | VERIFIED (source) |
| Payload extractor | `state/experiments/inspect_champion_df.py` | Oct 19 era | VERIFIED (source) |
| lamhuy public notebook (payload source) | `tmp_kernel_lamhuy/solar-filament-unet-segmentation-0-55.ipynb` | 2026-10-19 23:54 | VERIFIED (file) |
| Kaggle submission receipts 001–013 | `state/kaggle_submission_0NN_receipt.json` | Oct 16–23 | VERIFIED (API-bound) |
| Upload preflights (champion family) | `state/uploads/champion-*/preflight.json` | Oct 20–21 | VERIFIED (hash-bound) |
| Experiment log (267 events) | `state/EXPERIMENT_LOG.jsonl` | Oct 15–23 | SUPPORTED |
| Submission log | `state/SUBMISSION_LOG.md` | Oct 16–23 | SUPPORTED |
| LB investigation (integrity findings) | `state/LB_INVESTIGATION_20261026.md` | Oct 25–31 | SUPPORTED |
| Project state file | `state/STATE.md` (98 KB) | Oct 15–20 | SUPPORTED |
| Public leaderboard snapshot | `state/filament-segmentation-2026-publicleaderboard-2026-10-20T20_44_03.csv` | Oct 20 | VERIFIED (file) |
| Cosine95 checkpoint (gap-fill model) | `state/kaggle-upload-hybrid-1025/cosine95_long_fold1_16384.pt` | Oct 26 copy | VERIFIED (SHA-256 `671ebe63…` matches era receipt) |
| Dataset manifest + folds | `state/kaggle-upload-hybrid-1025/solar_dataset_manifest.json`, `solar_folds.json` | Oct 29 | VERIFIED (file) |
| Honest flagship weights (final package) | `final_package/weights/alldata_convnext_32768.pt`, `alldata_convnext_16384.pt`, `convnext_long_fold1_16384.pt` | Oct 26–Nov 5 | VERIFIED (file + manifest) |
| Final package (honest artifact) | `final_package/` (README, PUBLISH.md, notebooks, src, report PDF) | Oct 26–Nov 10 | VERIFIED (file) |
| Champion-family run CSVs (Nov 9–10) | `runs/candidate-086…`, `runs/champion-080/082/084…` | Nov 9–10 | VERIFIED (file) |

## 2. The scored 0.56 CSV — provenance chain (all VERIFIED, hash-bound)

| Link | Evidence | Hash/ID |
|---|---|---|
| Kaggle submission | `state/kaggle_submission_012_receipt.json` + `state/uploads/champion-064-adaptive-upload-001/preflight.json` | ref **57468408**, 2026-10-21 05:56:19 UTC, public **0.56**, CSV SHA-256 **`e17c2e26655ff6200807181d4ea69f4c3b6884fd60cfdbad01a20bb9423ad0c6`** |
| Builder | `build_adaptive_physics_champion.py`, ALPHA=0.25, shaved_pixels_total **255,289** | receipt 012 |
| Input CSV | `runs/champion-055-submission-001/submission.csv` (deleted) | SHA-256 **`5bf460da16419888bae2676d80664fd5a7104ba489efa6d7121194df4ba2bce0`** (preflight) |
| champion-055 builder | `audit_055_candidate.py` | payload + Cosine95 gap fill |
| Payload | cell 5 of the lamhuy notebook, base85+lzma blob → `runs/champion_055_database.csv` (deleted) | re-extracted 2026-11-13: 1342 rows / 178 stems, SHA-256 `7e657038…` |
| Gap fill | Cosine95 rot4 `predictions.json` (deleted with `data/`): 3 instances on `20160513133714Th`; none on `20180507140210Th` | receipt-bound CSV `b22848ef…`, 1149 rows / 170 images, 10 zero-detection images |

### The 179/180 coverage discrepancy — explained

`20180507140210Th` received **zero detections** from the payload AND from the Cosine95
fallback model, so the CSV contains no row for it (1345 rows over 179 distinct stems).
Kaggle accepted and scored it (0.56). This was a known, audited behavior at submission
time (`fresh_mask_audit.zero_detection_images` in every preflight), not an accident
discovered later. The Nov 9 "Candidate 86" submission (ref in Nov receipts, 1346 rows)
later added a recovered prediction for that image at the same 0.56.

## 3. What was NOT recovered

- The original CSV bytes (`e17c2e26…`) — deleted from `runs/` (gitignored), absent from
  the Recycle Bin, and the Kaggle API does not expose submission-file downloads
  (both REST probes returned 404). Byte-exact verification therefore relies on
  regeneration (see REPRODUCTION_REPORT.md).
- `runs/champion-055-submission-001/submission.csv` and
  `runs/champion_055_database.csv` (deleted, gitignored).
- `data/cosine95-project/runs/cosine95-long-fold1-test-001/predictions.json`
  (retained-listed in the Oct 20 eviction receipt but the whole `data/` tree is gone).
- The October 2026 ZCode session transcripts: no filament sessions exist in
  `~/.zcode/cli/db/db.sqlite` (only a scheduled-automation session from Oct 25
  survives); the per-session rollout/agent stores contain no champion-era content.
- The organizer Overleaf template source (login-gated; not downloaded at the time).

## 4. Preservation actions taken

- No file in `solar_filament/` was modified or deleted by the recovery session.
- Re-extracted payload and reproduction outputs are written under
  `recovery/` (separate from the historical project tree).
- Competition data re-downloaded from the organizer mount into `recovery/downloads/`
  (660 MB zip, SHA-256 of the test-image set cross-checked per-image by the original
  manifest `pixel_sha256`/`sha256` fields).

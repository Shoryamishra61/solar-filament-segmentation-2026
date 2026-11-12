# RELEASE_GATES.md — final-release verification (2026-11-13)

## Gate A — Provenance: **PASS**
- [x] Training annotations: official competition train JSON only, hash-bound in
      every run record (`annotation_sha256` in `results/*.json`).
- [x] Pretrained weights: generic ImageNet backbone only, file SHA-pinned
      (`4bee155ff6…`), no task-specific external checkpoints.
- [x] No test-ground-truth lookup, no hidden-label contamination in the honest
      pipeline (code audited; the contamination finding is about a THIRD-PARTY
      payload documented under `audit/`, never used as training signal).
- [x] Public MAGFiLO data used only as an audit measurement, never as labels.

## Gate B — Correctness: **PASS**
- [x] Source executes: `pytest tests/` 3/3; all reproduction scripts ran
      end-to-end; repo `scripts/verify_submission.py` PASS on both CSVs.
- [x] All predictions derived from organizer-mounted images (test set
      re-downloaded 2026-11-13, per-image SHA manifest match).
- [x] Official RLE format + metric parity (organizer-scorer replica verified
      to ~1e-9 in the era; all-pairs scorer used for audit measurements).
- [x] Original CSV reproduction measured, not asserted: byte-exact ×4
      (champion-055, champion-064, champion-060 control, determinism re-run);
      flagship preserved hash-exactly + structurally reproduced (949/1015
      masks byte-identical on different GPU class).

## Gate C — Independent reproducibility: **PASS with one pending user step**
- [x] Fresh clone: repo has no absolute local paths (redacted/sanitized);
      notebook uses `/kaggle/input` mounts + repo-relative fallbacks.
- [x] Fresh dependency install: `requirements.txt` pins the exact era stack
      (verified by building that stack from scratch for this audit).
- [x] Weights: committed via Git LFS with `.gitattributes`; SHA-256 manifest.
      **Pending:** verify unauthenticated LFS retrieval after the user pushes;
      fallback documented (GitHub Release assets + manifest hashes).
- [x] Notebook executes without undocumented manual artifacts (logic
      re-executed locally end-to-end; Kaggle path documented).
- [x] Predicted CSV compared against the Kaggle-submitted CSV (both artifacts).

## Gate D — Scientific quality: **PASS**
- [x] Actual method explained (report §Method; audit documents the historical chain).
- [x] Claims backed by measurements; negative results reported.
- [x] Real validation (month-grouped, embargoed, reserved-holdout protocol);
      honest limitations incl. the contamination analysis and evaluator
      vulnerabilities.

## Gate E — Final-submission completeness: **PARTIAL — user actions pending**
- [x] Exact scored CSVs preserved byte-exactly (`final/`, not committed).
- [x] Public repository and notebook prepared (git history ready; **push pending user**).
- [x] Version-pinned requirements supplied.
- [x] Report compiled (3 pages, IEEE 2-column).
- [ ] **Organizer Overleaf template**: NOT used — the template source requires
      an Overleaf login this environment does not have. The report satisfies
      the content requirements; if the organizers strictly require their
      template, the user must export it from
      https://www.overleaf.com/read/vjztpvrnbrdh and the content must be
      ported (≈1 h of work; content is template-independent).
- [x] Google Form fields prepared (`evidence/FORM_PREP.md`); **not submitted**
      (user must complete authentication/CAPTCHA and the submission-choice).
- [ ] Kaggle selection state: user must decide which submission(s) remain
      selected (0.56 audit artifact vs 0.37 honest flagship).

# COMPETITOR_EVIDENCE.md — Harshit Kulkarni: leaderboard 0.62 vs repository 0.40

Compiled 2026-11-13. Facts only; no accusation is made or implied. "Verified"
= directly observable; "Unknown" = no public evidence either way.

## Verified facts

1. **Public leaderboard** (downloaded 2026-11-13 via the authenticated Kaggle
   API): team "Harshit Kulkarni" (`harshitkulkarni07`) is **rank 4 with 0.62**,
   TeamId 17100753, 65 submissions, last submission 2026-11-10 10:24:24 UTC.
2. **The linked repository**
   [`kulharshit21/solar-filament-final-draft`](https://github.com/kulharshit21/solar-filament-final-draft)
   documents a different score: its README and INTEGRITY.md state, verbatim,
   "public score: 0.40" for the canonical `submission_w11_champion.csv`,
   "independently resubmitted to Kaggle on 10 November 2026", with OOF PQ
   0.4622 (SQ 0.6857, RQ 0.6741, TP/FP/FN 1031/454/543) and a stated SHA-256
   for the exact CSV.
3. The repository describes a **YOLO11x-seg → Mask2Former Swin-L → SAM 2.1
   Hiera-L ensemble** trained on official competition data, with date-grouped
   folds and annotator-aware validation. Code, configs, and an integrity
   verifier are public; weights and the canonical CSV are distributed out-of-
   band (Google Drive) with SHA-256 manifests.
4. The account `harshitkulkarni07` has **no public Kaggle notebooks** for this
   competition (API listing returned none), so no public artifact documents
   the pipeline behind the 0.62 entry.
5. For calibration: the top public score (0.70, team "IONode Online") matches
   a documented public-evaluator exploit (seven empty masks per photograph;
   responsible disclosure `artkomissar/please-fix-the-leaderboard`, 52 votes),
   and this project's own measurements put every cleanly trained pipeline at
   0.34–0.37 public and ≤ 0.44 development PQ — while a payload known to be
   informed by test annotations reaches 0.65+ against the public test
   annotation subset (see `evidence/VALIDATION_RESULTS.md`).

## Answers to the specific questions

| Question | Finding |
|---|---|
| Does the repository document an older/lower-scoring submission? | It documents 0.40 as its best and only canonical artifact; it also mentions a newer "Python4 candidate" scoring 0.38 that was excluded. |
| Is there another public notebook/submission describing the 0.62 pipeline? | None found (no public kernels for the account; nothing in the repo claims 0.62). |
| Do leaderboard and repository refer to different scoring states? | Yes — they cannot be the same submission: the repo's canonical CSV is stated to score 0.40 on resubmission the same day the leaderboard entry scored 0.62. |
| Does the implementation satisfy final-submission reproducibility requirements? | The repo provides source, pinned configs, an integrity verifier, weight manifests with SHA-256, and a stated exact-CSV hash; weights live on Google Drive rather than the repository itself, so unauthenticated reproducibility depends on that link remaining available. Not independently verified. |
| Are required notebooks or weights missing from the public repository? | Weights and the exact CSV are not in the repository (by design, per its ARTIFACTS.md separation). |

## Unknowns (no public evidence)

- How the 0.62 submission was produced; whether it shares the repository's
  pipeline; whether it was trained/derived with information beyond the
  competition training data.
- Whether the 0.40 resubmission record referenced by the repository is
  the same entry that later moved to 0.62, or a different submission.

## Position of this project

Our objective is a prize-eligible, scientifically defensible submission — the
honest flagship (0.37 public, byte-verified reproducibility, clean provenance)
plus this audit trail. We do not copy other competitors and do not exploit the
documented evaluator weaknesses.

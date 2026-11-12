# final/ — deliverable staging (local only)

This directory is **not committed** (CSVs are gitignored; regenerate them from
the pinned weights instead — both reproduce byte-exactly).

| File | What it is | SHA-256 |
|---|---|---|
| `Solar_Filament_Technical_Report.pdf` | the 3-page technical report (IEEE 2-column source in `report/`) | — |
| `submission_57468408_public_0.56.csv` | byte-exact regeneration of the historical 0.56 submission (Kaggle ref 57468408, 2026-10-21). 414,750 bytes, 1345 rows / 179 photos. **Audit artifact — contamination-suspected payload; not the honest submission.** | `e17c2e26655ff6200807181d4ea69f4c3b6884fd60cfdbad01a20bb9423ad0c6` |
| `submission_57826506_public_0.37.csv` | byte-exact regeneration of the honest flagship submission (Kaggle ref 57826506). 1015 rows / 176 photos, disjoint. | `d6b9cfa9492e8be6a2d67ac66f72066347eec542662953ce3e8353bc70fcf814` |

Which CSV to attach to the organizer's Google Form follows from which
submission(s) the user keeps selected on Kaggle — see
`evidence/FORM_PREP.md` (never submitted automatically).

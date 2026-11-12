# Google Form submission package — prepared, NOT submitted

Organizer form: https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform

## Field values prepared

| Field | Value | Status |
|---|---|---|
| Participant name | Shoryakumar Mishra | from account context |
| Kaggle username | shoryakumarmishra | verified via API auth |
| Institution | SRM Institute of Science and Technology | from mission brief |
| Country | India | from mission brief |
| Public repository link | **PENDING — user must create/push** (see blockers) | blocked on publish approval |
| Exact Kaggle-submitted CSV | two candidates staged, see below | user decision required |
| Final technical report PDF | `final/Solar_Filament_Technical_Report.pdf` (3 pages, IEEE 2-column) | ready |
| Verification summary | `evidence/` (recovery, reproduction, validation) | ready |

## The exact-CSV decision (user must choose)

Kaggle's final evaluation scores the **selected** submissions (default: best
public + best private of the participant's choices). Current state on Kaggle:

- Best public = **0.56** — the historical payload-derived artifact
  (ref 57468408, CSV SHA `e17c2e26…`; a later equivalent "Candidate 86" ref
  adds the 180th image at the same 0.56). This is what the leaderboard rank 19
  reflects.
- Honest flagship = **0.37** (ref 57826506, CSV SHA `d6b9cfa9…`) — the
  artifact this repository's weights and notebook byte-exactly reproduce, and
  the only one defensible under the contamination audit.

Both CSVs are preserved byte-exactly:

- `final/submission_57468408_public_0.56.csv` (regenerated, SHA-verified `e17c2e26…`)
- `final/submission_57826506_public_0.37.csv` (regenerated, SHA-verified `d6b9cfa9…`)

The form must be completed by the user in a browser (authentication/CAPTCHA),
together with the repository publish decision. Nothing was submitted.

## Steps remaining for the user

1. Review `evidence/VALIDATION_RESULTS.md` §2 and choose which submission(s)
   remain selected on Kaggle (this changes the final-evaluation story).
2. Create the GitHub repository (suggested name
   `solar-filament-segmentation-2026`), push this folder, make it PUBLIC.
3. Verify LFS weight retrieval from a fresh clone; if LFS quota blocks, attach
   the three checkpoints as GitHub Release assets and note it in the README.
4. Submit the Google Form with the staged values above.
5. Optional: export the organizer Overleaf template and re-render the report
   in it (the template source requires an Overleaf login this environment
   does not have).

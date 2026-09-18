# SKILLS.md — Skills Demonstrated / Applied in This Build

| # | Skill | Where applied |
|---|---|---|
| 1 | Feature-level multimodal fusion (the actual technique, not a black box) | `src/train.py` — `hstack([csr_matrix(X_tab_train), X_text_train])`, a literal, inspectable concatenation of two modalities' features into one matrix |
| 2 | Controlled comparison (fixing the algorithm to isolate the variable under test) | `src/train.py` — all 3 models use identical `Ridge(alpha=RIDGE_ALPHA, ...)`, so leaderboard differences are attributable only to feature availability, not algorithm choice |
| 3 | Constructing verifiable synthetic data (ground truth for a specific claim) | `data/generate_data.py` — price is deliberately split into a tabular component and a text-only component, making "fusion needs both" checkable rather than assumed |
| 4 | Leakage-safe text vectorization | `src/train.py` — `TfidfVectorizer` fit only on the training split, `.transform()` (not re-fit) on test |
| 5 | Fixed-category encoding for reliable live inference | `src/data_prep.py: build_tabular_features()` — single-row inference produces the same columns training did |
| 6 | Per-request comparison, not just an aggregate metric | `src/predict.py: predict_price()` returns all 3 models' predictions for the same input, making the value of fusion visible per-listing |
| 7 | Testing a structural property, not just an outcome | `tests/test_api.py: test_tabular_only_is_blind_to_text_signal` — asserts the tabular-only prediction is bit-identical across two different descriptions, proving the model structurally cannot see text, not just that it "performs worse" |
| 8 | API input validation via typed literals | `app.py: ListingRequest` — `Literal[...]` on `room_type`/`neighborhood`, HTTP 422 on an invalid category |
| 9 | Documentation-as-artifact | This file, plus `PROMPTS.md`, `IMPLEMENTATION_PLANS.md`, `WALKTHROUGH.md`, `AUDIT_REPORT.md`, `README.md` |

## Explicitly not attempted here (see `IMPLEMENTATION_PLANS.md` §8)

The actual AutoGluon Multimodal package, an image modality, a
deep-learning text encoder, and hyperparameter search were all
plausible parts of the source repo's larger vision but were
intentionally left out — per your explicit instruction to keep this
"not fancy," with the main aim being a simple, working frontend and
backend.

# 🏠🔤 Multimodal AutoML Suite

A small, real, end-to-end CRISP-DM data science project: trains and
compares a tabular-only model, a text-only model, and a **fusion**
model (both modalities combined), on the exact technique real AutoGluon
Multimodal is built on — feature-level fusion — and proves fusion
genuinely adds value, both in aggregate metrics and per-prediction.

Reproduced (scoped down, "not fancy," simple frontend + backend) from
Project 14 of
[`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples).
**Note**: this project's exact original prompt could not be retrieved
(same GitHub access restriction as project 13) — see
[`PROMPTS.md`](PROMPTS.md) for what's known and what's this build's own
interpretation, including why the real (heavy) AutoGluon Multimodal
package was deliberately not used.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt

# 1. Generate data (synthetic — see data/generate_data.py to swap in real data)
python data/generate_data.py

# 2. Train and compare all 3 models
cd src && python train.py && cd ..

# 3. Run the app
uvicorn app:app --reload --port 8014
```

Open **http://127.0.0.1:8014** — see the leaderboard, then try changing
only the description text while keeping tabular fields the same and
watch tabular-only stay frozen while fusion shifts.

Run the tests any time with:
```bash
python -m pytest tests/
```

## What's in the box

```
14_autogluon_multimodal_automl_suite/
├── README.md                  ← you are here
├── PROMPTS.md                 ← what's known about the original ask, honestly caveated
├── IMPLEMENTATION_PLANS.md    ← architecture & technical spec
├── WALKTHROUGH.md             ← code-ordered deep-dive tour (text, see note below)
├── SKILLS.md                  ← practices applied, with file/line pointers
├── AUDIT_REPORT.md            ← fusion-correctness audit, with commands to re-run it yourself
├── requirements.txt
├── data/
│   └── generate_data.py       ← synthetic listings generator (tabular + text, engineered so price needs both)
├── src/
│   ├── data_prep.py           ← fixed-category tabular encoding
│   ├── train.py                ← trains & compares 3 models (same algorithm, different features)
│   └── predict.py             ← live scoring through all 3 models for comparison
├── app.py                     ← FastAPI backend
├── static/                    ← plain HTML/CSS/JS frontend (leaderboard, side-by-side prediction)
├── models/                    ← 3 trained models, scaler, vectorizer, metrics.json
└── tests/
    └── test_api.py            ← automated tests (4/4 passing)
```

## The headline result

| Model | R2 | MAE |
|---|---|---|
| Tabular only | 0.616 | $29.97 |
| Text only | 0.252 | $42.67 |
| **Fusion (both)** | **0.955** | **$10.02** |

More convincing than the metric alone: two listings with **identical**
tabular fields but different description text get the **exact same**
tabular-only prediction (it structurally cannot see the text) but
different fusion predictions — verified across multiple listing
configurations in `AUDIT_REPORT.md`, not just one cherry-picked example.

## Honest limitations of this build

- **This project's exact original prompt is unknown** — see `PROMPTS.md`.
- **No actual AutoGluon Multimodal dependency.** The real package needs
  PyTorch, Transformers, and timm — several GB and conflicts with "keep
  it simple." This build reimplements the actual underlying technique
  (feature-level fusion) directly with scikit-learn instead.
- **Data is synthetic**, deliberately engineered so price depends on
  both modalities (documented in `data/generate_data.py`), because this
  build environment has no internet access to Kaggle.
- **No video walkthrough is included.** `WALKTHROUGH.md` is a written,
  code-ordered substitute, with a shot list at the bottom.
- **This zip wasn't pushed to GitHub for you.** To publish it yourself:
  ```bash
  cd 14_autogluon_multimodal_automl_suite
  git init && git add . && git commit -m "Multimodal AutoML suite"
  git branch -M main
  git remote add origin https://github.com/<you>/<repo>.git
  git push -u origin main
  ```
- Deliberately **simple**, per your instruction: no image modality, no
  deep-learning text encoder, no database — see
  `IMPLEMENTATION_PLANS.md` §8 for the full list of what was left out and why.

## License

MIT — do whatever you like with it.

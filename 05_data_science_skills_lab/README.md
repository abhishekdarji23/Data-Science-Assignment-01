# 🧪 Data Science Skills Mastery Lab

A small, real, end-to-end CRISP-DM data science project: a catalog of 16
independently-executable data science skills (missing-value analysis,
outlier detection, feature scaling, model training, cross-validation,
and more) that run **live** against a Titanic-style dataset and render as
an actual dashboard — metric cards, tables, and bar charts — not raw JSON.

Reproduced (scoped down, "not fancy," simple frontend + backend) from
Project 05 of
[`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples).
See [`PROMPTS.md`](PROMPTS.md) for exactly what was asked and how this
build's scope differs from the original — including the original repo's
own follow-up complaint about raw JSON output, which this build was built
to directly address.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt

# Generate data (synthetic — see data/generate_data.py to swap in real Kaggle data)
python data/generate_data.py

# Run the app (no separate training step — every skill runs live, per click)
uvicorn app:app --reload --port 8005
```

Open **http://127.0.0.1:8005** — 16 skill cards, grouped by CRISP-DM
phase. Click "Execute skill" on any of them to run it live.

Run the tests any time with:
```bash
python -m pytest tests/
```

## What's in the box

```
05_data_science_skills_lab/
├── README.md                  ← you are here
├── PROMPTS.md                 ← what was asked, and how scope was adjusted
├── IMPLEMENTATION_PLANS.md    ← architecture & technical spec
├── WALKTHROUGH.md             ← code-ordered deep-dive tour (text, see note below)
├── SKILLS.md                  ← the 16 skills demonstrated + engineering practices, with file/line pointers
├── AUDIT_REPORT.md            ← ground-truth recovery audit, with commands to re-run it yourself
├── requirements.txt
├── data/
│   └── generate_data.py       ← synthetic Titanic-style data generator (CRISP-DM: Data Understanding)
├── src/
│   ├── data_prep.py           ← dataset loading (deliberately no cleaning — see docstring)
│   └── skills.py               ← the 16-skill registry (Data Prep, Modeling, Evaluation skills live here)
├── app.py                     ← FastAPI backend (CRISP-DM: Deployment)
├── static/                    ← plain HTML/CSS/JS frontend (skill cards + friendly result rendering)
└── tests/
    └── test_api.py            ← automated smoke tests (7/7 passing, loops over every registered skill)
```

## CRISP-DM summary

This project's structure *is* the CRISP-DM summary — every skill is
tagged with the phase it belongs to, and the catalog groups them that way
in the UI:

1. **Data understanding** (8 skills) — dataset overview, missing values,
   duplicates, descriptive stats, outliers, categorical counts,
   correlation, class balance.
2. **Data preparation** (3 skills) — missing-value imputation, one-hot
   encoding, feature scaling.
3. **Modeling** (2 skills) — train/test split, baseline vs. logistic
   regression.
4. **Evaluation** (3 skills) — 5-fold cross-validation, confusion
   matrix + precision/recall/F1, feature importance.

Full CRISP-DM phase table (with file/line pointers) in
`IMPLEMENTATION_PLANS.md` §4.

## Honest limitations of this build

- **Data is synthetic**, with a genuine (non-random) survival model and
  deliberately injected missing values/duplicates, because this build
  environment has no internet access to Kaggle. Drop a real
  `data/titanic_real.csv` in and re-run — no code changes needed. See
  `data/generate_data.py`'s docstring.
- **Only 16 skills, not 54, and no installed external skill packages.**
  The original prompt asked to install two specific GitHub skill
  repositories and demonstrate every skill across 5 Kaggle datasets —
  this assistant can't install and integrate someone else's arbitrary
  GitHub package into a running product on your behalf. 16
  self-contained skills across every CRISP-DM phase is a genuine,
  representative sample instead.
- **No video walkthrough is included.** Recording narration and uploading
  to YouTube isn't something this assistant can do. `WALKTHROUGH.md` is a
  written, code-ordered substitute, with a shot list at the bottom if you
  want to record one yourself in a few minutes.
- **This zip wasn't pushed to GitHub for you** — there's no GitHub write
  access available here. To publish it yourself:
  ```bash
  cd 05_data_science_skills_lab
  git init && git add . && git commit -m "Data science skills mastery lab"
  git branch -M main
  git remote add origin https://github.com/<you>/<repo>.git
  git push -u origin main
  ```
- Deliberately **simple**, per your instruction: no React build, no
  charting library (bar charts are plain CSS), no database — see
  `IMPLEMENTATION_PLANS.md` §8 for the full list of what was left out and why.

## License

MIT — do whatever you like with it.

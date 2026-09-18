# 🕵️ CRISP-DM NYC Taxi Audit Platform

A small, real, live audit platform: runs a taxi trip-duration pipeline
in either a `clean` mode (correct practice throughout) or a deliberately
`flawed` mode (4 specific, real violations), then runs 10 independent
CRISP-DM/methodology checks against whichever one you pick — and proves
the checks actually catch the violations, rather than only ever being
demonstrated against data that already passes.

Reproduced (scoped down, "not fancy," simple frontend + backend) from
Project 13 of
[`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples).
**Note**: this project's exact original prompt could not be retrieved
(GitHub access restrictions blocked it) — see [`PROMPTS.md`](PROMPTS.md)
for exactly what's known, and what's this build's own interpretation.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt

# Generate the audit-subject dataset
python data/generate_data.py

# Run the app (no separate training step — every audit run trains fresh)
uvicorn app:app --reload --port 8013
```

Open **http://127.0.0.1:8013** — run the audit against the clean
pipeline (100% compliance), the flawed one (50%), or compare both side
by side.

Run the tests any time with:
```bash
python -m pytest tests/
```

## What's in the box

```
13_crispdm_nyc_taxi_audit_platform/
├── README.md                  ← you are here
├── PROMPTS.md                 ← what's known about the original ask, honestly caveated
├── IMPLEMENTATION_PLANS.md    ← architecture & technical spec
├── WALKTHROUGH.md             ← code-ordered deep-dive tour (text, see note below)
├── SKILLS.md                  ← practices applied, with file/line pointers
├── AUDIT_REPORT.md            ← auditing the auditor — includes a real bug found & fixed
├── requirements.txt
├── data/
│   └── generate_data.py       ← synthetic NYC-taxi-style data generator
├── src/
│   ├── pipeline.py             ← the audit SUBJECT: clean or flawed variant
│   ├── audit_checks.py        ← 10 independent checks across 4 CRISP-DM phases
│   └── audit_runner.py        ← ties pipeline + checks together, produces the report
├── app.py                     ← FastAPI backend
├── static/                    ← plain HTML/CSS/JS dashboard
└── tests/
    └── test_api.py            ← automated tests (7/7 passing, verified stable across repeats)
```

## What makes this a genuine audit, not a demo

The subject pipeline has a `flawed` mode with 4 real, specific
violations: a feature derived directly from the target (leakage), a
random shuffle split on genuinely time-series data, duplicate rows left
in training data, and no random seed. Every audit check is proven
against BOTH modes — see `AUDIT_REPORT.md` for the full verification,
including an actual threshold-selection bug that was found (via running
the flawed pipeline 10 times and seeing a borderline case) and fixed
during this build, not just asserted correct.

**A genuinely useful finding**: the flawed (leaky) pipeline's R2 comes
out HIGHER than the clean pipeline's (≈0.97 vs ≈0.89) — leakage doesn't
just violate methodology, it produces misleadingly optimistic metrics.
This is exactly why the audit includes both a structural check (is the
leaky feature present) and a behavioral check (is the R2 suspiciously
high) — neither alone would be a complete leakage detector.

## Honest limitations of this build

- **This project's exact original prompt is unknown.** Unlike every
  other project in this series, GitHub's access restrictions prevented
  retrieving the verbatim source prompt. `PROMPTS.md` says this plainly
  and explains what this build's scope is actually based on.
- **Data is synthetic**, self-contained to this project, because this
  build environment has no internet access to Kaggle.
- **No video walkthrough is included.** Recording narration and uploading
  to YouTube isn't something this assistant can do. `WALKTHROUGH.md` is a
  written, code-ordered substitute (doubling as the "video script" the
  source description mentioned), with a shot list at the bottom.
- **This zip wasn't pushed to GitHub for you** — there's no GitHub write
  access available here. To publish it yourself:
  ```bash
  cd 13_crispdm_nyc_taxi_audit_platform
  git init && git add . && git commit -m "CRISP-DM NYC taxi audit platform"
  git branch -M main
  git remote add origin https://github.com/<you>/<repo>.git
  git push -u origin main
  ```
- Deliberately **simple**, per your instruction: no React build, no
  persisted audit history, no database — see `IMPLEMENTATION_PLANS.md`
  §9 for the full list of what was left out and why.

## License

MIT — do whatever you like with it.

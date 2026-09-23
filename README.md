
# 🌟 Data Science & ML Portfolio — Reproduced Systems

A scoped-down, "not fancy," verified-working reproduction of
[`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples)'s
**16 Full-Stack Data Science, Machine Learning, and TypeScript Systems
(Projects 00 through 15)**, rebuilt with a plain backend + simple
frontend for each, following CRISP-DM where applicable.

Every built project swaps the original's heavy dependencies (PyTorch,
AutoGluon, Chronos, PatchTST, TFT, React/Vite) for lightweight,
verifiable equivalents where the original's exact stack would conflict
with "keep it simple" — see each project's own `PROMPTS.md` for the
specific substitution and reasoning. Every metric and result quoted
below was actually run and checked (see each project's own
`AUDIT_REPORT.md`), not copied from the source repo's claims.

---

## 🏛️ Systems Portfolio Index (16 Projects)

| # | System Title & Directory | Domain & Methodology (as actually built) | Video Walkthrough |
|---|---|---|---|
| **0** | [**Dynamic Todo Workspace**](./00_dynamic_todo_workspace) | Full-stack reactive task workspace | [🎬 link](https://drive.google.com/file/d/10pG4lQvrkzXTnZEO8bt0UZm0QjceVTk4/view?usp=sharing) |
| **1** | [**NYC Taxi Trip Prediction**](./01_nyc_taxi_trip_prediction) | CRISP-DM regression — GradientBoostingRegressor, time-based split, R²≈0.89 | [🎬 link](https://drive.google.com/file/d/10h-7P43ea8H4qHm2d767YA-2tQoPivgi/view?usp=sharing) |
| **2** | [**NanoLlama SFT LLM**](./02_nano_llm_transformer) | PyTorch autoregressive transformer | [🎬 link](https://drive.google.com/file/d/1WvWEBYZmksFTUGb1UITLmavEvalLAMA8/view?usp=sharing) |
| **3** | [**Customer Segmentation Clustering**](./03_customer_segmentation_clustering) | KMeans clustering, k chosen by silhouette score (k=4) | [🎬 link](https://drive.google.com/file/d/1ZQosOepGijNVYayt8xTfHuohQb7UcTS7/view?usp=sharing) |
| **4** | [**Market Basket Pattern Mining**](./04_associative_pattern_mining) | Apriori association rules (mlxtend), lift-ranked | [🎬 link](https://drive.google.com/file/d/1KpAIXI-hI-pdGzioXtEtF_8H0Bbr1G2d/view?usp=sharing) |
| **5** | [**DS Skills Mastery Lab**](./05_data_science_skills_lab) | 16 live-executable analytical skills across 4 CRISP-DM phases | [🎬 link](https://drive.google.com/file/d/1qFxzlDX50hKyjZXjRm9Neo6abVvFzZp3/view?usp=sharing) |
| **6** | [**Anomaly Detection**](./06_anomaly_detection) | Isolation Forest vs. LOF vs. z-score baseline, honestly compared | [🎬 link](https://drive.google.com/file/d/1m3xQdabQeZKBayyyl5jIl-S-HahQrY0o/view?usp=sharing) |
| **7** | [**AutoML Stacking**](./07_automl_autogluon) | 5-model zoo + out-of-fold stacking (scikit-learn, not AutoGluon) | [🎬 link](https://drive.google.com/file/d/1kq3fGvP5_d6XKYJXmU0fUAIflYngXHDW/view?usp=sharing) |
| **8** | [**DS Visual Foundations**](./08_datascience_visual_mastery) | Naive Bayes, model evaluation, gradient descent, backprop — live sims + graded quizzes | [🎬 link](https://drive.google.com/file/d/1QHTd-xBoO_b1GinGt4kM7K8UtwjHFawG/view?usp=sharing) |
| **9** | [**FlowForge DAG Engine**](./09_flowforge_dag_engine) | Type-safe DAG builder (branded types, generics) + Kahn's algorithm, TypeScript/Node | [🎬 link](https://drive.google.com/file/d/1baio0OUDAyO6C22cLBjrUK9LRy167ct6/view?usp=sharing) |
| **10** | [**CRISP-DM Master's Curriculum**](./10_crispdm_masters_curriculum) | 7-phase CRISP-DM teaching platform | [🎬 link](https://drive.google.com/file/d/1CDUN32N2xsnQRKs7u22-F32mAZNnhhND/view?usp=sharing) |
| **11** | [**Enterprise DS Audit**](./11_enterprise_ds_audit) | Data-quality & leakage scorecard | [🎬 link](https://drive.google.com/file/d/1TGyoBP5hQyUnS0fciuh__Iqf7ZFfUVuj/view?usp=sharing) |
| **12** | [**TimePulse Forecasting**](./12_timeseries_forecasting) | Multi-horizon forecasting | [🎬 link](https://drive.google.com/file/d/1x9K5zd8b8__OAW78OcH2PXXvv9RcRSZ5/view?usp=sharing) |
| **13** | [**CRISP-DM NYC Taxi Audit Platform**](./13_crispdm_nyc_taxi_audit_platform) | Live audit engine — 10 checks, clean vs. deliberately-flawed pipeline (100% vs. 50%) | [🎬 link](https://drive.google.com/file/d/1ppzhI9EfKTalcxDwaswyOA01s3dnWsrp/view?usp=sharing) |
| **14** | [**Multimodal AutoML Suite**](./14_autogluon_multimodal_automl_suite) | Tabular + text fusion (TF-IDF + Ridge), fusion R²≈0.96 vs. 0.62/0.25 single-modality | [🎬 link](https://drive.google.com/file/d/1ko4GJrljXcbMA_0IL50jzkbnDN_LIxRR/view?usp=sharing) |
| **15** | [**SPY Time Series Forecasting**](./15_spy_timeseries_sota_forecasting) | Naive/MA/ETS/ARIMA/GBM, honest 1-step-ahead backtest (naive is a hard baseline) | [🎬 link](https://drive.google.com/file/d/1c-vokhMzO2Lqk4SKse9iqE02haE7wcXB/view?usp=sharing) |

---

## 🎯 Prompt Catalog, Implementation Plans & Audit Reports

Unlike the source repo (one global set of these files for all 16
systems), **each built project here ships its own, project-specific
set** in its own folder:

* **`PROMPTS.md`** — what was asked for that project, and exactly how/why this build's scope differs from the original (dependency substitutions, data source, etc.)
* **`IMPLEMENTATION_PLANS.md`** — architecture, data schema, modeling approach, and verification performed
* **`WALKTHROUGH.md`** — a code-ordered deep-dive tour, plus a shot list for recording your own video walkthrough
* **`SKILLS.md`** — the specific engineering practices applied, each with a file/line pointer
* **`AUDIT_REPORT.md`** — checks actually run against that project's code (leakage, reproducibility, honest-comparison checks), including any real bugs found and fixed along the way

---

## 🛠️ Quick Start

Every built project here is a **single-port, full-stack app** — one
FastAPI (or, for Project 9, Node.js) process serves both the API and
the plain HTML/JS/CSS frontend. There's no separate frontend build step
or second port to run.

```bash
# Example: NYC Taxi Trip Prediction
cd 01_nyc_taxi_trip_prediction
pip install -r requirements.txt
python data/generate_data.py       # first time only
cd src && python train.py && cd .. # first time only
uvicorn app:app --reload --port 8001
# open http://127.0.0.1:8001

# Example: FlowForge DAG Engine (TypeScript, zero npm dependencies to run)
cd 09_flowforge_dag_engine
node src/server.ts
# open http://127.0.0.1:8009
```

Each project's own `README.md` has the exact commands and honest
limitations for that build.

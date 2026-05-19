# ChemBench User Guide

Welcome to **ChemBench** — a standardized benchmark suite for machine learning in chemical engineering. This guide walks you through installation, core concepts, and the end-to-end workflow from data loading to interactive results exploration.

---

## Table of Contents

1. [Installation](#installation)
2. [Architecture Overview](#architecture-overview)
3. [Quick Start (5-Minute Tutorial)](#quick-start-5-minute-tutorial)
4. [Running Benchmarks](#running-benchmarks)
5. [Analyzing Results](#analyzing-results)
6. [Viewing Results (Streamlit Dashboard)](#viewing-results-streamlit-dashboard)
7. [Supported Datasets](#supported-datasets)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- **Python 3.10+** (recommended)
- **Git**
- Optional but recommended for molecular datasets: **RDKit** (`pip install rdkit`)

### Clone and set up the environment

```bash
git clone https://github.com/IAyaanHere/ChemBench.git
cd ChemBench
```

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Install ChemBench as an editable package (registers the `chembench` module):

```bash
pip install -e .
```

### Optional dependencies

| Package | Purpose |
|---------|---------|
| `lightgbm` | LightGBM baseline (included in `requirements.txt`) |
| `torch` | Deep learning baselines (FCNN, CNN1D, GNN) |
| `streamlit`, `plotly` | Interactive dashboard (`app.py`) |
| `rdkit` | Scaffold splits and SMILES validation for molecular data |

---

## Architecture Overview

ChemBench is organized as a modular pipeline. Each layer has a single responsibility, making it easy to extend datasets or swap models without rewriting the benchmark harness.

```
chembench/
├── data/           # Loading, cleaning, and splitting
├── models/         # Baseline model implementations
├── evaluation/     # Metrics, plots, and evaluation reports
└── datasets/       # Standardized processed data per benchmark
```

### `chembench/data/`

| Module | Role |
|--------|------|
| `ChemBenchDataLoader` | Resolves processed CSV paths and exposes features/targets |
| `DataCleaner` | Missing-value imputation, outlier handling, SMILES validation |
| `DataSplitter` | Random, temporal, and scaffold-based train/val/test splits |

### `chembench/models/`

| Module | Role |
|--------|------|
| `baselines/sklearn_models.py` | Linear, Random Forest, Gradient Boosting, LightGBM, MLP |
| `baselines/dl_models.py` | PyTorch models (FCNN, CNN1D, GNN) via `PyTorchWrapper` |
| `baselines/base.py` | Shared `BaselineModel` API: `fit`, `predict`, `evaluate` |

### `chembench/evaluation/`

| Module | Role |
|--------|------|
| `evaluator.py` | `ModelEvaluator` — confusion matrices, prediction plots, markdown reports |

### Scripts and outputs (repository root)

| Path | Role |
|------|------|
| `scripts/run_benchmarks.py` | Full multi-model benchmark orchestration |
| `scripts/analyze_results.py` | Generates `results/analysis.md` from the leaderboard |
| `app.py` | Streamlit dashboard for interactive exploration |
| `results/` | Leaderboard CSV/JSON/MD and analysis artifacts |

---

## Quick Start (5-Minute Tutorial)

This example loads the **ESOL** solubility dataset, splits it with scaffold partitioning, and trains a **Random Forest** baseline.

```python
from chembench.data.loader import ChemBenchDataLoader
from chembench.data.cleaner import DataCleaner
from chembench.data.splitter import DataSplitter
from chembench.models.baselines.sklearn_models import RandomForestModel

# 1. Load data
loader = ChemBenchDataLoader("esol")
df = loader.load_data()
target_col = loader.get_targets().columns[0]
print(f"Loaded {len(df)} rows; target = {target_col}")

# 2. Clean
cleaner = DataCleaner()
df = cleaner.handle_missing(df, strategy="mean")
df = cleaner.validate_chemistry(df, smiles_col="smiles")

# 3. Split (scaffold split preserves chemical novelty)
splitter = DataSplitter()
train, val, test = splitter.scaffold_split(df, smiles_col="smiles", test_size=0.2, val_size=0.1)

# 4. Prepare numeric features
feature_cols = (
    train.drop(columns=[target_col, "smiles"])
    .select_dtypes(include=["number"])
    .columns.tolist()
)

X_train = train[feature_cols]
y_train = train[target_col]
X_test = test[feature_cols]
y_test = test[target_col]

# 5. Train and evaluate
model = RandomForestModel(n_estimators=100)
model.fit(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
print(metrics)  # e.g. {'mse': ..., 'mae': ..., 'rmse': ...}
```

Expected output includes regression metrics (`mse`, `mae`, `rmse`) because ESOL is a continuous solubility prediction task.

---

## Running Benchmarks

The benchmark script trains **all eight baseline models** on a dataset (or on the full suite), records timing and memory, and exports a leaderboard.

### Single dataset

```bash
python scripts/run_benchmarks.py --dataset esol
```

Other examples:

```bash
python scripts/run_benchmarks.py --dataset tennessee_eastman
python scripts/run_benchmarks.py --dataset chemixhub/solubility
python scripts/run_benchmarks.py --dataset qm9
```

### Full suite (10 datasets)

```bash
python scripts/run_benchmarks.py --all
```

This runs benchmarks for:

- `tennessee_eastman`, `qm9`, `chemixhub/solubility`
- MoleculeNet: `esol`, `freesolv`, `lipophilicity`, `hiv`, `bace`, `tox21`, `bbbp`

### CLI options

| Flag | Description |
|------|-------------|
| `--dataset <name>` | Run one dataset (default: `esol` if neither flag is set) |
| `--all` | Run all canonical datasets |
| `--output-dir <path>` | Output directory (default: `./results`) |

### Outputs

After a successful run, you will find:

| File | Description |
|------|-------------|
| `results/leaderboard.csv` | Machine-readable results table |
| `results/leaderboard.json` | JSON export for tooling |
| `results/leaderboard.md` | Markdown summary by dataset |

The benchmark loop is fault-tolerant: if one model fails on a dataset, the error is logged and the run continues.

---

## Analyzing Results

Generate a formal **Results & Discussion** report from the leaderboard:

```bash
python scripts/analyze_results.py
```

This writes `results/analysis.md`, including per-dataset winners, aggregate timing statistics, and speed-vs-accuracy commentary.

---

## Viewing Results (Streamlit Dashboard)

Launch the interactive dashboard from the project root:

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically **http://localhost:8501**).

### Dashboard features

- **Sidebar filters** — dataset and model family (Traditional ML vs Deep Learning)
- **Leaderboard tab** — sortable table with CSV download
- **Analysis tab** — Plotly bar and scatter charts (scores and train time)
- **Model Details tab** — rendered `results/analysis.md` report

Ensure `results/leaderboard.csv` exists (run benchmarks first) and that `streamlit` and `plotly` are installed.

---

## Supported Datasets

| Dataset | Type | Task |
|---------|------|------|
| `tennessee_eastman` | Process | Fault detection (classification) |
| `qm9` | Molecular | Quantum properties (regression) |
| `chemixhub/<task>` | Mixture | Property prediction (regression) |
| `esol`, `freesolv`, `lipophilicity` | Molecular | Property regression |
| `hiv`, `bace`, `bbbp` | Molecular | Classification |
| `tox21` | Molecular | Multi-task toxicity |

Processed files live under `chembench/datasets/<name>/processed/`. See each dataset’s `README.md` for domain-specific notes.

---

## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| `ModuleNotFoundError: chembench` | Run `pip install -e .` from the repo root |
| RDKit / scaffold split errors | `pip install rdkit` |
| LightGBM missing | `pip install lightgbm` |
| Empty leaderboard in dashboard | Run `python scripts/run_benchmarks.py` first |
| QM9 path not found | Ensure `chembench/datasets/qm9/processed/qm9_molecules.csv` exists |

For contributions, new datasets, and code standards, see [CONTRIBUTING.md](../CONTRIBUTING.md) in the repository root.

---

*ChemBench — Standardized ML benchmarks for chemical engineering.*

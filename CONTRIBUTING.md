# Contributing to ChemBench

Thank you for your interest in contributing to **ChemBench**! This project aims to be a reliable, open benchmark suite for machine learning in chemical engineering. We welcome bug reports, feature ideas, documentation improvements, new datasets, and model contributions.

---

## Table of  Contents

1. [Code of Conduct](#code-of-conduct)
2. [How to Report Bugs](#how-to-report-bugs)
3. [How to Suggest Features](#how-to-suggest-features)
4. [Development Setup](#development-setup)
5. [Adding a New Dataset](#adding-a-new-dataset)
6. [Code Style Guidelines](#code-style-guidelines)
7. [Pull Request Process](#pull-request-process)
8. [Pull Request Template](#pull-request-template)

---

## Code of Conduct

Be respectful, constructive, and inclusive. Focus feedback on the work, not the person. Maintainers may close issues or PRs that violate these expectations.

---

## How to Report Bugs

1. **Search existing issues** to avoid duplicates.
2. Open a new issue with:
   - A clear, descriptive title
   - Steps to reproduce (commands, dataset name, Python version)
   - Expected vs actual behavior
   - Full error traceback (if applicable)
   - Environment details (OS, Python version, key package versions)

Example:

```text
**Steps to reproduce**
1. python scripts/run_benchmarks.py --dataset lipophilicity

**Expected**
Leaderboard row for Random Forest with valid MSE.

**Actual**
ValueError: No numeric feature columns...

**Environment**
- Windows 11, Python 3.11.5
- chembench 0.1.0 (editable install)
```

---

## How to Suggest Features

Open an issue labeled as a feature request (or use the feature template if available) and include:

- **Problem** — What limitation are you hitting today?
- **Proposed solution** — How should ChemBench behave?
- **Alternatives considered** — Other designs you evaluated
- **Impact** — Who benefits (researchers, industry, educators)?

Large features (new model families, new split strategies) benefit from a short design comment before you open a PR.

---

## Development Setup

```bash
git clone https://github.com/IAyaanHere/ChemBench.git
cd ChemBench
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
```

Run tests or smoke checks relevant to your change (e.g. a single-dataset benchmark):

```bash
python scripts/run_benchmarks.py --dataset esol
```

---

## Adding a New Dataset

New benchmarks must follow the ChemBench directory layout so `ChemBenchDataLoader` and `run_benchmarks.py` can discover them automatically.

### Directory structure

```text
chembench/datasets/<your_dataset>/
├── README.md                 # Domain description, citations, split notes
├── processed/
│   ├── <your_dataset>.csv    # Primary processed table (or task-specific CSVs)
│   └── metadata.json         # Required schema manifest
├── exploratory/              # Optional EDA outputs
└── raw/                      # Optional raw downloads (gitignored if large)
```

### Processed data requirements

- Store **clean, analysis-ready** tables under `processed/`.
- Include a **target column** with a stable name documented in `metadata.json`.
- For **molecular** data: include a `smiles` or `mol` column.
- For **process** data: feature columns should be numeric; include `label` or a documented target name.
- For **mixture** data: use `chemixhub/<task>` naming or document task CSV filenames in metadata.

### `metadata.json` format

Place `metadata.json` in `chembench/datasets/<your_dataset>/processed/`.

**Single-task dataset (minimal example):**

```json
{
  "dataset_name": "my_dataset",
  "num_rows": 1000,
  "num_columns": 8,
  "columns": [
    "feature_a",
    "feature_b",
    "smiles",
    "target_property"
  ]
}
```

**Multi-task dataset (ChemixHub-style example):**

```json
{
  "dataset_name": "MyMixtureHub",
  "description": "Short description of the benchmark",
  "tasks": [
    {
      "name": "solubility",
      "property": "Solubility (mol/L)",
      "num_samples": 3500,
      "file": "solubility.csv"
    }
  ]
}
```

### Registering the dataset in code

1. Add the dataset name to `ChemBenchDataLoader` constants in `chembench/data/loader.py` if it introduces a new type (`molecular`, `process`, `mixture`).
2. Implement `_target_columns()` logic if the target column name is non-standard.
3. Add the dataset to `ALL_BENCHMARK_DATASETS` in `scripts/run_benchmarks.py` if it should run with `--all`.
4. Document usage in `docs/user_guide.md` and add a dataset `README.md`.

### Validation checklist

- [ ] `metadata.json` is valid JSON and matches the processed CSV(s)
- [ ] `ChemBenchDataLoader("<name>").load_data()` succeeds
- [ ] `python scripts/run_benchmarks.py --dataset <name>` completes without dataset-level failures
- [ ] README cites data sources and license terms

---

## Code Style Guidelines

ChemBench follows **PEP 8** with these conventions:

| Topic | Guideline |
|-------|-----------|
| **Formatting** | 4-space indentation; max line length **100** characters where practical |
| **Imports** | Standard library → third party → `chembench` (isort-style grouping) |
| **Naming** | `snake_case` for functions/variables; `PascalCase` for classes |
| **Types** | Type hints encouraged on public functions |
| **Docstrings** | Module-level and public class/method docstrings for APIs |
| **Logging** | Use the `logging` module in scripts; avoid bare `print()` in library code |
| **Dependencies** | Add new runtime deps to `requirements.txt` and `setup.py` when appropriate |

### Style tools (recommended)

```bash
pip install ruff black
ruff check chembench scripts
black chembench scripts
```

Match the style of surrounding files — prefer small, focused diffs over large refactors in contribution PRs.

---

## Pull Request Process

1. **Fork** the repository and create a branch from `main`:
   - `feature/<short-description>`
   - `fix/<short-description>`
   - `docs/<short-description>`
2. **Make focused changes** — one logical concern per PR when possible.
3. **Update documentation** if you change CLI flags, dataset layout, or public APIs.
4. **Verify** benchmarks or tests affected by your change.
5. **Open a PR** using the template below and link related issues (`Fixes #123`).

Maintainers will review for correctness, reproducibility, and alignment with ChemBench’s benchmarking goals. You may be asked to rebase or split large PRs.

---

## Pull Request Template

Copy the following into your PR description:

```markdown
## Summary

<!-- What does this PR do? (1–3 sentences) -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] New dataset
- [ ] Documentation
- [ ] Refactor / chore

## Related issues

<!-- Fixes #123 / Relates to #456 -->

## Changes made

-
-

## Testing performed

<!-- Commands run, datasets tested, screenshots for UI changes -->

- [ ] `python scripts/run_benchmarks.py --dataset <name>`
- [ ] Other: 

## Checklist

- [ ] Code follows PEP 8 / project conventions
- [ ] Documentation updated (if applicable)
- [ ] `metadata.json` added/updated for new datasets
- [ ] No secrets or large binary files committed
- [ ] Dataset licenses and citations documented
```

---

Thank you for helping make ChemBench a stronger benchmark for the chemical engineering ML community.

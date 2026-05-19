<div align="center">

# 🧪 ChemBench — Standardized Benchmark Suite for ML in Chemical Engineering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch & Scikit-Learn](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Scikit--Learn-orange.svg)]()
[![Dashboard: Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)]()

### Bridging Artificial Intelligence and Chemical Engineering

An open-source benchmark infrastructure providing curated datasets, baseline models, standardized preprocessing pipelines, and interactive evaluation tools for Chemical Engineering ML research.

</div>

---

# 📌 Why ChemBench?

Machine Learning research in Chemical Engineering suffers from:
- Scattered datasets
- Repetitive preprocessing
- No standardized benchmarks

ChemBench provides a unified framework for:
- Dataset loading
- Data preprocessing
- Standardized train/test splits
- Baseline benchmarking
- Evaluation metrics
- Interactive visualization

---

# ✨ Features

- 📦 Curated benchmark datasets
- 🤖 Traditional ML + Deep Learning baselines
- 🧹 Standardized preprocessing pipelines
- 📊 Streamlit dashboard
- ⚡ Unified API for experiments

---

# ⚡ Installation

```bash
git clone https://github.com/IAyaanHere/ChemBench.git

cd ChemBench

pip install -e .
```

---

# 🚀 Quick Start

```python
from chembench.data.loader import ChemBenchDataLoader
from chembench.models import RandomForestModel

loader = ChemBenchDataLoader("esol")

X_train, X_test, y_train, y_test = loader.get_splits()

model = RandomForestModel()
model.fit(X_train, y_train)

metrics = model.evaluate(X_test, y_test)

print(metrics)
```

---

# 📈 Launch Dashboard

```bash
streamlit run app.py
```

---

# 🗃️ Supported Datasets

| Dataset | Domain | Task |
|---|---|---|
| Tennessee Eastman Process | Process Control | Classification |
| QM9 | Quantum Chemistry | Regression |
| ESOL | Molecular Solubility | Regression |
| MoleculeNet | Molecular ML | Multi-task |
| CheMixHub | Mixture Properties | Regression |

---

# 🛠️ Tech Stack

- Python
- PyTorch
- Scikit-Learn
- Pandas
- RDKit
- Streamlit

---

# 🤝 Contributing

Contributions are welcome.

You can contribute by:
- Adding datasets
- Implementing models
- Improving pipelines
- Fixing bugs
- Improving documentation

---

# 📄 License

MIT License

---

<div align="center">

### Built for AI-driven Chemical Engineering ⚗️

⭐ Star the repository if you like the project.

</div>

# 🧪 ChemBench: The Standardized Benchmark Suite for ML in Chemical Engineering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**ChemBench** is an open-source framework and benchmark suite designed to bridge the gap between advanced Machine Learning and core Chemical Engineering. It provides standardized, unified access to complex process engineering datasets, robust baselines, and evaluation metrics.

## ✨ Features
- 📦 **Curated Datasets:** Tennessee Eastman Process (TEP), QM9, CheMixHub, and more.
- 🧹 **Built-in Preprocessing:** Ready-to-use techniques for handling class imbalance and time-series normalization.
- 📊 **Standardized Evaluation:** Domain-specific metrics alongside standard ML metrics.

## 💻 Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/IAyaanHere/ChemBench.git
cd ChemBench
pip install -e .
```

## 🚀 Quick Start

```python
from chembench.datasets import DatasetLoader
from chembench.baselines import RandomForestBaseline

# Load standardized data
loader = DatasetLoader(name='tennessee_eastman')
X_train, X_test, y_train, y_test = loader.get_splits()

# Train baseline
model = RandomForestBaseline()
model.fit(X_train, y_train)
```

# Contributing to ChemBench 🧪

First off, thank you for considering contributing to ChemBench! It's researchers and developers like you who make open-source chemical engineering tools reliable and standardized.

## 📊 Adding a New Dataset
We welcome new datasets relevant to Chemical Engineering, Process Control, Thermodynamics, and Molecular ML. To add a dataset, please ensure:
1. **Public Availability:** The data must be openly accessible (e.g., CC0, MIT, Public Domain).
2. **Relevance:** Must solve a specific ChemE/Chemistry problem.
3. **Minimum Size:** At least 500 samples to ensure robust benchmarking.
4. **Documentation:** You must provide a comprehensive description in `DATASETS.md`.

**Steps to add:**
1. Write a download script in `scripts/download_<dataset>.py`.
2. Ensure the data is cleaned and split into `train.csv`, `val.csv`, and `test.csv`.
3. Submit a Pull Request.

## 🤖 Adding a Baseline Model
If you have a new ML architecture (e.g., a novel Graph Neural Network or Time-Series Transformer) that outperforms our baselines:
1. Create a new class in `chembench/baselines/` inheriting from our `BaselineModel`.
2. Include the hyperparameter configuration.
3. Ensure it integrates smoothly with the `Evaluator` class.

## 💻 Code Style & Pull Requests
- We follow **PEP 8** for Python code.
- Please use `black` for code formatting.
- Ensure all tests pass before submitting a PR.
- Write clear, descriptive commit messages.

Thank you for helping bridge the gap between Machine Learning and Chemical Engineering! 🚀

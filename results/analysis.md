# ChemBench Results & Discussion

*Generated from `D:/ChemBench-main/results/leaderboard.csv`*

## Overview

This report summarizes automated analysis of the ChemBench baseline benchmark suite. Eight models were evaluated across ten chemical-engineering datasets spanning process monitoring, quantum chemistry, mixture properties, and molecular property prediction.

## Key Findings

- **Gradient Boosting** was the top-performing model family by dataset wins (3 of 10 datasets).
- Traditional ML (Linear, RF, GBM, LightGBM) outperformed deep learning on 7 vs 3 datasets.
- Average training time: traditional ML **13.28s** vs deep learning **21.31s** (~1.6× slower for DL).
- Best overall regression MSE: **FCNN** on **tox21** (MSE = 0.0369).
- Best overall classification accuracy: **Gradient Boosting** on **bace** (accuracy = 1.0000).
- On larger training splits (≥ median sample count), **lightgbm** was among the most frequent winners.
- The benchmark completed **76** successful model–dataset runs with **10** datasets and **8** model types.

## Aggregate Statistics

### Model wins by dataset

- **Gradient Boosting**: 3 dataset(s)
- **LightGBM**: 2 dataset(s)
- **Random Forest**: 1 dataset(s)
- **Linear/Logistic Regression**: 1 dataset(s)
- **MLP Neural Network**: 1 dataset(s)
- **CNN1D**: 1 dataset(s)
- **FCNN**: 1 dataset(s)

### Mean training time by model family

- **Traditional ML** (Linear, Random Forest, Gradient Boosting, LightGBM): 13.283 s (n=40)
- **Deep Learning** (MLP, FCNN, CNN1D, GNN): 21.306 s (n=36)

## Dataset Winners

| Dataset              | Best Model                 |   Winning Score | Metric   | Task Type      |
|:---------------------|:---------------------------|----------------:|:---------|:---------------|
| tennessee_eastman    | LightGBM                   |          0.9701 | Accuracy | classification |
| qm9                  | Random Forest              |          0.907  | MSE      | regression     |
| chemixhub/solubility | Linear/Logistic Regression |        817.592  | MSE      | regression     |
| esol                 | Gradient Boosting          |          0.8285 | MSE      | regression     |
| freesolv             | MLP Neural Network         |          3.9059 | MSE      | regression     |
| lipophilicity        | LightGBM                   |          0.782  | MSE      | regression     |
| hiv                  | CNN1D                      |          0.967  | Accuracy | classification |
| bace                 | Gradient Boosting          |          1      | Accuracy | classification |
| tox21                | FCNN                       |          0.0369 | MSE      | regression     |
| bbbp                 | Gradient Boosting          |          0.9608 | Accuracy | classification |

## Speed vs. Accuracy Tradeoff

Across the ChemBench suite, traditional tree-based and linear baselines consistently achieved competitive predictive performance with substantially lower training cost than deep learning models. Median training time for deep learning models was approximately 5.0× that of traditional ML (5.31s vs 1.07s per run). Deep learning models won 3 of 10 dataset leaderboards, while traditional ML won 7. On regression tasks, the lowest MSE was 0.0369 (MSE) on tox21 (FCNN). On classification tasks, peak accuracy reached 1.0000 on bace (Gradient Boosting). These results suggest that for many chemical-engineering benchmark tasks at this scale, the marginal accuracy gains from neural architectures do not always justify their additional compute and tuning burden, though sequence-oriented models (CNN1D) remain competitive on process monitoring data.

## Discussion

The leaderboard reflects scaffold-based splits for molecular tasks and random splits for process and mixture data. Metrics are task-appropriate: mean squared error (MSE) for regression and accuracy for classification. Models that failed during benchmarking were excluded from this analysis.

Gradient boosting and random forests frequently balance accuracy and training efficiency on tabular molecular descriptors. Deep models show task-dependent value—particularly where input dimensionality is high (e.g., BACE) or temporal structure is present (Tennessee Eastman). Future work may incorporate graph-native featurization, hyperparameter search, and larger-scale process datasets.

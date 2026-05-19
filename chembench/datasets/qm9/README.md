# QM9

## Overview

QM9 is a comprehensive dataset of 130K molecules with their quantum mechanical properties calculated using density functional theory.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Dataset | qm9 |
| Sample Rows (from first CSV) | 130831 |
| Columns | 20 |
| Target Type | Regression/Classification |
| Split | Train/Val/Test |

## File Structure

```
qm9/
├── exploratory/
│   ├── atom_count_dist.png
│   ├── correlation_heatmap.png
│   ├── mw_vs_gap.png
│   ├── property_distributions.png
│   ├── qm9_eda_report.md
├── processed/
│   ├── qm9_molecules.csv
│   ├── test.csv
│   ├── train.csv
│   ├── val.csv
├── raw/
│   ├── data_v3.pt
│   ├── pre_filter.pt
│   ├── pre_transform.pt
│   ├── qm9_v3.pt
```

## Feature Descriptions

The QM9 dataset includes quantum mechanical properties:

- **smiles**: SMILES string representation
- **molecule_id**: Unique identifier
- Quantum properties: HOMO-LUMO gap, atomization energy, dipole moment, and more

## Usage Examples

### Loading the Dataset

```python
from chembench.data.loader import ChemBenchDataLoader

loader = ChemBenchDataLoader('qm9')
df = loader.load_data()
print(f"Dataset shape: {df.shape}")
```

### Cleaning the Data

```python
from chembench.data.cleaner import DataCleaner

cleaner = DataCleaner()
df = cleaner.handle_missing(df, strategy='mean')
df = cleaner.remove_outliers(df, method='iqr', threshold=1.5)
```

### Splitting the Data

```python
from chembench.data.splitter import DataSplitter

splitter = DataSplitter()
train, val, test, y_train, y_val, y_test = splitter.random_split(
    df, target_col='label', test_size=0.2, val_size=0.1
)
```

### Training a Model

```python
from sklearn.ensemble import RandomForestClassifier

X_train = train.drop('label', axis=1)
y_train = train['label']

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

## References

For more information about this dataset, refer to the main ChemBench documentation at the project root.

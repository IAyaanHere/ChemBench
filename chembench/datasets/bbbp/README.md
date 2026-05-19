# BBBP

## Overview

BBBP (Blood-Brain Barrier Penetration) is a dataset for predicting molecules that can cross the blood-brain barrier.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Dataset Name | bbbp |
| Number of Rows | 2050 |
| Number of Columns | 4 |
| Target Type | Regression/Classification |
| Split | Train/Val/Test |

## File Structure

```
bbbp/
├── exploratory/
│   ├── metadata.json
│   ├── mw_vs_atoms.png
│   ├── target_distribution.png
├── processed/
│   ├── bbbp.csv
│   ├── metadata.json
```

## Feature Descriptions

The dataset contains the following features:

- **num**: [Add description]
- **name**: [Add description]
- **p_np**: [Add description]
- **smiles**: [Add description]

## Usage Examples

### Loading the Dataset

```python
from chembench.data.loader import ChemBenchDataLoader

loader = ChemBenchDataLoader('bbbp')
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
train, val, test = splitter.scaffold_split(
    df, smiles_col='smiles', test_size=0.2, val_size=0.1
)
```

### Training a Model

```python
from sklearn.ensemble import RandomForestRegressor

X_train = train[['Molecular Weight', 'Polar Surface Area']]
y_train = train['measured log solubility in mols per litre']

model = RandomForestRegressor()
model.fit(X_train, y_train)
```

## References

For more information about this dataset, refer to the main ChemBench documentation at the project root.

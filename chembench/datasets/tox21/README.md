# TOX21

## Overview

Tox21 is a large-scale molecular toxicity dataset from the NCATS Tox21 challenge, containing predictions across multiple toxicity endpoints.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Dataset Name | tox21 |
| Number of Rows | 7831 |
| Number of Columns | 14 |
| Target Type | Regression/Classification |
| Split | Train/Val/Test |

## File Structure

```
tox21/
├── exploratory/
│   ├── metadata.json
│   ├── mw_vs_atoms.png
│   ├── target_distribution.png
├── processed/
│   ├── metadata.json
│   ├── tox21.csv
```

## Feature Descriptions

The dataset contains the following features:

- **NR-AR**: [Add description]
- **NR-AR-LBD**: [Add description]
- **NR-AhR**: [Add description]
- **NR-Aromatase**: [Add description]
- **NR-ER**: [Add description]
- **NR-ER-LBD**: [Add description]
- **NR-PPAR-gamma**: [Add description]
- **SR-ARE**: [Add description]
- **SR-ATAD5**: [Add description]
- **SR-HSE**: [Add description]
- **SR-MMP**: [Add description]
- **SR-p53**: [Add description]
- **mol_id**: [Add description]
- **smiles**: [Add description]

## Usage Examples

### Loading the Dataset

```python
from chembench.data.loader import ChemBenchDataLoader

loader = ChemBenchDataLoader('tox21')
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

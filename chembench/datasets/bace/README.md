# BACE

## Overview

BACE is a dataset for predicting human beta-secretase inhibitors, useful for Alzheimer's disease research.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Dataset Name | bace |
| Number of Rows | 1513 |
| Number of Columns | 595 |
| Target Type | Regression/Classification |
| Split | Train/Val/Test |

## File Structure

```
bace/
├── exploratory/
│   ├── metadata.json
│   ├── mw_vs_atoms.png
│   ├── target_distribution.png
├── processed/
│   ├── bace.csv
│   ├── metadata.json
├── README.md
```

## Feature Descriptions

The dataset contains the following features:

- **mol**: [Add description]
- **CID**: [Add description]
- **Class**: [Add description]
- **Model**: [Add description]
- **pIC50**: [Add description]
- **MW**: [Add description]
- **AlogP**: [Add description]
- **HBA**: [Add description]
- **HBD**: [Add description]
- **RB**: [Add description]
- **HeavyAtomCount**: [Add description]
- **ChiralCenterCount**: [Add description]
- **ChiralCenterCountAllPossible**: [Add description]
- **RingCount**: [Add description]
- **PSA**: [Add description]
- **Estate**: [Add description]
- **MR**: [Add description]
- **Polar**: [Add description]
- **sLi_Key**: [Add description]
- **ssBe_Key**: [Add description]
- **ssssBem_Key**: [Add description]
- **sBH2_Key**: [Add description]
- **ssBH_Key**: [Add description]
- **sssB_Key**: [Add description]
- **ssssBm_Key**: [Add description]
- **sCH3_Key**: [Add description]
- **dCH2_Key**: [Add description]
- **ssCH2_Key**: [Add description]
- **tCH_Key**: [Add description]
- **dsCH_Key**: [Add description]
- **aaCH_Key**: [Add description]
- **sssCH_Key**: [Add description]
- **ddC_Key**: [Add description]
- **tsC_Key**: [Add description]
- **dssC_Key**: [Add description]
- **aasC_Key**: [Add description]
- **aaaC_Key**: [Add description]
- **ssssC_Key**: [Add description]
- **sNH3_Key**: [Add description]
- **sNH2_Key**: [Add description]
- **ssNH2_Key**: [Add description]
- **dNH_Key**: [Add description]
- **ssNH_Key**: [Add description]
- **aaNH_Key**: [Add description]
- **tN_Key**: [Add description]
- **sssNH_Key**: [Add description]
- **dsN_Key**: [Add description]
- **aaN_Key**: [Add description]
- **sssN_Key**: [Add description]
- **ddsN_Key**: [Add description]

## Usage Examples

### Loading the Dataset

```python
from chembench.data.loader import ChemBenchDataLoader

loader = ChemBenchDataLoader('bace')
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

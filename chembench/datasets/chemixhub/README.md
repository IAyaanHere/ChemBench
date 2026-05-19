# CheMixHub

## Overview

CheMixHub is a dataset containing physical and chemical properties of binary mixtures.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Dataset Name | CheMixHub |
| Number of Rows | N/A |
| Number of Columns | N/A |
| Target Type | Regression/Classification |
| Split | Train/Val/Test |

## File Structure

```
chemixhub/
├── exploratory/
│   ├── chemixhub_eda_report.md
│   ├── mixture_complexity.png
│   ├── task_distributions.png
│   ├── temp_pressure_ranges.png
├── processed/
│   ├── boiling_point.csv
│   ├── critical_properties.csv
│   ├── density.csv
│   ├── flash_point.csv
│   ├── heat_capacity.csv
│   ├── metadata.json
│   ├── partition_coefficient.csv
│   ├── PROCESSING_REPORT.md
│   ├── refractive_index.csv
│   ├── SCHEMA.md
│   ├── solubility.csv
│   ├── surface_tension.csv
│   ├── thermal_conductivity.csv
│   ├── viscosity.csv
├── raw/
│   ├── github_repo
```

## Feature Descriptions

Features are derived from the molecular structure or process measurements. Please refer to the dataset reference or metadata.json for detailed descriptions.

## Usage Examples

### Loading the Dataset

```python
from chembench.data.loader import ChemBenchDataLoader

loader = ChemBenchDataLoader('chemixhub')
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

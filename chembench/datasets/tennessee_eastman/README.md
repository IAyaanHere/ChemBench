# Tennessee Eastman Process (TEP)

## Overview

Tennessee Eastman Process (TEP) is a benchmark dataset for process fault detection and diagnosis in chemical engineering. It contains sensor readings and process variables from normal and faulty operations.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Dataset | tennessee_eastman |
| Sample Rows (from first CSV) | 21120 |
| Columns | 53 |
| Target Type | Regression/Classification |
| Split | Train/Val/Test |

## File Structure

```
tennessee_eastman/
├── exploratory/
│   ├── correlation_heatmap.png
│   ├── timeseries_plot.png
├── processed/
│   ├── test.csv
│   ├── train.csv
│   ├── val.csv
├── raw/
│   ├── fault_free_testing.csv
│   ├── fault_free_training.csv
│   ├── faulty_testing.csv
│   ├── faulty_training.csv
```

## Feature Descriptions

The TEP dataset includes 52 process variables from sensors and control systems:

- **XMEAS(1-41)**: Process measurements (temperatures, pressures, flows, concentrations)
- **XMV(1-11)**: Manipulated variables (controller outputs)
- **label**: 0 for normal operation, 1 for fault condition

For detailed documentation on each variable, refer to the original TEP documentation.

## Usage Examples

### Loading the Dataset

```python
from chembench.data.loader import ChemBenchDataLoader

loader = ChemBenchDataLoader('tennessee_eastman')
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

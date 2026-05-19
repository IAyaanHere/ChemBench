import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetDocumenter:
    def __init__(self):
        self.datasets_root = Path(__file__).resolve().parent.parent / 'chembench' / 'datasets'

    def get_all_datasets(self):
        """Return list of dataset directories."""
        datasets = []
        for item in sorted(self.datasets_root.iterdir()):
            if item.is_dir() and not item.name.startswith('_'):
                datasets.append(item)
        return datasets

    def load_metadata(self, dataset_dir: Path) -> Optional[Dict]:
        """Load metadata.json for a dataset, return None if not found."""
        metadata_path = dataset_dir / 'processed' / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding='utf-8') as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning('Failed to load metadata for %s: %s', dataset_dir.name, exc)
        return None

    def generate_readme(self, dataset_dir: Path, metadata: Optional[Dict]) -> str:
        """Generate README content from dataset metadata."""
        dataset_name = dataset_dir.name
        title = self._get_title(dataset_name, metadata)
        description = self._get_description(dataset_name)
        stats_table = self._generate_stats_table(metadata, dataset_dir)
        file_structure = self._get_file_structure(dataset_dir)
        features_section = self._get_features_section(metadata, dataset_name)
        usage_section = self._get_usage_section(dataset_name)

        readme = f"""# {title}

## Overview

{description}

## Dataset Statistics

{stats_table}

## File Structure

```
{file_structure}
```

## Feature Descriptions

{features_section}

## Usage Examples

{usage_section}

## References

For more information about this dataset, refer to the main ChemBench documentation at the project root.
"""
        return readme

    @staticmethod
    def _get_title(dataset_name: str, metadata: Optional[Dict]) -> str:
        """Get dataset title."""
        if metadata and 'dataset_name' in metadata:
            name = metadata['dataset_name']
            if name.lower() == 'chemi' or name.lower() == 'chemixhub':
                return 'CheMixHub'
            return name.upper()
        dataset_titles = {
            'tennessee_eastman': 'Tennessee Eastman Process (TEP)',
            'esol': 'ESOL',
            'freesolv': 'FreeSolv',
            'lipophilicity': 'Lipophilicity',
            'hiv': 'HIV',
            'bace': 'BACE',
            'tox21': 'Tox21',
            'bbbp': 'BBBP',
            'qm9': 'QM9',
            'chemixhub': 'CheMixHub',
        }
        return dataset_titles.get(dataset_name, dataset_name.upper())

    @staticmethod
    def _get_description(dataset_name: str) -> str:
        """Get dataset description."""
        descriptions = {
            'tennessee_eastman': 'Tennessee Eastman Process (TEP) is a benchmark dataset for process fault detection and diagnosis in chemical engineering. It contains sensor readings and process variables from normal and faulty operations.',
            'esol': 'ESOL (Estimated Solubility) is a molecular dataset containing organic compounds with their predicted and measured solubility properties in aqueous solution.',
            'freesolv': 'FreeSolv is a dataset of calculated and experimental hydration free energies for small organic molecules.',
            'lipophilicity': 'Lipophilicity is a molecular dataset containing hydration free energies and octanol-water partition coefficients for pharmaceutical compounds.',
            'hiv': 'HIV is a dataset for predicting HIV inhibitors, containing molecular structures and their activity labels.',
            'bace': 'BACE is a dataset for predicting human beta-secretase inhibitors, useful for Alzheimer\'s disease research.',
            'tox21': 'Tox21 is a large-scale molecular toxicity dataset from the NCATS Tox21 challenge, containing predictions across multiple toxicity endpoints.',
            'bbbp': 'BBBP (Blood-Brain Barrier Penetration) is a dataset for predicting molecules that can cross the blood-brain barrier.',
            'qm9': 'QM9 is a comprehensive dataset of 130K molecules with their quantum mechanical properties calculated using density functional theory.',
            'chemixhub': 'CheMixHub is a dataset containing physical and chemical properties of binary mixtures.',
        }
        return descriptions.get(dataset_name, f'This is the {dataset_name} dataset.')

    @staticmethod
    def _generate_stats_table(metadata: Optional[Dict], dataset_dir: Path) -> str:
        """Generate dataset statistics table."""
        if metadata is None:
            # Fallback: try to compute stats from processed files
            return DatasetDocumenter._compute_stats_fallback(dataset_dir)

        num_rows = metadata.get('num_rows') or metadata.get('rows', 'N/A')
        num_cols = metadata.get('num_columns') or metadata.get('columns', 'N/A')
        # Handle case where 'columns' is an int
        if isinstance(num_cols, int):
            num_cols_str = str(num_cols)
        else:
            num_cols_str = str(len(num_cols)) if isinstance(num_cols, list) else str(num_cols)
        dataset_name = metadata.get('dataset_name', 'Unknown')

        table = f"""| Metric | Value |
|--------|-------|
| Dataset Name | {dataset_name} |
| Number of Rows | {num_rows} |
| Number of Columns | {num_cols_str} |
| Target Type | Regression/Classification |
| Split | Train/Val/Test |"""

        return table

    @staticmethod
    def _compute_stats_fallback(dataset_dir: Path) -> str:
        """Compute basic stats from CSV files when metadata is missing."""
        try:
            import pandas as pd
            processed_dir = dataset_dir / 'processed'
            if processed_dir.exists():
                csv_files = list(processed_dir.glob('*.csv'))
                if csv_files:
                    df = pd.read_csv(csv_files[0])
                    num_rows = len(df)
                    num_cols = len(df.columns)
                    return f"""| Metric | Value |
|--------|-------|
| Dataset | {dataset_dir.name} |
| Sample Rows (from first CSV) | {num_rows} |
| Columns | {num_cols} |
| Target Type | Regression/Classification |
| Split | Train/Val/Test |"""
        except Exception:
            pass

        return """| Metric | Value |
|--------|-------|
| Dataset | See processed CSV files |
| Sample Rows | N/A |
| Columns | N/A |
| Target Type | Regression/Classification |
| Split | Train/Val/Test |"""

    @staticmethod
    def _get_file_structure(dataset_dir: Path) -> str:
        """Generate file structure tree."""
        structure_lines = [f"{dataset_dir.name}/"]
        for item in sorted(dataset_dir.iterdir()):
            if item.name.startswith('_') or item.name.startswith('.'):
                continue
            if item.is_dir():
                structure_lines.append(f"├── {item.name}/")
                for subitem in sorted(item.iterdir()):
                    if not subitem.name.startswith('_') and not subitem.name.startswith('.'):
                        structure_lines.append(f"│   ├── {subitem.name}")
            else:
                structure_lines.append(f"├── {item.name}")
        return '\n'.join(structure_lines)

    @staticmethod
    def _get_features_section(metadata: Optional[Dict], dataset_name: str) -> str:
        """Generate feature descriptions section."""
        if metadata:
            # Try 'columns' key first (list of column names)
            if 'columns' in metadata and isinstance(metadata['columns'], list):
                columns = metadata['columns']
                feature_lines = ['The dataset contains the following features:']
                feature_lines.append('')
                for col in columns:
                    if col.lower() not in {'label', 'target', 'y'}:
                        feature_lines.append(f'- **{col}**: [Add description]')
                return '\n'.join(feature_lines)
            # Try 'features' key (list of feature names)
            if 'features' in metadata and isinstance(metadata['features'], list):
                features = metadata['features']
                feature_lines = ['The dataset contains the following features:']
                feature_lines.append('')
                for feat in features:
                    if feat.lower() not in {'label', 'target', 'y'}:
                        feature_lines.append(f'- **{feat}**: [Add description]')
                return '\n'.join(feature_lines)

        dataset_features = {
            'tennessee_eastman': '''The TEP dataset includes 52 process variables from sensors and control systems:

- **XMEAS(1-41)**: Process measurements (temperatures, pressures, flows, concentrations)
- **XMV(1-11)**: Manipulated variables (controller outputs)
- **label**: 0 for normal operation, 1 for fault condition

For detailed documentation on each variable, refer to the original TEP documentation.''',
            'esol': '''The ESOL dataset includes molecular descriptors computed from SMILES strings:

- **smiles**: SMILES string representation of the molecule
- **Molecular Weight**: Computed molecular weight
- **Polar Surface Area**: PSA descriptor
- **Number of H-Bond Donors/Acceptors**: Structural features
- **measured log solubility in mols per litre**: Target variable (measured solubility)''',
            'qm9': '''The QM9 dataset includes quantum mechanical properties:

- **smiles**: SMILES string representation
- **molecule_id**: Unique identifier
- Quantum properties: HOMO-LUMO gap, atomization energy, dipole moment, and more''',
        }

        return dataset_features.get(
            dataset_name,
            'Features are derived from the molecular structure or process measurements. Please refer to the dataset reference or metadata.json for detailed descriptions.',
        )

    @staticmethod
    def _get_usage_section(dataset_name: str) -> str:
        """Generate usage examples section."""
        usage = f'''### Loading the Dataset

```python
from chembench.data.loader import ChemBenchDataLoader

loader = ChemBenchDataLoader('{dataset_name}')
df = loader.load_data()
print(f"Dataset shape: {{df.shape}}")
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
'''

        if dataset_name in {'esol', 'freesolv', 'lipophilicity', 'hiv', 'bace', 'tox21', 'bbbp'}:
            usage += f'''train, val, test = splitter.scaffold_split(
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
```'''
        else:
            usage += f'''train, val, test, y_train, y_val, y_test = splitter.random_split(
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
```'''

        return usage

    def generate_for_dataset(self, dataset_dir: Path) -> bool:
        """Generate README for a single dataset."""
        logger.info('Generating documentation for %s...', dataset_dir.name)
        metadata = self.load_metadata(dataset_dir)
        readme_content = self.generate_readme(dataset_dir, metadata)
        readme_path = dataset_dir / 'README.md'

        try:
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            logger.info('✓ Generated %s', readme_path)
            return True
        except Exception as exc:
            logger.error('Failed to write README for %s: %s', dataset_dir.name, exc)
            return False

    def generate_for_all(self) -> int:
        """Generate READMEs for all datasets."""
        datasets = self.get_all_datasets()
        logger.info('Found %d datasets', len(datasets))

        success_count = 0
        for dataset_dir in datasets:
            if self.generate_for_dataset(dataset_dir):
                success_count += 1

        logger.info('Generated documentation for %d/%d datasets', success_count, len(datasets))
        return success_count


def main():
    parser = argparse.ArgumentParser(
        description='Generate README documentation for ChemBench datasets.',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate documentation for all datasets',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Generate documentation for a specific dataset (e.g., esol, tennessee_eastman)',
    )

    args = parser.parse_args()

    documenter = DatasetDocumenter()

    if args.all:
        logger.info('Generating documentation for all datasets...')
        documenter.generate_for_all()
    elif args.dataset:
        dataset_dir = documenter.datasets_root / args.dataset
        if dataset_dir.exists():
            documenter.generate_for_dataset(dataset_dir)
        else:
            logger.error('Dataset not found: %s', args.dataset)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

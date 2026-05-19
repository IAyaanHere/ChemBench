import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_ALIASES = {
    'tep': 'tennessee_eastman',
    'tennessee-eastman': 'tennessee_eastman',
}

PROCESS_DATASETS = {'tennessee_eastman'}
MOLECULAR_DATASETS = {
    'qm9',
    'esol',
    'freesolv',
    'lipophilicity',
    'hiv',
    'bace',
    'tox21',
    'bbbp',
}
MIXTURE_DATASETS = {'chemixhub'}

MOLECULENET_DATASETS = {'esol', 'freesolv', 'lipophilicity', 'hiv', 'bace', 'tox21', 'bbbp'}


class ChemBenchDataLoader:
    def __init__(self, dataset_name: str, task_name: Optional[str] = None):
        self.original_name = dataset_name
        normalized_name, task_name = self._normalize_dataset_name(dataset_name, task_name)
        self.dataset_name = normalized_name
        self.task_name = task_name
        self.dataset_type = self._detect_dataset_type(self.dataset_name)
        self.processed_path = self._resolve_processed_path()
        self._data_frame: Optional[pd.DataFrame] = None

    def _normalize_dataset_name(self, dataset_name: str, task_name: Optional[str]) -> tuple[str, Optional[str]]:
        name = dataset_name.strip().lower()
        if '/' in name:
            root_name, remainder = name.split('/', 1)
            if root_name == 'chemixhub':
                task_name = remainder
                name = 'chemixhub'
        if name in DATASET_ALIASES:
            name = DATASET_ALIASES[name]
        return name, task_name

    def _detect_dataset_type(self, name: str) -> str:
        if name in PROCESS_DATASETS:
            return 'process'
        if name in MOLECULAR_DATASETS or name.startswith('qm9'):
            return 'molecular'
        if name in MIXTURE_DATASETS:
            return 'mixture'
        raise ValueError(f'Invalid or unsupported dataset name: {name}')

    def _resolve_processed_path(self) -> Path:
        root = Path(__file__).resolve().parent.parent / 'datasets'
        dataset_dir = root / self.dataset_name / 'processed'
        if not dataset_dir.exists():
            raise FileNotFoundError(f'Processed dataset folder does not exist: {dataset_dir}')

        if self.dataset_type == 'mixture':
            csv_files = sorted(dataset_dir.glob('*.csv'))
            if not csv_files:
                raise FileNotFoundError(f'No processed CSV files found in: {dataset_dir}')
            if self.task_name:
                task_path = dataset_dir / f'{self.task_name}.csv'
                if task_path.exists():
                    return task_path
                raise FileNotFoundError(
                    f'ChemixHub task not found: {self.task_name}. Available tasks: {[p.stem for p in csv_files]}'
                )
            if len(csv_files) == 1:
                return csv_files[0]
            raise ValueError(
                'ChemixHub contains multiple processed CSV files. Use dataset_name="chemixhub/<task>" or task_name="<task>".'
            )

        csv_path = dataset_dir / f'{self.dataset_name}.csv'
        if csv_path.exists():
            return csv_path

        if self.dataset_name == 'qm9' or self.dataset_name.startswith('qm9'):
            for candidate in ('qm9_molecules.csv', 'train.csv'):
                qm9_candidate = dataset_dir / candidate
                if qm9_candidate.exists():
                    return qm9_candidate

        csv_files = sorted(dataset_dir.glob('*.csv'))
        if len(csv_files) == 1:
            return csv_files[0]

        if self.dataset_type == 'process':
            for candidate in ['train.csv', 'test.csv', 'val.csv']:
                candidate_path = dataset_dir / candidate
                if candidate_path.exists():
                    return candidate_path
            if csv_files:
                return csv_files[0]

        raise FileNotFoundError(
            f'Processed CSV file not found for dataset: {self.dataset_name}. Expected one of: {csv_path}, or a single CSV in {dataset_dir}.'
        )

    def load_data(self) -> pd.DataFrame:
        if self._data_frame is None:
            logger.info('Loading data for dataset: %s from %s', self.original_name, self.processed_path)
            self._data_frame = pd.read_csv(self.processed_path)
            logger.info('Loaded %s rows and %s columns from %s',
                        len(self._data_frame), len(self._data_frame.columns), self.processed_path)
        return self._data_frame

    def get_features(self) -> pd.DataFrame:
        df = self.load_data()
        if self.dataset_type == 'molecular':
            smiles_col = self._find_smiles_column(df)
            return df[[smiles_col]].copy()
        if self.dataset_type == 'process':
            if 'label' in df.columns:
                return df.drop(columns=['label']).copy()
            feature_columns = [col for col in df.columns if col not in self._target_columns(df)]
            return df[feature_columns].copy()
        if self.dataset_type == 'mixture':
            return df.copy()
        raise ValueError(f'Unknown dataset type: {self.dataset_type}')

    def get_targets(self) -> pd.DataFrame:
        df = self.load_data()
        if self.dataset_type == 'molecular':
            target_cols = self._target_columns(df)
            return df[target_cols].copy()
        if self.dataset_type == 'process':
            if 'label' in df.columns:
                return df[['label']].copy()
            return df[self._target_columns(df)].copy()
        if self.dataset_type == 'mixture':
            return df.copy()
        raise ValueError(f'Unknown dataset type: {self.dataset_type}')

    def _find_smiles_column(self, df: pd.DataFrame) -> str:
        for candidate in ['smiles', 'mol']:
            if candidate in df.columns:
                return candidate
        raise KeyError('Molecular dataset missing required SMILES column (smiles or mol)')

    def _target_columns(self, df: pd.DataFrame) -> List[str]:
        if self.dataset_type == 'molecular':
            if self.dataset_name == 'esol':
                return ['measured log solubility in mols per litre']
            if self.dataset_name == 'freesolv':
                return ['expt']
            if self.dataset_name == 'lipophilicity':
                return ['exp']
            if self.dataset_name == 'hiv':
                return ['HIV_active'] if 'HIV_active' in df.columns else ['activity']
            if self.dataset_name == 'bace':
                return ['Class'] if 'Class' in df.columns else ['pIC50']
            if self.dataset_name == 'tox21':
                return [col for col in df.columns if col not in {'mol_id', 'smiles', 'mol'}]
            if self.dataset_name == 'bbbp':
                return ['p_np'] if 'p_np' in df.columns else ['p_np']
            if self.dataset_name.startswith('qm9'):
                return [col for col in df.columns if col not in {'smiles', 'mol', 'molecule_id'}]
            smiles_col = self._find_smiles_column(df)
            return [col for col in df.columns if col != smiles_col]

        if self.dataset_type == 'process':
            if 'label' in df.columns:
                return ['label']
            target_cols = [col for col in df.columns if col.lower().startswith(('y', 'target', 'label'))]
            if target_cols:
                return target_cols
            return [col for col in df.columns if col.lower().startswith(('xmv', 'xmeas', 'output', 'y'))]

        if self.dataset_type == 'mixture':
            return [col for col in df.columns if col.lower() not in {'smiles', 'molecule_id', 'task'}]

        return []


if __name__ == '__main__':
    sample_names = ['tennessee_eastman', 'esol', 'lipophilicity', 'chemixhub/solubility']
    for name in sample_names:
        try:
            loader = ChemBenchDataLoader(name)
            df = loader.load_data()
            print(f'{name}: data shape =', df.shape)
            print(f'{name}: features shape =', loader.get_features().shape)
            print(f'{name}: targets shape =', loader.get_targets().shape)
        except Exception as exc:
            print(f'ERROR loading {name}:', exc)

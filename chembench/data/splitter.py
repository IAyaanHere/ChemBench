import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSplitter:
    def __init__(self):
        pass

    def random_split(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        if target_col not in df.columns:
            raise KeyError(f'Target column not found: {target_col}')

        stratify = None
        if self._is_categorical(df[target_col]):
            stratify = df[target_col]
            logger.info('Using stratified split for categorical target %s', target_col)

        train_val_size = 1.0 - test_size
        if train_val_size <= 0 or val_size <= 0 or test_size <= 0:
            raise ValueError('test_size and val_size must be positive and sum to less than 1.0')

        df_train_val, df_test = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
            shuffle=True,
        )

        stratify_val = None
        if stratify is not None:
            stratify_val = df_train_val[target_col]

        val_relative_size = val_size / train_val_size
        df_train, df_val = train_test_split(
            df_train_val,
            test_size=val_relative_size,
            stratify=stratify_val,
            random_state=random_state,
            shuffle=True,
        )

        logger.info(
            'Random split finished: train=%s, val=%s, test=%s',
            df_train.shape,
            df_val.shape,
            df_test.shape,
        )

        return (
            df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True),
            df_train[target_col].reset_index(drop=True),
            df_val[target_col].reset_index(drop=True),
            df_test[target_col].reset_index(drop=True),
        )

    def temporal_split(
        self,
        df: pd.DataFrame,
        time_col: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if time_col not in df.columns:
            raise KeyError(f'Time column not found: {time_col}')

        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            raise ValueError('DataFrame is empty')

        n_test = max(1, math.ceil(n * test_size))
        n_val = max(1, math.ceil(n * val_size))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError('Not enough rows to create train/val/test splits with the given sizes')

        train = df_sorted.iloc[:n_train].reset_index(drop=True)
        val = df_sorted.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test = df_sorted.iloc[n_train + n_val:].reset_index(drop=True)

        logger.info(
            'Temporal split finished: train=%s, val=%s, test=%s',
            train.shape,
            val.shape,
            test.shape,
        )

        return train, val, test

    def scaffold_split(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
        except ImportError as exc:
            raise ImportError('RDKit is required for scaffold splitting. Install rdkit and retry.') from exc

        if smiles_col not in df.columns:
            raise KeyError(f'SMILES column not found: {smiles_col}')

        if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1.0:
            raise ValueError('test_size and val_size must be positive and sum to less than 1.0')

        df = df.reset_index(drop=True).copy()
        scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, smiles in df[smiles_col].items():
            mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
            scaffold = (
                MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                if mol is not None
                else 'invalid'
            )
            scaffold_to_indices[scaffold].append(idx)

        groups = sorted(scaffold_to_indices.items(), key=lambda item: len(item[1]), reverse=True)
        n = len(df)
        n_test = math.ceil(n * test_size)
        n_val = math.ceil(n * val_size)
        n_train = n - n_test - n_val

        splited: Dict[str, List[int]] = {'train': [], 'val': [], 'test': []}
        counts = {'train': 0, 'val': 0, 'test': 0}
        targets = {'train': n_train, 'val': n_val, 'test': n_test}

        for scaffold, indices in groups:
            best_split = min(
                ['train', 'val', 'test'],
                key=lambda split: self._split_overfill_ratio(counts, targets, split, len(indices)),
            )
            splited[best_split].extend(indices)
            counts[best_split] += len(indices)

        train = df.loc[splited['train']].reset_index(drop=True)
        val = df.loc[splited['val']].reset_index(drop=True)
        test = df.loc[splited['test']].reset_index(drop=True)

        logger.info(
            'Scaffold split finished: train=%s, val=%s, test=%s; scaffolds=%s',
            train.shape,
            val.shape,
            test.shape,
            len(groups),
        )

        return train, val, test

    @staticmethod
    def _split_overfill_ratio(counts: Dict[str, int], targets: Dict[str, int], split: str, group_size: int) -> float:
        if targets[split] <= 0:
            return float('inf')
        return (counts[split] + group_size) / targets[split]

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        if series.dtype.name == 'category':
            return True
        if series.dtype == object:
            return True
        unique = series.dropna().unique()
        return len(unique) <= 20

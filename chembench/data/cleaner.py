import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    def __init__(self):
        pass

    def handle_missing(self, df: pd.DataFrame, strategy: str = 'mean', fill_value: Optional[float] = None) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            logger.info('No numeric columns found for missing-value imputation.')
            return df

        if strategy == 'mean':
            impute_values = df[numeric_cols].mean()
        elif strategy == 'median':
            impute_values = df[numeric_cols].median()
        elif strategy == 'constant':
            if fill_value is None:
                raise ValueError('fill_value must be provided when strategy="constant"')
            impute_values = pd.Series(fill_value, index=numeric_cols)
        else:
            raise ValueError(f'Unsupported missing-value strategy: {strategy}')

        df[numeric_cols] = df[numeric_cols].fillna(impute_values)
        logger.info('Imputed missing values for %d numeric columns using %s strategy.', len(numeric_cols), strategy)
        return df

    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            logger.info('No numeric columns found for outlier removal.')
            return df

        if method != 'iqr':
            raise ValueError(f'Unsupported outlier removal method: {method}')

        initial_rows = len(df)
        q1 = df[numeric_cols].quantile(0.25)
        q3 = df[numeric_cols].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        mask = pd.Series(True, index=df.index)
        for col in numeric_cols:
            col_mask = df[col].between(lower_bound[col], upper_bound[col], inclusive='both')
            mask &= col_mask

        df = df[mask].copy()
        removed = initial_rows - len(df)
        logger.info('Removed %d outlier rows using %s method on %d numeric columns.', removed, method, len(numeric_cols))
        return df

    def normalize_features(self, df: pd.DataFrame, features: List[str], method: str = 'standard') -> pd.DataFrame:
        df = df.copy()
        if not features:
            raise ValueError('features list must not be empty for normalization')

        missing_features = [feat for feat in features if feat not in df.columns]
        if missing_features:
            raise KeyError(f'Normalization features not found in dataframe: {missing_features}')

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f'Unsupported normalization method: {method}')

        df[features] = scaler.fit_transform(df[features])
        logger.info('Normalized %d feature columns using %s scaler.', len(features), method)
        return df

    def validate_chemistry(self, df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError('RDKit is required for chemistry validation. Install rdkit and retry.') from exc

        if smiles_col not in df.columns:
            raise KeyError(f'SMILES column not found: {smiles_col}')

        df = df.copy()
        valid_mask = df[smiles_col].apply(lambda sm: self._is_valid_smiles(sm, Chem))
        invalid_count = len(df) - valid_mask.sum()
        df = df[valid_mask].copy()
        logger.info('Validated SMILES strings in column %s; dropped %d invalid rows.', smiles_col, invalid_count)
        return df

    @staticmethod
    def _is_valid_smiles(smiles: str, Chem) -> bool:
        if pd.isna(smiles):
            return False
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None

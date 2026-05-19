"""
Comprehensive benchmarking for ChemBench baseline models.

Loads data via ChemBenchDataLoader + DataSplitter, trains all eight baselines where
applicable, records timing and memory, and exports a leaderboard (CSV / MD / JSON).

Usage:
    python scripts/run_benchmarks.py --dataset esol
    python scripts/run_benchmarks.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import tracemalloc
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from chembench.data.cleaner import DataCleaner
from chembench.data.loader import ChemBenchDataLoader
from chembench.data.splitter import DataSplitter
from chembench.models.baselines.base import BaselineModel
from chembench.models.baselines.dl_models import CNN1D, FullyConnectedNN, GNNBase, PyTorchWrapper
from chembench.models.baselines.sklearn_models import (
    GradientBoostingModel,
    LightGBMModel,
    LinearRegressionModel,
    MLPModel,
    RandomForestModel,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Canonical 10 datasets for --all (ChemixHub uses loader task suffix).
ALL_BENCHMARK_DATASETS: List[str] = [
    'tennessee_eastman',
    'qm9',
    'chemixhub/solubility',
    'esol',
    'freesolv',
    'lipophilicity',
    'hiv',
    'bace',
    'tox21',
    'bbbp',
]

DATASET_KIND = {
    'tennessee_eastman': 'process',
    'qm9': 'molecular',
    'esol': 'molecular',
    'freesolv': 'molecular',
    'lipophilicity': 'molecular',
    'hiv': 'molecular',
    'bace': 'molecular',
    'tox21': 'molecular',
    'bbbp': 'molecular',
    'qm9': 'molecular',
}

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False


class CNN1DTabularAdapter(nn.Module):
    """Wrap CNN1D so 2D tabular tensors (batch, n_features) from PyTorchWrapper work."""

    def __init__(self, n_features: int, output_dim: int) -> None:
        super().__init__()
        self.inner = CNN1D(input_dim=1, output_dim=output_dim, seq_length=n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return self.inner(x)


class GNNTabularAdapter(nn.Module):
    """GNN baseline on molecular descriptor matrices (FCNN fallback when graphs are unavailable)."""

    def __init__(self, n_features: int, output_dim: int) -> None:
        super().__init__()
        gnn = GNNBase(input_dim=1, output_dim=output_dim, hidden_dim=128)
        if getattr(gnn, 'use_fallback', False):
            self.network = gnn.network
        else:
            # Descriptor tabular inputs are not batched PyG graphs; use the documented fallback.
            self.network = FullyConnectedNN(
                input_dim=n_features,
                output_dim=output_dim,
                hidden_dim=128,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ChemBenchPyTorchWrapper(PyTorchWrapper):
    """PyTorchWrapper with CrossEntropy-friendly labels and class predictions."""

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> None:
        if self.task_type is None:
            self.task_type = self._detect_task_type(y_train)

        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train

        X_train_np = self.scaler.fit_transform(X_train_np)

        if X_val is None:
            n_train = int(len(X_train_np) * (1 - self.val_split))
            indices = np.random.permutation(len(X_train_np))
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            X_train_split = X_train_np[train_indices]
            y_train_split = y_train_np[train_indices]
            X_val_split = X_train_np[val_indices]
            y_val_split = y_train_np[val_indices]
        else:
            X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val
            X_val_split = self.scaler.transform(X_val_np)
            y_val_split = y_val_np
            X_train_split = X_train_np
            y_train_split = y_train_np

        if self.task_type == 'classification':
            y_train_tensor = torch.LongTensor(np.asarray(y_train_split).astype(np.int64).ravel())
            y_val_tensor = torch.LongTensor(np.asarray(y_val_split).astype(np.int64).ravel())
        else:
            y_train_tensor = torch.FloatTensor(
                y_train_split.reshape(-1, 1) if len(y_train_split.shape) == 1 else y_train_split
            )
            y_val_tensor = torch.FloatTensor(
                y_val_split.reshape(-1, 1) if len(y_val_split.shape) == 1 else y_val_split
            )

        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(np.ascontiguousarray(X_train_split, dtype=np.float32)),
            y_train_tensor,
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(np.ascontiguousarray(X_val_split, dtype=np.float32)),
            y_val_tensor,
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.pytorch_model.parameters(), lr=self.learning_rate)
        if self.task_type == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.pytorch_model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.pytorch_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)

            self.pytorch_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.pytorch_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_loss /= max(len(val_loader), 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                logger.info('Early stopping at epoch %d', epoch + 1)
                break

        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f'Model {self.model_name} is not fitted yet')
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        X_np = self.scaler.transform(X_np)
        X_tensor = torch.FloatTensor(X_np).to(self.device)
        self.pytorch_model.eval()
        with torch.no_grad():
            outputs = self.pytorch_model(X_tensor)
        predictions = outputs.cpu().numpy()
        if self.task_type == 'regression':
            return predictions.squeeze()
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            return np.argmax(predictions, axis=1)
        return (predictions.squeeze() >= 0.5).astype(np.int64)


def _detect_task_type(y: pd.Series) -> str:
    return BaselineModel._detect_task_type(y)


def _classification_output_dim(y_train: np.ndarray) -> int:
    classes = np.unique(np.asarray(y_train).astype(np.int64))
    return int(max(2, len(classes)))


def _compute_sklearn_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics: Dict[str, float] = {}
    if task_type == 'regression':
        mse = float(mean_squared_error(y_true, y_pred))
        metrics['mse'] = mse
        metrics['rmse'] = float(np.sqrt(mse))
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
    else:
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    return metrics


def _peak_memory_mb(rss_before: float, rss_after: float, tracemalloc_peak: int) -> float:
    rss_delta_mb = max(0.0, (rss_after - rss_before) / (1024 * 1024))
    trace_mb = tracemalloc_peak / (1024 * 1024)
    return round(max(rss_delta_mb, trace_mb), 4)


def _rss_mb() -> float:
    if _HAS_PSUTIL:
        return float(psutil.Process(os.getpid()).memory_info().rss)
    return 0.0


def _load_dataset_frame(loader: ChemBenchDataLoader) -> pd.DataFrame:
    if loader.dataset_name == 'qm9':
        qm9_path = loader.processed_path.parent / 'qm9_molecules.csv'
        if qm9_path.exists():
            logger.info('Using QM9 processed file: %s', qm9_path)
            return pd.read_csv(qm9_path)
    return loader.load_data()


def _resolve_primary_target(loader: ChemBenchDataLoader, df: pd.DataFrame) -> str:
    if loader.dataset_type == 'mixture':
        for candidate in ('value', 'target_property', 'target'):
            if candidate in df.columns:
                return candidate
        numeric = df.select_dtypes(include=['number']).columns.tolist()
        if numeric:
            return numeric[-1]
    if loader.dataset_name == 'qm9':
        for candidate in ('U0', 'U', 'H', 'G', 'Gap'):
            if candidate in df.columns:
                return candidate
    targets_df = loader.get_targets()
    return targets_df.columns[0]


def _sanitize_feature_columns(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X.columns = [
        str(c).replace(' ', '_').replace('[', '(').replace(']', ')').replace('{', '(').replace('}', ')')
        for c in X.columns
    ]
    return X


def _morgan_fingerprint_features(df: pd.DataFrame, smiles_col: str, n_bits: int = 512) -> pd.DataFrame:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rows: List[np.ndarray] = []
    for smiles in df[smiles_col].astype(str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.float32))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            rows.append(np.array(fp, dtype=np.float32))
    return pd.DataFrame(
        rows,
        columns=[f'fp_{i}' for i in range(n_bits)],
        index=df.index,
    )


def _split_to_feature_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    smiles_col: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def featurize(split_df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [target_col]
        for col in (
            'mol_id',
            'mixture_id',
            'target_property',
            'Compound ID',
            'CMPD_CHEMBLID',
            'smiles',
            'mol',
            'SMILES',
        ):
            if col in split_df.columns:
                drop_cols.append(col)
        feats = split_df.drop(columns=drop_cols, errors='ignore')
        numeric_cols = feats.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            return _sanitize_feature_columns(feats[numeric_cols])
        if smiles_col and smiles_col in split_df.columns:
            return _morgan_fingerprint_features(split_df, smiles_col)
        raise ValueError('No numeric or SMILES features available after preprocessing')

    return featurize(train_df), featurize(val_df), featurize(test_df)


def models_for_dataset(dataset_name: str, n_features: int = 0) -> List[str]:
    """Eight baselines with dataset-aware emphasis (all are still attempted)."""
    kind = DATASET_KIND.get(dataset_name, 'mixture')
    base = ['linear', 'random_forest', 'gradient_boosting', 'lightgbm', 'mlp', 'fcnn', 'cnn1d', 'gnn']
    if n_features < 8:
        base = [m for m in base if m != 'cnn1d']
    return base


def build_model(
    model_key: str,
    n_features: int,
    task_type: str,
    y_train: np.ndarray,
) -> Optional[BaselineModel]:
    output_dim = 1 if task_type == 'regression' else _classification_output_dim(y_train)
    try:
        if model_key == 'linear':
            return LinearRegressionModel()
        if model_key == 'random_forest':
            return RandomForestModel(n_estimators=100)
        if model_key == 'gradient_boosting':
            return GradientBoostingModel(n_estimators=100)
        if model_key == 'lightgbm':
            return LightGBMModel(num_leaves=31)
        if model_key == 'mlp':
            return MLPModel(hidden_layer_sizes=(100, 50))
        if model_key == 'fcnn':
            net = FullyConnectedNN(input_dim=n_features, output_dim=output_dim, hidden_dim=128)
            return ChemBenchPyTorchWrapper(
                model_name='FCNN',
                pytorch_model=net,
                task_type=task_type,
                learning_rate=0.001,
                batch_size=512,
                epochs=50,
                early_stopping_patience=3,
            )
        if model_key == 'cnn1d':
            net = CNN1DTabularAdapter(n_features=n_features, output_dim=output_dim)
            return ChemBenchPyTorchWrapper(
                model_name='CNN1D',
                pytorch_model=net,
                task_type=task_type,
                learning_rate=0.001,
                batch_size=512,
                epochs=50,
                early_stopping_patience=3,
            )
        if model_key == 'gnn':
            net = GNNTabularAdapter(n_features=n_features, output_dim=output_dim)
            return ChemBenchPyTorchWrapper(
                model_name='GNN',
                pytorch_model=net,
                task_type=task_type,
                learning_rate=0.001,
                batch_size=512,
                epochs=50,
                early_stopping_patience=3,
            )
    except Exception as exc:
        logger.warning('Could not instantiate model %s: %s', model_key, exc)
        return None
    return None


class BenchmarkRunner:
    def __init__(
        self,
        output_dir: Path,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed
        self.results: List[Dict[str, Any]] = []

    def load_and_prepare_data(
        self,
        dataset_name: str,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, str]:
        loader = ChemBenchDataLoader(dataset_name)
        df = _load_dataset_frame(loader)
        target_col = _resolve_primary_target(loader, df)

        cleaner = DataCleaner()
        df = cleaner.handle_missing(df, strategy='mean')

        smiles_col = None
        for candidate in ('smiles', 'mol', 'SMILES'):
            if candidate in df.columns:
                smiles_col = candidate
                break
        if loader.dataset_type == 'molecular' and smiles_col and loader.dataset_name != 'qm9':
            df = cleaner.validate_chemistry(df, smiles_col)

        splitter = DataSplitter()
        use_scaffold = loader.dataset_type == 'molecular' and smiles_col and loader.dataset_name != 'qm9'
        if use_scaffold:
            train_df, val_df, test_df = splitter.scaffold_split(
                df,
                smiles_col=smiles_col,
                test_size=self.test_size,
                val_size=self.val_size,
            )
        else:
            train_df, val_df, test_df, _, _, _ = splitter.random_split(
                df,
                target_col=target_col,
                test_size=self.test_size,
                val_size=self.val_size,
                random_state=self.random_seed,
            )

        if loader.dataset_name == 'tox21':
            extra_targets = [c for c in loader.get_targets().columns if c != target_col]
            train_df = train_df.drop(columns=extra_targets, errors='ignore')
            val_df = val_df.drop(columns=extra_targets, errors='ignore')
            test_df = test_df.drop(columns=extra_targets, errors='ignore')

        X_train, X_val, X_test = _split_to_feature_matrices(
            train_df, val_df, test_df, target_col, smiles_col
        )
        y_train = train_df[target_col].copy()
        y_val = val_df[target_col].copy()
        y_test = test_df[target_col].copy()

        return X_train, y_train, X_val, y_val, X_test, y_test, target_col

    def train_and_evaluate_one(
        self,
        model: BaselineModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str,
        label_encoder: Optional[LabelEncoder],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            'model_name': model.model_name,
            'train_time_s': None,
            'inference_time_ms_per_sample': None,
            'peak_memory_mb': None,
            'metrics': {},
            'status': 'failed',
            'error': None,
        }

        try:
            y_train_fit = y_train
            y_val_fit = y_val
            y_test_eval = y_test
            if label_encoder is not None:
                y_train_fit = pd.Series(label_encoder.transform(y_train.astype(str)))
                y_val_fit = pd.Series(label_encoder.transform(y_val.astype(str)))
                y_test_eval = pd.Series(label_encoder.transform(y_test.astype(str)))

            tracemalloc.start()
            rss_before = _rss_mb()
            t0 = time.perf_counter()

            if isinstance(model, ChemBenchPyTorchWrapper):
                model.fit(X_train, y_train_fit, X_val=X_val, y_val=y_val_fit)
            else:
                model.fit(X_train, y_train_fit)

            train_time = time.perf_counter() - t0
            rss_after = _rss_mb()
            _, trace_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            result['train_time_s'] = round(float(train_time), 4)
            result['peak_memory_mb'] = _peak_memory_mb(rss_before, rss_after, trace_peak)

            t1 = time.perf_counter()
            y_pred = model.predict(X_test)
            infer_elapsed = time.perf_counter() - t1
            n_test = max(len(X_test), 1)
            result['inference_time_ms_per_sample'] = round(infer_elapsed / n_test * 1000.0, 6)

            y_true = y_test_eval.values if isinstance(y_test_eval, pd.Series) else np.asarray(y_test_eval)
            y_pred_arr = np.asarray(y_pred)
            if label_encoder is not None and task_type == 'classification':
                y_true = label_encoder.transform(pd.Series(y_true).astype(str))

            result['metrics'] = _compute_sklearn_metrics(y_true, y_pred_arr, task_type)
            result['status'] = 'success'
        except Exception:
            result['error'] = traceback.format_exc()
            logger.error('Model %s failed: %s', model.model_name, result['error'])
        return result

    def benchmark_dataset(self, dataset_name: str) -> None:
        logger.info('Benchmarking dataset: %s', dataset_name)
        try:
            X_train, y_train, X_val, y_val, X_test, y_test, target_col = self.load_and_prepare_data(dataset_name)
        except Exception:
            logger.error('Failed to load or split %s:\n%s', dataset_name, traceback.format_exc())
            return

        task_type = _detect_task_type(pd.Series(y_train.values))
        label_encoder: Optional[LabelEncoder] = None
        if task_type == 'classification' and (y_train.dtype == object or y_train.dtype.name == 'category'):
            label_encoder = LabelEncoder()
            label_encoder.fit(pd.concat([y_train, y_val, y_test], axis=0).astype(str))

        n_features = X_train.shape[1]
        model_keys = models_for_dataset(dataset_name, n_features=n_features)

        for model_key in tqdm(model_keys, desc=f'{dataset_name} models', leave=False):
            try:
                model = build_model(model_key, n_features, task_type, y_train.values)
                if model is None:
                    raise RuntimeError(f'build_model returned None for {model_key}')
                if model.task_type is None:
                    model.task_type = task_type

                row = self.train_and_evaluate_one(
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    task_type,
                    label_encoder,
                )
                row.update(
                    {
                        'dataset': dataset_name,
                        'target_col': target_col,
                        'task_type': task_type,
                        'model_key': model_key,
                        'n_features': n_features,
                        'train_samples': int(len(X_train)),
                        'val_samples': int(len(X_val)),
                        'test_samples': int(len(X_test)),
                    }
                )
                self.results.append(row)
            except Exception:
                err = traceback.format_exc()
                logger.error('Unhandled failure for %s on %s:\n%s', model_key, dataset_name, err)
                self.results.append(
                    {
                        'dataset': dataset_name,
                        'target_col': target_col,
                        'task_type': task_type,
                        'model_key': model_key,
                        'model_name': model_key,
                        'n_features': n_features,
                        'train_samples': int(len(X_train)),
                        'val_samples': int(len(X_val)),
                        'test_samples': int(len(X_test)),
                        'train_time_s': None,
                        'inference_time_ms_per_sample': None,
                        'peak_memory_mb': None,
                        'metrics': {},
                        'status': 'failed',
                        'error': err,
                    }
                )

    def build_leaderboard(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for r in self.results:
            flat = {
                'dataset': r.get('dataset'),
                'model_key': r.get('model_key'),
                'model_name': r.get('model_name'),
                'target_col': r.get('target_col'),
                'task_type': r.get('task_type'),
                'status': r.get('status'),
                'train_time_s': r.get('train_time_s'),
                'inference_time_ms_per_sample': r.get('inference_time_ms_per_sample'),
                'peak_memory_mb': r.get('peak_memory_mb'),
                'train_samples': r.get('train_samples'),
                'val_samples': r.get('val_samples'),
                'test_samples': r.get('test_samples'),
                'n_features': r.get('n_features'),
                'error': r.get('error'),
            }
            for k, v in (r.get('metrics') or {}).items():
                flat[k] = v
            rows.append(flat)

        leaderboard = pd.DataFrame(rows)
        if leaderboard.empty:
            return leaderboard
        if 'mse' in leaderboard.columns:
            leaderboard = leaderboard.sort_values(['dataset', 'mse'], ascending=[True, True], na_position='last')
        elif 'accuracy' in leaderboard.columns:
            leaderboard = leaderboard.sort_values(['dataset', 'accuracy'], ascending=[True, False], na_position='last')
        return leaderboard

    def export_leaderboard(self, leaderboard: pd.DataFrame, name: str = 'leaderboard') -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / f'{name}.csv'
        leaderboard.to_csv(csv_path, index=False)
        json_path = self.output_dir / f'{name}.json'
        records = leaderboard.replace({np.nan: None}).to_dict(orient='records')
        json_path.write_text(json.dumps(records, indent=2), encoding='utf-8')
        md_path = self.output_dir / f'{name}.md'
        lines = [
            '# ChemBench Leaderboard',
            '',
            f'Generated: {datetime.now().isoformat()}',
            '',
        ]
        if not leaderboard.empty and 'status' in leaderboard.columns:
            lines.append(f"Successful runs: {(leaderboard['status'] == 'success').sum()}")
            lines.append(f"Failed runs: {(leaderboard['status'] == 'failed').sum()}")
            lines.append('')
        for ds in leaderboard['dataset'].dropna().unique() if not leaderboard.empty else []:
            lines.append(f'## {ds}')
            lines.append('')
            sub = leaderboard[leaderboard['dataset'] == ds]
            try:
                lines.append(sub.to_markdown(index=False))
            except ImportError:
                lines.append(sub.to_string(index=False))
            lines.append('')
        md_path.write_text('\n'.join(lines), encoding='utf-8')
        logger.info('Wrote %s, %s, %s', csv_path, md_path, json_path)


def _resolve_datasets(args: argparse.Namespace, runner: BenchmarkRunner) -> List[str]:
    if args.all:
        return list(ALL_BENCHMARK_DATASETS)
    if args.dataset:
        return [args.dataset.strip()]
    return ['esol']


def main() -> None:
    os.chdir(PROJECT_ROOT)

    parser = argparse.ArgumentParser(description='ChemBench baseline benchmarks')
    parser.add_argument('--dataset', type=str, help='Run a single dataset (e.g. esol, chemixhub/solubility)')
    parser.add_argument('--all', action='store_true', help='Run all ten canonical datasets')
    parser.add_argument('--output-dir', type=str, default=str(PROJECT_ROOT / 'results'))
    args = parser.parse_args()

    runner = BenchmarkRunner(output_dir=Path(args.output_dir))
    datasets = _resolve_datasets(args, runner)

    logger.info('ChemBench benchmark run starting; datasets=%s', datasets)
    for ds in tqdm(datasets, desc='Datasets'):
        try:
            runner.benchmark_dataset(ds)
        except Exception:
            logger.error('Dataset-level failure for %s:\n%s', ds, traceback.format_exc())

    leaderboard = runner.build_leaderboard()
    runner.export_leaderboard(leaderboard)
    if not leaderboard.empty:
        print(leaderboard.head(min(20, len(leaderboard))).to_string(index=False))


if __name__ == '__main__':
    main()

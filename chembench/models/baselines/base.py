import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaselineModel(ABC):
    """Abstract base class for baseline models."""

    def __init__(self, model_name: str, task_type: Optional[str] = None):
        self.model_name = model_name
        self.task_type = task_type  # 'regression' or 'classification'
        self.model: Optional[Any] = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on data."""
        pass

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if not self.is_fitted:
            raise RuntimeError(f'Model {self.model_name} is not fitted yet')

        y_pred = self.predict(X_test)
        metrics = {}

        if self.task_type == 'regression':
            metrics['mse'] = float(mean_squared_error(y_test, y_pred))
            metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            logger.info('%s (regression) - MSE: %.4f, MAE: %.4f, RMSE: %.4f',
                        self.model_name, metrics['mse'], metrics['mae'], metrics['rmse'])

        elif self.task_type == 'classification':
            metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
            metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            if hasattr(self.model, 'predict_proba'):
                try:
                    y_proba = self.model.predict_proba(X_test)
                    if y_proba.shape[1] == 2:
                        metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba[:, 1]))
                except Exception as exc:
                    logger.warning('Could not compute ROC-AUC: %s', exc)

            logger.info('%s (classification) - Accuracy: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f',
                        self.model_name, metrics['accuracy'], metrics['f1'], metrics['precision'], metrics['recall'])

        return metrics

    def save_model(self, filepath: Path) -> None:
        """Save the model to disk."""
        if not self.is_fitted:
            logger.warning('Attempting to save unfitted model %s', self.model_name)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info('Model %s saved to %s', self.model_name, filepath)
        except Exception as exc:
            logger.error('Failed to save model %s: %s', self.model_name, exc)
            raise

    def load_model(self, filepath: Path) -> None:
        """Load the model from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f'Model file not found: {filepath}')

        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            self.is_fitted = True
            logger.info('Model %s loaded from %s', self.model_name, filepath)
        except Exception as exc:
            logger.error('Failed to load model %s: %s', self.model_name, exc)
            raise

    @staticmethod
    def _detect_task_type(y: pd.Series) -> str:
        """Auto-detect task type from target variable."""
        if y.dtype in ['int64', 'int32', 'int16', 'int8']:
            unique_values = y.nunique()
            if unique_values <= 20:
                return 'classification'
        if y.dtype in ['float64', 'float32']:
            return 'regression'
        if y.dtype == 'object' or y.dtype.name == 'category':
            return 'classification'
        return 'classification'

    def __repr__(self) -> str:
        return f'{self.model_name} (task_type={self.task_type}, is_fitted={self.is_fitted})'

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

from chembench.models.baselines.base import BaselineModel


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LinearRegressionModel(BaselineModel):
    """Linear or Logistic Regression baseline."""

    def __init__(self, task_type: Optional[str] = None, **kwargs):
        super().__init__('Linear/Logistic Regression', task_type)
        self.kwargs = kwargs

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        if self.task_type is None:
            self.task_type = self._detect_task_type(y_train)

        try:
            if self.task_type == 'regression':
                self.model = LinearRegression(**self.kwargs)
                logger.info('Initialized LinearRegression for regression task')
            else:
                self.model = LogisticRegression(max_iter=1000, **self.kwargs)
                logger.info('Initialized LogisticRegression for classification task')

            self.model.fit(X_train, y_train)
            self.is_fitted = True
            logger.info('LinearRegression model fitted successfully')
        except Exception as exc:
            logger.error('Failed to fit LinearRegression: %s', exc)
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model is not fitted yet')
        return self.model.predict(X)


class RandomForestModel(BaselineModel):
    """Random Forest baseline."""

    def __init__(self, task_type: Optional[str] = None, n_estimators: int = 100, **kwargs):
        super().__init__('Random Forest', task_type)
        self.n_estimators = n_estimators
        self.kwargs = kwargs

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        if self.task_type is None:
            self.task_type = self._detect_task_type(y_train)

        try:
            if self.task_type == 'regression':
                self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42, **self.kwargs)
                logger.info('Initialized RandomForestRegressor for regression task')
            else:
                self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42, **self.kwargs)
                logger.info('Initialized RandomForestClassifier for classification task')

            self.model.fit(X_train, y_train)
            self.is_fitted = True
            logger.info('Random Forest model fitted successfully')
        except Exception as exc:
            logger.error('Failed to fit Random Forest: %s', exc)
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model is not fitted yet')
        return self.model.predict(X)


class GradientBoostingModel(BaselineModel):
    """Gradient Boosting baseline."""

    def __init__(self, task_type: Optional[str] = None, n_estimators: int = 100, **kwargs):
        super().__init__('Gradient Boosting', task_type)
        self.n_estimators = n_estimators
        self.kwargs = kwargs

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        if self.task_type is None:
            self.task_type = self._detect_task_type(y_train)

        try:
            if self.task_type == 'regression':
                self.model = GradientBoostingRegressor(n_estimators=self.n_estimators, random_state=42, **self.kwargs)
                logger.info('Initialized GradientBoostingRegressor for regression task')
            else:
                self.model = GradientBoostingClassifier(n_estimators=self.n_estimators, random_state=42, **self.kwargs)
                logger.info('Initialized GradientBoostingClassifier for classification task')

            self.model.fit(X_train, y_train)
            self.is_fitted = True
            logger.info('Gradient Boosting model fitted successfully')
        except Exception as exc:
            logger.error('Failed to fit Gradient Boosting: %s', exc)
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model is not fitted yet')
        return self.model.predict(X)


class LightGBMModel(BaselineModel):
    """LightGBM baseline."""

    def __init__(self, task_type: Optional[str] = None, num_leaves: int = 31, **kwargs):
        super().__init__('LightGBM', task_type)
        self.num_leaves = num_leaves
        self.kwargs = kwargs

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            logger.error('LightGBM is not installed. Install it with: pip install lightgbm')
            raise ImportError('LightGBM is required for this model') from exc

        if self.task_type is None:
            self.task_type = self._detect_task_type(y_train)

        try:
            if self.task_type == 'regression':
                self.model = lgb.LGBMRegressor(num_leaves=self.num_leaves, random_state=42, **self.kwargs)
                logger.info('Initialized LGBMRegressor for regression task')
            else:
                self.model = lgb.LGBMClassifier(num_leaves=self.num_leaves, random_state=42, **self.kwargs)
                logger.info('Initialized LGBMClassifier for classification task')

            self.model.fit(X_train, y_train)
            self.is_fitted = True
            logger.info('LightGBM model fitted successfully')
        except Exception as exc:
            logger.error('Failed to fit LightGBM: %s', exc)
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model is not fitted yet')
        return self.model.predict(X)


class MLPModel(BaselineModel):
    """Multi-Layer Perceptron (Neural Network) baseline."""

    def __init__(self, task_type: Optional[str] = None, hidden_layer_sizes: tuple = (100, 50), **kwargs):
        super().__init__('MLP Neural Network', task_type)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.kwargs = kwargs

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        if self.task_type is None:
            self.task_type = self._detect_task_type(y_train)

        try:
            if self.task_type == 'regression':
                self.model = MLPRegressor(
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    random_state=42,
                    max_iter=500,
                    **self.kwargs
                )
                logger.info('Initialized MLPRegressor for regression task')
            else:
                self.model = MLPClassifier(
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    random_state=42,
                    max_iter=500,
                    **self.kwargs
                )
                logger.info('Initialized MLPClassifier for classification task')

            self.model.fit(X_train, y_train)
            self.is_fitted = True
            logger.info('MLP model fitted successfully')
        except Exception as exc:
            logger.error('Failed to fit MLP: %s', exc)
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model is not fitted yet')
        return self.model.predict(X)

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chembench.data.cleaner import DataCleaner
from chembench.data.loader import ChemBenchDataLoader
from chembench.data.splitter import DataSplitter
from chembench.models.baselines.sklearn_models import (
    GradientBoostingModel,
    LightGBMModel,
    LinearRegressionModel,
    MLPModel,
    RandomForestModel,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'LinearRegressionModel': LinearRegressionModel,
    'RandomForestModel': RandomForestModel,
    'GradientBoostingModel': GradientBoostingModel,
    'LightGBMModel': LightGBMModel,
    'MLPModel': MLPModel,
}

MOLECULAR_DATASETS = {'esol', 'freesolv', 'lipophilicity', 'hiv', 'bace', 'tox21', 'bbbp', 'qm9'}


class TrainingPipeline:
    def __init__(self, config_path: Path, dataset_override: Optional[str] = None):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        if dataset_override:
            self.config['dataset']['name'] = dataset_override
            logger.info('Dataset overridden to: %s', dataset_override)

        self.dataset_name = self.config['dataset']['name']
        self.save_dir = Path(self.config['logging']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info('Save directory: %s', self.save_dir)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f'Config file not found: {self.config_path}')

        with open(self.config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info('Loaded configuration from %s', self.config_path)
        return config

    def _load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean data using ChemBenchDataLoader and DataCleaner."""
        logger.info('Loading dataset: %s', self.dataset_name)
        loader = ChemBenchDataLoader(self.dataset_name)
        df = loader.load_data()
        logger.info('Loaded dataset with shape: %s', df.shape)

        cleaner = DataCleaner()
        logger.info('Starting data cleaning pipeline')

        # Validate chemistry if molecular dataset
        if self.dataset_name in MOLECULAR_DATASETS:
            smiles_col = 'smiles' if 'smiles' in df.columns else 'mol'
            df = cleaner.validate_chemistry(df, smiles_col=smiles_col)
            logger.info('SMILES validation complete')

        # Handle missing values
        df = cleaner.handle_missing(df, strategy='mean')

        # Remove outliers (skip for TEP as anomalies are the target)
        if self.dataset_name not in {'tennessee_eastman', 'tep'}:
            df = cleaner.remove_outliers(df, method='iqr', threshold=1.5)

        # Normalize numeric features
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        target_col = self._get_target_column(df)
        feature_cols = [col for col in numeric_cols if col != target_col]
        if feature_cols:
            df = cleaner.normalize_features(df, features=feature_cols, method='standard')

        logger.info('Data cleaning complete. Final shape: %s', df.shape)
        return df

    def _get_target_column(self, df: pd.DataFrame) -> str:
        """Get target column from config or auto-detect."""
        target_col = self.config['dataset'].get('target_col')
        if target_col:
            return target_col

        # Auto-detect from common naming patterns
        for col in df.columns:
            if col.lower() in {'label', 'target', 'y'}:
                logger.info('Auto-detected target column: %s', col)
                return col

        # Last numeric column as fallback
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            target_col = numeric_cols[-1]
            logger.info('Auto-detected target column (last numeric): %s', target_col)
            return target_col

        raise ValueError('Could not auto-detect target column')

    def _split_data(self, df: pd.DataFrame) -> tuple:
        """Split data using DataSplitter."""
        logger.info('Splitting dataset')
        splitter = DataSplitter()
        target_col = self._get_target_column(df)
        test_size = self.config['training'].get('test_size', 0.2)
        val_size = self.config['training'].get('val_size', 0.1)

        # Choose split strategy based on dataset type
        use_scaffold_split = self.config['training'].get('use_scaffold_split', True)
        if use_scaffold_split and self.dataset_name in MOLECULAR_DATASETS:
            logger.info('Using scaffold-based splitting for molecular dataset')
            smiles_col = 'smiles' if 'smiles' in df.columns else 'mol'
            train, val, test = splitter.scaffold_split(df, smiles_col=smiles_col,
                                                        test_size=test_size, val_size=val_size)
        else:
            logger.info('Using stratified random splitting')
            train, val, test, y_train, y_val, y_test = splitter.random_split(
                df, target_col=target_col, test_size=test_size, val_size=val_size
            )

        logger.info('Split complete: train=%s, val=%s, test=%s', train.shape, val.shape, test.shape)
        return train, val, test, target_col

    def _train_model(self, model_config: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Instantiate and train a single model."""
        model_class_name = model_config.get('class')
        if model_class_name not in MODEL_CLASSES:
            raise ValueError(f'Unknown model class: {model_class_name}')

        model_class = MODEL_CLASSES[model_class_name]
        model_params = model_config.get('params', {})

        logger.info('Initializing %s with params: %s', model_class_name, model_params)
        model = model_class(**model_params)

        logger.info('Fitting %s', model_class_name)
        model.fit(X_train, y_train)
        return model

    def run(self) -> Dict[str, Dict[str, float]]:
        """Execute the full training pipeline."""
        logger.info('=' * 80)
        logger.info('Starting Baseline Training Pipeline')
        logger.info('Dataset: %s', self.dataset_name)
        logger.info('=' * 80)

        # Load and clean data
        df = self._load_and_clean_data()

        # Split data
        train, val, test, target_col = self._split_data(df)

        # Prepare train/test sets
        X_train = train.drop(target_col, axis=1)
        y_train = train[target_col]
        X_test = test.drop(target_col, axis=1)
        y_test = test[target_col]

        # Remove non-numeric columns (e.g., SMILES, mol_id, Compound ID)
        numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        X_train = X_train[numeric_cols]
        X_test = X_test[numeric_cols]

        logger.info('X_train shape: %s, y_train shape: %s', X_train.shape, y_train.shape)
        logger.info('X_test shape: %s, y_test shape: %s', X_test.shape, y_test.shape)

        # Train models
        all_results = {}
        model_configs = self.config.get('models', [])

        logger.info('Training %d models', len(model_configs))
        for model_config in tqdm(model_configs, desc='Training Models'):
            model_name = model_config.get('name')
            logger.info('\n' + '=' * 80)
            logger.info('Training model: %s', model_name)
            logger.info('=' * 80)

            try:
                model = self._train_model(model_config, X_train, y_train)

                # Evaluate
                logger.info('Evaluating %s on test set', model_name)
                metrics = model.evaluate(X_test, y_test)

                all_results[model_name] = metrics

                # Save model
                model_path = self.save_dir / f'{model_name}_{self.dataset_name}.pkl'
                model.save_model(model_path)

                logger.info('Model %s complete. Metrics: %s', model_name, metrics)
            except Exception as exc:
                logger.error('Failed to train model %s: %s', model_name, exc)
                all_results[model_name] = {'error': str(exc)}

        # Save results summary
        results_path = self.save_dir / f'results_{self.dataset_name}.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        logger.info('Results saved to %s', results_path)

        logger.info('\n' + '=' * 80)
        logger.info('Training Pipeline Complete!')
        logger.info('=' * 80)
        self._print_results_summary(all_results)

        return all_results

    @staticmethod
    def _print_results_summary(all_results: Dict[str, Dict[str, float]]) -> None:
        """Print summary of all results."""
        logger.info('\n--- RESULTS SUMMARY ---\n')
        for model_name, metrics in all_results.items():
            logger.info('Model: %s', model_name)
            if 'error' in metrics:
                logger.error('  Error: %s', metrics['error'])
            else:
                for metric_name, metric_value in metrics.items():
                    logger.info('  %s: %.4f', metric_name, metric_value)
            logger.info('')


def main():
    parser = argparse.ArgumentParser(description='Train baseline models on ChemBench datasets.')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Override dataset name in config (e.g., esol, tennessee_eastman)',
    )

    args = parser.parse_args()

    try:
        pipeline = TrainingPipeline(args.config, dataset_override=args.dataset)
        pipeline.run()
    except Exception as exc:
        logger.error('Pipeline failed: %s', exc)
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
Training script for Deep Learning baseline models.

Usage:
    python scripts/train_dl_baselines.py
    python scripts/train_dl_baselines.py --dataset esol --model fcnn
"""

import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from chembench.data.loader import ChemBenchDataLoader
from chembench.data.cleaner import DataCleaner
from chembench.data.splitter import DataSplitter
from chembench.models.baselines.dl_models import FullyConnectedNN, CNN1D, PyTorchWrapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class DeepLearningTrainingPipeline:
    """Pipeline for training deep learning baseline models."""

    def __init__(
        self,
        dataset_name: str = 'esol',
        model_type: str = 'fcnn',
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        random_seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.random_seed = random_seed

        # Set seeds for reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Output directory
        self.output_dir = Path('./results/baselines/')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            'Initialized DeepLearningTrainingPipeline: dataset=%s, model=%s',
            dataset_name,
            model_type,
        )

    def load_and_clean_data(self):
        """Load and clean dataset."""
        logger.info('='*60)
        logger.info('STEP 1: Loading and Cleaning Data')
        logger.info('='*60)

        # Load data
        loader = ChemBenchDataLoader(self.dataset_name)
        df = loader.load_data()
        logger.info('✓ Loaded %d rows × %d columns', len(df), len(df.columns))

        # Get target column
        targets_df = loader.get_targets()
        target_col = targets_df.columns[0]

        # Clean data
        cleaner = DataCleaner()
        df_cleaned = cleaner.handle_missing(df, strategy='mean')
        logger.info('✓ Handled missing values')

        # Validate chemistry if molecular dataset
        if self.dataset_name in {'esol', 'freesolv', 'lipophilicity', 'hiv', 'bace', 'tox21', 'bbbp', 'qm9'}:
            smiles_col = 'smiles' if 'smiles' in df_cleaned.columns else 'mol'
            if smiles_col in df_cleaned.columns:
                initial_rows = len(df_cleaned)
                df_cleaned = cleaner.validate_chemistry(df_cleaned, smiles_col)
                logger.info('✓ Validated chemistry: %d → %d rows', initial_rows, len(df_cleaned))

        logger.info('✓ Final dataset: %d rows × %d columns', len(df_cleaned), len(df_cleaned.columns))

        return df_cleaned, target_col

    def split_data(self, df, target_col):
        """Split data into train/val/test sets."""
        logger.info('')
        logger.info('='*60)
        logger.info('STEP 2: Splitting Data')
        logger.info('='*60)

        splitter = DataSplitter()

        # Use scaffold split for molecular data
        if self.dataset_name in {'esol', 'freesolv', 'lipophilicity', 'hiv', 'bace', 'tox21', 'bbbp', 'qm9'}:
            smiles_col = 'smiles' if 'smiles' in df.columns else 'mol'
            train, val, test = splitter.scaffold_split(
                df,
                smiles_col=smiles_col,
                test_size=0.2,
                val_size=0.1,
            )
            logger.info('✓ Used scaffold split')
        else:
            train, val, test, _, _, _ = splitter.random_split(
                df,
                target_col=target_col,
                test_size=0.2,
                val_size=0.1,
            )
            logger.info('✓ Used random split')

        logger.info('✓ Split sizes: train=%d, val=%d, test=%d', len(train), len(val), len(test))

        return train, val, test

    def prepare_features(self, df, target_col):
        """Prepare features by selecting numeric columns only."""
        X = df.drop(target_col, axis=1)
        
        # Keep only numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X = X[numeric_cols]
        
        y = df[target_col]
        
        return X, y

    def build_model(self, input_dim: int, output_dim: int = 1):
        """Build PyTorch model based on model_type."""
        logger.info('')
        logger.info('='*60)
        logger.info('STEP 3: Building Model')
        logger.info('='*60)

        if self.model_type == 'fcnn':
            pytorch_model = FullyConnectedNN(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=128,
                num_layers=3,
                dropout=0.3,
            )
            logger.info('✓ Built FCNN model: %d → 128 → 128 → 128 → %d', input_dim, output_dim)

        elif self.model_type == 'cnn1d':
            pytorch_model = CNN1D(
                input_dim=input_dim,
                output_dim=output_dim,
                seq_length=input_dim,
            )
            logger.info('✓ Built CNN1D model')

        else:
            raise ValueError(f'Unknown model type: {self.model_type}')

        # Wrap in PyTorchWrapper
        wrapper = PyTorchWrapper(
            model_name=f'dl_{self.model_type}',
            pytorch_model=pytorch_model,
            task_type='regression',
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.epochs,
            early_stopping_patience=self.early_stopping_patience,
            val_split=0.2,
        )

        return wrapper

    def train_model(self, model, X_train, y_train, X_val, y_val):
        """Train the model."""
        logger.info('')
        logger.info('='*60)
        logger.info('STEP 4: Training Model')
        logger.info('='*60)

        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        logger.info('✓ Training completed')

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model on test set."""
        logger.info('')
        logger.info('='*60)
        logger.info('STEP 5: Evaluating Model')
        logger.info('='*60)

        metrics = model.evaluate(X_test, y_test)

        for metric_name, metric_value in metrics.items():
            logger.info('  %s: %.6f', metric_name, metric_value)

        return metrics

    def run(self):
        """Run the full training pipeline."""
        logger.info('')
        logger.info('='*70)
        logger.info('ChemBench Deep Learning Baseline Training')
        logger.info('='*70)
        logger.info('Dataset: %s', self.dataset_name.upper())
        logger.info('Model: %s', self.model_type.upper())
        logger.info('Timestamp: %s', datetime.now().isoformat())
        logger.info('='*70)

        try:
            # Load and clean
            df, target_col = self.load_and_clean_data()

            # Split data
            train, val, test = self.split_data(df, target_col)

            # Prepare features
            X_train, y_train = self.prepare_features(train, target_col)
            X_val, y_val = self.prepare_features(val, target_col)
            X_test, y_test = self.prepare_features(test, target_col)

            logger.info('✓ Feature shape: %s', X_train.shape)

            # Build model
            input_dim = X_train.shape[1]
            model = self.build_model(input_dim=input_dim, output_dim=1)

            # Train
            self.train_model(model, X_train, y_train, X_val, y_val)

            # Evaluate
            metrics = self.evaluate_model(model, X_test, y_test)

            # Save results
            results = {
                'timestamp': datetime.now().isoformat(),
                'dataset': self.dataset_name,
                'model': self.model_type,
                'input_dim': input_dim,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test),
                'hyperparameters': {
                    'epochs': self.epochs,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate,
                    'early_stopping_patience': self.early_stopping_patience,
                },
                'metrics': metrics,
            }

            results_file = self.output_dir / f'dl_{self.model_type}_{self.dataset_name}_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info('')
            logger.info('='*70)
            logger.info('TRAINING COMPLETED SUCCESSFULLY')
            logger.info('='*70)
            logger.info('Results saved to: %s', results_file)
            logger.info('='*70)

        except Exception as exc:
            logger.error('Training failed with error: %s', exc, exc_info=True)
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train deep learning baseline models')
    parser.add_argument('--dataset', default='esol', help='Dataset name (default: esol)')
    parser.add_argument('--model', default='fcnn', help='Model type: fcnn, cnn1d (default: fcnn)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    args = parser.parse_args()

    # Run training pipeline
    pipeline = DeepLearningTrainingPipeline(
        dataset_name=args.dataset,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
        random_seed=args.seed,
    )

    pipeline.run()


if __name__ == '__main__':
    main()

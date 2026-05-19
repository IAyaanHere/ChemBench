"""
Evaluation Runner Script
Loads trained models and generates comprehensive evaluation visualizations and reports.
"""

import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chembench.data.loader import ChemBenchDataLoader
from chembench.data.splitter import DataSplitter
from chembench.evaluation import ModelEvaluator
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


class EvaluationRunner:
    """Orchestrates model evaluation pipeline."""
    
    def __init__(self, dataset_name: str = "esol", eval_dir: str = "./results/evaluation"):
        """
        Initialize evaluation runner.
        
        Args:
            dataset_name: Name of the dataset
            eval_dir: Base directory for evaluation results
        """
        self.dataset_name = dataset_name
        self.eval_dir = Path(eval_dir) / dataset_name
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path("./results/baselines")
        self.evaluator = ModelEvaluator(dpi=300)
        
    def load_data(self) -> tuple:
        """
        Load dataset and create test split.
        
        Returns:
            Tuple of (X_test, y_test, feature_names, target_name)
        """
        print(f"\n{'='*60}")
        print(f"Loading {self.dataset_name.upper()} Dataset")
        print(f"{'='*60}")
        
        # Load dataset
        loader = ChemBenchDataLoader(self.dataset_name)
        df = loader.load_data()
        print(f"✓ Loaded {len(df)} rows × {len(df.columns)} columns")
        
        # Get target column from loader
        targets_df = loader.get_targets()
        target_col = targets_df.columns[0]
        
        if target_col is None:
            raise ValueError(f"Could not determine target column for {self.dataset_name}")
        
        # Split data using DataSplitter
        splitter = DataSplitter()
        
        # Determine if molecular or process dataset
        if self.dataset_name in {'esol', 'freesolv', 'lipophilicity', 'hiv', 'bace', 'tox21', 'bbbp', 'qm9'}:
            # Use scaffold split for molecular data
            train, val, test = splitter.scaffold_split(
                df, 
                smiles_col='smiles' if 'smiles' in df.columns else 'mol',
                test_size=0.2, 
                val_size=0.1
            )
        else:
            # Use random split for process data
            train, val, test, _, _, _ = splitter.random_split(
                df, 
                target_col=target_col,
                test_size=0.2, 
                val_size=0.1
            )
        
        # Extract features and targets
        X_test = test.drop(target_col, axis=1)
        y_test = test[target_col]
        
        # Keep only numeric columns
        numeric_cols = X_test.select_dtypes(include=['number']).columns.tolist()
        X_test = X_test[numeric_cols]
        
        feature_names = X_test.columns.tolist()
        
        print(f"✓ Test split: {len(X_test)} samples × {len(feature_names)} features")
        print(f"✓ Target: {target_col}")
        
        return X_test.values, y_test.values, feature_names, target_col
    
    def load_model(self, model_name: str):
        """
        Load a trained model from pickle file.
        
        Args:
            model_name: Name of the model (e.g., 'random_forest', 'linear')
            
        Returns:
            Loaded model object
        """
        model_path = self.models_dir / f"{model_name}_{self.dataset_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    def get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> dict:
        """
        Calculate evaluation metrics based on task type.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if task_type == "regression":
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
        else:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # ROC-AUC only for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
                except:
                    pass
        
        return metrics
    
    def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list
    ) -> dict:
        """
        Evaluate a single model with visualizations.
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test targets
            feature_names: List of feature names
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name.replace('_', ' ').title()}")
        print(f"{'='*60}")
        
        # Load model
        model = self.load_model(model_name)
        print(f"✓ Model loaded from {self.models_dir / f'{model_name}_{self.dataset_name}.pkl'}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Detect task type
        task_type = self._detect_task_type(y_test)
        print(f"✓ Task type: {task_type}")
        
        # Calculate metrics
        metrics = self.get_metrics(y_test, y_pred, task_type)
        print(f"✓ Metrics calculated:")
        for metric_name, metric_value in metrics.items():
            print(f"  - {metric_name}: {metric_value:.6f}")
        
        # Generate visualizations
        plot_paths = {}
        
        print(f"✓ Generating visualizations...")
        
        # Predictions plot
        pred_plot = self.evaluator.plot_predictions(
            y_test, y_pred,
            self.eval_dir,
            model_name,
            task_type
        )
        plot_paths['predictions'] = pred_plot
        print(f"  ✓ Predictions plot: {pred_plot.name}")
        
        # Residuals plot (regression only)
        if task_type == "regression":
            residuals_plot = self.evaluator.plot_residuals(
                y_test, y_pred,
                self.eval_dir,
                model_name
            )
            plot_paths['residuals'] = residuals_plot
            print(f"  ✓ Residuals plot: {residuals_plot.name}")
        
        # Feature importance (tree-based models only)
        if hasattr(model, 'feature_importances_'):
            fi_plot = self.evaluator.plot_feature_importance(
                model,
                feature_names,
                self.eval_dir,
                model_name,
                top_n=15
            )
            if fi_plot:
                plot_paths['feature_importance'] = fi_plot
                print(f"  ✓ Feature importance plot: {fi_plot.name}")
        
        # Generate report
        report_path = self.evaluator.generate_report(
            model_name,
            task_type,
            metrics,
            y_test,
            y_pred,
            feature_names=feature_names,
            plot_paths=plot_paths,
            save_dir=self.eval_dir
        )
        print(f"  ✓ Evaluation report: {report_path.name}")
        
        return {
            'model_name': model_name,
            'task_type': task_type,
            'metrics': metrics,
            'plots': plot_paths,
            'report': report_path
        }
    
    def _detect_task_type(self, y: np.ndarray) -> str:
        """Detect whether task is regression or classification."""
        if np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) <= 20:
            return "classification"
        return "regression"
    
    def run(self, models: list = None):
        """
        Run evaluation pipeline for specified models.
        
        Args:
            models: List of model names to evaluate (default: random_forest, linear)
        """
        if models is None:
            models = ['random_forest', 'linear']
        
        print(f"\n{'#'*60}")
        print(f"# ChemBench Model Evaluation")
        print(f"# Dataset: {self.dataset_name.upper()}")
        print(f"# Models: {', '.join(models)}")
        print(f"{'#'*60}")
        
        # Load data once
        X_test, y_test, feature_names, target_col = self.load_data()
        
        # Evaluate each model
        results = {}
        for model_name in tqdm(models, desc="Evaluating models"):
            try:
                result = self.evaluate_model(model_name, X_test, y_test, feature_names)
                results[model_name] = result
            except Exception as e:
                print(f"✗ Error evaluating {model_name}: {e}")
                continue
        
        # Summary
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Models evaluated: {len(results)}/{len(models)}")
        print(f"Results saved to: {self.eval_dir}")
        print(f"\nGenerated files:")
        for model_name, result in results.items():
            print(f"\n  {model_name.replace('_', ' ').title()}:")
            for plot_type, plot_path in result['plots'].items():
                print(f"    - {plot_path.name}")
            print(f"    - {result['report'].name}")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained ChemBench models"
    )
    parser.add_argument(
        "--dataset",
        default="esol",
        help="Dataset name (default: esol)"
    )
    parser.add_argument(
        "--models",
        nargs='+',
        default=['random_forest', 'linear'],
        help="Models to evaluate (default: random_forest linear)"
    )
    parser.add_argument(
        "--eval-dir",
        default="./results/evaluation",
        help="Base directory for evaluation results"
    )
    
    args = parser.parse_args()
    
    runner = EvaluationRunner(
        dataset_name=args.dataset,
        eval_dir=args.eval_dir
    )
    
    runner.run(models=args.models)


if __name__ == "__main__":
    main()

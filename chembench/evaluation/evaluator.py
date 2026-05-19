"""
Model Evaluation Module
Provides comprehensive evaluation tools for regression and classification models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from typing import Union, Tuple, Dict, Any, Optional
from datetime import datetime


class ModelEvaluator:
    """
    Comprehensive model evaluation framework.
    Generates publication-quality visualizations and reports for model predictions.
    """
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid", dpi: int = 300):
        """
        Initialize evaluator with styling preferences.
        
        Args:
            style: Matplotlib style to use
            dpi: DPI for saved plots (default 300 for publication quality)
        """
        self.dpi = dpi
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")
        sns.set_palette("husl")
        
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_dir: Union[str, Path],
        model_name: str,
        task_type: str = "regression"
    ) -> Path:
        """
        Generate prediction visualization.
        
        For regression: Scatter plot of actual vs predicted values.
        For classification: Confusion matrix heatmap.
        
        Args:
            y_true: True target values
            y_pred: Predicted values or predicted classes
            save_dir: Directory to save plot
            model_name: Name of the model (for file naming)
            task_type: 'regression' or 'classification'
            
        Returns:
            Path to saved plot
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if task_type == "regression":
            return self._plot_regression_predictions(y_true, y_pred, save_dir, model_name)
        else:
            return self._plot_classification_confusion_matrix(y_true, y_pred, save_dir, model_name)
    
    def _plot_regression_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_dir: Path,
        model_name: str
    ) -> Path:
        """Generate scatter plot for regression predictions."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Labels and title
        ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}: Actual vs Predicted Values', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Save
        filepath = save_dir / f"{model_name}_predictions.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _plot_classification_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_dir: Path,
        model_name: str
    ) -> Path:
        """Generate confusion matrix heatmap for classification."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Confusion matrix heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar_kws={'label': 'Count'},
            ax=ax,
            linewidths=0.5
        )
        
        ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}: Confusion Matrix', fontsize=14, fontweight='bold')
        
        # Save
        filepath = save_dir / f"{model_name}_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_dir: Union[str, Path],
        model_name: str
    ) -> Path:
        """
        Generate residual plot for regression tasks.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            save_dir: Directory to save plot
            model_name: Name of the model
            
        Returns:
            Path to saved plot
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
        axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1].hist(residuals, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'{model_name}: Residual Analysis', fontsize=14, fontweight='bold')
        
        # Save
        filepath = save_dir / f"{model_name}_residuals.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: list,
        save_dir: Union[str, Path],
        model_name: str,
        top_n: int = 15
    ) -> Optional[Path]:
        """
        Extract and plot feature importances for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            save_dir: Directory to save plot
            model_name: Name of the model
            top_n: Number of top features to display
            
        Returns:
            Path to saved plot or None if model doesn't support feature importance
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if model has feature importances
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importances = model.feature_importances_
        
        # Get top N features
        top_indices = np.argsort(importances)[-top_n:]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Horizontal bar plot
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_importances, color='steelblue', edgecolor='black', linewidth=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}: Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Save
        filepath = save_dir / f"{model_name}_feature_importance.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def generate_report(
        self,
        model_name: str,
        task_type: str,
        metrics: Dict[str, float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: Optional[list] = None,
        plot_paths: Optional[Dict[str, Path]] = None,
        save_dir: Union[str, Path] = "."
    ) -> Path:
        """
        Generate a comprehensive Markdown evaluation report.
        
        Args:
            model_name: Name of the model
            task_type: 'regression' or 'classification'
            metrics: Dictionary of metric_name: metric_value
            y_true: True target values
            y_pred: Predicted values
            feature_names: Optional list of feature names
            plot_paths: Optional dict of plot_type: path_to_plot
            save_dir: Directory to save report
            
        Returns:
            Path to saved report
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        report_lines = [
            f"# Evaluation Report: {model_name}",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Task Type:** {task_type.capitalize()}",
            f"\n**Data Points:** {len(y_true)}",
        ]
        
        # Metrics Section
        report_lines.append("\n## Model Performance Metrics\n")
        report_lines.append("| Metric | Value |")
        report_lines.append("|--------|-------|")
        for metric_name, metric_value in metrics.items():
            report_lines.append(f"| {metric_name} | {metric_value:.6f} |")
        
        # Visualizations Section
        if plot_paths:
            report_lines.append("\n## Visualizations\n")
            if "predictions" in plot_paths:
                rel_path = Path(plot_paths["predictions"]).name
                report_lines.append(f"### Predictions Plot")
                report_lines.append(f"![Predictions]({rel_path})\n")
            
            if "residuals" in plot_paths:
                rel_path = Path(plot_paths["residuals"]).name
                report_lines.append(f"### Residual Analysis")
                report_lines.append(f"![Residuals]({rel_path})\n")
            
            if "feature_importance" in plot_paths:
                rel_path = Path(plot_paths["feature_importance"]).name
                report_lines.append(f"### Feature Importance")
                report_lines.append(f"![Feature Importance]({rel_path})\n")
        
        # Additional Statistics
        if task_type == "regression":
            report_lines.append("\n## Prediction Statistics\n")
            report_lines.append(f"- **Mean Absolute Error:** {np.mean(np.abs(y_true - y_pred)):.6f}")
            report_lines.append(f"- **Std Dev of Predictions:** {np.std(y_pred):.6f}")
            report_lines.append(f"- **Std Dev of Residuals:** {np.std(y_true - y_pred):.6f}")
            report_lines.append(f"- **Min/Max True Values:** {y_true.min():.6f} / {y_true.max():.6f}")
            report_lines.append(f"- **Min/Max Predicted Values:** {y_pred.min():.6f} / {y_pred.max():.6f}\n")
        else:
            report_lines.append("\n## Classification Statistics\n")
            unique_classes = np.unique(y_true)
            report_lines.append(f"- **Number of Classes:** {len(unique_classes)}")
            report_lines.append(f"- **Classes:** {unique_classes.tolist()}")
            report_lines.append(f"- **Predictions Accuracy:** {np.mean(y_true == y_pred):.6f}\n")
        
        # Footer
        report_lines.append("\n---")
        report_lines.append("*Report generated by ChemBench Evaluation Framework*")
        
        # Write report
        report_path = save_dir / f"{model_name}_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return report_path

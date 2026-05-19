import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
RAW_DATA_DIR = "chembench/datasets/tennessee_eastman/raw/"
EXPLORATORY_DIR = "chembench/datasets/tennessee_eastman/exploratory/"
DPI = 300

# CSV file names
CSV_FILES = {
    "fault_free_training": "fault_free_training.csv",
    "fault_free_testing": "fault_free_testing.csv",
    "faulty_training": "faulty_training.csv",
    "faulty_testing": "faulty_testing.csv",
}

def load_datasets():
    """Load all dataset CSV files."""
    datasets = {}
    try:
        for key, filename in CSV_FILES.items():
            filepath = os.path.join(RAW_DATA_DIR, filename)
            logger.info(f"Loading {filename}...")
            df = pd.read_csv(filepath, header=None)
            logger.info(f"  Raw shape: {df.shape}")

            # Remove a header-like first row if it contains sequential integer values
            if df.shape[0] > 1:
                first_row = df.iloc[0]
                try:
                    first_row_int = first_row.astype(int)
                    expected_header = pd.Series(range(df.shape[1]), dtype=int)
                    if first_row_int.equals(expected_header):
                        logger.info(f"  Detected header row in {filename}; removing it.")
                        df = df.iloc[1:].reset_index(drop=True)
                        logger.info(f"  Shape after removing header row: {df.shape}")
                except Exception:
                    pass

            # Transpose if dataset appears to be oriented as features x time
            if df.shape[0] < df.shape[1]:
                logger.info(f"Fixing Matrix Shape: Transposing {key}")
                df = df.T.reset_index(drop=True)
                logger.info(f"  Shape after transpose: {df.shape}")

            datasets[key] = df
            logger.info(f"  Final shape: {df.shape}")
            logger.info(f"  Missing values: {df.isnull().sum().sum()}")
        return datasets
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        sys.exit(1)

def print_statistics(datasets):
    """Print basic statistics for each dataset."""
    logger.info("\n" + "="*60)
    logger.info("DATASET STATISTICS")
    logger.info("="*60)
    
    for key, df in datasets.items():
        logger.info(f"\n{key.upper()}")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Data type: {df.dtypes.unique()}")
        logger.info(f"  Missing values: {df.isnull().sum().sum()}")
        logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

def create_exploratory_dir():
    """Create the exploratory directory if it doesn't exist."""
    try:
        os.makedirs(EXPLORATORY_DIR, exist_ok=True)
        logger.info(f"Exploratory directory ready: {EXPLORATORY_DIR}")
    except Exception as e:
        logger.error(f"Error creating directory: {e}")
        sys.exit(1)

def plot_correlation_heatmap(datasets):
    """Generate correlation heatmap of first 10 features from fault_free_training."""
    try:
        logger.info("Generating correlation heatmap...")
        df = datasets["fault_free_training"]
        
        # Select first 10 features
        df_subset = df.iloc[:, :10]
        
        # Compute correlation matrix
        corr_matrix = df_subset.corr()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        plt.title("Correlation Heatmap - First 10 Features (Fault-Free Training)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(EXPLORATORY_DIR, "correlation_heatmap.png")
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info(f"Correlation heatmap saved: {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error generating correlation heatmap: {e}")

def plot_timeseries(datasets):
    """Generate time-series plot of first 3 features over first 300 time steps from faulty_training."""
    try:
        logger.info("Generating time-series plot...")
        df = datasets["faulty_training"]
        
        # Select first 3 features and first 300 time steps
        df_subset = df.iloc[:300, :3]
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot each feature
        for idx, col in enumerate(df_subset.columns):
            axes[idx].plot(df_subset[col], linewidth=1.5, color="steelblue")
            axes[idx].set_ylabel(f"Feature {col}", fontsize=11, fontweight="bold")
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim(0, 300)
        
        axes[-1].set_xlabel("Time Steps", fontsize=11, fontweight="bold")
        fig.suptitle("Time-Series Plot - First 3 Features (Faulty Training, 300 Time Steps)", 
                     fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(EXPLORATORY_DIR, "timeseries_plot.png")
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info(f"Time-series plot saved: {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error generating time-series plot: {e}")

def main():
    """Main function to run the EDA."""
    logger.info("Starting Tennessee Eastman Process EDA...")
    logger.info(f"Raw data directory: {RAW_DATA_DIR}")
    logger.info(f"Exploratory output directory: {EXPLORATORY_DIR}")
    
    # Create output directory
    create_exploratory_dir()
    
    # Load datasets
    datasets = load_datasets()
    
    # Print statistics
    print_statistics(datasets)
    
    # Generate visualizations
    plot_correlation_heatmap(datasets)
    plot_timeseries(datasets)
    
    logger.info("="*60)
    logger.info("EDA completed successfully!")
    logger.info(f"Plots saved to: {EXPLORATORY_DIR}")
    logger.info("="*60)

if __name__ == "__main__":
    main()

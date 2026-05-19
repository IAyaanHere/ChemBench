import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
RAW_DATA_DIR = "chembench/datasets/tennessee_eastman/raw/"
PROCESSED_DATA_DIR = "chembench/datasets/tennessee_eastman/processed/"
TRAIN_VAL_SPLIT = 0.9

# CSV file names
RAW_FILES = {
    "fault_free_training": "fault_free_training.csv",
    "fault_free_testing": "fault_free_testing.csv",
    "faulty_training": "faulty_training.csv",
    "faulty_testing": "faulty_testing.csv",
}

def load_and_preprocess_datasets():
    """Load and preprocess all datasets."""
    datasets = {}
    try:
        for key, filename in RAW_FILES.items():
            filepath = os.path.join(RAW_DATA_DIR, filename)
            logger.info(f"Loading {filename}...")
            df = pd.read_csv(filepath, header=None)
            logger.info(f"  Initial shape: {df.shape}")
            
            # Skip the first row (header row from the CSV file)
            df = df.iloc[1:].reset_index(drop=True)
            logger.info(f"  After removing header row: {df.shape}")
            
            # Check if fault_free_training needs to be transposed
            if key == "fault_free_training" and df.shape[0] == 52 and df.shape[1] >= 500:
                logger.info("Fixing Matrix Shape: Transposing fault_free_training")
                df = df.T
                df = df.reset_index(drop=True)
                logger.info(f"  After transpose: {df.shape}")
            
            # Convert column names to string indices (aligned to 52 features for all datasets)
            df.columns = [str(i) for i in range(df.shape[1])]
            logger.info(f"  Final shape: {df.shape}")
            datasets[key] = df
        
        return datasets
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        sys.exit(1)

def create_labeled_datasets(datasets):
    """Create labeled datasets by assigning labels (0 for fault-free, 1 for faulty)."""
    try:
        logger.info("\nCreating labeled datasets...")
        
        # Fault-free training (label=0)
        ff_train = datasets["fault_free_training"].copy()
        ff_train["label"] = 0
        logger.info(f"Fault-free training: {ff_train.shape}")
        
        # Fault-free testing (label=0)
        ff_test = datasets["fault_free_testing"].copy()
        ff_test["label"] = 0
        logger.info(f"Fault-free testing: {ff_test.shape}")
        
        # Faulty training (label=1)
        faulty_train = datasets["faulty_training"].copy()
        faulty_train["label"] = 1
        logger.info(f"Faulty training: {faulty_train.shape}")
        
        # Faulty testing (label=1)
        faulty_test = datasets["faulty_testing"].copy()
        faulty_test["label"] = 1
        logger.info(f"Faulty testing: {faulty_test.shape}")
        
        return ff_train, ff_test, faulty_train, faulty_test
    except Exception as e:
        logger.error(f"Error creating labeled datasets: {e}")
        sys.exit(1)

def concatenate_datasets(ff_train, ff_test, faulty_train, faulty_test):
    """Concatenate training and testing datasets."""
    try:
        logger.info("\nConcatenating datasets...")
        
        # Full training set
        full_train = pd.concat([ff_train, faulty_train], ignore_index=True)
        logger.info(f"Full training set: {full_train.shape}")
        
        # Full testing set
        full_test = pd.concat([ff_test, faulty_test], ignore_index=True)
        logger.info(f"Full testing set: {full_test.shape}")
        
        return full_train, full_test
    except Exception as e:
        logger.error(f"Error concatenating datasets: {e}")
        sys.exit(1)

def split_and_scale(full_train, full_test):
    """Perform stratified train/val split and scale all sets."""
    try:
        logger.info("\nPerforming stratified train/val split (90/10)...")
        
        # Separate features and labels
        X_train_full = full_train.drop("label", axis=1)
        y_train_full = full_train["label"]
        
        X_test = full_test.drop("label", axis=1)
        y_test = full_test["label"]
        
        # Stratified split: 90% train, 10% val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=1 - TRAIN_VAL_SPLIT,
            stratify=y_train_full,
            random_state=42
        )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        logger.info(f"Testing set: {X_test.shape}")
        logger.info(f"  Train label distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"  Val label distribution: {y_val.value_counts().to_dict()}")
        logger.info(f"  Test label distribution: {y_test.value_counts().to_dict()}")
        
        # Fit scaler ONLY on the training set
        logger.info("\nFitting StandardScaler on training set...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        logger.info(f"  Mean (training): {scaler.mean_[:5]}... (showing first 5)")
        logger.info(f"  Std (training): {scaler.scale_[:5]}... (showing first 5)")
        
        # Transform validation and test sets using the fitted scaler
        logger.info("Transforming validation and test sets...")
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    except Exception as e:
        logger.error(f"Error in scaling and splitting: {e}")
        sys.exit(1)

def save_processed_datasets(X_train, X_val, X_test, y_train, y_val, y_test):
    """Save processed datasets to CSV files."""
    try:
        logger.info("\nSaving processed datasets...")
        
        # Create output directory
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        logger.info(f"Output directory created: {PROCESSED_DATA_DIR}")
        
        # Convert scaled arrays to DataFrames and add labels
        train_df = pd.DataFrame(X_train)
        train_df["label"] = y_train.reset_index(drop=True)
        
        val_df = pd.DataFrame(X_val)
        val_df["label"] = y_val.reset_index(drop=True)
        
        test_df = pd.DataFrame(X_test)
        test_df["label"] = y_test.reset_index(drop=True)
        
        # Save to CSV
        train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
        val_path = os.path.join(PROCESSED_DATA_DIR, "val.csv")
        test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        logger.info(f"Training set saved: {train_path}")
        
        val_df.to_csv(val_path, index=False)
        logger.info(f"Validation set saved: {val_path}")
        
        test_df.to_csv(test_path, index=False)
        logger.info(f"Testing set saved: {test_path}")
    except Exception as e:
        logger.error(f"Error saving processed datasets: {e}")
        sys.exit(1)

def main():
    """Main preprocessing pipeline."""
    logger.info("="*60)
    logger.info("Starting Tennessee Eastman Process Preprocessing")
    logger.info("="*60)
    logger.info(f"Raw data directory: {RAW_DATA_DIR}")
    logger.info(f"Processed data directory: {PROCESSED_DATA_DIR}")
    
    # Step 1: Load and preprocess
    datasets = load_and_preprocess_datasets()
    
    # Step 2: Create labeled datasets
    ff_train, ff_test, faulty_train, faulty_test = create_labeled_datasets(datasets)
    
    # Step 3: Concatenate datasets
    full_train, full_test = concatenate_datasets(ff_train, ff_test, faulty_train, faulty_test)
    
    # Step 4: Split and scale
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(full_train, full_test)
    
    # Step 5: Save processed datasets
    save_processed_datasets(X_train, X_val, X_test, y_train, y_val, y_test)
    
    logger.info("="*60)
    logger.info("Preprocessing completed successfully!")
    logger.info("="*60)

if __name__ == "__main__":
    main()

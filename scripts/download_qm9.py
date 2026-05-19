import pandas as pd
import logging
from pathlib import Path
import numpy as np
from torch_geometric.datasets import QM9

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create directories
    raw_dir = project_root / 'chembench' / 'datasets' / 'qm9' / 'raw'
    processed_dir = project_root / 'chembench' / 'datasets' / 'qm9' / 'processed'
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Raw directory: {raw_dir}")
    logger.info(f"Processed directory: {processed_dir}")
    
    # Download QM9 dataset
    logger.info("Downloading QM9 dataset...")
    try:
        dataset = QM9(root=str(raw_dir))
        logger.info(f"QM9 dataset loaded successfully. Total molecules: {len(dataset)}")
    except Exception as e:
        logger.error(f"Error downloading QM9 dataset: {e}")
        raise
    
    # Extract data
    logger.info("Extracting SMILES and target properties...")
    data_list = []
    
    try:
        for idx, data in enumerate(dataset):
            if idx % 5000 == 0:
                logger.info(f"Processing molecule {idx}/{len(dataset)}")
            
            # Extract SMILES from the data object
            smiles = data.smiles if hasattr(data, 'smiles') else f"molecule_{idx}"
            
            # Extract 19 target properties
            y = data.y.numpy() if hasattr(data.y, 'numpy') else data.y.cpu().detach().numpy()
            y = y.flatten()  # Ensure it's 1D
            
            # Create row with SMILES and 19 properties
            row = {'SMILES': smiles}
            property_names = [
                'U0', 'U', 'H', 'G', 'Cv',  # Index 0-4
                'D', 'Mu', 'Alpha', 'HOMO', 'LUMO',  # Index 5-9
                'Gap', 'R2', 'ZPVE', 'omega1', 'omega2',  # Index 10-14
                'omega3', 'omega4', 'omega5', 'omega6'  # Index 15-18
            ]
            
            for i in range(min(19, len(y))):
                prop_name = property_names[i] if i < len(property_names) else f'Property_{i}'
                row[prop_name] = float(y[i].item()) if hasattr(y[i], 'item') else float(y[i])
            
            data_list.append(row)
        
        logger.info(f"Successfully extracted {len(data_list)} molecules")
    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        raise
    
    # Create DataFrame
    logger.info("Creating DataFrame...")
    df = pd.DataFrame(data_list)
    logger.info(f"DataFrame created with shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Save full dataset
    full_csv_path = processed_dir / 'qm9_molecules.csv'
    df.to_csv(full_csv_path, index=False)
    logger.info(f"Full dataset saved to {full_csv_path}")
    
    # Create train/val/test splits (80/10/10)
    logger.info("Creating train/val/test splits (80/10/10)...")
    n = len(df)
    indices = np.random.permutation(n)
    
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    logger.info(f"Train set: {len(train_df)} samples ({len(train_df)/n*100:.1f}%)")
    logger.info(f"Val set: {len(val_df)} samples ({len(val_df)/n*100:.1f}%)")
    logger.info(f"Test set: {len(test_df)} samples ({len(test_df)/n*100:.1f}%)")
    
    # Save splits
    train_csv_path = processed_dir / 'train.csv'
    val_csv_path = processed_dir / 'val.csv'
    test_csv_path = processed_dir / 'test.csv'
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    logger.info(f"Train split saved to {train_csv_path}")
    logger.info(f"Val split saved to {val_csv_path}")
    logger.info(f"Test split saved to {test_csv_path}")
    logger.info("QM9 dataset processing completed successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("QM9 DATASET PROCESSING SUMMARY")
    print("="*60)
    print(f"Total molecules: {len(df)}")
    print(f"Features: {df.shape[1]}")
    print(f"Train/Val/Test split: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"Output directory: {processed_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

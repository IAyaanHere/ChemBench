import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def final_cleanup():
    # 1. Delete empty lingering root directories
    for old_dir in ['chembench/datasets/processed', 'chembench/datasets/exploratory']:
        p = Path(old_dir)
        if p.exists():
            shutil.rmtree(p)
            logging.info(f"Deleted old root directory: {p}")

    # 2. Fix PyTorch Geometric nested folders in QM9
    qm9_raw = Path('chembench/datasets/qm9/raw')
    nested_raw = qm9_raw / 'raw'
    nested_proc = qm9_raw / 'processed'

    try:
        # Move files from qm9/raw/raw/ to qm9/raw/
        if nested_raw.exists():
            for f in nested_raw.iterdir():
                shutil.move(str(f), str(qm9_raw / f.name))
            nested_raw.rmdir()
            logging.info("Flattened nested raw folder in QM9.")

        # Move files from qm9/raw/processed/ to qm9/raw/
        if nested_proc.exists():
            for f in nested_proc.iterdir():
                shutil.move(str(f), str(qm9_raw / f.name))
            nested_proc.rmdir()
            logging.info("Flattened nested processed folder in QM9.")
            
    except Exception as e:
        logging.error(f"Error while moving files: {e}")

if __name__ == "__main__":
    final_cleanup()
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def fix_structure():
    # 1. Fix ESOL folder capitalization
    datasets_dir = Path('chembench/datasets')
    esol_upper = datasets_dir / 'ESOL'
    esol_lower = datasets_dir / 'esol'

    if esol_upper.exists() and esol_upper.is_dir():
        logging.info("Found uppercase 'ESOL' folder. Renaming to 'esol'...")
        esol_upper.rename(esol_lower)
        logging.info("✅ Renamed ESOL to esol.")

    # 2. Move integrate_manual_downloads.py to scripts/
    root_script = Path('integrate_manual_downloads.py')
    target_script = Path('scripts/integrate_manual_downloads.py')

    if root_script.exists():
        logging.info("Found integrate_manual_downloads.py in root. Moving to scripts/...")
        shutil.move(str(root_script), str(target_script))
        logging.info("✅ Moved script to scripts/ folder.")

    logging.info("🎉 Structure analysis and cleanup complete. Everything is uniform now!")

if __name__ == "__main__":
    fix_structure()
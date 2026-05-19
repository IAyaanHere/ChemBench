import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clear_github_clutter():
    # Target raw github repo folder
    repo_dir = Path('chembench/datasets/chemixhub/raw/github_repo')
    
    if not repo_dir.exists():
        logging.warning(f"Path not found: {repo_dir}")
        return

    # Items we DON'T need (based on your screenshot)
    clutter_items = [
        '.git', '.gitignore', 'LICENSE', 'README.md', 
        'setup.cfg', 'setup.py', 'src', 'scripts', 'media', 'config'
    ]

    for item in clutter_items:
        item_path = repo_dir / item
        if item_path.exists():
            try:
                if item_path.is_file():
                    item_path.unlink()
                elif item_path.is_dir():
                    shutil.rmtree(item_path)
                logging.info(f"🗑️ Deleted: {item}")
            except Exception as e:
                logging.error(f"Failed to delete {item}: {e}")

    logging.info("✅ Cleanup complete! Only essential raw data is left.")

if __name__ == "__main__":
    clear_github_clutter()

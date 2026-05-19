import logging
from pathlib import Path
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get project root
    project_root = Path(__file__).parent.parent
    datasets_dir = project_root / 'chembench' / 'datasets'
    
    # Define old and new paths
    old_raw_dir = datasets_dir / 'raw'
    old_processed_tep_dir = datasets_dir / 'processed' / 'tep'
    old_exploratory_tep_dir = datasets_dir / 'exploratory' / 'tep'
    
    new_tep_base_dir = datasets_dir / 'tennessee_eastman'
    new_raw_dir = new_tep_base_dir / 'raw'
    new_processed_dir = new_tep_base_dir / 'processed'
    new_exploratory_dir = new_tep_base_dir / 'exploratory'
    
    logger.info("="*70)
    logger.info("STARTING DATASET STRUCTURE REFACTORING")
    logger.info("="*70)
    
    # Step 1: Create new directory structure
    logger.info("\nStep 1: Creating new directory structure...")
    new_raw_dir.mkdir(parents=True, exist_ok=True)
    new_processed_dir.mkdir(parents=True, exist_ok=True)
    new_exploratory_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Created {new_tep_base_dir}")
    logger.info(f"  ├── raw/")
    logger.info(f"  ├── processed/")
    logger.info(f"  └── exploratory/")
    
    # Step 2: Move raw TEP CSV files
    logger.info("\nStep 2: Moving raw TEP CSV files...")
    if old_raw_dir.exists():
        for csv_file in old_raw_dir.glob('*.csv'):
            dest_file = new_raw_dir / csv_file.name
            logger.info(f"Moving {csv_file.name}...")
            shutil.move(str(csv_file), str(dest_file))
            logger.info(f"✓ Moved to {dest_file}")
    else:
        logger.warning(f"Old raw directory not found: {old_raw_dir}")
    
    # Step 3: Move processed TEP files
    logger.info("\nStep 3: Moving processed TEP files...")
    if old_processed_tep_dir.exists():
        for csv_file in old_processed_tep_dir.glob('*.csv'):
            dest_file = new_processed_dir / csv_file.name
            logger.info(f"Moving {csv_file.name}...")
            shutil.move(str(csv_file), str(dest_file))
            logger.info(f"✓ Moved to {dest_file}")
    else:
        logger.warning(f"Old processed directory not found: {old_processed_tep_dir}")
    
    # Step 4: Move exploratory TEP files
    logger.info("\nStep 4: Moving exploratory TEP files...")
    if old_exploratory_tep_dir.exists():
        for file in old_exploratory_tep_dir.glob('*'):
            if file.is_file():
                dest_file = new_exploratory_dir / file.name
                logger.info(f"Moving {file.name}...")
                shutil.move(str(file), str(dest_file))
                logger.info(f"✓ Moved to {dest_file}")
    else:
        logger.warning(f"Old exploratory directory not found: {old_exploratory_tep_dir}")
    
    # Step 5: Update Python scripts with new paths
    logger.info("\nStep 5: Scanning and updating Python scripts...")
    scripts_dir = project_root / 'scripts'
    
    # Define path replacements
    replacements = [
        ('chembench/datasets/tennessee_eastman/processed/', 'chembench/datasets/tennessee_eastman/processed/'),
        ("'chembench/datasets/tennessee_eastman/processed/'", "'chembench/datasets/tennessee_eastman/processed/'"),
        ('chembench/datasets/tennessee_eastman/raw/', 'chembench/datasets/tennessee_eastman/raw/'),
        ("'chembench/datasets/tennessee_eastman/raw/'", "'chembench/datasets/tennessee_eastman/raw/'"),
        ('chembench/datasets/tennessee_eastman/exploratory/', 'chembench/datasets/tennessee_eastman/exploratory/'),
        ("'chembench/datasets/tennessee_eastman/exploratory/'", "'chembench/datasets/tennessee_eastman/exploratory/'"),
    ]
    
    updated_files = []
    for py_file in scripts_dir.glob('*.py'):
        try:
            content = py_file.read_text()
            original_content = content
            
            # Apply all replacements
            for old_path, new_path in replacements:
                if old_path in content:
                    content = content.replace(old_path, new_path)
                    logger.info(f"  In {py_file.name}: {old_path} → {new_path}")
            
            # Write back if changes were made
            if content != original_content:
                py_file.write_text(content)
                updated_files.append(py_file.name)
                logger.info(f"✓ Updated {py_file.name}")
        except Exception as e:
            logger.error(f"Error updating {py_file.name}: {e}")
    
    if updated_files:
        logger.info(f"✓ Updated {len(updated_files)} script(s): {', '.join(updated_files)}")
    else:
        logger.info("No scripts needed updating.")
    
    # Step 6: Remove old empty directories
    logger.info("\nStep 6: Cleaning up old directories...")
    
    dirs_to_remove = []
    
    # Remove old tep subdirectories if empty
    if old_processed_tep_dir.exists() and not list(old_processed_tep_dir.glob('*')):
        dirs_to_remove.append(old_processed_tep_dir)
    
    if old_exploratory_tep_dir.exists() and not list(old_exploratory_tep_dir.glob('*')):
        dirs_to_remove.append(old_exploratory_tep_dir)
    
    # Remove old raw directory if empty (except .gitkeep)
    if old_raw_dir.exists():
        remaining_files = [f for f in old_raw_dir.glob('*') if f.name != '.gitkeep']
        if not remaining_files:
            dirs_to_remove.append(old_raw_dir)
    
    for dir_path in dirs_to_remove:
        try:
            if dir_path.is_dir():
                shutil.rmtree(dir_path)
                logger.info(f"✓ Removed empty directory: {dir_path.relative_to(project_root)}")
        except Exception as e:
            logger.error(f"Error removing {dir_path}: {e}")
    
    # Step 7: Verify new structure
    logger.info("\nStep 7: Verifying new structure...")
    logger.info(f"✓ New TEP dataset location: {new_tep_base_dir.relative_to(project_root)}/")
    
    if new_raw_dir.exists():
        raw_files = list(new_raw_dir.glob('*.csv'))
        logger.info(f"  ├── raw/ ({len(raw_files)} files)")
        for f in raw_files:
            logger.info(f"      ├── {f.name}")
    
    if new_processed_dir.exists():
        processed_files = list(new_processed_dir.glob('*.csv'))
        logger.info(f"  ├── processed/ ({len(processed_files)} files)")
        for f in processed_files:
            logger.info(f"      ├── {f.name}")
    
    if new_exploratory_dir.exists():
        exploratory_files = list(new_exploratory_dir.glob('*'))
        logger.info(f"  └── exploratory/ ({len(exploratory_files)} files)")
        for f in exploratory_files:
            logger.info(f"      └── {f.name}")
    
    logger.info("\n" + "="*70)
    logger.info("REFACTORING COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    print("\n" + "="*70)
    print("DATASET STRUCTURE REFACTORING SUMMARY")
    print("="*70)
    print(f"New dataset root: chembench/datasets/tennessee_eastman/")
    print(f"  ├── raw/ - Contains raw TEP CSV files")
    print(f"  ├── processed/ - Contains train/val/test splits")
    print(f"  └── exploratory/ - Contains EDA visualizations")
    print(f"\nUpdated scripts: {', '.join(updated_files) if updated_files else 'None'}")
    print("="*70)

if __name__ == "__main__":
    main()

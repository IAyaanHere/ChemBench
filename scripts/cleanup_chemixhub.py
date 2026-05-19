"""
CleanUp CheMixHub Dataset Structure

Reorganizes the CheMixHub dataset by moving the cloned repository source 
into the unified directory structure.

Current structure:
    chembench/datasets/
    ├── chemixhub_source/      (raw cloned repo)
    └── chemixhub/
        ├── processed/
        ├── exploratory/
        └── raw/               (will be created)

Target structure:
    chembench/datasets/
    └── chemixhub/
        ├── raw/
        │   └── github_repo/   (renamed from chemixhub_source)
        ├── processed/
        └── exploratory/
"""

import logging
import shutil
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def handle_remove_readonly(func, path, exc):
    """Handle read-only files during directory removal"""
    import stat
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR | stat.S_IREAD)
        func(path)
    else:
        raise


def cleanup_chemixhub_structure():
    """Move chemixhub_source into chemixhub/raw/github_repo/"""
    
    logger.info("=" * 70)
    logger.info("STARTING CHEMIXHUB CLEANUP")
    logger.info("=" * 70)
    
    # Define paths
    project_root = Path(__file__).parent.parent
    source_dir = project_root / 'chembench' / 'datasets' / 'chemixhub_source'
    chemixhub_dir = project_root / 'chembench' / 'datasets' / 'chemixhub'
    raw_dir = chemixhub_dir / 'raw'
    target_dir = raw_dir / 'github_repo'
    
    logger.info(f"\nProject root: {project_root}")
    logger.info(f"Source directory: {source_dir}")
    logger.info(f"Target directory: {target_dir}")
    
    # Check if source exists
    if not source_dir.exists():
        logger.warning(f"\n✗ Source directory does not exist: {source_dir}")
        logger.warning("No cleanup needed.")
        return False
    
    logger.info(f"\n✓ Source directory exists: {source_dir}")
    
    # Check if target already exists
    if target_dir.exists():
        logger.warning(f"\n✗ Target directory already exists: {target_dir}")
        logger.info("The copy operation may have already completed.")
        logger.info("Attempting to remove the source directory only...")
        
        # Try to remove the source directory
        try:
            shutil.rmtree(str(source_dir), onerror=handle_remove_readonly)
            logger.info(f"✓ Successfully removed source directory: {source_dir}")
        except Exception as e:
            logger.error(f"✗ Failed to remove source directory: {e}")
            return False
    else:
        # Create raw directory if it doesn't exist
        try:
            raw_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Created raw directory: {raw_dir}")
        except Exception as e:
            logger.error(f"✗ Failed to create raw directory: {e}")
            return False
        
        # Move source to target
        try:
            logger.info(f"\nMoving: {source_dir} -> {target_dir}")
            logger.info("Note: Using copy + delete approach to handle git read-only files...")
            
            # Copy directory recursively
            shutil.copytree(str(source_dir), str(target_dir), dirs_exist_ok=False)
            logger.info(f"✓ Successfully copied source directory")
            
            # Remove source directory with error handler for read-only files
            shutil.rmtree(str(source_dir), onerror=handle_remove_readonly)
            logger.info(f"✓ Successfully removed source directory")
            
        except Exception as e:
            logger.error(f"✗ Failed during move operation: {e}")
            # Clean up partial copy if it failed
            if target_dir.exists():
                try:
                    shutil.rmtree(str(target_dir), onerror=handle_remove_readonly)
                    logger.info("Cleaned up partial copy.")
                except Exception as cleanup_e:
                    logger.error(f"Failed to clean up: {cleanup_e}")
            return False
    
    # Verify the move
    if not target_dir.exists():
        logger.error(f"✗ Verification failed: {target_dir} does not exist")
        return False
    
    if source_dir.exists():
        logger.error(f"✗ Verification failed: {source_dir} still exists")
        return False
    
    logger.info(f"✓ Verification successful: {target_dir} exists")
    logger.info(f"✓ Source directory removed: {source_dir} no longer exists")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("CHEMIXHUB CLEANUP COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info("\nBefore:")
    logger.info("  chembench/datasets/chemixhub_source/")
    logger.info("  chembench/datasets/chemixhub/processed/")
    logger.info("\nAfter:")
    logger.info("  chembench/datasets/chemixhub/")
    logger.info("  ├── raw/")
    logger.info("  │   └── github_repo/          (moved from chemixhub_source)")
    logger.info("  ├── processed/")
    logger.info("  └── exploratory/")
    logger.info("=" * 70)
    
    return True


if __name__ == '__main__':
    success = cleanup_chemixhub_structure()
    exit(0 if success else 1)

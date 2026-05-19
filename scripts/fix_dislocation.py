"""
Fix Structural Dislocation in ChemBench Project

Moves misplaced directories and files from chembench/ root to chembench/datasets/:
- chembench/qm9/ -> chembench/datasets/qm9/
- chembench/tennessee_eastman/ -> chembench/datasets/tennessee_eastman/
- chembench/DATASETS.md -> chembench/datasets/DATASETS.md
"""

import shutil
from pathlib import Path


def fix_dislocation():
    """Move misplaced items back to chembench/datasets/"""
    
    print("=" * 70)
    print("FIXING STRUCTURAL DISLOCATION")
    print("=" * 70)
    
    # Define project root
    project_root = Path(__file__).parent.parent
    chembench_dir = project_root / 'chembench'
    datasets_dir = chembench_dir / 'datasets'
    
    print(f"\nProject root: {project_root}")
    print(f"Target location: {datasets_dir}")
    
    # Ensure datasets directory exists
    datasets_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Ensured {datasets_dir} exists\n")
    
    # Define items to move
    items_to_move = [
        ('qm9', 'directory'),
        ('tennessee_eastman', 'directory'),
        ('DATASETS.md', 'file'),
    ]
    
    moved_count = 0
    
    for item_name, item_type in items_to_move:
        source = chembench_dir / item_name
        target = datasets_dir / item_name
        
        # Check if source exists
        if not source.exists():
            print(f"⊘ {item_name}: Not found at {source}")
            continue
        
        # Check if target already exists
        if target.exists():
            print(f"✓ {item_name}: Already at {target}")
            moved_count += 1
            continue
        
        # Move the item
        try:
            shutil.move(str(source), str(target))
            print(f"✓ Moved {item_type}: {item_name}")
            print(f"  From: {source}")
            print(f"  To:   {target}\n")
            moved_count += 1
        except Exception as e:
            print(f"✗ Failed to move {item_name}: {e}\n")
    
    # Summary
    print("=" * 70)
    print(f"COMPLETED: {moved_count}/{len(items_to_move)} items processed successfully")
    print("=" * 70)
    
    return moved_count == len(items_to_move)


if __name__ == '__main__':
    success = fix_dislocation()
    exit(0 if success else 1)

from pathlib import Path
import shutil
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_problematic_name(name: str) -> bool:
    return '\\' in name or '/' in name


def processed_empty(path: Path) -> bool:
    proc = path / 'processed'
    if not proc.exists():
        return True
    if not proc.is_dir():
        return True
    # consider empty if no files inside
    return not any(proc.iterdir())


def main():
    root = Path(__file__).parent.parent / 'chembench' / 'datasets'
    if not root.exists():
        logger.error('Datasets root not found: %s', root)
        return

    removed = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if is_problematic_name(child.name):
            logger.info('Removing directory with problematic name: %s', child)
            shutil.rmtree(child)
            removed.append(str(child))
            continue

        # Remove incomplete dataset roots (no processed data or processed empty)
        if processed_empty(child):
            logger.info('Removing incomplete dataset folder: %s', child)
            shutil.rmtree(child)
            removed.append(str(child))

    if removed:
        logger.info('Removed %d folders', len(removed))
    else:
        logger.info('No problematic or incomplete folders found')


if __name__ == '__main__':
    main()

import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import deepchem as dc
except Exception as exc:
    raise ImportError("DeepChem is required. Activate the venv where deepchem is installed.") from exc

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DATASETS = {
    'ESOL': 'load_delaney',
    'FreeSolv': 'load_freesolv',
    'Lipophilicity': 'load_lipophilicity',
    'HIV': 'load_hiv',
    'BACE': 'load_bace',
    'Tox21': 'load_tox21',
    'BBBP': 'load_bbbp',
}

FEATURIZER = 'ECFP'


def get_loader(loader_name: str):
    loader = getattr(dc.molnet, loader_name, None)
    if loader is None:
        raise AttributeError(f"DeepChem does not expose loader '{loader_name}'")
    return loader


def make_target_entry(y_row):
    # y_row may be scalar-like, numpy scalar, or array-like
    try:
        arr = np.asarray(y_row)
    except Exception:
        return y_row
    if arr.ndim == 0:
        return float(arr) if np.isfinite(arr) else None
    if arr.size == 1:
        val = arr.flatten()[0]
        try:
            return float(val)
        except Exception:
            return val
    # multi-task: return list
    return arr.flatten().tolist()


def standardize_dataset(dataset, split_name: str, tasks):
    n = len(dataset)
    ids = getattr(dataset, 'ids', None)
    if ids is None:
        ids = [f'{split_name}_{i:06d}' for i in range(n)]
    else:
        ids = [str(x) for x in ids]

    # Attempt to obtain smiles from ids when possible; otherwise keep ids as placeholder
    smiles = ids

    # handle y which can be ragged or have unexpected shape
    raw_y = getattr(dataset, 'y', None)
    targets = []
    if raw_y is None:
        targets = [None] * n
    else:
        # raw_y may be a numpy array, list of lists, or nested dtype
        for i in range(n):
            try:
                row = raw_y[i]
            except Exception:
                row = None
            targets.append(make_target_entry(row))

    data = {
        'molecule_id': ids,
        'smiles': smiles,
        'target': [json.dumps(t) if isinstance(t, list) else t for t in targets],
        'split': [split_name] * n,
    }

    return pd.DataFrame(data)


def process_dataset(dataset_name: str, loader_name: str, base_dir: Path):
    logger.info(f"Starting dataset: {dataset_name}")
    loader = get_loader(loader_name)
    output_dir = Path('chembench') / 'datasets' / dataset_name.lower() / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Loading {dataset_name} with featurizer={FEATURIZER}")
        tasks, datasets, transformers = loader(featurizer=FEATURIZER, reload=False)
        train_dataset, valid_dataset, test_dataset = datasets
        tcounts = [len(x) if x is not None else 0 for x in (train_dataset, valid_dataset, test_dataset)]
        logger.info(f"Loaded {dataset_name}: train={tcounts[0]}, valid={tcounts[1]}, test={tcounts[2]}")
    except Exception as exc:
        logger.error(f"Failed to load {dataset_name}: {exc}")
        return None

    frames = []
    for split_name, split_dataset in zip(['train', 'val', 'test'], [train_dataset, valid_dataset, test_dataset]):
        if split_dataset is None:
            continue
        try:
            df_split = standardize_dataset(split_dataset, split_name, tasks)
            frames.append(df_split)
        except Exception as exc:
            logger.error(f"Failed to standardize {dataset_name} split {split_name}: {exc}")
            # skip this split
            continue

    if not frames:
        logger.warning(f"No valid splits for {dataset_name}; skipping CSV export")
        return None

    combined = pd.concat(frames, ignore_index=True)
    csv_path = output_dir / f'{dataset_name.lower()}.csv'
    combined.to_csv(csv_path, index=False)
    logger.info(f"Saved standardized CSV for {dataset_name}: {csv_path}")

    # metadata
    metadata = {
        'dataset_name': dataset_name,
        'loader': loader_name,
        'featurizer': FEATURIZER,
        'tasks': tasks,
        'num_samples': int(len(combined)),
        'split_counts': {
            'train': int((combined['split'] == 'train').sum()),
            'val': int((combined['split'] == 'val').sum()),
            'test': int((combined['split'] == 'test').sum()),
        },
    }

    metadata_path = output_dir / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logger.info(f"Saved metadata for {dataset_name}: {metadata_path}")

    return {
        'dataset_name': dataset_name,
        'csv_path': str(csv_path),
        'metadata_path': str(metadata_path),
        'num_samples': int(len(combined)),
    }


def main():
    project_root = Path(__file__).parent.parent
    datasets_root = project_root / 'chembench' / 'datasets'
    datasets_root.mkdir(parents=True, exist_ok=True)

    results = []
    iterator = tqdm(DATASETS.items(), desc='Downloading MoleculeNet') if tqdm else DATASETS.items()

    for dataset_name, loader_name in iterator:
        try:
            result = process_dataset(dataset_name, loader_name, datasets_root)
            if result is not None:
                results.append(result)
        except Exception as exc:
            logger.error(f"Unhandled error while processing {dataset_name}: {exc}")

    summary_path = datasets_root / 'moleculenet_download_summary.json'
    summary_path.write_text(json.dumps({'datasets': results}, indent=2), encoding='utf-8')
    logger.info(f"Download summary saved: {summary_path}")
    logger.info('MoleculeNet download process completed.')


if __name__ == '__main__':
    main()


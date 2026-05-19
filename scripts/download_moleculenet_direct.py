from pathlib import Path
import json
import logging

import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


URLS = {
    'esol': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
    'freesolv': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv',
    'lipophilicity': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv',
    'hiv': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv',
    'bace': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv',
    'tox21': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
    'bbbp': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',
}


def save_dataset(name: str, url: str, root: Path):
    out_dir = root / name / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'{name}.csv'
    meta_path = out_dir / 'metadata.json'

    try:
        logger.info('Downloading %s from %s', name, url)
        df = pd.read_csv(url)
        df.to_csv(csv_path, index=False)

        metadata = {
            'dataset_name': name,
            'num_rows': int(df.shape[0]),
            'num_columns': int(df.shape[1]),
            'columns': list(df.columns[:50]),
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(f'SUCCESS: {name} -> {csv_path}')
        logger.info('Saved %s: %s', name, csv_path)
        return True
    except Exception as exc:
        print(f'FAIL: {name} -> {exc}')
        logger.error('Failed to download %s: %s', name, exc)
        return False


def main():
    root = Path('chembench') / 'datasets'
    root.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, url in URLS.items():
        ok = save_dataset(name, url, root)
        results[name] = {'success': ok}

    summary_path = root / 'moleculenet_direct_download_summary.json'
    summary_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    logger.info('Download summary written to %s', summary_path)


if __name__ == '__main__':
    main()

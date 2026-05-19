import logging
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chembench.data.cleaner import DataCleaner
from chembench.data.loader import ChemBenchDataLoader


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _numeric_feature_columns(df, exclude: List[str]) -> List[str]:
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]


def clean_esol():
    logger.info('Starting ESOL cleaning pipeline')
    loader = ChemBenchDataLoader('esol')
    df = loader.load_data()
    logger.info('ESOL original shape: %s', df.shape)

    cleaner = DataCleaner()
    df = cleaner.validate_chemistry(df, smiles_col='smiles')
    df = cleaner.handle_missing(df, strategy='mean')
    df = cleaner.remove_outliers(df, method='iqr', threshold=1.5)

    target_column = 'measured log solubility in mols per litre'
    features = _numeric_feature_columns(df, exclude=[target_column])
    df = cleaner.normalize_features(df, features=features, method='standard')

    logger.info('ESOL cleaned shape: %s', df.shape)
    print('ESOL: shape before cleaning =', loader.load_data().shape)
    print('ESOL: shape after cleaning =', df.shape)
    return df


def clean_tep():
    logger.info('Starting Tennessee Eastman cleaning pipeline')
    loader = ChemBenchDataLoader('tennessee_eastman')
    df = loader.load_data()
    logger.info('TEP original shape: %s', df.shape)

    cleaner = DataCleaner()
    df = cleaner.handle_missing(df, strategy='mean')
    df = cleaner.remove_outliers(df, method='iqr', threshold=1.5)

    features = _numeric_feature_columns(df, exclude=['label'])
    df = cleaner.normalize_features(df, features=features, method='standard')

    logger.info('TEP cleaned shape: %s', df.shape)
    print('TEP: shape before cleaning =', loader.load_data().shape)
    print('TEP: shape after cleaning =', df.shape)
    return df


def main():
    project_root = Path(__file__).resolve().parent.parent
    logger.info('Project root detected at %s', project_root)

    esol_cleaned = clean_esol()
    tep_cleaned = clean_tep()

    logger.info('Cleaning complete. ESOL rows after cleaning: %s; TEP rows after cleaning: %s',
                len(esol_cleaned), len(tep_cleaned))


if __name__ == '__main__':
    main()

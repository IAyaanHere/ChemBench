import logging
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chembench.data.cleaner import DataCleaner
from chembench.data.loader import ChemBenchDataLoader
from chembench.data.splitter import DataSplitter


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _numeric_feature_columns(df, exclude: List[str]) -> List[str]:
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]


def clean_esol(df):
    cleaner = DataCleaner()
    df = cleaner.validate_chemistry(df, smiles_col='smiles')
    df = cleaner.handle_missing(df, strategy='mean')
    df = cleaner.remove_outliers(df, method='iqr', threshold=1.5)

    target_column = 'measured log solubility in mols per litre'
    features = _numeric_feature_columns(df, exclude=[target_column])
    df = cleaner.normalize_features(df, features=features, method='standard')
    return df


def clean_tep(df):
    cleaner = DataCleaner()
    df = cleaner.handle_missing(df, strategy='mean')
    features = _numeric_feature_columns(df, exclude=['label'])
    df = cleaner.normalize_features(df, features=features, method='standard')
    return df


def main():
    project_root = Path(__file__).resolve().parent.parent
    logger.info('Project root detected at %s', project_root)

    # ESOL pipeline
    logger.info('Loading ESOL dataset')
    esol_loader = ChemBenchDataLoader('esol')
    esol_df = esol_loader.load_data()
    logger.info('ESOL raw shape: %s', esol_df.shape)

    esol_cleaned = clean_esol(esol_df)
    splitter = DataSplitter()
    esol_train, esol_val, esol_test = splitter.scaffold_split(esol_cleaned, smiles_col='smiles')

    logger.info('ESOL splits: train=%s, val=%s, test=%s', esol_train.shape, esol_val.shape, esol_test.shape)
    print('ESOL train:', esol_train.shape)
    print('ESOL val:  ', esol_val.shape)
    print('ESOL test: ', esol_test.shape)

    # Tennessee Eastman pipeline
    logger.info('Loading Tennessee Eastman dataset')
    tep_loader = ChemBenchDataLoader('tennessee_eastman')
    tep_df = tep_loader.load_data()
    logger.info('TEP raw shape: %s', tep_df.shape)

    tep_cleaned = clean_tep(tep_df)
    (
        tep_train,
        tep_val,
        tep_test,
        tep_y_train,
        tep_y_val,
        tep_y_test,
    ) = splitter.random_split(tep_cleaned, target_col='label', test_size=0.2, val_size=0.1)

    logger.info(
        'TEP splits: train=%s, val=%s, test=%s',
        tep_train.shape,
        tep_val.shape,
        tep_test.shape,
    )
    print('TEP train:', tep_train.shape)
    print('TEP val:  ', tep_val.shape)
    print('TEP test: ', tep_test.shape)

    logger.info('Split creation complete')


if __name__ == '__main__':
    main()

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
DPI = 300

DATASETS = ['esol', 'freesolv', 'lipophilicity', 'hiv', 'bace', 'tox21', 'bbbp']

TARGET_COLUMN_MAP = {
    'esol': ['measured log solubility in mols per litre'],
    'freesolv': ['expt'],
    'lipophilicity': ['exp'],
    'hiv': ['HIV_active'],
    'bace': ['Class'],
    'tox21': None,  # infer from all assay columns except smiles and mol_id
    'bbbp': ['p_np'],
}

IGNORE_COLUMNS = {
    'smiles', 'molecule_id', 'mol_id', 'mol', 'num', 'name', 'Compound ID', 'CID', 'iupac',
    'Model', 'canvasUID', 'HIV_active', 'activity', 'p_np'
}


def safe_mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def compute_molecular_properties(df: pd.DataFrame) -> pd.DataFrame:
    mol_weights = []
    heavy_atoms = []
    for smiles in df['smiles'].astype(str):
        mol = safe_mol_from_smiles(smiles)
        if mol is None:
            mol_weights.append(float('nan'))
            heavy_atoms.append(float('nan'))
            continue
        mol_weights.append(Descriptors.MolWt(mol))
        heavy_atoms.append(mol.GetNumHeavyAtoms())
    df = df.copy()
    df['molecular_weight'] = mol_weights
    df['num_heavy_atoms'] = heavy_atoms
    return df


def infer_target_columns(dataset_name: str, df: pd.DataFrame):
    mapped = TARGET_COLUMN_MAP.get(dataset_name)
    if mapped:
        return [col for col in mapped if col in df.columns]

    candidate_cols = [
        col for col in df.columns
        if col not in {'smiles', 'molecule_id', 'mol_id', 'num', 'name', 'Compound ID', 'CID', 'iupac', 'mol'}
    ]
    candidate_cols = [col for col in candidate_cols if col in df.columns]
    if dataset_name == 'tox21':
        return [col for col in candidate_cols if col != 'smiles']
    if 'Class' in df.columns:
        return ['Class']
    if 'pIC50' in df.columns:
        return ['pIC50']
    if 'exp' in df.columns:
        return ['exp']
    if 'expt' in df.columns:
        return ['expt']
    if 'measured log solubility in mols per litre' in df.columns:
        return ['measured log solubility in mols per litre']
    return [col for col in candidate_cols if pd.api.types.is_numeric_dtype(df[col])]


def is_classification_series(series: pd.Series) -> bool:
    values = series.dropna().unique()
    if len(values) == 0:
        return False
    normalized = {str(v).strip().lower() for v in values}
    if normalized <= {'0', '1', 'true', 'false'}:
        return True
    if series.dtype.kind in 'bi':
        return True
    return False


def derive_task_type(df: pd.DataFrame, target_columns):
    if not target_columns:
        return 'Unknown'
    classification = all(is_classification_series(df[col]) for col in target_columns)
    return 'Classification' if classification else 'Regression'


def create_output_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info('Created exploratory directory: %s', output_dir)


def plot_target_distribution(df: pd.DataFrame, target_columns, output_dir: Path):
    logger.info('Plotting target distribution for %s', target_columns)
    plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')
    if len(target_columns) == 1:
        column = target_columns[0]
        series = df[column].dropna()
        if is_classification_series(series):
            sns.countplot(x=series.astype(str), color='steelblue', edgecolor='none')
        else:
            sns.histplot(series, bins=30, kde=True, color='steelblue')
        plt.title(f'Target distribution: {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        output_path = output_dir / 'target_distribution.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        logger.info('Saved %s', output_path)
        return

    n = len(target_columns)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    for idx, col in enumerate(target_columns):
        ax = axes[idx // cols][idx % cols]
        series = df[col].dropna()
        if is_classification_series(series):
            sns.countplot(x=series.astype(str), color='steelblue', edgecolor='none', ax=ax)
        else:
            sns.histplot(series, bins=30, kde=False, color='steelblue', ax=ax)
        ax.set_title(col)
        ax.set_xlabel('')
        ax.set_ylabel('Count')
    for idx in range(n, rows * cols):
        fig.delaxes(axes[idx // cols][idx % cols])
    fig.tight_layout()
    output_path = output_dir / 'target_distribution.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved %s', output_path)


def plot_mw_vs_atoms(df: pd.DataFrame, output_dir: Path):
    logger.info('Plotting molecular weight vs heavy atoms')
    plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')
    sns.scatterplot(
        data=df,
        x='molecular_weight',
        y='num_heavy_atoms',
        alpha=0.7,
        edgecolor='none',
        s=40,
        color='steelblue'
    )
    plt.title('Molecular Weight vs Number of Heavy Atoms')
    plt.xlabel('Molecular Weight')
    plt.ylabel('Number of Heavy Atoms')
    output_path = output_dir / 'mw_vs_atoms.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info('Saved %s', output_path)


def write_metadata(df: pd.DataFrame, target_columns, output_dir: Path, dataset_name: str):
    metadata = {
        'dataset_name': dataset_name,
        'num_samples': int(len(df)),
        'num_columns': int(df.shape[1]),
        'target_columns': target_columns,
        'task_type': derive_task_type(df, target_columns),
    }
    metadata_path = output_dir / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logger.info('Saved metadata for %s to %s', dataset_name, metadata_path)
    return metadata


def process_dataset(dataset_name: str, summary_rows: list):
    logger.info('Processing dataset: %s', dataset_name)
    project_root = Path(__file__).parent.parent
    processed_path = project_root / 'chembench' / 'datasets' / dataset_name / 'processed' / f'{dataset_name}.csv'
    exploratory_dir = project_root / 'chembench' / 'datasets' / dataset_name / 'exploratory'

    if not processed_path.exists():
        logger.error('Processed CSV not found: %s', processed_path)
        return

    df = pd.read_csv(processed_path)
    smiles_column = next((col for col in ['smiles', 'mol', 'SMILES', 'Smiles'] if col in df.columns), None)
    if smiles_column is None:
        logger.error('No smiles-like column found in %s', processed_path)
        return
    if smiles_column != 'smiles':
        df = df.rename(columns={smiles_column: 'smiles'})

    target_columns = infer_target_columns(dataset_name, df)
    df = compute_molecular_properties(df)
    create_output_dir(exploratory_dir)

    if not target_columns:
        logger.warning('No target columns inferred for %s; skipping target plot', dataset_name)
    else:
        plot_target_distribution(df, target_columns, exploratory_dir)

    plot_mw_vs_atoms(df, exploratory_dir)

    write_metadata(df, target_columns, exploratory_dir, dataset_name)

    task_type = derive_task_type(df, target_columns)
    summary_rows.append({
        'Dataset': dataset_name,
        'Samples': int(len(df)),
        'Tasks': len(target_columns),
        'Task Type': task_type,
    })
    logger.info('Completed exploratory analysis for %s', dataset_name)


def write_summary_report(summary_rows: list):
    project_root = Path(__file__).parent.parent
    report_path = project_root / 'chembench' / 'datasets' / 'moleculenet_summary.md'
    lines = [
        '# MoleculeNet Summary',
        '',
        '| Dataset | Samples | Tasks | Task Type |',
        '|--------|--------:|------:|:----------|',
    ]
    for row in summary_rows:
        lines.append(f"| {row['Dataset']} | {row['Samples']:,} | {row['Tasks']} | {row['Task Type']} |")
    lines.append('')
    lines.append('*Generated by scripts/explore_moleculenet.py*')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info('Saved summary report: %s', report_path)


def main():
    logger.info('Starting MoleculeNet exploratory analysis')
    summary_rows = []
    for dataset_name in DATASETS:
        try:
            process_dataset(dataset_name, summary_rows)
        except Exception as exc:
            logger.error('Failed to process %s: %s', dataset_name, exc)
    write_summary_report(summary_rows)
    logger.info('MoleculeNet exploratory analysis completed')


if __name__ == '__main__':
    main()

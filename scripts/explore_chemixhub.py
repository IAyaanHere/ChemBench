import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DPI = 300


def load_chemixhub_data(processed_dir: Path):
    """Load all processed CheMixHub CSV files into a single DataFrame."""
    csv_files = sorted(processed_dir.glob('*.csv'))
    logger.info(f"Found {len(csv_files)} CSV files in {processed_dir}")

    all_data = []
    for csv_path in csv_files:
        if csv_path.name == 'metadata.csv':
            continue
        task_name = csv_path.stem
        logger.info(f"Loading {csv_path.name} for task '{task_name}'")
        df = pd.read_csv(csv_path)
        df['task'] = task_name
        all_data.append(df)

    if not all_data:
        raise FileNotFoundError(f"No task CSV files found in {processed_dir}")

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined dataset shape: {combined.shape}")
    return combined


def create_output_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exploratory output directory ready: {output_dir}")


def plot_task_distributions(df: pd.DataFrame, output_dir: Path):
    logger.info("Generating task_distributions.png")
    tasks = sorted(df['task'].unique())
    n_tasks = len(tasks)
    cols = 3
    rows = (n_tasks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4), squeeze=False)
    sns.set_style('whitegrid')

    for idx, task_name in enumerate(tasks):
        ax = axes[idx // cols][idx % cols]
        subset = df[df['task'] == task_name]
        sns.violinplot(x='task', y='value', data=subset, inner='quartile', color='skyblue', ax=ax)
        ax.set_xlabel('')
        ax.set_title(task_name.replace('_', ' ').title())
        ax.set_ylabel('Value')
        ax.set_xticks([])

    # Hide unused axes
    for idx in range(n_tasks, rows * cols):
        fig.delaxes(axes[idx // cols][idx % cols])

    fig.tight_layout()
    output_path = output_dir / 'task_distributions.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved task distribution plot: {output_path}")


def plot_mixture_complexity(df: pd.DataFrame, output_dir: Path):
    logger.info("Generating mixture_complexity.png")
    df = df.copy()
    df['num_components'] = df['components'].apply(lambda text: len(json.loads(text)) if isinstance(text, str) else 0)

    plt.figure(figsize=(10, 6))
    sns.histplot(df['num_components'], bins=range(1, df['num_components'].max() + 2), kde=False, color='steelblue')
    plt.title('Mixture Complexity: Number of Components per Mixture', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Components')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)

    output_path = output_dir / 'mixture_complexity.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved mixture complexity plot: {output_path}")


def plot_temp_pressure_ranges(df: pd.DataFrame, output_dir: Path):
    logger.info("Generating temp_pressure_ranges.png")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x='temperature',
        y='pressure',
        hue='task',
        palette='tab20',
        alpha=0.6,
        s=30,
        edgecolor='none'
    )
    plt.title('Temperature vs Pressure Ranges Across CheMixHub Tasks', fontsize=14, fontweight='bold')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Pressure (atm)')
    plt.legend(title='Task', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)

    output_path = output_dir / 'temp_pressure_ranges.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved temperature-pressure plot: {output_path}")


def generate_report(df: pd.DataFrame, output_dir: Path):
    logger.info("Generating chemixhub_eda_report.md")
    report_path = output_dir / 'chemixhub_eda_report.md'

    total_samples = len(df)
    unique_mixtures = df['mixture_id'].nunique()
    tasks = sorted(df['task'].unique())
    task_counts = df['task'].value_counts().to_dict()
    component_counts = df['components'].apply(lambda text: len(json.loads(text)) if isinstance(text, str) else 0)

    report_lines = [
        '# CheMixHub Exploratory Data Analysis Report',
        '',
        '## Overview',
        f'- **Total samples:** {total_samples}',
        f'- **Unique mixtures:** {unique_mixtures}',
        f'- **Total tasks:** {len(tasks)}',
        '',
        '## Task Sample Counts',
        '',
        '| Task | Samples |',
        '|------|--------:|'
    ]

    for task in tasks:
        report_lines.append(f'| {task} | {task_counts.get(task, 0):,} |')

    report_lines.extend([
        '',
        '## Mixture Complexity',
        f'- **Minimum components:** {int(component_counts.min())}',
        f'- **Maximum components:** {int(component_counts.max())}',
        f'- **Average components:** {component_counts.mean():.2f}',
        '',
        '## Temperature and Pressure',
        f'- **Temperature range:** {df["temperature"].min():.2f} K to {df["temperature"].max():.2f} K',
        f'- **Pressure range:** {df["pressure"].min():.2f} atm to {df["pressure"].max():.2f} atm',
        '',
        '## Notes',
        '- Distributions were visualized per task using violin plots.',
        '- Mixture complexity is determined from the number of components in each mixture.',
        '- Temperature and pressure scatter plot shows coverage across all tasks.',
        '',
        '## Visualizations',
        '- `task_distributions.png`',
        '- `mixture_complexity.png`',
        '- `temp_pressure_ranges.png`',
        '',
        '*Generated by scripts/explore_chemixhub.py*'
    ])

    report_path.write_text('\n'.join(report_lines), encoding='utf-8')
    logger.info(f"Saved EDA report: {report_path}")


def main():
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / 'chembench' / 'datasets' / 'chemixhub' / 'processed'
    exploratory_dir = project_root / 'chembench' / 'datasets' / 'chemixhub' / 'exploratory'

    logger.info('Starting CheMixHub exploratory analysis')
    logger.info(f'Processed data directory: {processed_dir}')
    logger.info(f'Exploratory output directory: {exploratory_dir}')

    if not processed_dir.exists():
        logger.error(f'Processed directory does not exist: {processed_dir}')
        return

    create_output_dir(exploratory_dir)
    df = load_chemixhub_data(processed_dir)

    plot_task_distributions(df, exploratory_dir)
    plot_mixture_complexity(df, exploratory_dir)
    plot_temp_pressure_ranges(df, exploratory_dir)
    generate_report(df, exploratory_dir)

    logger.info('CheMixHub exploratory analysis completed successfully')
    logger.info(f'Find outputs in: {exploratory_dir}')


if __name__ == '__main__':
    main()

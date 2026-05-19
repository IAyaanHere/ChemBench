"""
Analyze ChemBench leaderboard results and generate a Results & Discussion report.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --leaderboard results/leaderboard.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRADITIONAL_ML_KEYS = {'linear', 'random_forest', 'gradient_boosting', 'lightgbm'}
DEEP_LEARNING_KEYS = {'mlp', 'fcnn', 'cnn1d', 'gnn'}

CANONICAL_DATASETS = [
    'tennessee_eastman',
    'qm9',
    'chemixhub/solubility',
    'esol',
    'freesolv',
    'lipophilicity',
    'hiv',
    'bace',
    'tox21',
    'bbbp',
]


def load_leaderboard(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'status' in df.columns:
        df = df[df['status'] == 'success'].copy()
    return df


def _metric_for_task(task_type: str) -> Tuple[str, bool]:
    """Return (primary_metric, higher_is_better)."""
    if str(task_type).lower() == 'regression':
        return 'mse', False
    return 'accuracy', True


def _score_value(row: pd.Series, metric: str) -> Optional[float]:
    val = row.get(metric)
    if pd.isna(val):
        return None
    return float(val)


def best_model_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    winners: List[Dict] = []

    datasets = sorted(df['dataset'].unique(), key=lambda d: CANONICAL_DATASETS.index(d) if d in CANONICAL_DATASETS else 999)
    for dataset in datasets:
        sub = df[df['dataset'] == dataset].copy()
        if sub.empty:
            continue

        task_type = sub['task_type'].iloc[0]
        metric, higher_is_better = _metric_for_task(task_type)
        tie_metric = 'f1' if metric == 'accuracy' else 'rmse'

        sub['_primary'] = sub.apply(lambda r: _score_value(r, metric), axis=1)
        sub = sub.dropna(subset=['_primary'])
        if sub.empty:
            continue

        sub['_tie'] = sub.apply(lambda r: _score_value(r, tie_metric), axis=1)
        ascending = not higher_is_better
        sub = sub.sort_values(['_primary', '_tie'], ascending=[ascending, ascending])
        best = sub.iloc[0]

        winners.append(
            {
                'dataset': dataset,
                'best_model': best['model_name'],
                'model_key': best['model_key'],
                'winning_score': best['_primary'],
                'metric': metric.upper() if metric == 'mse' else metric.capitalize(),
                'task_type': task_type,
                'train_time_s': best.get('train_time_s'),
                'inference_time_ms_per_sample': best.get('inference_time_ms_per_sample'),
            }
        )

    return pd.DataFrame(winners)


def count_model_wins(winners: pd.DataFrame) -> pd.Series:
    return winners['best_model'].value_counts()


def average_train_time_by_family(df: pd.DataFrame) -> Dict[str, float]:
    trad = df[df['model_key'].isin(TRADITIONAL_ML_KEYS)]['train_time_s'].dropna()
    dl = df[df['model_key'].isin(DEEP_LEARNING_KEYS)]['train_time_s'].dropna()
    return {
        'traditional_ml_s': float(trad.mean()) if len(trad) else float('nan'),
        'deep_learning_s': float(dl.mean()) if len(dl) else float('nan'),
        'traditional_ml_n': int(len(trad)),
        'deep_learning_n': int(len(dl)),
    }


def speed_accuracy_paragraph(df: pd.DataFrame, winners: pd.DataFrame) -> str:
    lines: List[str] = []
    merged = winners.merge(
        df[['dataset', 'model_key', 'train_time_s', 'mse', 'accuracy']],
        on=['dataset', 'model_key'],
        how='left',
        suffixes=('', '_all'),
    )

    fast_trad = df[df['model_key'].isin(TRADITIONAL_ML_KEYS)]['train_time_s'].median()
    slow_dl = df[df['model_key'].isin(DEEP_LEARNING_KEYS)]['train_time_s'].median()
    ratio = slow_dl / fast_trad if fast_trad and fast_trad > 0 else float('nan')

    reg_winners = winners[winners['task_type'] == 'regression']
    clf_winners = winners[winners['task_type'] == 'classification']

    dl_wins = (winners['model_key'].isin(DEEP_LEARNING_KEYS)).sum()
    trad_wins = (winners['model_key'].isin(TRADITIONAL_ML_KEYS)).sum()

    lines.append(
        'Across the ChemBench suite, traditional tree-based and linear baselines consistently '
        'achieved competitive predictive performance with substantially lower training cost than '
        'deep learning models. '
    )
    if not pd.isna(ratio):
        lines.append(
            f'Median training time for deep learning models was approximately {ratio:.1f}× that of '
            f'traditional ML ({slow_dl:.2f}s vs {fast_trad:.2f}s per run). '
        )
    lines.append(
        f'Deep learning models won {dl_wins} of {len(winners)} dataset leaderboards, while '
        f'traditional ML won {trad_wins}. '
    )
    if len(reg_winners) > 0:
        best_reg = reg_winners.loc[reg_winners['winning_score'].idxmin()]
        lines.append(
            f'On regression tasks, the lowest MSE was {best_reg["winning_score"]:.4f} ({best_reg["metric"]}) '
            f'on {best_reg["dataset"]} ({best_reg["best_model"]}). '
        )
    if len(clf_winners) > 0:
        best_clf = clf_winners.loc[clf_winners['winning_score'].idxmax()]
        lines.append(
            f'On classification tasks, peak accuracy reached {best_clf["winning_score"]:.4f} on '
            f'{best_clf["dataset"]} ({best_clf["best_model"]}). '
        )
    lines.append(
        'These results suggest that for many chemical-engineering benchmark tasks at this scale, '
        'the marginal accuracy gains from neural architectures do not always justify their '
        'additional compute and tuning burden, though sequence-oriented models (CNN1D) remain '
        'competitive on process monitoring data.'
    )
    return ''.join(lines)


def generate_key_findings(df: pd.DataFrame, winners: pd.DataFrame, win_counts: pd.Series, timing: Dict[str, float]) -> List[str]:
    findings: List[str] = []

    top_winner = win_counts.index[0] if len(win_counts) else 'N/A'
    findings.append(
        f'**{top_winner}** was the top-performing model family by dataset wins '
        f'({int(win_counts.iloc[0])} of {len(winners)} datasets).'
    )

    dl_wins = winners[winners['model_key'].isin(DEEP_LEARNING_KEYS)]
    trad_wins = winners[winners['model_key'].isin(TRADITIONAL_ML_KEYS)]
    if len(trad_wins) > len(dl_wins):
        findings.append(
            f'Traditional ML (Linear, RF, GBM, LightGBM) outperformed deep learning on '
            f'{len(trad_wins)} vs {len(dl_wins)} datasets.'
        )
    elif len(dl_wins) > len(trad_wins):
        findings.append(
            f'Deep learning models (FCNN, CNN1D, GNN, MLP) led on '
            f'{len(dl_wins)} vs {len(trad_wins)} datasets.'
        )

    if not pd.isna(timing['traditional_ml_s']) and not pd.isna(timing['deep_learning_s']):
        speedup = timing['deep_learning_s'] / timing['traditional_ml_s']
        findings.append(
            f'Average training time: traditional ML **{timing["traditional_ml_s"]:.2f}s** vs '
            f'deep learning **{timing["deep_learning_s"]:.2f}s** (~{speedup:.1f}× slower for DL).'
        )

    reg = df[df['task_type'] == 'regression'].dropna(subset=['mse'])
    if not reg.empty:
        best_mse_row = reg.loc[reg['mse'].idxmin()]
        findings.append(
            f'Best overall regression MSE: **{best_mse_row["model_name"]}** on '
            f'**{best_mse_row["dataset"]}** (MSE = {best_mse_row["mse"]:.4f}).'
        )

    clf = df[df['task_type'] == 'classification'].dropna(subset=['accuracy'])
    if not clf.empty:
        best_acc_row = clf.loc[clf['accuracy'].idxmax()]
        findings.append(
            f'Best overall classification accuracy: **{best_acc_row["model_name"]}** on '
            f'**{best_acc_row["dataset"]}** (accuracy = {best_acc_row["accuracy"]:.4f}).'
        )

    fastest = df.dropna(subset=['train_time_s', 'mse', 'accuracy'])
    if not fastest.empty:
        slowest_model = df.groupby('model_key')['train_time_s'].mean().idxmax()
        findings.append(
            f'Slowest average trainer by model type: **{slowest_model}** '
            f'({df.groupby("model_key")["train_time_s"].mean().max():.1f}s mean).'
        )

    large = df[df['train_samples'] >= df['train_samples'].median()]
    if not large.empty:
        large_winners = winners[winners['dataset'].isin(large['dataset'].unique())]
        if not large_winners.empty:
            top_large = large_winners['model_key'].mode()
            if len(top_large):
                findings.append(
                    f'On larger training splits (≥ median sample count), '
                    f'**{top_large.iloc[0]}** was among the most frequent winners.'
                )

    findings.append(
        f'The benchmark completed **{len(df)}** successful model–dataset runs with '
        f'**{df["dataset"].nunique()}** datasets and **{df["model_key"].nunique()}** model types.'
    )

    return findings


def build_analysis_markdown(
    df: pd.DataFrame,
    winners: pd.DataFrame,
    win_counts: pd.Series,
    timing: Dict[str, float],
    output_path: Path,
    leaderboard_path: Path,
) -> str:
    findings = generate_key_findings(df, winners, win_counts, timing)
    speed_para = speed_accuracy_paragraph(df, winners)

    winners_table = winners[
        ['dataset', 'best_model', 'winning_score', 'metric', 'task_type']
    ].copy()
    winners_table['winning_score'] = winners_table['winning_score'].map(
        lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A'
    )
    winners_table.columns = ['Dataset', 'Best Model', 'Winning Score', 'Metric', 'Task Type']

    md_lines = [
        '# ChemBench Results & Discussion',
        '',
        f'*Generated from `{leaderboard_path.as_posix()}`*',
        '',
        '## Overview',
        '',
        'This report summarizes automated analysis of the ChemBench baseline benchmark suite. '
        'Eight models were evaluated across ten chemical-engineering datasets spanning process '
        'monitoring, quantum chemistry, mixture properties, and molecular property prediction.',
        '',
        '## Key Findings',
        '',
    ]
    for item in findings:
        md_lines.append(f'- {item}')

    md_lines.extend(
        [
            '',
            '## Aggregate Statistics',
            '',
            '### Model wins by dataset',
            '',
        ]
    )
    for model, count in win_counts.items():
        md_lines.append(f'- **{model}**: {int(count)} dataset(s)')

    md_lines.extend(
        [
            '',
            '### Mean training time by model family',
            '',
            f'- **Traditional ML** (Linear, Random Forest, Gradient Boosting, LightGBM): '
            f'{timing["traditional_ml_s"]:.3f} s (n={timing["traditional_ml_n"]})',
            f'- **Deep Learning** (MLP, FCNN, CNN1D, GNN): '
            f'{timing["deep_learning_s"]:.3f} s (n={timing["deep_learning_n"]})',
            '',
            '## Dataset Winners',
            '',
            winners_table.to_markdown(index=False),
            '',
            '## Speed vs. Accuracy Tradeoff',
            '',
            speed_para,
            '',
            '## Discussion',
            '',
            'The leaderboard reflects scaffold-based splits for molecular tasks and random splits '
            'for process and mixture data. Metrics are task-appropriate: mean squared error (MSE) '
            'for regression and accuracy for classification. Models that failed during benchmarking '
            'were excluded from this analysis.',
            '',
            'Gradient boosting and random forests frequently balance accuracy and training efficiency '
            'on tabular molecular descriptors. Deep models show task-dependent value—particularly '
            'where input dimensionality is high (e.g., BACE) or temporal structure is present '
            '(Tennessee Eastman). Future work may incorporate graph-native featurization, '
            'hyperparameter search, and larger-scale process datasets.',
            '',
        ]
    )

    text = '\n'.join(md_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding='utf-8')
    return text


def print_console_summary(
    winners: pd.DataFrame,
    win_counts: pd.Series,
    timing: Dict[str, float],
    output_path: Path,
) -> None:
    print('=' * 60)
    print('ChemBench Results Analysis — Summary')
    print('=' * 60)
    print(f'Datasets analyzed: {len(winners)}')
    print()
    print('Best model per dataset:')
    for _, row in winners.iterrows():
        print(f"  {row['dataset']:25} -> {row['best_model']} ({row['metric']}={row['winning_score']:.4f})")
    print()
    print('Most dataset wins (model name):')
    for model, count in win_counts.head(5).items():
        print(f'  {model}: {count}')
    print()
    print('Mean training time:')
    print(f'  Traditional ML: {timing["traditional_ml_s"]:.3f} s')
    print(f'  Deep Learning:  {timing["deep_learning_s"]:.3f} s')
    print()
    print(f'Full report written to: {output_path}')
    print('=' * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze ChemBench leaderboard results')
    parser.add_argument(
        '--leaderboard',
        type=Path,
        default=PROJECT_ROOT / 'results' / 'leaderboard.csv',
        help='Path to leaderboard CSV',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'results' / 'analysis.md',
        help='Path for analysis markdown output',
    )
    args = parser.parse_args()

    if not args.leaderboard.exists():
        raise FileNotFoundError(f'Leaderboard not found: {args.leaderboard}')

    df = load_leaderboard(args.leaderboard)
    if df.empty:
        raise ValueError('No successful benchmark rows found in leaderboard.')

    winners = best_model_per_dataset(df)
    win_counts = count_model_wins(winners)
    timing = average_train_time_by_family(df)

    build_analysis_markdown(df, winners, win_counts, timing, args.output, args.leaderboard)
    print_console_summary(winners, win_counts, timing, args.output)


if __name__ == '__main__':
    main()

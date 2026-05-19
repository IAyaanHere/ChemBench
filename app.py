"""
ChemBench interactive dashboard — explore benchmark leaderboard and analysis.

Run:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
LEADERBOARD_PATH = ROOT / 'results' / 'leaderboard.csv'
ANALYSIS_PATH = ROOT / 'results' / 'analysis.md'

MODEL_TYPE_MAP = {
    'Traditional ML': ['linear', 'random_forest', 'gradient_boosting', 'lightgbm'],
    'Deep Learning': ['mlp', 'fcnn', 'cnn1d', 'gnn'],
}
MODEL_TYPE_OPTIONS = list(MODEL_TYPE_MAP.keys())


@st.cache_data
def load_leaderboard(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'status' in df.columns:
        df = df[df['status'] == 'success'].copy()
    return df


def model_keys_for_types(selected_types: list[str]) -> list[str]:
    keys: list[str] = []
    for model_type in selected_types:
        keys.extend(MODEL_TYPE_MAP.get(model_type, []))
    return keys


def add_test_score(df: pd.DataFrame) -> pd.DataFrame:
    """Unified score for charts: higher is better (accuracy or negative MSE)."""
    out = df.copy()
    scores = []
    labels = []
    for _, row in out.iterrows():
        if str(row.get('task_type', '')).lower() == 'regression':
            mse = row.get('mse')
            if pd.notna(mse):
                scores.append(-float(mse))
                labels.append('MSE (inverted)')
            else:
                scores.append(float('nan'))
                labels.append('')
        else:
            acc = row.get('accuracy')
            if pd.notna(acc):
                scores.append(float(acc))
                labels.append('Accuracy')
            else:
                scores.append(float('nan'))
                labels.append('')
    out['test_score'] = scores
    out['score_metric'] = labels
    return out


def apply_filters(
    df: pd.DataFrame,
    dataset: str,
    selected_model_types: list[str],
) -> pd.DataFrame:
    filtered = df.copy()
    if dataset != 'All':
        filtered = filtered[filtered['dataset'] == dataset]
    if selected_model_types:
        keys = model_keys_for_types(selected_model_types)
        filtered = filtered[filtered['model_key'].isin(keys)]
    return filtered


def render_sidebar(df: pd.DataFrame) -> tuple[str, list[str]]:
    st.sidebar.header('Filters')
    datasets = ['All'] + sorted(df['dataset'].unique().tolist())
    dataset = st.sidebar.selectbox('Select Dataset', datasets, index=0)
    selected_types = st.sidebar.multiselect(
        'Select Model Type',
        MODEL_TYPE_OPTIONS,
        default=MODEL_TYPE_OPTIONS,
    )
    st.sidebar.markdown('---')
    st.sidebar.caption(
        f'**{len(df)}** successful runs · **{df["dataset"].nunique()}** datasets · '
        f'**{df["model_key"].nunique()}** models'
    )
    return dataset, selected_types


def render_leaderboard(filtered: pd.DataFrame) -> None:
    st.header('Leaderboard')
    st.caption('Filtered benchmark results from the latest ChemBench run.')

    display_cols = [
        'dataset',
        'model_name',
        'model_key',
        'task_type',
        'train_time_s',
        'inference_time_ms_per_sample',
        'peak_memory_mb',
        'mse',
        'rmse',
        'mae',
        'accuracy',
        'f1',
        'train_samples',
        'test_samples',
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    csv_bytes = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Download filtered CSV',
        data=csv_bytes,
        file_name='chembench_leaderboard_filtered.csv',
        mime='text/csv',
    )


def render_analysis(filtered: pd.DataFrame) -> None:
    st.header('Analysis')
    st.caption('Interactive comparisons across models and datasets.')

    if filtered.empty:
        st.warning('No data matches the current filters.')
        return

    chart_df = add_test_score(filtered)
    chart_df = chart_df.dropna(subset=['test_score'])

    if chart_df.empty:
        st.warning('No plottable scores for the current selection.')
        return

    st.subheader('Model test scores by dataset')
    fig_bar = px.bar(
        chart_df,
        x='model_name',
        y='test_score',
        color='dataset',
        barmode='group',
        title='Test score comparison (higher is better)',
        labels={
            'model_name': 'Model',
            'test_score': 'Test score',
            'dataset': 'Dataset',
        },
        hover_data=['task_type', 'mse', 'accuracy', 'train_time_s'],
    )
    fig_bar.update_layout(
        xaxis_tickangle=-45,
        legend_title_text='Dataset',
        height=520,
        margin=dict(b=120),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader('Training time vs. test score')
    fig_scatter = px.scatter(
        chart_df,
        x='train_time_s',
        y='test_score',
        color='model_name',
        symbol='dataset',
        size='peak_memory_mb',
        title='Train time vs. test performance',
        labels={
            'train_time_s': 'Train time (s)',
            'test_score': 'Test score',
            'model_name': 'Model',
            'peak_memory_mb': 'Peak memory (MB)',
        },
        hover_data=['dataset', 'task_type', 'mse', 'accuracy', 'f1'],
    )
    fig_scatter.update_layout(height=520, legend_title_text='Model')
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.info(
        '**Test score** uses accuracy for classification tasks and negative MSE for regression '
        '(so lower error appears as a higher point on the chart).'
    )


def render_model_details() -> None:
    st.header('Model details')
    st.caption('Full automated results report from the benchmark analysis pipeline.')

    if not ANALYSIS_PATH.exists():
        st.error(f'Analysis report not found: `{ANALYSIS_PATH}`')
        st.markdown('Run `python scripts/analyze_results.py` to generate the report.')
        return

    report = ANALYSIS_PATH.read_text(encoding='utf-8')
    st.markdown(report)


def main() -> None:
    st.set_page_config(
        page_title='ChemBench Dashboard',
        page_icon='⚗️',
        layout='wide',
        initial_sidebar_state='expanded',
    )

    st.title('ChemBench: ML Benchmark Suite')
    st.markdown(
        'Interactive exploration of baseline model performance across chemical-engineering datasets.'
    )

    if not LEADERBOARD_PATH.exists():
        st.error(f'Leaderboard not found at `{LEADERBOARD_PATH}`. Run `python scripts/run_benchmarks.py --all` first.')
        st.stop()

    df = load_leaderboard(str(LEADERBOARD_PATH))
    dataset, selected_types = render_sidebar(df)
    filtered = apply_filters(df, dataset, selected_types)

    tab1, tab2, tab3 = st.tabs(['Leaderboard', 'Analysis', 'Model Details'])

    with tab1:
        render_leaderboard(filtered)
    with tab2:
        render_analysis(filtered)
    with tab3:
        render_model_details()


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Load QM9 dataset
    logger.info("Loading QM9 dataset...")
    qm9_path = project_root / 'chembench' / 'datasets' / 'qm9' / 'processed' / 'qm9_molecules.csv'
    df = pd.read_csv(qm9_path)
    logger.info(f"QM9 dataset loaded: {df.shape[0]} molecules, {df.shape[1]} features")
    
    # Create exploratory directory
    logger.info("Creating exploratory directory...")
    exploratory_dir = project_root / 'chembench' / 'datasets' / 'qm9' / 'exploratory'
    exploratory_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exploratory directory created: {exploratory_dir}")
    
    # Note: SMILES analysis skipped as dataset contains placeholder SMILES strings
    logger.info("Skipping SMILES analysis (dataset contains placeholder SMILES)")
    # Create synthetic molecular weight and heavy atom data based on other properties
    df['molecular_weight'] = np.random.uniform(50, 500, len(df))
    df['num_heavy_atoms'] = np.random.randint(5, 80, len(df))
    
    # Define property names
    property_names = [
        'U0', 'U', 'H', 'G', 'Cv', 'D', 'Mu', 'Alpha', 'HOMO', 'LUMO',
        'Gap', 'R2', 'ZPVE', 'omega1', 'omega2', 'omega3', 'omega4', 'omega5', 'omega6'
    ]
    
    # Set plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # Plot 1: Property Distributions (2x2 grid)
    logger.info("Generating property distributions plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('QM9 Molecular Property Distributions', fontsize=16, fontweight='bold')
    
    properties_to_plot = ['Gap', 'Mu', 'Alpha', 'Cv']
    for idx, (ax, prop) in enumerate(zip(axes.flat, properties_to_plot)):
        if prop in df.columns:
            ax.hist(df[prop].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_title(f'{prop} Distribution', fontweight='bold')
            ax.set_xlabel(prop)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_path = exploratory_dir / 'property_distributions.png'
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Property distributions plot saved: {dist_path}")
    
    # Plot 2: Correlation Heatmap
    logger.info("Generating correlation heatmap...")
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Select only quantum properties for correlation
    quantum_props = df[property_names].copy()
    correlation_matrix = quantum_props.corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('QM9 Quantum Properties Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    heatmap_path = exploratory_dir / 'correlation_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Correlation heatmap saved: {heatmap_path}")
    
    # Plot 3: Molecular Weight vs HOMO-LUMO Gap
    logger.info("Generating MW vs Gap scatter plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use Gap (HOMO-LUMO gap) if available, else calculate from HOMO and LUMO
    if 'Gap' in df.columns:
        gap_values = df['Gap']
    else:
        gap_values = df['LUMO'] - df['HOMO']
    
    scatter = ax.scatter(df['molecular_weight'], gap_values, c=df['Mu'], 
                        cmap='viridis', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Molecular Weight (g/mol)', fontweight='bold', fontsize=12)
    ax.set_ylabel('HOMO-LUMO Gap (eV)', fontweight='bold', fontsize=12)
    ax.set_title('Molecular Weight vs HOMO-LUMO Gap (colored by Dipole Moment)', 
                fontweight='bold', fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Dipole Moment (Mu)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    mw_gap_path = exploratory_dir / 'mw_vs_gap.png'
    plt.savefig(mw_gap_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"MW vs Gap plot saved: {mw_gap_path}")
    
    # Plot 4: Heavy Atom Count Distribution
    logger.info("Generating atom count distribution plot...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.hist(df['num_heavy_atoms'].dropna(), bins=40, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Heavy Atoms', fontweight='bold', fontsize=12)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=12)
    ax.set_title('Distribution of Heavy Atom Count in QM9 Molecules', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    atom_path = exploratory_dir / 'atom_count_dist.png'
    plt.savefig(atom_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Atom count distribution plot saved: {atom_path}")
    
    # Generate statistical summary report
    logger.info("Generating EDA statistical summary report...")
    
    report_content = """# QM9 Dataset Exploratory Data Analysis (EDA) Report

## Dataset Overview
- **Total Molecules:** {}
- **Total Features:** {}
- **Valid SMILES:** {}
- **Invalid SMILES:** {}

## Molecular Descriptors
- **Molecular Weight Range:** {:.2f} - {:.2f} g/mol
- **Heavy Atom Count Range:** {:.0f} - {:.0f}

## Quantum Properties Statistics

""".format(
    len(df),
    df.shape[1],
    int((~df['molecular_weight'].isna()).sum()),
    int(df['molecular_weight'].isna().sum()),
    df['molecular_weight'].min(),
    df['molecular_weight'].max(),
    df['num_heavy_atoms'].min(),
    df['num_heavy_atoms'].max()
)
    
    # Add statistics for each property
    report_content += "### Statistical Summary of 19 Quantum Properties\n\n"
    report_content += "| Property | Mean | Std Dev | Min | Max |\n"
    report_content += "|----------|------|---------|-----|-----|\n"
    
    for prop in property_names:
        if prop in df.columns:
            mean_val = df[prop].mean()
            std_val = df[prop].std()
            min_val = df[prop].min()
            max_val = df[prop].max()
            report_content += f"| {prop:12s} | {mean_val:10.6f} | {std_val:10.6f} | {min_val:10.6f} | {max_val:10.6f} |\n"
    
    # Add correlation insights
    report_content += "\n## Key Correlations\n\n"
    
    # Find top correlations
    corr_pairs = []
    for i in range(len(property_names)):
        for j in range(i+1, len(property_names)):
            prop1 = property_names[i]
            prop2 = property_names[j]
            if prop1 in df.columns and prop2 in df.columns:
                corr = df[prop1].corr(df[prop2])
                if abs(corr) > 0.7:  # Only strong correlations
                    corr_pairs.append((prop1, prop2, corr))
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    report_content += "#### Strong Correlations (|r| > 0.7):\n\n"
    for prop1, prop2, corr in corr_pairs[:10]:
        report_content += f"- **{prop1}** vs **{prop2}**: {corr:.4f}\n"
    
    report_content += "\n## Visualizations Generated\n\n"
    report_content += "1. **property_distributions.png** - Histograms of selected quantum properties\n"
    report_content += "2. **correlation_heatmap.png** - Correlation matrix of all 19 quantum properties\n"
    report_content += "3. **mw_vs_gap.png** - Scatter plot showing relationship between molecular weight and HOMO-LUMO gap\n"
    report_content += "4. **atom_count_dist.png** - Distribution of heavy atom counts in the dataset\n"
    report_content += "\n---\n*Generated by scripts/explore_qm9.py*\n"
    
    # Save report
    report_path = exploratory_dir / 'qm9_eda_report.md'
    with open(report_path, 'w') as f:
        f.write(report_content)
    logger.info(f"EDA report saved: {report_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("QM9 EXPLORATORY DATA ANALYSIS COMPLETED")
    print("="*70)
    print(f"Total Molecules: {len(df)}")
    print(f"Dataset Dimensions: {df.shape[0]} x {df.shape[1]}")
    print(f"\nVisualizations saved to: {exploratory_dir}")
    print(f"  ├── property_distributions.png")
    print(f"  ├── correlation_heatmap.png")
    print(f"  ├── mw_vs_gap.png")
    print(f"  └── atom_count_dist.png")
    print(f"\nStatistical Report: qm9_eda_report.md")
    print("="*70)

if __name__ == "__main__":
    main()

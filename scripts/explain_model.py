import pandas as pd
import logging
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Load training data to get feature names
    logger.info("Loading training data to extract feature names...")
    train_path = project_root / 'chembench' / 'datasets' / 'processed' / 'tep' / 'train.csv'
    df_train = pd.read_csv(train_path)
    feature_names = [col for col in df_train.columns if col != 'label']
    logger.info(f"Extracted {len(feature_names)} feature names")
    
    # Load the trained Random Forest model
    logger.info("Loading trained Random Forest model...")
    model_path = project_root / 'chembench' / 'models' / 'baselines' / 'random_forest_baseline.joblib'
    model = joblib.load(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
    
    # Extract feature importances
    logger.info("Extracting feature importances from the model...")
    importances = model.feature_importances_
    logger.info(f"Feature importances extracted: {len(importances)} values")
    
    # Create DataFrame with feature importances
    logger.info("Creating feature importance DataFrame...")
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Print Top 10 features
    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*60)
    top_10 = importance_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"{idx+1:2d}. {row['Feature']:20s} - {row['Importance']:.6f}")
    print("="*60)
    logger.info("Top 10 features printed to console")
    
    # Create and save horizontal bar chart for Top 15 features
    logger.info("Creating horizontal bar chart for Top 15 features...")
    top_15 = importance_df.head(15)
    
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    bars = plt.barh(range(len(top_15)), top_15['Importance'].values, color='steelblue')
    plt.yticks(range(len(top_15)), top_15['Feature'].values)
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title('Top 15 Most Important Features - Random Forest Baseline', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_15.iterrows()):
        plt.text(row['Importance'], i, f" {row['Importance']:.4f}", va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Create explainability directory if it doesn't exist
    logger.info("Creating explainability directory...")
    explainability_dir = project_root / 'chembench' / 'models' / 'explainability'
    explainability_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Explainability directory ready at {explainability_dir}")
    
    # Save the figure
    chart_path = explainability_dir / 'feature_importance.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    logger.info(f"Feature importance chart saved to {chart_path}")
    
    print(f"\n✓ Feature importance visualization saved to chembench/models/explainability/feature_importance.png")

if __name__ == "__main__":
    main()

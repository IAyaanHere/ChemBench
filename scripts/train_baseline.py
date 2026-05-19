import pandas as pd
import logging
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Load training data
    logger.info("Loading training data...")
    train_path = project_root / 'chembench' / 'datasets' / 'tennessee_eastman' / 'processed' / 'train.csv'
    df_train = pd.read_csv(train_path)
    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']
    logger.info(f"Training data loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Load test data
    logger.info("Loading test data...")
    test_path = project_root / 'chembench' / 'datasets' / 'tennessee_eastman' / 'processed' / 'test.csv'
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop('label', axis=1)
    y_test = df_test['label']
    logger.info(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Initialize and train the model
    logger.info("Initializing RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    logger.info("Training the model...")
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    # Make predictions
    logger.info("Making predictions on test data...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    report = classification_report(y_test, y_pred)

    # Print formatted results
    print("\n" + "="*50)
    print("BASELINE MODEL TRAINING RESULTS")
    print("="*50)
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"F1 Score (Binary): {f1:.4f}")
    print("\nClassification Report:")
    print("-"*30)
    print(report)
    print("="*50)

    # Save the trained model
    model_dir = project_root / 'chembench' / 'models' / 'baselines'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'random_forest_baseline.joblib'
    joblib.dump(model, model_path)
    logger.info(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    main()
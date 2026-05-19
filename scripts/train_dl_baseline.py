import pandas as pd
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Load training data
    logger.info("Loading training data...")
    train_path = project_root / 'chembench' / 'datasets' / 'tennessee_eastman' / 'processed' / 'train.csv'
    df_train = pd.read_csv(train_path)
    X_train = df_train.drop('label', axis=1).values
    y_train = df_train['label'].values
    logger.info(f"Training data loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Load test data
    logger.info("Loading test data...")
    test_path = project_root / 'chembench' / 'datasets' / 'tennessee_eastman' / 'processed' / 'test.csv'
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop('label', axis=1).values
    y_test = df_test['label'].values
    logger.info(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Convert to tensors
    logger.info("Converting data to PyTorch tensors...")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    logger.info("Tensors created successfully")
    
    # Create DataLoader
    logger.info("Creating DataLoader with batch size 64...")
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    logger.info(f"DataLoader created with {len(train_loader)} batches")
    
    # Define MLP model
    logger.info("Building MLP model architecture...")
    model = nn.Sequential(
        nn.Linear(52, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    logger.info("MLP model: Input(52) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(1) -> Sigmoid")
    
    # Define loss function and optimizer
    logger.info("Initializing loss function and optimizer...")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    logger.info("Loss: BCELoss | Optimizer: Adam (lr=0.001)")
    
    # Training loop
    logger.info("Starting training loop for 10 epochs...")
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")
    
    logger.info("Training completed successfully")
    
    # Evaluate on test data
    logger.info("Evaluating model on test data...")
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = (test_outputs.numpy() >= 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, test_predictions)
    f1 = f1_score(y_test, test_predictions)
    logger.info(f"Test Accuracy: {accuracy:.4f} | Test F1 Score: {f1:.4f}")
    
    # Print formatted results
    print("\n" + "="*50)
    print("DEEP LEARNING BASELINE MODEL RESULTS")
    print("="*50)
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("="*50)
    
    # Save model
    logger.info("Saving model state dictionary...")
    model_dir = project_root / 'chembench' / 'models' / 'baselines'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'mlp_baseline.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    main()

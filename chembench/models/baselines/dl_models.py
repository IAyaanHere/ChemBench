"""
Deep Learning Baseline Models for ChemBench.

Includes:
- FullyConnectedNN (FCNN): For tabular/molecular property data
- CNN1D: For time-series process data (e.g., TEP)
- GNNBase: For graph-based molecular representations
- PyTorchWrapper: Base class integrating PyTorch models with sklearn-like interface
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from chembench.models.baselines.base import BaselineModel

logger = logging.getLogger(__name__)


# ============================================================================
# PyTorch Neural Network Architectures
# ============================================================================

class FullyConnectedNN(nn.Module):
    """Fully Connected Neural Network for tabular data.
    
    Architecture:
    - Input layer: input_dim features
    - Hidden layers: [hidden_dim, hidden_dim, hidden_dim]
    - Output layer: output_dim
    - Activation: ReLU
    - Regularization: Dropout(0.3)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers
        layers = []
        prev_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for time-series data.
    
    Architecture:
    - Conv1D layers: 32, 64 filters with kernel_size=3
    - MaxPooling after each conv layer
    - Fully connected layers: 128 → output_dim
    - Activation: ReLU
    - Regularization: Dropout(0.3)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_length: int = 52,
        num_filters: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length

        # Conv layers
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)

        # Calculate flattened size
        flat_size = (seq_length // 4) * (num_filters * 2)

        # FC layers
        self.fc1 = nn.Linear(flat_size, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, seq_length, input_dim)
                Will be transposed to (batch_size, input_dim, seq_length)
        """
        # Transpose to (batch, channels, length) for Conv1d
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class GNNBase(nn.Module):
    """Graph Neural Network base class for molecular graphs.
    
    Uses GCNConv for graph convolution on molecular adjacency matrices.
    Architecture:
    - GCN layers: input_dim → 128 → 64
    - Global pooling (mean)
    - FC layers: 64 → output_dim
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        try:
            from torch_geometric.nn import GCNConv
        except ImportError:
            logger.warning('torch_geometric not installed. GNNBase will use MLP fallback.')
            # Fallback to FCNN if torch_geometric not available
            self.use_fallback = True
            self.network = FullyConnectedNN(input_dim, output_dim, hidden_dim)
            return

        self.use_fallback = False

        # GCN layers
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim // 2)

        # FC layers after pooling
        self.fc1 = nn.Linear(hidden_dim // 2, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features (batch_size, num_nodes, input_dim)
            edge_index: Edge indices for graph connectivity
        """
        if self.use_fallback:
            return self.network(x)

        from torch_geometric.nn import global_mean_pool

        # If edge_index not provided, use fully connected graph
        if edge_index is None:
            batch_size, num_nodes, _ = x.shape
            edge_index = self._create_fc_edge_index(batch_size, num_nodes)
            edge_index = edge_index.to(x.device)

        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        x = torch.relu(x)

        # Global mean pooling
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    @staticmethod
    def _create_fc_edge_index(batch_size: int, num_nodes: int) -> torch.Tensor:
        """Create fully connected edge index for batch of graphs."""
        edge_list = []
        node_offset = 0

        for _ in range(batch_size):
            # Create fully connected graph for this sample
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    edge_list.append([node_offset + i, node_offset + j])
                    edge_list.append([node_offset + j, node_offset + i])

            node_offset += num_nodes

        if not edge_list:
            return torch.tensor([[], []], dtype=torch.long)

        return torch.tensor(edge_list, dtype=torch.long).t()


# ============================================================================
# PyTorch Wrapper for BaselineModel Integration
# ============================================================================

class PyTorchWrapper(BaselineModel):
    """Wrapper for PyTorch models to integrate with sklearn-like BaselineModel interface.
    
    Features:
    - Automatic device management (CPU/CUDA)
    - Training loop with validation and early stopping
    - Feature normalization (StandardScaler)
    - Metrics computation inherited from BaselineModel
    """

    def __init__(
        self,
        model_name: str,
        pytorch_model: nn.Module,
        task_type: str = 'regression',
        device: Optional[str] = None,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        early_stopping_patience: int = 5,
        val_split: float = 0.2,
    ):
        super().__init__(model_name, task_type)
        self.pytorch_model = pytorch_model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs  # max_epochs: training stops at this cap or early stopping
        self.early_stopping_patience = early_stopping_patience
        self.val_split = val_split
        self.scaler = StandardScaler()
        self.model = pytorch_model.to(self.device)

        logger.info(
            'Initialized PyTorchWrapper: model=%s, task=%s, device=%s',
            model_name,
            task_type,
            self.device,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> None:
        """Train the PyTorch model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (if None, will be split from train)
            y_val: Validation targets
        """
        # Auto-detect task type if not set
        if self.task_type is None:
            self.task_type = self._detect_task_type(y_train)
            logger.info('Auto-detected task type: %s', self.task_type)

        # Convert to numpy
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train

        # Normalize features
        X_train_np = self.scaler.fit_transform(X_train_np)

        # Split validation set if not provided
        if X_val is None:
            n_train = int(len(X_train_np) * (1 - self.val_split))
            indices = np.random.permutation(len(X_train_np))
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]

            X_train_split = X_train_np[train_indices]
            y_train_split = y_train_np[train_indices]
            X_val_split = X_train_np[val_indices]
            y_val_split = y_train_np[val_indices]
        else:
            X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val
            X_val_split = self.scaler.transform(X_val_np)
            y_val_split = y_val_np
            X_train_split = X_train_np
            y_train_split = y_train_np

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_split),
            torch.FloatTensor(y_train_split.reshape(-1, 1) if len(y_train_split.shape) == 1 else y_train_split),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_split),
            torch.FloatTensor(y_val_split.reshape(-1, 1) if len(y_val_split.shape) == 1 else y_val_split),
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Setup optimizer and loss
        optimizer = optim.Adam(self.pytorch_model.parameters(), lr=self.learning_rate)

        if self.task_type == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        logger.info(
            'Starting training: epochs=%d, batch_size=%d, lr=%f',
            self.epochs,
            self.batch_size,
            self.learning_rate,
        )

        for epoch in range(self.epochs):
            # Train phase
            self.pytorch_model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.pytorch_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            self.pytorch_model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.pytorch_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    'Epoch %d/%d - Train Loss: %.6f, Val Loss: %.6f',
                    epoch + 1,
                    self.epochs,
                    train_loss,
                    val_loss,
                )

            if patience_counter >= self.early_stopping_patience:
                logger.info('Early stopping at epoch %d', epoch + 1)
                break

        self.is_fitted = True
        logger.info('Training completed. Best validation loss: %.6f', best_val_loss)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions as numpy array
        """
        if not self.is_fitted:
            raise RuntimeError(f'Model {self.model_name} is not fitted yet')

        # Convert to numpy
        X_np = X.values if isinstance(X, pd.DataFrame) else X

        # Normalize using fitted scaler
        X_np = self.scaler.transform(X_np)

        # Create tensor
        X_tensor = torch.FloatTensor(X_np).to(self.device)

        # Predict
        self.pytorch_model.eval()
        with torch.no_grad():
            outputs = self.pytorch_model(X_tensor)

        # Convert to numpy
        predictions = outputs.cpu().numpy()

        # For regression, squeeze to 1D
        if self.task_type == 'regression':
            predictions = predictions.squeeze()

        return predictions

    def save_model(self, filepath: Path) -> None:
        """Save PyTorch model to disk.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            logger.warning('Attempting to save unfitted model %s', self.model_name)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.save(
                {
                    'model_state': self.pytorch_model.state_dict(),
                    'scaler': self.scaler,
                    'task_type': self.task_type,
                    'learning_rate': self.learning_rate,
                },
                filepath,
            )
            logger.info('PyTorch model %s saved to %s', self.model_name, filepath)
        except Exception as exc:
            logger.error('Failed to save model %s: %s', self.model_name, exc)
            raise

    def load_model(self, filepath: Path) -> None:
        """Load PyTorch model from disk.
        
        Args:
            filepath: Path to model file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f'Model file not found: {filepath}')

        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.pytorch_model.load_state_dict(checkpoint['model_state'])
            self.scaler = checkpoint['scaler']
            self.task_type = checkpoint['task_type']
            self.is_fitted = True
            logger.info('PyTorch model %s loaded from %s', self.model_name, filepath)
        except Exception as exc:
            logger.error('Failed to load model %s: %s', self.model_name, exc)
            raise

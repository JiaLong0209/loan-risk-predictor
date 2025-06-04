import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from .base_model import BaseModel
from tqdm import tqdm

class MLPModel(BaseModel):
    """Multi-Layer Perceptron model implementation."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load hyperparameters from config
        self.hyperparams = self.config.get(f"models.hyperparameters.{self.model_name}")

    def _create_model(self, input_size: int) -> nn.Module:
        """Create the MLP model architecture."""
        class MLP(nn.Module):
            def __init__(self, input_size, hidden_layers, dropout_rate):
                super(MLP, self).__init__()
                
                # Create layers dynamically based on hidden_layers
                layers = []
                prev_size = input_size
                
                for hidden_size in hidden_layers:
                    layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_size = hidden_size
                
                # Add final layer
                layers.append(nn.Linear(prev_size, 1))
                layers.append(nn.Sigmoid())
                
                self.layers = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.layers(x)
        
        return MLP(
            input_size=input_size,
            hidden_layers=self.hyperparams['hidden_layers'],
            dropout_rate=self.hyperparams['dropout_rate']
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        # Create model
        self.model = self._create_model(X_train.shape[1]).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hyperparams['learning_rate']
        )
        
        # Training loop
        batch_size = self.hyperparams['batch_size']
        n_samples = X_train.shape[0]
        
        self.model.train()
        # Add progress bar for epochs
        pbar = tqdm(range(self.hyperparams['epochs']), desc=f'Training {self.model_name}')
        for epoch in pbar:
            # Shuffle data
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            # Record average loss for the epoch
            avg_loss = epoch_loss / (n_samples / batch_size)
            self.record_loss(avg_loss)
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Plot and save the training loss curve
        self.plot_train_loss()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Convert to PyTorch tensor
        X = torch.FloatTensor(X).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        
        return predictions.cpu().numpy()

    def save(self, path: str) -> None:
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the model from disk."""
        if self.model is None:
            raise ValueError("Model not initialized")
        self.model.load_state_dict(torch.load(path)) 
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
from .base_model import BaseModel
from tqdm import tqdm

class DLSTMMLPModel(BaseModel):
    """Hybrid model combining LSTM for temporal feature extraction and MLP for final prediction."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = None
        
        self.model_name = "d_lstm_mlp"

        # Load hyperparameters from config
        self.hyperparams = self.config.get(f"models.hyperparameters.{self.model_name}")
    
    def _create_model(self, input_size: int) -> nn.Module:
        """Create the hybrid LSTM-MLP model architecture."""
        class HybridModel(nn.Module):
            def __init__(self, input_size, lstm_hidden_size, lstm_num_layers, lstm_dropout_rate, lstm_bidirectional,
                         mlp_hidden_layers, mlp_dropout_rate):
                super(HybridModel, self).__init__()
                
                # LSTM component for temporal feature extraction
                self.lstm = nn.LSTM(
                    input_size,
                    lstm_hidden_size,
                    lstm_num_layers,
                    batch_first=True,
                    dropout=lstm_dropout_rate,
                    bidirectional=lstm_bidirectional
                )
                
                # Calculate LSTM output size (doubled due to bidirectional)
                lstm_output_size = (lstm_hidden_size* 2 if lstm_bidirectional else lstm_hidden_size)
                
                # MLP component for final prediction
                mlp_layers = []
                prev_size = lstm_output_size
                
                for hidden_size in mlp_hidden_layers:
                    mlp_layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        nn.Dropout(mlp_dropout_rate)
                    ])
                    prev_size = hidden_size
                
                # Add final layer
                mlp_layers.extend([
                    nn.Linear(prev_size, 1),
                    nn.Sigmoid()
                ])
                
                self.mlp = nn.Sequential(*mlp_layers)
            
            def forward(self, x):
                # LSTM forward pass
                num_layers =  (self.lstm.num_layers* 2 if self.lstm.bidirectional else self.lstm.num_layers)
                h0 = torch.zeros(num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
                c0 = torch.zeros(num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
                
                lstm_out, _ = self.lstm(x, (h0, c0))
                lstm_features = lstm_out[:, -1, :]  # Get the last time step
                
                # MLP forward pass
                return self.mlp(lstm_features)
        
        return HybridModel(
            input_size=input_size,
            lstm_hidden_size=self.hyperparams['lstm']['hidden_size'],
            lstm_num_layers=self.hyperparams['lstm']['num_layers'],
            lstm_dropout_rate=self.hyperparams['lstm']['dropout_rate'],
            lstm_bidirectional=self.hyperparams['lstm']['bidirectional'],
            mlp_hidden_layers=self.hyperparams['mlp']['hidden_layers'],
            mlp_dropout_rate=self.hyperparams['mlp']['dropout_rate'],
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        # Reshape input for LSTM (batch_size, sequence_length, features)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        # Store input size and create model
        self.input_size = X_train.shape[2]
        self.model = self._create_model(self.input_size).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hyperparams['lstm']['learning_rate']
        )
        
        # Training loop
        batch_size = self.hyperparams['lstm']['batch_size']
        n_samples = X_train.shape[0]
        
        self.model.train()
        # Add progress bar for epochs
        pbar = tqdm(range(self.hyperparams['lstm']['epochs']), desc=f'Training {self.model_name}')
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
        
        # Reshape input for LSTM
        X = X.reshape(X.shape[0], 1, X.shape[1])
        
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
        
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'hyperparams': self.hyperparams
        }
        torch.save(save_data, path)
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.input_size = checkpoint['input_size']
        self.hyperparams = checkpoint['hyperparams']
        
        self.model = self._create_model(self.input_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict']) 
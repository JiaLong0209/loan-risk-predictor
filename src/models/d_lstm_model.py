import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from .base_model import BaseModel
from tqdm import tqdm

class DLSTMModel(BaseModel):
    """Deep LSTM model implementation."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = None
        
        self.model_name = "d_lstm"
        # Load hyperparameters from config
        self.hyperparams = self.config.get(f"models.hyperparameters.{self.model_name}")
    
    def _create_model(self, input_size: int) -> nn.Module:
        """Create the LSTM model architecture."""
        class LSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout_rate, bidirectional):
                super(LSTM, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout_rate,
                    bidirectional=bidirectional
                )
                
                # Adjust output size based on bidirectional
                output_size = hidden_size * 2 if bidirectional else hidden_size
                
                self.fc1 = nn.Linear(output_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)
                self.dropout = nn.Dropout(dropout_rate)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                num_layers = self.num_layers * 2 if self.lstm.bidirectional else self.num_layers
                h0 = torch.zeros(num_layers, x.size(0), self.hidden_size).to(x.device) 
                c0 = torch.zeros(num_layers, x.size(0), self.hidden_size).to(x.device)
                
                out, _ = self.lstm(x, (h0, c0))
                out = out[:, -1, :]  # Get the last time step
                
                out = torch.relu(self.fc1(out))
                out = self.dropout(out)
                out = torch.relu(self.fc2(out))
                out = self.dropout(out)
                out = self.fc3(out)
                return self.sigmoid(out)
        
        return LSTM(
            input_size=input_size,
            hidden_size=self.hyperparams['hidden_size'],
            num_layers=self.hyperparams['num_layers'],
            dropout_rate=self.hyperparams['dropout_rate'],
            bidirectional=self.hyperparams['bidirectional'],
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
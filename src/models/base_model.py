from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Optional
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import os
from ..config.config import ConfigManager

class BaseModel(ABC):
    """Base class for all models implementing the Strategy pattern."""
    
    def __init__(self):
        """Initialize the base model with common attributes."""
        self.config = ConfigManager()
        self.train_losses: List[float] = []
        self.model_name: str = self.__class__.__name__.replace('Model', '').lower()
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model on the given data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the given data."""
        pass

    def find_best_threshold(self, y_true, y_probs):
        """
        Find the threshold that gives the best F1 score.
        y_true : array-like, true binary labels (0 or 1)
        y_probs: array-like, predicted probabilities (float between 0 and 1)
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)  # avoid division by zero

        best_index = f1s.argmax()
        best_threshold = thresholds[best_index]
        
        print(f"Best Threshold = {best_threshold:.3f}")
        print(f"Precision = {precisions[best_index]:.3f}")
        print(f"Recall = {recalls[best_index]:.3f}")
        print(f"F1 Score = {f1s[best_index]:.3f}")
        
        return best_threshold

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float]:
        """Evaluate the model."""
        predictions = self.predict(X_test)
        # Ensure predictions are 1D array
        predictions = predictions.reshape(-1)

        best_thresh = self.find_best_threshold(y_test, predictions)
        predictions = (predictions > best_thresh).astype(int)
        return self._calculate_metrics(y_test, predictions)
    
    def plot_train_loss(self, save_path: Optional[str] = None) -> None:
        """Plot and optionally save the training loss curve.
        
        Args:
            save_path: Optional path to save the plot. If None, uses the default graph directory from config.
        """
        if not self.train_losses:
            print(f"No training losses recorded for {self.model_name}")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', color='#2ecc71', linewidth=2)
        plt.title(f'Training Loss Curve - {self.model_name.upper()}', fontsize=12)
        plt.xlabel('Epoch', fontsize=10)

        epochs = len(self.train_losses)
        # plt.xticks(np.linspace(0, epochs - 1, num=10).tolist() + [epochs - 1], )
        # plt.xticks(ticks=range(0, epochs, 2), labels=[str(i + 1) for i in range(0, epochs, 2)])
        plt.xticks([])
        plt.xlim(0, epochs-1)

        plt.ylabel('Loss', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        # Add final loss value annotation
        final_loss = self.train_losses[-1]
        plt.annotate(f'Final Loss: {final_loss:.4f}',
                    xy=(len(self.train_losses)-1, final_loss),
                    xytext=(len(self.train_losses)-1, final_loss*1.1),
                    arrowprops=dict(facecolor='#333333', shrink=0.05, width=1, headwidth=8, headlength=8),
                    fontsize=8)

        # Save the plot if path is provided or use default
        if save_path is None:
            graph_dir = self.config.get("data.train_loss_dir")
            if not os.path.exists(graph_dir):
                os.makedirs(graph_dir)
            save_path = os.path.join(graph_dir, f"{self.model_name}_train_loss.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def record_loss(self, loss: float) -> None:
        """Record a training loss value.
        
        Args:
            loss: The loss value to record
        """
        self.train_losses.append(loss)
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk."""
        pass
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate accuracy, recall, precision, and f1_score."""
        # Debug logging
        print(f"True labels shape: {y_true.shape}, unique values: {np.unique(y_true)}")
        print(f"Predictions shape: {y_pred.shape}, unique values: {np.unique(y_pred)}")
        print(f"True labels distribution: {np.bincount(y_true.astype(int))}")
        print(f"Predictions distribution: {np.bincount(y_pred.astype(int))}")
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Debug logging for confusion matrix components
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Total samples: {tp + tn + fp + fn}")

        cm = confusion_matrix(y_true, y_pred)
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Previously sensitivity
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Previously specificity
        f1 = f1_score(y_true, y_pred)
        
        return accuracy, recall, precision, f1, cm
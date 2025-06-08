import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif
from typing import List, Tuple
from scipy.special import psi
import torch
import torch.nn as nn
from tqdm import tqdm 
from ..config.config import ConfigManager
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

class FeatureEngineer:
    """Feature engineering and selection utilities."""
    
    def __init__(self):
        self.config = ConfigManager()
        self.feature_weights = None
        self.selected_indices = None
        self.max_samples = self.config.get('feature_engineering.max_samples')  # Maximum samples for MI calculation

    def calculate_kraskov_mi(self, X: np.ndarray, y: np.ndarray, k: int = 5) -> np.ndarray:
        """Calculate Kraskov mutual information between each feature and the target using KNN and optional subsampling."""
        n_samples, n_features = X.shape
        mi_scores = np.zeros(n_features)

        # Subsample to reduce memory and computation (stratified to preserve class distribution)
        if n_samples > self.max_samples:
            X_sub, _, y_sub, _ = train_test_split(
                X, y,
                train_size=self.max_samples,
                stratify=y,
                random_state=42
            )
        else:
            X_sub, y_sub = X, y

        n_sub_samples = X_sub.shape[0]

        for i in tqdm(range(n_features)):
            # Use 1D feature vector for KNN
            feature_column = X_sub[:, i].reshape(-1, 1)

            # Fit KNN and get distances to k+1 neighbors (first neighbor is the point itself)
            nn = NearestNeighbors(n_neighbors=k + 1, metric='manhattan')  
            nn.fit(feature_column)
            distances, _ = nn.kneighbors(feature_column)
            epsilon = distances[:, k]  # Distance to the k-th neighbor

            # Count neighbors within epsilon for X and for y
            n_x = np.sum(np.abs(feature_column - feature_column.T) <= epsilon[:, None], axis=1)
            n_y = np.sum(np.abs(y_sub[:, None] - y_sub) <= epsilon[:, None], axis=1)

            # Kraskov mutual information estimate
            mi_scores[i] = psi(k) - np.mean(psi(n_x + 1) + psi(n_y + 1)) + psi(n_sub_samples)

        return mi_scores

    def weight_features(self, X: np.ndarray, y: np.ndarray, n_features: int = 5) -> np.ndarray:
        """Weight features using Kraskov mutual information."""
        # Calculate Kraskov MI scores
        k = self.config.get('feature_engineering.kraskov_mi.k_neighbors')
        mi_scores = self.calculate_kraskov_mi(X, y, k)

        # Normalize scores to get weights
        print(f"mi_scores: {mi_scores}")

        mi_scores = mi_scores - np.min(mi_scores)  # Shift to non-negative values
        print(f"mi_scores: {mi_scores}")
        self.feature_weights = mi_scores / np.sum(mi_scores)

        # total_score = np.sum(mi_scores)
        # self.feature_weights = mi_scores / total_score if total_score > 0 else np.ones_like(mi_scores) / len(mi_scores)
        
        # Select top features
        print(f"self.feature_weights.shape: {self.feature_weights.shape}")
        print(f"self.feature_weights: {self.feature_weights}")

        self.selected_indices = np.argsort(mi_scores)[-n_features:]

        print(f"self.selected_indices: {X[0:1, self.selected_indices]}")

        weighted_features = X[:, self.selected_indices] * self.feature_weights[self.selected_indices]
        print(f"weighted_features: {weighted_features[0:1, :]}")

        # weighted_features = self.standardize_features(weighted_features)
        weighted_features = self.normalize_features(weighted_features)
        print(f"weighted_features: {weighted_features[0:1, :]}")

        # weighted_features = X[:, self.selected_indices] + self.feature_weights[self.selected_indices]
        
        return weighted_features
    
    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using min-max scaling."""
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)

    def standardize_features(self, X: np.ndarray) -> np.ndarray:
        """Standardize features using Z-score normalization."""
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
    def process_features(self, X: np.ndarray, y: np.ndarray, n_features: int = 5) -> np.ndarray:
        """Process features through the complete pipeline.
        
        Args:
            X: Input features
            y: Target labels
            n_features: Number of features to select
            
        Returns:
            Processed features (weighted and selected)
        """
        # Normalize features
        X_normalized = self.normalize_features(X)
        
        # Weight and select features using Kraskov MI
        weighted_features = self.weight_features(X_normalized, y, n_features)
        
        return weighted_features 
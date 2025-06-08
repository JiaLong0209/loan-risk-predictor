from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from .config.config import ConfigManager
from .data_pipeline.data_repository import DataRepository
from .data_pipeline.feature_engineering import FeatureEngineer
from .models.model_factory import ModelFactory
from .utils.logger import Logger
from .models.autoencoder import AutoencoderModel
from .utils.test_utils import TestDataGenerator, DebugLogger
import pandas as pd
import os
import time  # Add time module
import json  # Add this import at the top with other imports
import matplotlib.pyplot as plt

class LoanRiskPredictor:
    """Main application class for loan risk prediction."""
    
    def __init__(self):
        self.config = ConfigManager()
        # Load config from config.yaml (adjust the path if needed)
        self.config.load_from_yaml("config.yaml")
        self.data_repo = DataRepository()
        self.logger = Logger()
        self.feature_engineer = FeatureEngineer()
        self.debug_logger = DebugLogger("loan_risk_predictor")
        # Set random seed for reproducibility
        np.random.seed(42)
        # Initialize timing metrics
        self.timing_metrics = {
            'fold_times': [],
            'total_time': 0.0
        }
        # Initialize training losses storage
        self.model_losses = {}

    def _train_d_lstm_mlp(self, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> Tuple[float, float, float, float, np.ndarray, Dict[str, List[float]]]:
        """Special training method for d_lstm_mlp model that combines D-LSTM and MLP.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            
        Returns:
            Tuple of (accuracy, recall, precision, f1_score, confusion_matrix, training_losses)
        """
        # Step 1: Train D-LSTM model
        self.logger.info("Training D-LSTM model...")
        d_lstm_model = ModelFactory.create_model("d_lstm")
        d_lstm_model.train(X_train, y_train)
        
        # Store LSTM training losses and plot them
        lstm_losses = d_lstm_model.train_losses.copy()
        graph_dir = self.config.get("data.train_loss_dir")
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        d_lstm_model.plot_train_loss(os.path.join(graph_dir, "d_lstm_mlp-lstm.train_loss.png"))
        
        # Step 2: Get D-LSTM predictions and append to features
        self.logger.info("Getting D-LSTM predictions...")
        d_lstm_train_pred = d_lstm_model.predict(X_train)
        d_lstm_val_pred = d_lstm_model.predict(X_val)
        
        # Reshape predictions to 2D if needed
        if len(d_lstm_train_pred.shape) == 1:
            d_lstm_train_pred = d_lstm_train_pred.reshape(-1, 1)
            d_lstm_val_pred = d_lstm_val_pred.reshape(-1, 1)
        
        # Combine original features with D-LSTM predictions
        X_train_combined = np.hstack([X_train, d_lstm_train_pred])
        X_val_combined = np.hstack([X_val, d_lstm_val_pred])
        
        # Step 3: Train MLP model on combined features
        self.logger.info(f"Training MLP model on combined features, shape = {X_train_combined.shape}")
        mlp_model = ModelFactory.create_model("mlp")
        mlp_model.train(X_train_combined, y_train)
        
        # Store MLP training losses and plot them
        mlp_losses = mlp_model.train_losses.copy()
        mlp_model.plot_train_loss(os.path.join(graph_dir, "d_lstm_mlp-mlp.train_loss.png"))
        
        # Step 4: Evaluate the final model
        acc, rec, prec, f1, cm = mlp_model.evaluate(X_val_combined, y_val)
        
        # Create a dictionary of training losses for both components
        training_losses = {
            'lstm': lstm_losses,
            'mlp': mlp_losses
        }
        
        return acc, rec, prec, f1, cm, training_losses

    def plot_all_training_losses(self) -> None:
        """Plot training losses for all models in a single figure."""
        if not self.model_losses:
            self.logger.warning("No training losses recorded for any model")
            return

        plt.figure(figsize=(12, 8))
        
        # Use a colormap for different models
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.model_losses)))
        
        # Plot each model's loss curve
        for (model_name, losses), color in zip(self.model_losses.items(), colors):
            plt.plot(losses, label=model_name.upper(), color=color, linewidth=2)
            
            # Add final loss value annotation
            final_loss = losses[-1]
            plt.annotate(f'{final_loss:.4f}',
                        xy=(len(losses)-1, final_loss),
                        xytext=(len(losses)-1, final_loss*1.1),
                        arrowprops=dict(facecolor='#333333', shrink=0.05, width=1),
                        fontsize=8)

        plt.title('Training Loss Curves for All Models', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        graph_dir = self.config.get("data.graph_dir")
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        plt.savefig(os.path.join(graph_dir, "all_models_training_loss.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _save_kfold_metrics(self, model_name: str, metrics: Dict[str, List[float]], n_folds: int) -> None:
        """Save k-fold metrics visualization for a model as a line chart."""
        try:
            # Create metrics directory if it doesn't exist
            metrics_dir = os.path.join(self.config.get("data.graph_dir"), "kfold_metrics")
            if not os.path.exists(metrics_dir):
                os.makedirs(metrics_dir)

            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8))
            # Plot metrics for each fold
            folds = range(1, n_folds + 1)
            metrics_to_plot = {
                'Accuracy': metrics['accuracy'],
                'Recall': metrics['recall'],
                'Precision': metrics['precision'],
                'F1-Score': metrics['f1_score']
            }

            # Plot individual metrics
            colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_to_plot)))
            for (metric_name, values), color in zip(metrics_to_plot.items(), colors):
                ax1.plot(folds, values, marker='o', label=metric_name, color=color, linewidth=2)
                
                # Add value labels
                for x, y in zip(folds, values):
                    ax1.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Customize first subplot
            ax1.set_title(f'{model_name} - Metrics Across Folds', fontsize=12, pad=20)
            ax1.set_xlabel('Fold', fontsize=10)
            ax1.set_ylabel('Score', fontsize=10)
            ax1.set_xticks(folds)
            ax1.set_ylim(0.0, 1.0)  # Set y-axis limits to 0.0-1.0
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            
            # Calculate and plot average metrics with standard deviation
            avg_metrics = {
                'Accuracy': np.mean(metrics['accuracy']),
                'Recall': np.mean(metrics['recall']),
                'Precision': np.mean(metrics['precision']),
                'F1-Score': np.mean(metrics['f1_score'])
            }
            
            std_metrics = {
                'Accuracy': np.std(metrics['accuracy']),
                'Recall': np.std(metrics['recall']),
                'Precision': np.std(metrics['precision']),
                'F1-Score': np.std(metrics['f1_score'])
            }
            
            # Plot average metrics with error bars
            x = np.arange(len(avg_metrics))
            width = 0.2
            
            for i, (metric_name, avg_value) in enumerate(avg_metrics.items()):
                ax2.bar(x[i], avg_value, width, yerr=std_metrics[metric_name],
                       color=colors[i], label=metric_name, capsize=5)
                
                # Add value labels
                ax2.text(x[i], avg_value, f'{avg_value:.3f}\n±{std_metrics[metric_name]:.3f}',
                        ha='center', va='bottom', fontsize=8)
            
            # Customize second subplot
            ax2.set_title('Average Metrics with Standard Deviation', fontsize=12, pad=20)
            ax2.set_xticks([])  # Remove x-ticks as they're not meaningful for averages
            ax2.set_ylabel('Score', fontsize=10)
            ax2.set_ylim(0.0, 1.0)  # Set y-axis limits to 0.0-1.0
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            
            # Adjust layout and save
            plt.tight_layout()
            chart_path = os.path.join(metrics_dir, f"{model_name}_kfold_metrics.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved k-fold metrics visualization to {chart_path}")

        except Exception as e:
            self.logger.error(f"Error saving k-fold metrics visualization: {str(e)}")

    def run(self, model_name: str = None, subsample_rate: float = 1.0, n_folds: int = 5,
            debug_mode: bool = False, use_feature_engineering: bool = True) -> Dict[str, List[float]]:
        """Run the loan risk prediction pipeline with K-fold cross-validation or single train/test split."""
        try:
            # Reset timing metrics
            self.timing_metrics = {
                'fold_times': [],
                'total_time': 0.0
            }
            
            # Start total timing
            total_start_time = time.time()
            
            # Load and preprocess data
            self.logger.info("================ RUN INFO ================")
            self.logger.info(f"model_name: {model_name}")
            self.logger.info(f"subsample_rate: {subsample_rate}")
            self.logger.info(f"n_folds: {n_folds}")
            self.logger.info(f"use_feature_engineering: {use_feature_engineering}")
            self.logger.info("============================================")

            self.logger.info("Loading and preprocessing data...")
            if debug_mode:
                data, labels = self.data_repo.load_data()
                self.debug_logger.log_data_info(data, "Debug Data")
                self.debug_logger.log_array_info(labels, "Debug Labels")
            else:
                data, labels = self.data_repo.load_data()

            processed_dir = self.config.get('data.processed_dir')

            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir)

            processed_path = os.path.join(processed_dir, self.config.get('data.normalized_data_path'))
            self.logger.info(f"{processed_path}")

            if os.path.exists(processed_path):
                self.logger.info(f"Using {processed_path} data.")
                normalized_data = pd.read_csv(processed_path).values
            else:
                normalized_data = self.data_repo.preprocess_data(data)

            if debug_mode:
                self.debug_logger.log_array_info(normalized_data, "Normalized Data")
            self.logger.info("Normalized Data Shape")
            self.logger.info(normalized_data.shape)
            
            # Subsample data if needed
            if subsample_rate < 1.0:
                self.logger.info(f"Subsampling data to {subsample_rate*100}%...")
                normalized_data, labels = self.data_repo.subsample_data(normalized_data, labels, subsample_rate)
                if debug_mode:
                    self.debug_logger.log_array_info(normalized_data, "Subsampled Data")
                    self.debug_logger.log_array_info(labels, "Subsampled Labels")
                self.logger.info("Subsample Data Shape")
                self.logger.info(normalized_data.shape)

            # Feature engineering
            if use_feature_engineering:
                fused_features_path = os.path.join(
                    processed_dir,
                    self.config.get('data.fused_features_path')
                )
                
                # Check if fused features exist and have matching shape
                if os.path.exists(fused_features_path):
                    n_features = self.config.get('features.n_features')
                    encoding_size = self.config.get('models.hyperparameters.autoencoder.encoding_size')

                    expected_fused_feature_shape = np.array((normalized_data.shape[0], n_features))

                    if self.config.get('feature_engineering.use_autoencoder'):
                        expected_fused_feature_shape[1] = encoding_size
                        self.logger.info(f"Expected Fused Features Shape: {expected_fused_feature_shape}")

                        if self.config.get('feature_engineering.append_autoencoder_features'):
                            expected_fused_feature_shape[1] += n_features
                            self.logger.info(f"Expected Fused Features Shape: {expected_fused_feature_shape}")

                    if self.config.get('feature_engineering.append_normalized_features'):
                        expected_fused_feature_shape[1] += normalized_data.shape[1]
                        self.logger.info(f"Expected Fused Features Shape: {expected_fused_feature_shape}")

                    try:
                        fused_features = pd.read_csv(fused_features_path).values
                        self.logger.info(f"Existed Fused Features Shape: {fused_features.shape}")
                        
                        self.logger.info(f"Expected Fused Features Shape: {expected_fused_feature_shape}")
                        if (fused_features.shape == expected_fused_feature_shape).all():
                            self.logger.info(f"Using cached fused features: {fused_features_path}")
                            encoded_features = fused_features
                        else:
                            raise ValueError("Feature shape mismatch")
                    except Exception as e:
                        self.logger.warning(f"Error loading cached features: {str(e)}")
                        self.logger.info("Performing feature engineering...")
                        encoded_features = self._perform_feature_engineering(
                            normalized_data, labels, debug_mode
                        )
                        # Save the new fused features
                        pd.DataFrame(encoded_features).to_csv(fused_features_path, index=False)
                else:
                    self.logger.info("No cached fused features found, performing feature engineering...")
                    encoded_features = self._perform_feature_engineering(
                        normalized_data, labels, debug_mode
                    )
                    # Save the fused features
                    pd.DataFrame(encoded_features).to_csv(fused_features_path, index=False)
            else:
                encoded_features = normalized_data
            
            # Initialize metrics storage
            metrics = {
                'accuracy': [],
                'recall': [],
                'precision': [],
                'f1_score': [],
                'confusion_matrices': [],
                'training_losses': []  # Add training losses to metrics
            }
            
            # Get model name from config if not provided
            if model_name is None:
                model_name = self.config.get('models.default_model')
            
            # Handle single train/test split
            if n_folds == 1:
                self.logger.info("Using single train/test split...")
                train_percentage = self.config.get('models.train_percentage', 80) / 100.0
                
                # Split data into train and test sets
                n_samples = len(encoded_features)
                n_train = int(n_samples * train_percentage)
                indices = np.random.permutation(n_samples)
                train_indices = indices[:n_train]
                test_indices = indices[n_train:]
                
                X_train = encoded_features[train_indices]
                y_train = labels[train_indices]
                X_test = encoded_features[test_indices]
                y_test = labels[test_indices]
                
                if debug_mode:
                    self.debug_logger.log_array_info(X_train, "Training Data")
                    self.debug_logger.log_array_info(y_train, "Training Labels")
                    self.debug_logger.log_array_info(X_test, "Test Data")
                    self.debug_logger.log_array_info(y_test, "Test Labels")
                
                # Start timing for training
                fold_start_time = time.time()
                
                self.logger.info("Training model...")
                
                # Special handling for d_lstm_mlp model
                if model_name == "d_lstm_mlp":
                    acc, rec, prec, f1, cm, training_losses = self._train_d_lstm_mlp(X_train, X_test, y_train, y_test)
                    metrics['training_losses'] = training_losses
                    
                    # Store metrics
                    metrics['accuracy'].append(acc)
                    metrics['recall'].append(rec)
                    metrics['precision'].append(prec)
                    metrics['f1_score'].append(f1)
                    metrics['confusion_matrices'].append(cm)
                    
                    # Calculate training time for single train/test split
                    fold_time = time.time() - fold_start_time
                    self.timing_metrics['fold_times'].append(fold_time)
                    
                    # Log training time
                    self.logger.info(f"Training time: {fold_time:.2f} seconds")
                    
                    if debug_mode:
                        self.debug_logger.log_metrics({
                            'accuracy': acc,
                            'recall': rec,
                            'precision': prec,
                            'f1_score': f1
                        }, "Test Results")
                        self.debug_logger.log_array_info(cm, "Test Confusion Matrix")
                    
                    self.logger.info("Test results:")
                    self.logger.info(f"Accuracy: {acc:.4f}")
                    self.logger.info(f"Recall: {rec:.4f}")
                    self.logger.info(f"Precision: {prec:.4f}")
                    self.logger.info(f"F1-Score: {f1:.4f}")
                    self.logger.log_confusion_matrix(cm, "Confusion Matrix: ")
                else:
                    # Create and train model
                    model = ModelFactory.create_model(model_name)
                    with tqdm(total=1, desc=f"Training {model_name}") as pbar:
                        model.train(X_train, y_train)
                        pbar.update(1)
                    
                    # Store model's training losses
                    metrics['training_losses'] = model.train_losses
                    
                    # Calculate training time
                    fold_time = time.time() - fold_start_time
                    self.timing_metrics['fold_times'].append(fold_time)
                    
                    # Log training time
                    self.logger.info(f"Training time: {fold_time:.2f} seconds")
                    
                    # Evaluate model
                    acc, rec, prec, f1, cm = model.evaluate(X_test, y_test)
                    
                    metrics['accuracy'].append(acc)
                    metrics['recall'].append(rec)
                    metrics['precision'].append(prec)
                    metrics['f1_score'].append(f1)
                    metrics['confusion_matrices'].append(cm)
                    
                    if debug_mode:
                        self.debug_logger.log_metrics({
                            'accuracy': acc,
                            'recall': rec,
                            'precision': prec,
                            'f1_score': f1
                        }, "Test Results")
                        self.debug_logger.log_array_info(cm, "Test Confusion Matrix")
                    
                    self.logger.info("Test results:")
                    self.logger.info(f"Accuracy: {acc:.4f}")
                    self.logger.info(f"Recall: {rec:.4f}")
                    self.logger.info(f"Precision: {prec:.4f}")
                    self.logger.info(f"F1-Score: {f1:.4f}")
                    self.logger.log_confusion_matrix(cm, "Confusion Matrix: ")
            
            else:
                # Perform K-fold cross-validation
                self.logger.info(f"Performing {n_folds}-fold cross-validation...")
                fold_losses = []  # Store losses for each fold
                
                for fold, (X_train, X_val, y_train, y_val) in enumerate(tqdm(self.data_repo.get_kfold_splits(encoded_features, labels, n_folds), 
                                                                            total=n_folds, desc="K-fold Cross Validation")):
                    # Start timing for this fold
                    fold_start_time = time.time()
                    
                    self.logger.info(f"Training fold {fold + 1}/{n_folds}...")
                    
                    if debug_mode:
                        self.debug_logger.log_array_info(X_train, f"Fold {fold + 1} Training Data")
                        self.debug_logger.log_array_info(y_train, f"Fold {fold + 1} Training Labels")
                        self.debug_logger.log_array_info(X_val, f"Fold {fold + 1} Validation Data")
                        self.debug_logger.log_array_info(y_val, f"Fold {fold + 1} Validation Labels")
                    
                    # Special handling for d_lstm_mlp model
                    if model_name == "d_lstm_mlp":
                        acc, rec, prec, f1, cm, training_losses = self._train_d_lstm_mlp(X_train, X_val, y_train, y_val)
                        fold_losses.append(training_losses)
                        
                        # Store metrics
                        metrics['accuracy'].append(acc)
                        metrics['recall'].append(rec)
                        metrics['precision'].append(prec)
                        metrics['f1_score'].append(f1)
                        metrics['confusion_matrices'].append(cm)
                        
                        # Calculate fold training time
                        fold_time = time.time() - fold_start_time
                        self.timing_metrics['fold_times'].append(fold_time)
                        
                        # Log fold timing
                        self.logger.info(f"Fold {fold + 1} training time: {fold_time:.2f} seconds")
                        
                        if debug_mode:
                            self.debug_logger.log_metrics({
                                'accuracy': acc,
                                'recall': rec,
                                'precision': prec,
                                'f1_score': f1
                            }, f"Fold {fold + 1} Results")
                            self.debug_logger.log_array_info(cm, f"Fold {fold + 1} Confusion Matrix")
                        
                        self.logger.info(f"Fold {fold + 1} results:")
                        self.logger.info(f"Accuracy: {acc:.4f}")
                        self.logger.info(f"Recall: {rec:.4f}")
                        self.logger.info(f"Precision: {prec:.4f}")
                        self.logger.info(f"F1-Score: {f1:.4f}")
                        self.logger.log_confusion_matrix(cm, "Confusion Matrix: ")
                    else:
                        # Create and train model
                        model = ModelFactory.create_model(model_name)
                        with tqdm(total=1, desc=f"Training {model_name}") as pbar:
                            model.train(X_train, y_train)
                            pbar.update(1)
                        
                        # Store fold's training losses
                        fold_losses.append(model.train_losses)
                        
                        # Calculate fold training time
                        fold_time = time.time() - fold_start_time
                        self.timing_metrics['fold_times'].append(fold_time)
                        
                        # Log fold timing
                        self.logger.info(f"Fold {fold + 1} training time: {fold_time:.2f} seconds")
                        
                        # Evaluate model
                        acc, rec, prec, f1, cm = model.evaluate(X_val, y_val)
                        
                        metrics['accuracy'].append(acc)
                        metrics['recall'].append(rec)
                        metrics['precision'].append(prec)
                        metrics['f1_score'].append(f1)
                        metrics['confusion_matrices'].append(cm)
                        
                        if debug_mode:
                            self.debug_logger.log_metrics({
                                'accuracy': acc,
                                'recall': rec,
                                'precision': prec,
                                'f1_score': f1
                            }, f"Fold {fold + 1} Results")
                            self.debug_logger.log_array_info(cm, f"Fold {fold + 1} Confusion Matrix")
                        
                        self.logger.info(f"Fold {fold + 1} results:")
                        self.logger.info(f"Accuracy: {acc:.4f}")
                        self.logger.info(f"Recall: {rec:.4f}")
                        self.logger.info(f"Precision: {prec:.4f}")
                        self.logger.info(f"F1-Score: {f1:.4f}")
                        self.logger.log_confusion_matrix(cm, "Confusion Matrix: ")
                
                # Average the losses across folds
                if fold_losses:
                    if model_name == "d_lstm_mlp":
                        # Average LSTM and MLP losses separately
                        try:
                            avg_lstm_losses = np.mean([fold['lstm'] for fold in fold_losses if isinstance(fold, dict) and 'lstm' in fold], axis=0)
                            avg_mlp_losses = np.mean([fold['mlp'] for fold in fold_losses if isinstance(fold, dict) and 'mlp' in fold], axis=0)
                            metrics['training_losses'] = {
                                'lstm': avg_lstm_losses.tolist() if isinstance(avg_lstm_losses, np.ndarray) else [],
                                'mlp': avg_mlp_losses.tolist() if isinstance(avg_mlp_losses, np.ndarray) else []
                            }
                        except Exception as e:
                            self.logger.warning(f"Error averaging d_lstm_mlp losses: {str(e)}")
                            metrics['training_losses'] = {'lstm': [], 'mlp': []}
                    else:
                        try:
                            avg_losses = np.mean(fold_losses, axis=0)
                            metrics['training_losses'] = avg_losses.tolist() if isinstance(avg_losses, np.ndarray) else []
                        except Exception as e:
                            self.logger.warning(f"Error averaging losses: {str(e)}")
                            metrics['training_losses'] = []
            
            # Calculate total training time
            total_time = time.time() - total_start_time
            self.timing_metrics['total_time'] = total_time
            
            # Log timing metrics
            self.logger.info("\nTraining Time Metrics:")
            self.logger.info(f"Total training time: {total_time:.2f} seconds")
            
            # Only calculate and log fold statistics if we have fold times
            if self.timing_metrics['fold_times']:
                avg_fold_time = np.mean(self.timing_metrics['fold_times'])
                min_fold_time = min(self.timing_metrics['fold_times'])
                max_fold_time = max(self.timing_metrics['fold_times'])
                
                self.logger.info(f"Average fold training time: {avg_fold_time:.2f} seconds")
                self.logger.info(f"Min fold training time: {min_fold_time:.2f} seconds")
                self.logger.info(f"Max fold training time: {max_fold_time:.2f} seconds")
            else:
                # For single train/test split, just log the total time
                self.logger.info("Single train/test split - no fold statistics available")
            
            # Calculate and log average metrics
            avg_metrics = {
                'accuracy': np.mean(metrics['accuracy']),
                'recall': np.mean(metrics['recall']),
                'precision': np.mean(metrics['precision']),
                'f1_score': np.mean(metrics['f1_score'])
            }
            
            # Calculate total confusion matrix
            total_cm = np.sum(metrics['confusion_matrices'], axis=0).astype(int)
            metrics['confusion_matrix'] = total_cm
            
            if debug_mode:
                self.debug_logger.log_metrics(avg_metrics, "Average Results")
                self.debug_logger.log_array_info(total_cm, "Total Confusion Matrix")
            
            self.logger.info("\nAverage results across all folds:")
            self.logger.info(f"Accuracy: {avg_metrics['accuracy']:.4f}")
            self.logger.info(f"Recall: {avg_metrics['recall']:.4f}")
            self.logger.info(f"Precision: {avg_metrics['precision']:.4f}")
            self.logger.info(f"F1-Score: {avg_metrics['f1_score']:.4f}")
            
            # Add timing metrics to the returned metrics dictionary
            metrics['timing'] = {
                'total_time': total_time,
                'fold_times': self.timing_metrics['fold_times'],
                'avg_fold_time': np.mean(self.timing_metrics['fold_times']) if self.timing_metrics['fold_times'] else total_time,
                'min_fold_time': min(self.timing_metrics['fold_times']) if self.timing_metrics['fold_times'] else total_time,
                'max_fold_time': max(self.timing_metrics['fold_times']) if self.timing_metrics['fold_times'] else total_time
            }
            
            # After calculating metrics and before returning
            if n_folds > 1:  # Only save k-fold metrics if we're doing k-fold cross validation
                self._save_kfold_metrics(model_name, metrics, n_folds)

            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in loan risk prediction: {str(e)}")
            raise

    def _perform_feature_engineering(self, normalized_data: np.ndarray, labels: np.ndarray, debug_mode: bool) -> np.ndarray:
        """Helper method to perform feature engineering and autoencoder encoding.
        
        Args:
            normalized_data: Normalized input features
            labels: Target labels
            debug_mode: Whether to enable debug logging
            
        Returns:
            Processed features (either weighted features or fused features)
        """
        n_features = self.config.get('features.n_features')
        append_autoencoder_features = self.config.get('feature_engineering.append_autoencoder_features', True)
        append_normalized_data = self.config.get('feature_engineering.append_normalized_features', True)
        
        with tqdm(total=3, desc="Feature Engineering") as pbar:
            # Step 1: Compute Kraskov MI and weight features
            self.logger.info("Computing Kraskov MI and weighting features...")
            weighted_features = self.feature_engineer.process_features(
                normalized_data, labels, n_features
            )
            pbar.update(1)
            if debug_mode:
                self.debug_logger.log_array_info(weighted_features, "Weighted Features")
            
            # Step 2: Autoencoder feature extraction
            if self.config.get('feature_engineering.use_autoencoder', True):
                self.logger.info("Performing autoencoder feature extraction...")

                use_weighted_input = self.config.get('feature_engineering.use_weighted_input_for_autoencoder')
                self.logger.info(f"use {'weighted_data' if use_weighted_input else 'normalized_data'} for autoencoder_input")

                autoencoder_input = weighted_features if use_weighted_input else normalized_data
                autoencoder = AutoencoderModel()

                autoencoder.train(autoencoder_input)
                # autoencoder.train(weighted_features)
                pbar.update(1)

                encoded_features = autoencoder.predict(autoencoder_input)

                pbar.update(1)
                
                if debug_mode:
                    self.debug_logger.log_array_info(encoded_features, "Encoded Features")
                
                # Combine features if configured
                if append_autoencoder_features:
                    final_features = np.hstack([weighted_features, encoded_features])
                    self.logger.info(f"Combined features shape: {final_features.shape}")
                else:
                    final_features = encoded_features
                    self.logger.info(f"Using only autoencoder features, shape: {final_features.shape}")
            else:
                final_features = weighted_features
                pbar.update(2)  # Skip autoencoder steps

            if append_normalized_data:
                final_features = np.hstack([normalized_data, final_features])   
                self.logger.info(f"Append normalized data, shape: {normalized_data.shape}")

            self.logger.info(f"Final features shape: {final_features.shape}")
                

            return final_features

    def run_all_models(self, subsample_rate: float = 1.0, n_folds: int = 5, debug_mode: bool = False) -> Dict[str, Dict[str, List[float]]]:
        """Run all available models with K-fold cross-validation."""
        results = {}
        total_start_time = time.time()
        
        for model_name in ModelFactory.get_available_models():
            try:
                model_start_time = time.time()
                self.logger.info(f"\nRunning {model_name} model...")
                metrics = self.run(model_name, subsample_rate, n_folds, debug_mode)
                model_time = time.time() - model_start_time
                self.logger.info(f"Total time for {model_name}: {model_time:.2f} seconds")
                results[model_name] = metrics
            except Exception as e:
                self.logger.error(f"Error running {model_name}: {str(e)}")
                continue
        
        total_time = time.time() - total_start_time
        self.logger.info(f"\nTotal time for all models: {total_time:.2f} seconds")
        
        return results 
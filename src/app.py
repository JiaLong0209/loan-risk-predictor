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

    def _train_d_lstm_mlp(self, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> Tuple[float, float, float, float, np.ndarray]:
        """Special training method for d_lstm_mlp model that combines D-LSTM and MLP.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            
        Returns:
            Tuple of (accuracy, recall, precision, f1_score, confusion_matrix)
        """
        # Step 1: Train D-LSTM model
        self.logger.info("Training D-LSTM model...")
        d_lstm_model = ModelFactory.create_model("d_lstm")
        d_lstm_model.train(X_train, y_train)
        
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
        
        # Step 4: Evaluate the final model
        return mlp_model.evaluate(X_val_combined, y_val)

    def run(self, model_name: str = None, subsample_rate: float = 1.0, n_folds: int = 5,
            debug_mode: bool = False, use_feature_engineering: bool = True) -> Dict[str, List[float]]:
        """Run the loan risk prediction pipeline with K-fold cross-validation."""
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

            # self.logger.info(f"{processed_dir, self.config.get('data.normalized_data_path')}")

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
                        # self.logger.info(f"Normalied Data Shape{normalized_data.shape}")
                        if (fused_features.shape == expected_fused_feature_shape).all() :
                        # if fused_features.shape[0] == normalized_data.shape[0] :
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
                'confusion_matrices': []
            }
            
            # Get model name from config if not provided
            if model_name is None:
                model_name = self.config.get('models.default_model')
            
            # Perform K-fold cross-validation
            self.logger.info(f"Performing {n_folds}-fold cross-validation...")
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
                    acc, rec, prec, f1, cm = self._train_d_lstm_mlp(X_train, X_val, y_train, y_val)
                else:
                    # Create and train model (original code)
                    model = ModelFactory.create_model(model_name)
                    with tqdm(total=1, desc=f"Training {model_name}") as pbar:
                        model.train(X_train, y_train)
                        pbar.update(1)
                    
                    # Evaluate model
                    acc, rec, prec, f1, cm = model.evaluate(X_val, y_val)
                
                # Calculate fold training time
                fold_time = time.time() - fold_start_time
                self.timing_metrics['fold_times'].append(fold_time)
                
                # Log fold timing
                self.logger.info(f"Fold {fold + 1} training time: {fold_time:.2f} seconds")
                
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
            
            # Calculate total training time
            total_time = time.time() - total_start_time
            self.timing_metrics['total_time'] = total_time
            
            # Log timing metrics
            self.logger.info("\nTraining Time Metrics:")
            self.logger.info(f"Total training time: {total_time:.2f} seconds")
            self.logger.info(f"Average fold training time: {np.mean(self.timing_metrics['fold_times']):.2f} seconds")
            self.logger.info(f"Min fold training time: {min(self.timing_metrics['fold_times']):.2f} seconds")
            self.logger.info(f"Max fold training time: {max(self.timing_metrics['fold_times']):.2f} seconds")
            
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
                'avg_fold_time': np.mean(self.timing_metrics['fold_times']),
                'min_fold_time': min(self.timing_metrics['fold_times']),
                'max_fold_time': max(self.timing_metrics['fold_times'])
            }
            
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
"""
LSTM-based time series forecasting module.

This module provides a robust LSTM implementation for financial
time series forecasting with advanced features and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime, timedelta
import warnings

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. LSTM functionality will be limited.")

# Scikit-learn imports
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. LSTM functionality will be limited.")

logger = logging.getLogger(__name__)


class LSTMForecaster:
    """
    Advanced LSTM-based time series forecaster for financial data.
    
    Implements a robust LSTM architecture with comprehensive
    preprocessing, training, and forecasting capabilities.
    
    Attributes:
        lookback_days (int): Number of days to look back for sequences
        model (keras.Model): Trained LSTM model
        scaler (MinMaxScaler): Data scaler for preprocessing
        is_trained (bool): Whether the model has been trained
        model_config (Dict): Model configuration parameters
    """
    
    def __init__(self, lookback_days: int = 60, **kwargs):
        """
        Initialize the LSTM Forecaster.
        
        Args:
            lookback_days: Number of days to look back for sequences
            **kwargs: Additional model configuration parameters
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM functionality")
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for LSTM functionality")
        
        self.lookback_days = lookback_days
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.model_config = {
            'lookback_days': lookback_days,
            'lstm_units': kwargs.get('lstm_units', [50, 50, 50]),
            'dropout_rate': kwargs.get('dropout_rate', 0.2),
            'learning_rate': kwargs.get('learning_rate', 0.001),
            'batch_size': kwargs.get('batch_size', 32),
            'epochs': kwargs.get('epochs', 100),
            'validation_split': kwargs.get('validation_split', 0.2),
            'patience': kwargs.get('patience', 10),
            'min_delta': kwargs.get('min_delta', 0.001)
        }
        
        logger.info(f"Initialized LSTM Forecaster with {lookback_days} day lookback")
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'Close',
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training and testing.
        
        Args:
            data: DataFrame with time series data
            target_column: Column to use as target variable
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test) arrays
            
        Raises:
            ValueError: If data is insufficient or invalid
        """
        if len(data) < self.lookback_days + 1:
            raise ValueError(f"Data must have at least {self.lookback_days + 1} observations")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Extract target variable
        target_data = data[target_column].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(target_data)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        # Split into train and test sets
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Prepared data: {len(X_train)} training, {len(X_test)} testing sequences")
        
        return X_train, y_train, X_test, y_test
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Scaled time series data
            
        Returns:
            Tuple of (X, y) arrays for LSTM training
        """
        X, y = [], []
        
        for i in range(self.lookback_days, len(data)):
            X.append(data[i - self.lookback_days:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (lookback_days, features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.model_config['lstm_units'][0],
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(self.model_config['dropout_rate']))
        
        # Additional LSTM layers
        for units in self.model_config['lstm_units'][1:]:
            model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(self.model_config['dropout_rate']))
        
        # Final LSTM layer
        model.add(LSTM(self.model_config['lstm_units'][-1]))
        model.add(Dropout(self.model_config['dropout_rate']))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.model_config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Built LSTM model with {model.count_params()} parameters")
        return model
    
    def train(self, data: pd.DataFrame, target_column: str = 'Close',
              test_size: float = 0.2, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            data: DataFrame with time series data
            target_column: Column to use as target variable
            test_size: Proportion of data to use for testing
            verbose: Whether to show training progress
            
        Returns:
            Dictionary with training history and metrics
            
        Raises:
            RuntimeError: If training fails
        """
        try:
            # Prepare data
            X_train, y_train, X_test, y_test = self.prepare_data(data, target_column, test_size)
            
            # Build model
            self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Set up callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.model_config['patience'],
                    min_delta=self.model_config['min_delta'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.model_config['patience'] // 2,
                    min_lr=1e-7
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.model_config['batch_size'],
                epochs=self.model_config['epochs'],
                validation_split=self.model_config['validation_split'],
                callbacks=callbacks,
                verbose=verbose
            )
            
            # Evaluate model
            train_loss, train_mae = self.model.evaluate(X_train, y_train, verbose=0)
            test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
            
            # Make predictions
            y_train_pred = self.predict(data.iloc[:len(X_train) + self.lookback_days], target_column)
            y_test_pred = self.predict(data.iloc[len(X_train):], target_column)
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                y_train, y_train_pred, y_test, y_test_pred
            )
            
            # Store test data for later use
            self._test_data = {
                'X_test': X_test,
                'y_test': y_test,
                'y_test_pred': y_test_pred
            }
            
            self.is_trained = True
            
            training_results = {
                'history': history.history,
                'metrics': metrics,
                'model_summary': self.model.summary(),
                'training_date': datetime.now().isoformat()
            }
            
            logger.info(f"LSTM model training completed. Test MAE: {test_mae:.4f}")
            return training_results
            
        except Exception as e:
            logger.error(f"LSTM training failed: {str(e)}")
            raise RuntimeError(f"LSTM training failed: {str(e)}")
    
    def predict(self, data: pd.DataFrame, target_column: str = 'Close') -> np.ndarray:
        """
        Make predictions using the trained LSTM model.
        
        Args:
            data: DataFrame with time series data
            target_column: Column to use as target variable
            
        Returns:
            Array of predictions
            
        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before making predictions")
        
        if len(data) < self.lookback_days:
            raise ValueError(f"Data must have at least {self.lookback_days} observations")
        
        # Extract target variable
        target_data = data[target_column].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.transform(target_data)
        
        # Create sequences for prediction
        X_pred = []
        for i in range(self.lookback_days, len(scaled_data)):
            X_pred.append(scaled_data[i - self.lookback_days:i])
        
        X_pred = np.array(X_pred)
        
        # Make predictions
        scaled_predictions = self.model.predict(X_pred, verbose=0)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(scaled_predictions)
        
        return predictions.flatten()
    
    def forecast_future(self, data: pd.DataFrame, target_column: str = 'Close',
                       periods: int = 30) -> pd.DataFrame:
        """
        Forecast future values beyond the available data.
        
        Args:
            data: DataFrame with historical time series data
            target_column: Column to use as target variable
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecasted values and confidence intervals
            
        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before making forecasts")
        
        # Get the last lookback_days of data
        last_data = data[target_column].tail(self.lookback_days).values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.transform(last_data)
        
        forecasts = []
        current_sequence = scaled_data.copy()
        
        for _ in range(periods):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, self.lookback_days, 1)
            
            # Make prediction
            scaled_pred = self.model.predict(X_pred, verbose=0)
            
            # Inverse transform
            pred = self.scaler.inverse_transform(scaled_pred)[0, 0]
            forecasts.append(pred)
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = scaled_pred
        
        # Create forecast DataFrame
        last_date = data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecasts
        })
        forecast_df.set_index('Date', inplace=True)
        
        # Add confidence intervals (simplified approach)
        forecast_df['Lower_95'] = forecast_df['Forecast'] * 0.95
        forecast_df['Upper_95'] = forecast_df['Forecast'] * 1.05
        
        logger.info(f"Generated {periods} period forecast")
        return forecast_df
    
    def _calculate_metrics(self, y_train: np.ndarray, y_train_pred: np.ndarray,
                          y_test: np.ndarray, y_test_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive model performance metrics.
        
        Args:
            y_train: Actual training values
            y_train_pred: Predicted training values
            y_test: Actual test values
            y_test_pred: Predicted test values
            
        Returns:
            Dictionary with performance metrics
        """
        # Training metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Test metrics
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Calculate MAPE
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        
        # Directional accuracy
        train_direction = np.mean(np.sign(np.diff(y_train)) == np.sign(np.diff(y_train_pred)))
        test_direction = np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(y_test_pred)))
        
        metrics = {
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'train_r2': float(train_r2),
            'train_mape': float(train_mape),
            'train_directional_accuracy': float(train_direction),
            'test_mae': float(test_mae),
            'test_rmse': float(test_rmse),
            'test_r2': float(test_r2),
            'test_mape': float(test_mape),
            'test_directional_accuracy': float(test_direction)
        }
        
        return metrics
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model and scaler.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("No trained model to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = filepath.with_suffix('.h5')
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = filepath.with_suffix('.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save configuration
        config_path = filepath.with_suffix('.json')
        config_data = {
            'model_config': self.model_config,
            'lookback_days': self.lookback_days,
            'training_date': datetime.now().isoformat(),
            'is_trained': self.is_trained
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load a trained model and scaler.
        
        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)
        
        # Load model
        model_path = filepath.with_suffix('.h5')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = load_model(model_path)
        
        # Load scaler
        scaler_path = filepath.with_suffix('.pkl')
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load configuration
        config_path = filepath.with_suffix('.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                self.model_config.update(config_data.get('model_config', {}))
                self.lookback_days = config_data.get('lookback_days', self.lookback_days)
                self.is_trained = config_data.get('is_trained', True)
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': 'LSTM',
            'lookback_days': self.lookback_days,
            'is_trained': self.is_trained,
            'model_config': self.model_config,
            'scaler_type': type(self.scaler).__name__
        }
        
        if self.model is not None:
            info.update({
                'total_parameters': self.model.count_params(),
                'model_layers': len(self.model.layers),
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape
            })
        
        return info
    
    def plot_training_history(self) -> None:
        """
        Plot the training history (loss and metrics).
        
        Note: This method requires matplotlib to be available.
        """
        try:
            import matplotlib.pyplot as plt
            
            if not hasattr(self, '_test_data') or 'history' not in self._test_data:
                logger.warning("No training history available for plotting")
                return
            
            history = self._test_data['history']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot loss
            ax1.plot(history['loss'], label='Training Loss')
            ax1.plot(history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot MAE
            ax2.plot(history['mae'], label='Training MAE')
            ax2.plot(history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")

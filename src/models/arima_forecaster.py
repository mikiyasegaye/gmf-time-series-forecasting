"""
ARIMA-based time series forecasting module.

This module provides a robust ARIMA implementation for financial
time series forecasting with automatic parameter optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime
import warnings

# Statsmodels imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn(
        "Statsmodels not available. ARIMA functionality will be limited.")

# PMDARIMA imports (for auto-ARIMA)
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except (ImportError, ValueError):
    PMDARIMA_AVAILABLE = False
    warnings.warn(
        "PMDARIMA not available due to compatibility issues. Auto-ARIMA functionality will be limited.")

# Scikit-learn imports
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "Scikit-learn not available. ARIMA functionality will be limited.")

logger = logging.getLogger(__name__)


class ARIMAForecaster:
    """
    Advanced ARIMA-based time series forecaster for financial data.

    Implements robust ARIMA modeling with automatic parameter
    optimization and comprehensive diagnostics.

    Attributes:
        order (Tuple[int, int, int]): ARIMA order (p, d, q)
        model (ARIMA): Fitted ARIMA model
        is_fitted (bool): Whether the model has been fitted
        model_config (Dict): Model configuration parameters
        diagnostics (Dict): Model diagnostic results
    """

    def __init__(self, order: Optional[Tuple[int, int, int]] = None, **kwargs):
        """
        Initialize the ARIMA Forecaster.

        Args:
            order: ARIMA order (p, d, q). If None, will be auto-determined
            **kwargs: Additional model configuration parameters
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "Statsmodels is required for ARIMA functionality")
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "Scikit-learn is required for ARIMA functionality")

        self.order = order
        self.model = None
        self.is_fitted = False
        self.model_config = {
            'order': order,
            'auto_optimize': kwargs.get('auto_optimize', True),
            'seasonal': kwargs.get('seasonal', False),
            'seasonal_order': kwargs.get('seasonal_order', (0, 0, 0, 0)),
            'max_p': kwargs.get('max_p', 5),
            'max_d': kwargs.get('max_d', 2),
            'max_q': kwargs.get('max_q', 5),
            'information_criterion': kwargs.get('information_criterion', 'aic'),
            'stepwise': kwargs.get('stepwise', True),
            'suppress_warnings': kwargs.get('suppress_warnings', True),
            'error_action': kwargs.get('error_action', 'ignore'),
            'trace': kwargs.get('trace', False)
        }
        self.diagnostics = {}

        logger.info(f"Initialized ARIMA Forecaster with order: {order}")

    def prepare_data(self, data: pd.DataFrame, target_column: str = 'Close',
                     test_size: float = 0.2) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare data for ARIMA training and testing.

        Args:
            data: DataFrame with time series data
            target_column: Column to use as target variable
            test_size: Proportion of data to use for testing

        Returns:
            Tuple of (train_series, test_series)

        Raises:
            ValueError: If data is insufficient or invalid
        """
        if len(data) < 50:  # Minimum observations for ARIMA
            raise ValueError("Data must have at least 50 observations")

        if target_column not in data.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in data")

        # Extract target variable
        target_series = data[target_column]

        # Check for stationarity
        is_stationary = self._check_stationarity(target_series)
        if not is_stationary:
            logger.warning("Data is not stationary. Consider differencing.")

        # Split into train and test sets
        split_idx = int(len(target_series) * (1 - test_size))
        train_series = target_series.iloc[:split_idx]
        test_series = target_series.iloc[split_idx:]

        logger.info(
            f"Prepared data: {len(train_series)} training, {len(test_series)} testing observations")

        return train_series, test_series

    def _check_stationarity(self, series: pd.Series, significance_level: float = 0.05) -> bool:
        """
        Check if a time series is stationary using ADF test.

        Args:
            series: Time series to test
            significance_level: Significance level for the test

        Returns:
            True if stationary, False otherwise
        """
        try:
            # Perform Augmented Dickey-Fuller test
            adf_result = adfuller(series.dropna())
            p_value = adf_result[1]

            is_stationary = p_value < significance_level

            # Store diagnostic information
            self.diagnostics['adf_test'] = {
                'statistic': adf_result[0],
                'p_value': p_value,
                'critical_values': adf_result[4],
                'is_stationary': is_stationary
            }

            return is_stationary

        except Exception as e:
            logger.warning(f"Error in stationarity test: {str(e)}")
            return True  # Assume stationary if test fails

    def _determine_differencing(self, series: pd.Series, max_d: int = 2) -> int:
        """
        Determine the optimal differencing order for stationarity.

        Args:
            series: Time series to analyze
            max_d: Maximum differencing order to try

        Returns:
            Optimal differencing order
        """
        current_series = series.copy()

        for d in range(max_d + 1):
            if d == 0:
                if self._check_stationarity(current_series):
                    return d
            else:
                current_series = current_series.diff().dropna()
                if self._check_stationarity(current_series):
                    return d

        return max_d

    def _auto_optimize_order(self, train_series: pd.Series) -> Tuple[int, int, int]:
        """
        Automatically optimize ARIMA order using PMDARIMA or grid search.

        Args:
            train_series: Training time series data

        Returns:
            Optimal ARIMA order (p, d, q)
        """
        if PMDARIMA_AVAILABLE and self.model_config['auto_optimize']:
            try:
                logger.info(
                    "Using PMDARIMA for automatic order optimization...")

                # Use auto_arima to find optimal parameters
                auto_model = auto_arima(
                    train_series,
                    start_p=0, start_q=0,
                    max_p=self.model_config['max_p'],
                    max_d=self.model_config['max_d'],
                    max_q=self.model_config['max_q'],
                    seasonal=self.model_config['seasonal'],
                    seasonal_order=self.model_config['seasonal_order'],
                    information_criterion=self.model_config['information_criterion'],
                    stepwise=self.model_config['stepwise'],
                    suppress_warnings=self.model_config['suppress_warnings'],
                    error_action=self.model_config['error_action'],
                    trace=self.model_config['trace']
                )

                optimal_order = auto_model.order
                logger.info(f"PMDARIMA optimal order: {optimal_order}")
                return optimal_order

            except Exception as e:
                logger.warning(
                    f"PMDARIMA optimization failed: {str(e)}. Falling back to manual optimization.")

        # Manual optimization fallback
        logger.info("Using manual order optimization...")

        # Determine differencing order
        d = self._determine_differencing(
            train_series, self.model_config['max_d'])

        # Simple grid search for p and q
        best_aic = float('inf')
        best_order = (1, d, 1)

        for p in range(self.model_config['max_p'] + 1):
            for q in range(self.model_config['max_q'] + 1):
                try:
                    temp_model = ARIMA(train_series, order=(p, d, q))
                    temp_fitted = temp_model.fit()

                    if temp_fitted.aic < best_aic:
                        best_aic = temp_fitted.aic
                        best_order = (p, d, q)

                except Exception:
                    continue

        logger.info(
            f"Manual optimization optimal order: {best_order} (AIC: {best_aic:.2f})")
        return best_order

    def fit(self, data: pd.DataFrame, target_column: str = 'Close',
            test_size: float = 0.2, verbose: bool = True) -> Dict[str, Any]:
        """
        Fit the ARIMA model to the data.

        Args:
            data: DataFrame with time series data
            target_column: Column to use as target variable
            test_size: Proportion of data to use for testing
            verbose: Whether to show fitting progress

        Returns:
            Dictionary with fitting results and diagnostics

        Raises:
            RuntimeError: If fitting fails
        """
        try:
            # Prepare data
            train_series, test_series = self.prepare_data(
                data, target_column, test_size)

            # Determine optimal order if not specified
            if self.order is None:
                self.order = self._auto_optimize_order(train_series)
                self.model_config['order'] = self.order

            # Fit ARIMA model
            logger.info(f"Fitting ARIMA{self.order} model...")

            self.model = ARIMA(train_series, order=self.order)
            fitted_model = self.model.fit()

            # Store fitted model
            self.model = fitted_model
            self.is_fitted = True

            # Make predictions
            train_pred = self.predict(
                data.iloc[:len(train_series)], target_column)
            test_pred = self.predict(
                data.iloc[len(train_series):], target_column)

            # Calculate metrics
            metrics = self._calculate_metrics(
                train_series, train_pred, test_series, test_pred
            )

            # Store test data for later use
            self._test_data = {
                'train_series': train_series,
                'test_series': test_series,
                'train_pred': train_pred,
                'test_pred': test_pred
            }

            # Perform model diagnostics
            diagnostics = self._perform_diagnostics(train_series)

            fitting_results = {
                'order': self.order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'metrics': metrics,
                'diagnostics': diagnostics,
                'fitting_date': datetime.now().isoformat()
            }

            logger.info(
                f"ARIMA{self.order} model fitting completed. AIC: {fitted_model.aic:.2f}")
            return fitting_results

        except Exception as e:
            logger.error(f"ARIMA fitting failed: {str(e)}")
            raise RuntimeError(f"ARIMA fitting failed: {str(e)}")

    def predict(self, data: pd.DataFrame, target_column: str = 'Close') -> pd.Series:
        """
        Make predictions using the fitted ARIMA model.

        Args:
            data: DataFrame with time series data
            target_column: Column to use as target variable

        Returns:
            Series of predictions

        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError(
                "Model must be fitted before making predictions")

        if target_column not in data.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in data")

        # Extract target variable
        target_series = data[target_column]

        # Make predictions
        predictions = self.model.predict(
            start=0,
            end=len(target_series) - 1,
            dynamic=False
        )

        # Align predictions with original data
        predictions.index = target_series.index

        return predictions

    def forecast_future(self, periods: int = 30, confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Forecast future values beyond the available data.

        Args:
            periods: Number of periods to forecast
            confidence_level: Confidence level for prediction intervals

        Returns:
            DataFrame with forecasted values and confidence intervals

        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before making forecasts")

        # Generate forecast
        forecast_result = self.model.forecast(
            steps=periods,
            alpha=1 - confidence_level
        )

        # Extract forecast values and confidence intervals
        forecast_values = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()

        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Forecast': forecast_values,
            'Lower_CI': conf_int.iloc[:, 0],
            'Upper_CI': conf_int.iloc[:, 1]
        })

        # Add confidence level information
        forecast_df.attrs['confidence_level'] = confidence_level

        logger.info(
            f"Generated {periods} period forecast with {confidence_level*100}% confidence")
        return forecast_df

    def _calculate_metrics(self, train_series: pd.Series, train_pred: pd.Series,
                           test_series: pd.Series, test_pred: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive model performance metrics.

        Args:
            train_series: Actual training values
            train_pred: Predicted training values
            test_series: Actual test values
            test_pred: Predicted test values

        Returns:
            Dictionary with performance metrics
        """
        # Training metrics
        train_mae = mean_absolute_error(train_series, train_pred)
        train_rmse = np.sqrt(mean_squared_error(train_series, train_pred))
        train_r2 = r2_score(train_series, train_pred)

        # Test metrics
        test_mae = mean_absolute_error(test_series, test_pred)
        test_rmse = np.sqrt(mean_squared_error(test_series, test_pred))
        test_r2 = r2_score(test_series, test_pred)

        # Calculate MAPE
        train_mape = np.mean(
            np.abs((train_series - train_pred) / train_series)) * 100
        test_mape = np.mean(
            np.abs((test_series - test_pred) / test_series)) * 100

        # Directional accuracy
        train_direction = np.mean(
            np.sign(np.diff(train_series)) == np.sign(np.diff(train_pred)))
        test_direction = np.mean(
            np.sign(np.diff(test_series)) == np.sign(np.diff(test_pred)))

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

    def _perform_diagnostics(self, residuals: pd.Series) -> Dict[str, Any]:
        """
        Perform comprehensive model diagnostics.

        Args:
            residuals: Model residuals

        Returns:
            Dictionary with diagnostic results
        """
        diagnostics = {}

        try:
            # Ljung-Box test for autocorrelation
            lb_test = acorr_ljungbox(
                residuals.dropna(), lags=10, return_df=True)
            diagnostics['ljung_box'] = {
                'statistic': lb_test['lb_stat'].iloc[-1],
                'p_value': lb_test['lb_pvalue'].iloc[-1]
            }

            # Residual statistics
            diagnostics['residual_stats'] = {
                'mean': float(residuals.mean()),
                'std': float(residuals.std()),
                'skewness': float(residuals.skew()),
                'kurtosis': float(residuals.kurtosis())
            }

            # Normality test (Jarque-Bera approximation)
            n = len(residuals)
            skewness = residuals.skew()
            kurtosis = residuals.kurtosis()

            jb_stat = n * (skewness**2 / 6 + (kurtosis - 3)**2 / 24)
            diagnostics['jarque_bera'] = {
                'statistic': float(jb_stat),
                # Approximate p-value
                'p_value': float(1 - np.exp(-jb_stat / 2))
            }

        except Exception as e:
            logger.warning(f"Error in diagnostics: {str(e)}")
            diagnostics['error'] = str(e)

        return diagnostics

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the fitted ARIMA model.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("No fitted model to save")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = filepath.with_suffix('.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save configuration and diagnostics
        config_path = filepath.with_suffix('.json')
        config_data = {
            'order': self.order,
            'model_config': self.model_config,
            'diagnostics': self.diagnostics,
            'fitting_date': datetime.now().isoformat(),
            'is_fitted': self.is_fitted
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"ARIMA model saved to {filepath}")

    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load a fitted ARIMA model.

        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)

        # Load model
        model_path = filepath.with_suffix('.pkl')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load configuration
        config_path = filepath.with_suffix('.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                self.order = config_data.get('order')
                self.model_config.update(config_data.get('model_config', {}))
                self.diagnostics = config_data.get('diagnostics', {})
                self.is_fitted = config_data.get('is_fitted', True)

        logger.info(f"ARIMA model loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.

        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': 'ARIMA',
            'order': self.order,
            'is_fitted': self.is_fitted,
            'model_config': self.model_config,
            'diagnostics': self.diagnostics
        }

        if self.model is not None:
            info.update({
                'aic': float(self.model.aic),
                'bic': float(self.model.bic),
                'hqic': float(self.model.hqic),
                'llf': float(self.model.llf),
                'nobs': int(self.model.nobs)
            })

        return info

    def plot_diagnostics(self) -> None:
        """
        Plot comprehensive model diagnostics.

        Note: This method requires matplotlib to be available.
        """
        try:
            import matplotlib.pyplot as plt

            if not self.is_fitted or self.model is None:
                logger.warning("No fitted model available for diagnostics")
                return

            # Create diagnostic plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Residuals plot
            residuals = self.model.resid
            axes[0, 0].plot(residuals)
            axes[0, 0].set_title('Residuals')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].grid(True)

            # Residuals histogram
            axes[0, 1].hist(residuals, bins=30, alpha=0.7)
            axes[0, 1].set_title('Residuals Distribution')
            axes[0, 1].set_xlabel('Residuals')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)

            # ACF plot
            plot_acf(residuals.dropna(), ax=axes[1, 0], lags=40)
            axes[1, 0].set_title('Autocorrelation Function')

            # PACF plot
            plot_pacf(residuals.dropna(), ax=axes[1, 1], lags=40)
            axes[1, 1].set_title('Partial Autocorrelation Function')

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting diagnostics: {str(e)}")

"""
Model evaluation and comparison module.

This module provides comprehensive evaluation and comparison
capabilities for time series forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import json
from datetime import datetime
import warnings

# Scikit-learn imports
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Model evaluation functionality will be limited.")

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for time series forecasting models.
    
    Provides detailed performance analysis, model comparison,
    and statistical validation for forecasting models.
    
    Attributes:
        models (Dict): Dictionary of models to evaluate
        evaluation_results (Dict): Comprehensive evaluation results
        comparison_metrics (pd.DataFrame): Model comparison metrics
    """
    
    def __init__(self):
        """
        Initialize the Model Evaluator.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for model evaluation functionality")
        
        self.models = {}
        self.evaluation_results = {}
        self.comparison_metrics = pd.DataFrame()
        
        logger.info("Initialized Model Evaluator")
    
    def add_model(self, name: str, model: Any, model_type: str = "Unknown") -> None:
        """
        Add a model to the evaluator.
        
        Args:
            name: Unique name for the model
            model: Model object to evaluate
            model_type: Type of the model (e.g., 'LSTM', 'ARIMA')
        """
        self.models[name] = {
            'model': model,
            'type': model_type,
            'added_date': datetime.now().isoformat()
        }
        
        logger.info(f"Added model '{name}' ({model_type}) to evaluator")
    
    def evaluate_model(self, name: str, y_true: Union[pd.Series, np.ndarray],
                      y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate a single model's performance.
        
        Args:
            name: Name of the model to evaluate
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
            
        Raises:
            KeyError: If model name not found
            ValueError: If input data is invalid
        """
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found in evaluator")
        
        # Convert to numpy arrays for consistency
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError("True and predicted values must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Input data cannot be empty")
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(y_true, y_pred)
        
        # Store evaluation results
        self.evaluation_results[name] = {
            'metrics': metrics,
            'evaluation_date': datetime.now().isoformat(),
            'data_points': len(y_true)
        }
        
        logger.info(f"Evaluated model '{name}' with {len(y_true)} data points")
        return metrics
    
    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with all performance metrics
        """
        # Basic error metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # R-squared
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # Custom metrics
        mape_robust = np.median(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Directional accuracy
        direction_accuracy = self._calculate_directional_accuracy(y_true, y_pred)
        
        # Volatility accuracy
        volatility_accuracy = self._calculate_volatility_accuracy(y_true, y_pred)
        
        # Trend accuracy
        trend_accuracy = self._calculate_trend_accuracy(y_true, y_pred)
        
        # Maximum error
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Mean error (bias)
        mean_error = np.mean(y_true - y_pred)
        
        # Error standard deviation
        error_std = np.std(y_true - y_pred)
        
        # Theil's U statistic
        theil_u = self._calculate_theil_u(y_true, y_pred)
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'mape_robust': float(mape_robust),
            'directional_accuracy': float(direction_accuracy),
            'volatility_accuracy': float(volatility_accuracy),
            'trend_accuracy': float(trend_accuracy),
            'max_error': float(max_error),
            'mean_error': float(mean_error),
            'error_std': float(error_std),
            'theil_u': float(theil_u)
        }
        
        return metrics
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (percentage of correct direction predictions).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Directional accuracy as a percentage
        """
        if len(y_true) < 2:
            return 0.0
        
        # Calculate direction changes
        true_direction = np.diff(y_true)
        pred_direction = np.diff(y_pred)
        
        # Count correct directions
        correct_directions = np.sum(np.sign(true_direction) == np.sign(pred_direction))
        total_directions = len(true_direction)
        
        return (correct_directions / total_directions) * 100 if total_directions > 0 else 0.0
    
    def _calculate_volatility_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate volatility accuracy (how well the model captures volatility).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Volatility accuracy score
        """
        if len(y_true) < 2:
            return 0.0
        
        # Calculate rolling volatility (20-period)
        window = min(20, len(y_true) // 4)
        if window < 2:
            return 0.0
        
        true_vol = pd.Series(y_true).rolling(window=window).std().dropna()
        pred_vol = pd.Series(y_pred).rolling(window=window).std().dropna()
        
        if len(true_vol) == 0:
            return 0.0
        
        # Calculate correlation between true and predicted volatility
        correlation = np.corrcoef(true_vol, pred_vol)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _calculate_trend_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate trend accuracy (how well the model captures trends).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Trend accuracy score
        """
        if len(y_true) < 10:
            return 0.0
        
        # Calculate linear trend using simple linear regression
        x = np.arange(len(y_true))
        
        # True trend
        true_trend_coef = np.polyfit(x, y_true, 1)[0]
        
        # Predicted trend
        pred_trend_coef = np.polyfit(x, y_pred, 1)[0]
        
        # Calculate trend accuracy
        if true_trend_coef == 0:
            return 100.0 if pred_trend_coef == 0 else 0.0
        
        trend_accuracy = (1 - abs(pred_trend_coef - true_trend_coef) / abs(true_trend_coef)) * 100
        return max(0.0, min(100.0, trend_accuracy))
    
    def _calculate_theil_u(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Theil's U statistic (forecast accuracy measure).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Theil's U statistic
        """
        # Calculate Theil's U
        numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
        denominator = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
        
        if denominator == 0:
            return 1.0
        
        theil_u = numerator / denominator
        return float(theil_u)
    
    def compare_models(self, metric: str = 'rmse', ascending: bool = True) -> pd.DataFrame:
        """
        Compare all evaluated models based on a specific metric.
        
        Args:
            metric: Metric to use for comparison
            ascending: Whether to sort in ascending order
            
        Returns:
            DataFrame with model comparison results
            
        Raises:
            ValueError: If no models have been evaluated
        """
        if not self.evaluation_results:
            raise ValueError("No models have been evaluated yet")
        
        # Create comparison DataFrame
        comparison_data = []
        
        for name, results in self.evaluation_results.items():
            model_info = self.models[name]
            metrics = results['metrics']
            
            row = {
                'Model': name,
                'Type': model_info['type'],
                'Data_Points': results['data_points'],
                'Evaluation_Date': results['evaluation_date']
            }
            
            # Add all metrics
            row.update(metrics)
            
            comparison_data.append(row)
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by specified metric
        if metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(by=metric, ascending=ascending)
        
        self.comparison_metrics = comparison_df
        
        logger.info(f"Generated model comparison for {len(comparison_df)} models")
        return comparison_df
    
    def get_best_model(self, metric: str = 'rmse') -> Tuple[str, Dict]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for ranking (lower is better for error metrics)
            
        Returns:
            Tuple of (model_name, model_results)
            
        Raises:
            ValueError: If no models have been evaluated
        """
        if not self.evaluation_results:
            raise ValueError("No models have been evaluated yet")
        
        # Get comparison results
        comparison_df = self.compare_models(metric=metric, ascending=True)
        
        if comparison_df.empty:
            raise ValueError("No valid comparison data available")
        
        # Get best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_model_results = self.evaluation_results[best_model_name]
        
        logger.info(f"Best model by {metric}: {best_model_name}")
        return best_model_name, best_model_results
    
    def generate_evaluation_report(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report (optional)
            
        Returns:
            Report content as string
        """
        if not self.evaluation_results:
            return "No models have been evaluated yet."
        
        # Generate report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Models Evaluated: {len(self.evaluation_results)}")
        report_lines.append("")
        
        # Model summary
        report_lines.append("MODEL SUMMARY:")
        report_lines.append("-" * 40)
        for name, info in self.models.items():
            report_lines.append(f"{name} ({info['type']}) - Added: {info['added_date']}")
        report_lines.append("")
        
        # Performance comparison
        if not self.comparison_metrics.empty:
            report_lines.append("PERFORMANCE COMPARISON:")
            report_lines.append("-" * 40)
            
            # Key metrics comparison
            key_metrics = ['mae', 'rmse', 'mape', 'r2', 'directional_accuracy']
            for metric in key_metrics:
                if metric in self.comparison_metrics.columns:
                    best_model = self.comparison_metrics.iloc[0]
                    report_lines.append(f"Best {metric.upper()}: {best_model['Model']} = {best_model[metric]:.4f}")
            
            report_lines.append("")
        
        # Detailed results
        report_lines.append("DETAILED RESULTS:")
        report_lines.append("-" * 40)
        
        for name, results in self.evaluation_results.items():
            report_lines.append(f"\n{name.upper()}:")
            report_lines.append(f"  Data Points: {results['data_points']}")
            report_lines.append(f"  Evaluation Date: {results['evaluation_date']}")
            
            metrics = results['metrics']
            report_lines.append("  Key Metrics:")
            for metric, value in metrics.items():
                if metric in ['mae', 'rmse', 'mape', 'r2', 'directional_accuracy']:
                    report_lines.append(f"    {metric.upper()}: {value:.4f}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_content = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report_content
    
    def plot_model_comparison(self, metrics: List[str] = None) -> None:
        """
        Plot model comparison charts.
        
        Args:
            metrics: List of metrics to plot (default: key metrics)
            
        Note: This method requires matplotlib to be available.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if self.comparison_metrics.empty:
                logger.warning("No comparison data available for plotting")
                return
            
            # Default metrics to plot
            if metrics is None:
                metrics = ['mae', 'rmse', 'mape', 'r2', 'directional_accuracy']
            
            # Filter available metrics
            available_metrics = [m for m in metrics if m in self.comparison_metrics.columns]
            
            if not available_metrics:
                logger.warning("No valid metrics available for plotting")
                return
            
            # Create subplots
            n_metrics = len(available_metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
            
            if n_metrics == 1:
                axes = [axes]
            
            # Plot each metric
            for i, metric in enumerate(available_metrics):
                ax = axes[i]
                
                # Create bar plot
                values = self.comparison_metrics[metric]
                models = self.comparison_metrics['Model']
                
                bars = ax.bar(range(len(models)), values)
                ax.set_title(f'{metric.upper()} Comparison')
                ax.set_xlabel('Models')
                ax.set_ylabel(metric.upper())
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=45, ha='right')
                
                # Color bars based on performance
                if metric in ['mae', 'rmse', 'mape']:  # Lower is better
                    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(bars)))
                else:  # Higher is better
                    colors = plt.cm.Greens(np.linspace(0.3, 0.8, len(bars)))
                
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting model comparison: {str(e)}")
    
    def export_results(self, output_path: Union[str, Path]) -> None:
        """
        Export evaluation results to JSON file.
        
        Args:
            output_path: Path to save the results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare export data
        export_data = {
            'export_date': datetime.now().isoformat(),
            'models': self.models,
            'evaluation_results': self.evaluation_results,
            'comparison_metrics': self.comparison_metrics.to_dict('records') if not self.comparison_metrics.empty else []
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Evaluation results exported to {output_path}")
    
    def import_results(self, input_path: Union[str, Path]) -> None:
        """
        Import evaluation results from JSON file.
        
        Args:
            input_path: Path to the results file
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Results file not found: {input_path}")
        
        # Load from JSON
        with open(input_path, 'r') as f:
            import_data = json.load(f)
        
        # Restore data
        self.models = import_data.get('models', {})
        self.evaluation_results = import_data.get('evaluation_results', {})
        
        # Restore comparison metrics
        comparison_data = import_data.get('comparison_metrics', [])
        if comparison_data:
            self.comparison_metrics = pd.DataFrame(comparison_data)
        
        logger.info(f"Evaluation results imported from {input_path}")
    
    def get_statistical_summary(self) -> Dict[str, Any]:
        """
        Get statistical summary of all model performances.
        
        Returns:
            Dictionary with statistical summary
        """
        if not self.evaluation_results:
            return {}
        
        # Collect all metrics
        all_metrics = {}
        for name, results in self.evaluation_results.items():
            for metric, value in results['metrics'].items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate statistics
        summary = {}
        for metric, values in all_metrics.items():
            values = np.array(values)
            summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return summary
    
    def clear_results(self) -> None:
        """
        Clear all evaluation results and comparison data.
        """
        self.evaluation_results.clear()
        self.comparison_metrics = pd.DataFrame()
        logger.info("Cleared all evaluation results")
    
    def remove_model(self, name: str) -> None:
        """
        Remove a model from the evaluator.
        
        Args:
            name: Name of the model to remove
        """
        if name in self.models:
            del self.models[name]
            logger.info(f"Removed model '{name}' from evaluator")
        
        if name in self.evaluation_results:
            del self.evaluation_results[name]
            logger.info(f"Removed evaluation results for '{name}'")
        
        # Update comparison metrics if they exist
        if not self.comparison_metrics.empty and name in self.comparison_metrics['Model'].values:
            self.comparison_metrics = self.comparison_metrics[self.comparison_metrics['Model'] != name]
            logger.info(f"Updated comparison metrics after removing '{name}'")

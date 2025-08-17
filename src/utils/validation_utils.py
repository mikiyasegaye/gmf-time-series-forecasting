"""
Validation utilities for data and models.

This module provides comprehensive validation capabilities for
financial data, models, and analysis results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime
import warnings
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ValidationUtils:
    """
    Comprehensive validation utilities for financial analysis.

    Provides data quality checks, model validation, and
    result verification capabilities.

    Attributes:
        validation_results (Dict): Results from validation checks
        validation_rules (Dict): Validation rules and thresholds
    """

    def __init__(self):
        """
        Initialize the Validation Utils.
        """
        self.validation_results = {}
        self.validation_rules = self._initialize_validation_rules()

        logger.info("Initialized Validation Utils")

    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """
        Initialize default validation rules.

        Returns:
            Dictionary with validation rules
        """
        rules = {
            'data_quality': {
                'min_observations': 100,
                'max_missing_pct': 0.05,  # 5%
                'max_outlier_pct': 0.01,  # 1%
                'min_date_range_days': 252,  # 1 year
                'max_price_deviation': 0.5,  # 50%
                'min_volume': 0
            },
            'model_validation': {
                'min_r2': 0.1,
                'max_mape': 0.5,  # 50%
                'min_directional_accuracy': 0.4,  # 40%
                'max_forecast_horizon': 252,  # 1 year
                'min_training_samples': 100
            },
            'portfolio_validation': {
                'min_assets': 2,
                'max_concentration': 0.5,  # 50%
                'min_diversification': 0.3,  # 30%
                'max_leverage': 1.0,
                'min_risk_free_rate': 0.0,
                'max_risk_free_rate': 0.1  # 10%
            }
        }

        return rules

    def validate_financial_data(self, data: pd.DataFrame, data_type: str = 'price') -> Dict[str, Any]:
        """
        Validate financial data quality and integrity.

        Args:
            data: DataFrame to validate
            data_type: Type of financial data ('price', 'returns', 'volume')

        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                'data_type': data_type,
                'validation_date': datetime.now().isoformat(),
                'checks_passed': 0,
                'checks_failed': 0,
                'total_checks': 0,
                'detailed_results': {},
                'overall_status': 'PENDING'
            }

            # Basic data structure checks
            structure_checks = self._validate_data_structure(data)
            validation_results['detailed_results']['structure'] = structure_checks
            validation_results['total_checks'] += len(structure_checks)

            # Data quality checks
            quality_checks = self._validate_data_quality(data, data_type)
            validation_results['detailed_results']['quality'] = quality_checks
            validation_results['total_checks'] += len(quality_checks)

            # Financial logic checks
            logic_checks = self._validate_financial_logic(data, data_type)
            validation_results['detailed_results']['logic'] = logic_checks
            validation_results['total_checks'] += len(logic_checks)

            # Calculate overall status
            total_passed = sum(
                1 for check in structure_checks.values() if check['status'] == 'PASS')
            total_passed += sum(1 for check in quality_checks.values()
                                if check['status'] == 'PASS')
            total_passed += sum(1 for check in logic_checks.values()
                                if check['status'] == 'PASS')

            validation_results['checks_passed'] = total_passed
            validation_results['checks_failed'] = validation_results['total_checks'] - total_passed

            # Determine overall status
            if validation_results['checks_failed'] == 0:
                validation_results['overall_status'] = 'PASS'
            # 10% failure threshold
            elif validation_results['checks_failed'] <= validation_results['total_checks'] * 0.1:
                validation_results['overall_status'] = 'WARNING'
            else:
                validation_results['overall_status'] = 'FAIL'

            # Store results
            self.validation_results[f'financial_data_{data_type}'] = validation_results

            logger.info(
                f"Financial data validation completed: {validation_results['overall_status']}")
            return validation_results

        except Exception as e:
            logger.error(f"Financial data validation failed: {str(e)}")
            return {'error': str(e)}

    def _validate_data_structure(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Validate basic data structure.

        Args:
            data: DataFrame to validate

        Returns:
            Dictionary with structure validation results
        """
        checks = {}

        # Check if data is empty
        checks['not_empty'] = {
            'status': 'PASS' if not data.empty else 'FAIL',
            'message': 'Data is not empty' if not data.empty else 'Data is empty',
            'value': len(data)
        }

        # Check minimum observations
        min_obs = self.validation_rules['data_quality']['min_observations']
        checks['min_observations'] = {
            'status': 'PASS' if len(data) >= min_obs else 'FAIL',
            'message': f'Data has {len(data)} observations (>= {min_obs})' if len(data) >= min_obs else f'Data has {len(data)} observations (< {min_obs})',
            'value': len(data)
        }

        # Check if index is datetime
        if not data.empty:
            checks['datetime_index'] = {
                'status': 'PASS' if isinstance(data.index, pd.DatetimeIndex) else 'FAIL',
                'message': 'Index is datetime' if isinstance(data.index, pd.DatetimeIndex) else 'Index is not datetime',
                'value': type(data.index).__name__
            }

        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] if 'price' in str(
            data.columns) else ['returns']
        missing_cols = [
            col for col in required_cols if col not in data.columns]
        checks['required_columns'] = {
            'status': 'PASS' if not missing_cols else 'FAIL',
            'message': f'All required columns present: {required_cols}' if not missing_cols else f'Missing columns: {missing_cols}',
            'value': missing_cols if missing_cols else 'All present'
        }

        return checks

    def _validate_data_quality(self, data: pd.DataFrame, data_type: str) -> Dict[str, Dict]:
        """
        Validate data quality metrics.

        Args:
            data: DataFrame to validate
            data_type: Type of financial data

        Returns:
            Dictionary with quality validation results
        """
        checks = {}

        if data.empty:
            return checks

        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        max_missing = self.validation_rules['data_quality']['max_missing_pct']
        checks['missing_values'] = {
            'status': 'PASS' if missing_pct <= max_missing else 'FAIL',
            'message': f'Missing values: {missing_pct:.2%} (<= {max_missing:.2%})' if missing_pct <= max_missing else f'Missing values: {missing_pct:.2%} (> {max_missing:.2%})',
            'value': f'{missing_pct:.2%}'
        }

        # Check for duplicates
        duplicate_count = data.duplicated().sum()
        checks['no_duplicates'] = {
            'status': 'PASS' if duplicate_count == 0 else 'FAIL',
            'message': f'No duplicate rows' if duplicate_count == 0 else f'{duplicate_count} duplicate rows found',
            'value': duplicate_count
        }

        # Check date range
        if isinstance(data.index, pd.DatetimeIndex):
            date_range = (data.index.max() - data.index.min()).days
            min_range = self.validation_rules['data_quality']['min_date_range_days']
            checks['date_range'] = {
                'status': 'PASS' if date_range >= min_range else 'FAIL',
                'message': f'Date range: {date_range} days (>= {min_range})' if date_range >= min_range else f'Date range: {date_range} days (< {min_range})',
                'value': date_range
            }

        # Check for outliers (simplified)
        if data_type == 'price' and 'Close' in data.columns:
            price_changes = data['Close'].pct_change().abs()
            outlier_threshold = self.validation_rules['data_quality']['max_outlier_pct']
            outlier_count = (price_changes > outlier_threshold).sum()
            checks['outlier_check'] = {
                'status': 'PASS' if outlier_count <= len(data) * 0.01 else 'WARNING',
                'message': f'Outliers: {outlier_count} extreme price changes' if outlier_count > 0 else 'No extreme price changes',
                'value': outlier_count
            }

        return checks

    def _validate_financial_logic(self, data: pd.DataFrame, data_type: str) -> Dict[str, Dict]:
        """
        Validate financial logic and consistency.

        Args:
            data: DataFrame to validate
            data_type: Type of financial data

        Returns:
            Dictionary with logic validation results
        """
        checks = {}

        if data.empty:
            return checks

        if data_type == 'price':
            # Check OHLC logic
            if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                ohlc_valid = (
                    (data['High'] >= data['Low']).all() and
                    (data['High'] >= data['Close']).all() and
                    (data['High'] >= data['Open']).all()
                )
                checks['ohlc_logic'] = {
                    'status': 'PASS' if ohlc_valid else 'FAIL',
                    'message': 'OHLC relationships are logical' if ohlc_valid else 'OHLC relationships are illogical',
                    'value': ohlc_valid
                }

            # Check for negative prices
            price_cols = [col for col in ['Open', 'High',
                                          'Low', 'Close'] if col in data.columns]
            if price_cols:
                negative_prices = (data[price_cols] <= 0).any().any()
                checks['positive_prices'] = {
                    'status': 'PASS' if not negative_prices else 'FAIL',
                    'message': 'All prices are positive' if not negative_prices else 'Negative prices found',
                    'value': not negative_prices
                }

        elif data_type == 'returns':
            # Check return bounds (reasonable range)
            if 'returns' in data.columns:
                # > 100% daily return
                extreme_returns = (data['returns'].abs() > 1.0).sum()
                checks['return_bounds'] = {
                    'status': 'PASS' if extreme_returns == 0 else 'WARNING',
                    'message': 'Returns within reasonable bounds' if extreme_returns == 0 else f'{extreme_returns} extreme returns found',
                    'value': extreme_returns
                }

        # Check for infinite values
        infinite_count = np.isinf(data.select_dtypes(
            include=[np.number])).sum().sum()
        checks['no_infinite_values'] = {
            'status': 'PASS' if infinite_count == 0 else 'FAIL',
            'message': 'No infinite values' if infinite_count == 0 else f'{infinite_count} infinite values found',
            'value': infinite_count
        }

        return checks

    def validate_model_performance(self, model_results: Dict[str, Any], model_type: str = 'forecasting') -> Dict[str, Any]:
        """
        Validate model performance metrics.

        Args:
            model_results: Dictionary with model results
            model_type: Type of model ('forecasting', 'classification', 'regression')

        Returns:
            Dictionary with model validation results
        """
        try:
            validation_results = {
                'model_type': model_type,
                'validation_date': datetime.now().isoformat(),
                'checks_passed': 0,
                'checks_failed': 0,
                'total_checks': 0,
                'detailed_results': {},
                'overall_status': 'PENDING'
            }

            # Extract metrics based on model type
            if model_type == 'forecasting':
                metrics_checks = self._validate_forecasting_metrics(
                    model_results)
            elif model_type == 'classification':
                metrics_checks = self._validate_classification_metrics(
                    model_results)
            else:
                metrics_checks = self._validate_regression_metrics(
                    model_results)

            validation_results['detailed_results']['metrics'] = metrics_checks
            validation_results['total_checks'] = len(metrics_checks)

            # Calculate overall status
            total_passed = sum(
                1 for check in metrics_checks.values() if check['status'] == 'PASS')
            validation_results['checks_passed'] = total_passed
            validation_results['checks_failed'] = validation_results['total_checks'] - total_passed

            # Determine overall status
            if validation_results['checks_failed'] == 0:
                validation_results['overall_status'] = 'PASS'
            # 20% failure threshold
            elif validation_results['checks_failed'] <= validation_results['total_checks'] * 0.2:
                validation_results['overall_status'] = 'WARNING'
            else:
                validation_results['overall_status'] = 'FAIL'

            # Store results
            self.validation_results[f'model_performance_{model_type}'] = validation_results

            logger.info(
                f"Model performance validation completed: {validation_results['overall_status']}")
            return validation_results

        except Exception as e:
            logger.error(f"Model performance validation failed: {str(e)}")
            return {'error': str(e)}

    def _validate_forecasting_metrics(self, model_results: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Validate forecasting model metrics.

        Args:
            model_results: Dictionary with forecasting results

        Returns:
            Dictionary with metric validation results
        """
        checks = {}

        # Check R-squared
        if 'r2' in model_results:
            r2 = model_results['r2']
            min_r2 = self.validation_rules['model_validation']['min_r2']
            checks['r2_threshold'] = {
                'status': 'PASS' if r2 >= min_r2 else 'FAIL',
                'message': f'R² = {r2:.3f} (>= {min_r2})' if r2 >= min_r2 else f'R² = {r2:.3f} (< {min_r2})',
                'value': r2
            }

        # Check MAPE
        if 'mape' in model_results:
            mape = model_results['mape']
            max_mape = self.validation_rules['model_validation']['max_mape']
            checks['mape_threshold'] = {
                'status': 'PASS' if mape <= max_mape else 'FAIL',
                'message': f'MAPE = {mape:.3f} (<= {max_mape})' if mape <= max_mape else f'MAPE = {mape:.3f} (> {max_mape})',
                'value': mape
            }

        # Check directional accuracy
        if 'directional_accuracy' in model_results:
            dir_acc = model_results['directional_accuracy']
            min_dir_acc = self.validation_rules['model_validation']['min_directional_accuracy']
            checks['directional_accuracy'] = {
                'status': 'PASS' if dir_acc >= min_dir_acc else 'FAIL',
                'message': f'Directional accuracy = {dir_acc:.3f} (>= {min_dir_acc})' if dir_acc >= min_dir_acc else f'Directional accuracy = {dir_acc:.3f} (< {min_dir_acc})',
                'value': dir_acc
            }

        # Check training samples
        if 'training_samples' in model_results:
            train_samples = model_results['training_samples']
            min_samples = self.validation_rules['model_validation']['min_training_samples']
            checks['training_samples'] = {
                'status': 'PASS' if train_samples >= min_samples else 'FAIL',
                'message': f'Training samples: {train_samples} (>= {min_samples})' if train_samples >= min_samples else f'Training samples: {train_samples} (< {min_samples})',
                'value': train_samples
            }

        return checks

    def _validate_classification_metrics(self, model_results: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Validate classification model metrics.

        Args:
            model_results: Dictionary with classification results

        Returns:
            Dictionary with metric validation results
        """
        checks = {}

        # Check accuracy
        if 'accuracy' in model_results:
            accuracy = model_results['accuracy']
            checks['accuracy_threshold'] = {
                'status': 'PASS' if accuracy >= 0.5 else 'FAIL',
                'message': f'Accuracy = {accuracy:.3f} (>= 0.5)' if accuracy >= 0.5 else f'Accuracy = {accuracy:.3f} (< 0.5)',
                'value': accuracy
            }

        # Check precision
        if 'precision' in model_results:
            precision = model_results['precision']
            checks['precision_threshold'] = {
                'status': 'PASS' if precision >= 0.3 else 'FAIL',
                'message': f'Precision = {precision:.3f} (>= 0.3)' if precision >= 0.3 else f'Precision = {precision:.3f} (< 0.3)',
                'value': precision
            }

        return checks

    def _validate_regression_metrics(self, model_results: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Validate regression model metrics.

        Args:
            model_results: Dictionary with regression results

        Returns:
            Dictionary with metric validation results
        """
        checks = {}

        # Check R-squared
        if 'r2' in model_results:
            r2 = model_results['r2']
            min_r2 = self.validation_rules['model_validation']['min_r2']
            checks['r2_threshold'] = {
                'status': 'PASS' if r2 >= min_r2 else 'FAIL',
                'message': f'R² = {r2:.3f} (>= {min_r2})' if r2 >= min_r2 else f'R² = {r2:.3f} (< {min_r2})',
                'value': r2
            }

        # Check RMSE
        if 'rmse' in model_results:
            rmse = model_results['rmse']
            checks['rmse_reasonable'] = {
                'status': 'PASS' if rmse >= 0 else 'FAIL',
                'message': f'RMSE = {rmse:.3f} (>= 0)' if rmse >= 0 else f'RMSE = {rmse:.3f} (< 0)',
                'value': rmse
            }

        return checks

    def validate_portfolio_constraints(self, portfolio_weights: Dict[str, float],
                                       constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate portfolio constraints and allocations.

        Args:
            portfolio_weights: Dictionary with asset weights
            constraints: Dictionary with portfolio constraints

        Returns:
            Dictionary with constraint validation results
        """
        try:
            validation_results = {
                'validation_date': datetime.now().isoformat(),
                'checks_passed': 0,
                'checks_failed': 0,
                'total_checks': 0,
                'detailed_results': {},
                'overall_status': 'PENDING'
            }

            # Check minimum assets
            min_assets = self.validation_rules['portfolio_validation']['min_assets']
            checks = {}
            checks['min_assets'] = {
                'status': 'PASS' if len(portfolio_weights) >= min_assets else 'FAIL',
                'message': f'Portfolio has {len(portfolio_weights)} assets (>= {min_assets})' if len(portfolio_weights) >= min_assets else f'Portfolio has {len(portfolio_weights)} assets (< {min_assets})',
                'value': len(portfolio_weights)
            }

            # Check weights sum to 1
            weight_sum = sum(portfolio_weights.values())
            checks['weights_sum_to_one'] = {
                'status': 'PASS' if abs(weight_sum - 1.0) < 1e-6 else 'FAIL',
                'message': f'Weights sum to {weight_sum:.6f} (≈ 1.0)' if abs(weight_sum - 1.0) < 1e-6 else f'Weights sum to {weight_sum:.6f} (≠ 1.0)',
                'value': weight_sum
            }

            # Check maximum concentration
            max_concentration = self.validation_rules['portfolio_validation']['max_concentration']
            max_weight = max(portfolio_weights.values())
            checks['max_concentration'] = {
                'status': 'PASS' if max_weight <= max_concentration else 'FAIL',
                'message': f'Max weight: {max_weight:.3f} (<= {max_concentration})' if max_weight <= max_concentration else f'Max weight: {max_weight:.3f} (> {max_concentration})',
                'value': max_weight
            }

            # Check for negative weights
            negative_weights = [asset for asset,
                                weight in portfolio_weights.items() if weight < 0]
            checks['no_negative_weights'] = {
                'status': 'PASS' if not negative_weights else 'FAIL',
                'message': 'No negative weights' if not negative_weights else f'Negative weights: {negative_weights}',
                'value': negative_weights if negative_weights else 'None'
            }

            validation_results['detailed_results'] = checks
            validation_results['total_checks'] = len(checks)

            # Calculate overall status
            total_passed = sum(1 for check in checks.values()
                               if check['status'] == 'PASS')
            validation_results['checks_passed'] = total_passed
            validation_results['checks_failed'] = validation_results['total_checks'] - total_passed

            # Determine overall status
            if validation_results['checks_failed'] == 0:
                validation_results['overall_status'] = 'PASS'
            # 20% failure threshold
            elif validation_results['checks_failed'] <= validation_results['total_checks'] * 0.2:
                validation_results['overall_status'] = 'WARNING'
            else:
                validation_results['overall_status'] = 'FAIL'

            # Store results
            self.validation_results['portfolio_constraints'] = validation_results

            logger.info(
                f"Portfolio constraint validation completed: {validation_results['overall_status']}")
            return validation_results

        except Exception as e:
            logger.error(f"Portfolio constraint validation failed: {str(e)}")
            return {'error': str(e)}

    def generate_validation_report(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate comprehensive validation report.

        Args:
            output_path: Path to save the report (optional)

        Returns:
            Report content as string
        """
        if not self.validation_results:
            return "No validation results available."

        # Generate report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(
            f"Total Validations: {len(self.validation_results)}")
        report_lines.append("")

        # Summary statistics
        total_checks = 0
        total_passed = 0
        total_failed = 0

        for validation_name, results in self.validation_results.items():
            if 'total_checks' in results:
                total_checks += results['total_checks']
                total_passed += results['checks_passed']
                total_failed += results['checks_failed']

        report_lines.append("OVERALL SUMMARY:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Checks: {total_checks}")
        report_lines.append(f"Checks Passed: {total_passed}")
        report_lines.append(f"Checks Failed: {total_failed}")
        report_lines.append(
            f"Success Rate: {total_passed/total_checks*100:.1f}%" if total_checks > 0 else "Success Rate: N/A")
        report_lines.append("")

        # Detailed results
        for validation_name, results in self.validation_results.items():
            report_lines.append(f"{validation_name.upper()}:")
            report_lines.append("-" * 40)
            report_lines.append(
                f"Status: {results.get('overall_status', 'UNKNOWN')}")
            report_lines.append(
                f"Checks Passed: {results.get('checks_passed', 0)}")
            report_lines.append(
                f"Checks Failed: {results.get('checks_failed', 0)}")

            if 'detailed_results' in results:
                report_lines.append("Detailed Results:")
                for check_name, check_result in results['detailed_results'].items():
                    if isinstance(check_result, dict):
                        status = check_result.get('status', 'UNKNOWN')
                        message = check_result.get('message', 'No message')
                        report_lines.append(
                            f"  {check_name}: {status} - {message}")

            report_lines.append("")

        report_lines.append("=" * 80)

        report_content = "\n".join(report_lines)

        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(report_content)

            logger.info(f"Validation report saved to {output_path}")

        return report_content

    def export_validation_results(self, output_path: Union[str, Path]) -> None:
        """
        Export validation results to JSON file.

        Args:
            output_path: Path to save the results
        """
        if not self.validation_results:
            logger.warning("No validation results to export")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export to JSON
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)

        logger.info(f"Validation results exported to {output_path}")

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validation results.

        Returns:
            Dictionary with validation summary
        """
        if not self.validation_results:
            return {'error': 'No validation results available'}

        summary = {
            'total_validations': len(self.validation_results),
            'validation_types': list(self.validation_results.keys()),
            'overall_status': self._calculate_overall_status(),
            'success_rate': self._calculate_success_rate()
        }

        return summary

    def _calculate_overall_status(self) -> str:
        """
        Calculate overall validation status.

        Returns:
            Overall status string
        """
        if not self.validation_results:
            return 'UNKNOWN'

        statuses = [results.get('overall_status', 'UNKNOWN')
                    for results in self.validation_results.values()]

        if 'FAIL' in statuses:
            return 'FAIL'
        elif 'WARNING' in statuses:
            return 'WARNING'
        elif all(status == 'PASS' for status in statuses):
            return 'PASS'
        else:
            return 'MIXED'

    def _calculate_success_rate(self) -> float:
        """
        Calculate overall success rate.

        Returns:
            Success rate as percentage
        """
        if not self.validation_results:
            return 0.0

        total_checks = 0
        total_passed = 0

        for results in self.validation_results.values():
            if 'total_checks' in results:
                total_checks += results['total_checks']
                total_passed += results['checks_passed']

        return (total_passed / total_checks * 100) if total_checks > 0 else 0.0

"""
Risk metrics calculation utilities.

This module provides comprehensive risk calculation utilities
for financial analysis and portfolio management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskMetrics:
    """
    Comprehensive risk metrics calculator for financial analysis.

    Provides a wide range of risk metrics including VaR, CVaR,
    drawdown analysis, and volatility measures.

    Attributes:
        returns_data (pd.DataFrame): Returns data for analysis
        risk_free_rate (float): Risk-free rate for calculations
        confidence_levels (List[float]): Confidence levels for VaR/CVaR
    """

    def __init__(self, returns_data: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize the Risk Metrics calculator.

        Args:
            returns_data: DataFrame with returns data
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.returns_data = returns_data.copy()
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = [0.95, 0.99, 0.999]

        logger.info(
            f"Initialized Risk Metrics calculator with {len(returns_data)} data points")

    def calculate_var(self, confidence_level: float = 0.95, method: str = 'historical') -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR).

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: Calculation method ('historical', 'parametric', 'monte_carlo')

        Returns:
            Dictionary with VaR results
        """
        try:
            if method == 'historical':
                var_value = np.percentile(
                    self.returns_data['returns'], (1 - confidence_level) * 100)
                var_annual = var_value * np.sqrt(252)

            elif method == 'parametric':
                # Assume normal distribution
                mean_return = self.returns_data['returns'].mean()
                std_return = self.returns_data['returns'].std()
                z_score = norm.ppf(confidence_level)
                var_value = mean_return - z_score * std_return
                var_annual = var_value * np.sqrt(252)

            elif method == 'monte_carlo':
                # Simplified Monte Carlo approach
                n_simulations = 10000
                simulated_returns = np.random.normal(
                    self.returns_data['returns'].mean(),
                    self.returns_data['returns'].std(),
                    n_simulations
                )
                var_value = np.percentile(
                    simulated_returns, (1 - confidence_level) * 100)
                var_annual = var_value * np.sqrt(252)

            else:
                raise ValueError(f"Unknown method: {method}")

            results = {
                'var_daily': float(var_value),
                'var_annual': float(var_annual),
                'confidence_level': confidence_level,
                'method': method
            }

            return results

        except Exception as e:
            logger.error(f"VaR calculation failed: {str(e)}")
            return {'error': str(e)}

    def calculate_cvar(self, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate Conditional Value at Risk (CVaR).

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Dictionary with CVaR results
        """
        try:
            # Calculate VaR first
            var_result = self.calculate_var(confidence_level, 'historical')
            if 'error' in var_result:
                return var_result

            var_daily = var_result['var_daily']

            # Calculate CVaR (expected loss beyond VaR)
            tail_returns = self.returns_data['returns'][self.returns_data['returns'] <= var_daily]

            if len(tail_returns) == 0:
                cvar_daily = var_daily
            else:
                cvar_daily = tail_returns.mean()

            cvar_annual = cvar_daily * np.sqrt(252)

            results = {
                'cvar_daily': float(cvar_daily),
                'cvar_annual': float(cvar_annual),
                'var_daily': float(var_daily),
                'confidence_level': confidence_level,
                'tail_observations': len(tail_returns)
            }

            return results

        except Exception as e:
            logger.error(f"CVaR calculation failed: {str(e)}")
            return {'error': str(e)}

    def calculate_drawdown_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive drawdown metrics.

        Returns:
            Dictionary with drawdown analysis
        """
        try:
            returns = self.returns_data['returns']
            cumulative_returns = (1 + returns).cumprod()

            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()

            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max

            # Maximum drawdown
            max_drawdown = drawdown.min()
            max_drawdown_date = drawdown.idxmin()

            # Current drawdown
            current_drawdown = drawdown.iloc[-1]

            # Drawdown duration
            drawdown_periods = (drawdown < 0).sum()
            total_periods = len(drawdown)
            drawdown_frequency = drawdown_periods / total_periods

            # Recovery time analysis
            recovery_analysis = self._analyze_recovery_times(drawdown)

            results = {
                'max_drawdown': float(max_drawdown),
                'max_drawdown_date': max_drawdown_date.isoformat() if hasattr(max_drawdown_date, 'isoformat') else str(max_drawdown_date),
                'current_drawdown': float(current_drawdown),
                'drawdown_frequency': float(drawdown_frequency),
                'total_drawdown_periods': int(drawdown_periods),
                'recovery_analysis': recovery_analysis
            }

            return results

        except Exception as e:
            logger.error(f"Drawdown calculation failed: {str(e)}")
            return {'error': str(e)}

    def _analyze_recovery_times(self, drawdown: pd.Series) -> Dict[str, Any]:
        """
        Analyze drawdown recovery times.

        Args:
            drawdown: Drawdown series

        Returns:
            Dictionary with recovery analysis
        """
        try:
            # Find drawdown periods
            drawdown_periods = []
            in_drawdown = False
            start_date = None

            for date, dd in drawdown.items():
                if dd < 0 and not in_drawdown:
                    # Start of drawdown
                    in_drawdown = True
                    start_date = date
                elif dd >= 0 and in_drawdown:
                    # End of drawdown
                    in_drawdown = False
                    if start_date is not None:
                        duration = (date - start_date).days
                        drawdown_periods.append(duration)

            # Handle case where still in drawdown
            if in_drawdown and start_date is not None:
                duration = (drawdown.index[-1] - start_date).days
                drawdown_periods.append(duration)

            if not drawdown_periods:
                return {'avg_recovery_days': 0, 'max_recovery_days': 0, 'total_recovery_periods': 0}

            recovery_analysis = {
                'avg_recovery_days': float(np.mean(drawdown_periods)),
                'max_recovery_days': int(max(drawdown_periods)),
                'min_recovery_days': int(min(drawdown_periods)),
                'total_recovery_periods': len(drawdown_periods)
            }

            return recovery_analysis

        except Exception as e:
            logger.warning(f"Recovery time analysis failed: {str(e)}")
            return {'error': str(e)}

    def calculate_volatility_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive volatility metrics.

        Returns:
            Dictionary with volatility analysis
        """
        try:
            returns = self.returns_data['returns']

            # Basic volatility measures
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)

            # Rolling volatility
            window_sizes = [5, 21, 63]  # 1 week, 1 month, 1 quarter
            rolling_vols = {}

            for window in window_sizes:
                if len(returns) >= window:
                    rolling_vol = returns.rolling(
                        window=window).std() * np.sqrt(252)
                    rolling_vols[f'{window}_day'] = {
                        'mean': float(rolling_vol.mean()),
                        'std': float(rolling_vol.std()),
                        'min': float(rolling_vol.min()),
                        'max': float(rolling_vol.max())
                    }

            # Volatility clustering
            volatility_clustering = self._calculate_volatility_clustering(
                returns)

            # GARCH-like volatility persistence
            volatility_persistence = self._calculate_volatility_persistence(
                returns)

            results = {
                'daily_volatility': float(daily_vol),
                'annual_volatility': float(annual_vol),
                'rolling_volatility': rolling_vols,
                'volatility_clustering': volatility_clustering,
                'volatility_persistence': volatility_persistence
            }

            return results

        except Exception as e:
            logger.error(f"Volatility calculation failed: {str(e)}")
            return {'error': str(e)}

    def _calculate_volatility_clustering(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate volatility clustering metrics.

        Args:
            returns: Returns series

        Returns:
            Dictionary with volatility clustering metrics
        """
        try:
            # Squared returns as volatility proxy
            squared_returns = returns ** 2

            # Autocorrelation at different lags
            lags = [1, 5, 10, 21]
            autocorrs = {}

            for lag in lags:
                if len(returns) > lag:
                    autocorr = squared_returns.autocorr(lag=lag)
                    autocorrs[f'lag_{lag}'] = float(
                        autocorr) if not np.isnan(autocorr) else 0.0

            # Ljung-Box test statistic
            if len(returns) > 10:
                lb_stat = len(returns) * (len(returns) + 2) * \
                    sum(autocorrs.values()) / (len(returns) - 1)
            else:
                lb_stat = 0.0

            return {
                'autocorrelations': autocorrs,
                'ljung_box_statistic': float(lb_stat)
            }

        except Exception as e:
            logger.warning(
                f"Volatility clustering calculation failed: {str(e)}")
            return {'error': str(e)}

    def _calculate_volatility_persistence(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate volatility persistence metrics.

        Args:
            returns: Returns series

        Returns:
            Dictionary with volatility persistence metrics
        """
        try:
            # Calculate absolute returns
            abs_returns = np.abs(returns)

            # Simple AR(1) coefficient for absolute returns
            if len(abs_returns) > 1:
                lagged_abs = abs_returns.shift(1).dropna()
                current_abs = abs_returns.iloc[1:]

                if len(lagged_abs) > 0 and len(current_abs) > 0:
                    # Simple correlation as persistence measure
                    persistence = current_abs.corr(lagged_abs)
                else:
                    persistence = 0.0
            else:
                persistence = 0.0

            # Half-life calculation (simplified)
            half_life = -np.log(2) / np.log(abs(persistence)
                                            ) if persistence != 0 else 0

            return {
                'persistence_coefficient': float(persistence),
                'half_life': float(half_life)
            }

        except Exception as e:
            logger.warning(
                f"Volatility persistence calculation failed: {str(e)}")
            return {'error': str(e)}

    def calculate_tail_risk_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive tail risk metrics.

        Returns:
            Dictionary with tail risk analysis
        """
        try:
            returns = self.returns_data['returns']

            # Tail risk at different confidence levels
            tail_metrics = {}

            for conf_level in self.confidence_levels:
                var_result = self.calculate_var(conf_level, 'historical')
                cvar_result = self.calculate_cvar(conf_level)

                if 'error' not in var_result and 'error' not in cvar_result:
                    tail_metrics[f'{int(conf_level*100)}%'] = {
                        'var': var_result['var_daily'],
                        'cvar': cvar_result['cvar_daily'],
                        'tail_ratio': abs(cvar_result['cvar_daily'] / var_result['var_daily']) if var_result['var_daily'] != 0 else 0
                    }

            # Expected shortfall
            expected_shortfall = returns[returns < 0].mean() if len(
                returns[returns < 0]) > 0 else 0

            # Tail dependence (simplified)
            tail_dependence = self._calculate_tail_dependence(returns)

            results = {
                'tail_risk_metrics': tail_metrics,
                'expected_shortfall': float(expected_shortfall),
                'tail_dependence': tail_dependence
            }

            return results

        except Exception as e:
            logger.error(f"Tail risk calculation failed: {str(e)}")
            return {'error': str(e)}

    def _calculate_tail_dependence(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate tail dependence metrics.

        Args:
            returns: Returns series

        Returns:
            Dictionary with tail dependence metrics
        """
        try:
            # Calculate quantiles
            q_05 = returns.quantile(0.05)
            q_95 = returns.quantile(0.95)

            # Left tail dependence
            left_tail = returns[returns <= q_05]
            left_tail_prob = len(left_tail) / len(returns)

            # Right tail dependence
            right_tail = returns[returns >= q_95]
            right_tail_prob = len(right_tail) / len(returns)

            # Tail asymmetry
            tail_asymmetry = abs(left_tail_prob - right_tail_prob)

            return {
                'left_tail_probability': float(left_tail_prob),
                'right_tail_probability': float(right_tail_prob),
                'tail_asymmetry': float(tail_asymmetry)
            }

        except Exception as e:
            logger.warning(f"Tail dependence calculation failed: {str(e)}")
            return {'error': str(e)}

    def calculate_comprehensive_risk_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.

        Returns:
            Dictionary with complete risk analysis
        """
        try:
            # Calculate all risk metrics
            var_metrics = {}
            cvar_metrics = {}

            for conf_level in self.confidence_levels:
                var_result = self.calculate_var(conf_level, 'historical')
                cvar_result = self.calculate_cvar(conf_level)

                if 'error' not in var_result:
                    var_metrics[f'{int(conf_level*100)}%'] = var_result

                if 'error' not in cvar_result:
                    cvar_metrics[f'{int(conf_level*100)}%'] = cvar_result

            # Compile comprehensive report
            risk_report = {
                'calculation_date': datetime.now().isoformat(),
                'data_summary': {
                    'total_observations': len(self.returns_data),
                    'date_range': f"{self.returns_data.index[0]} to {self.returns_data.index[-1]}",
                    'risk_free_rate': self.risk_free_rate
                },
                'var_analysis': var_metrics,
                'cvar_analysis': cvar_metrics,
                'drawdown_analysis': self.calculate_drawdown_metrics(),
                'volatility_analysis': self.calculate_volatility_metrics(),
                'tail_risk_analysis': self.calculate_tail_risk_metrics()
            }

            logger.info("Comprehensive risk report generated successfully")
            return risk_report

        except Exception as e:
            logger.error(
                f"Comprehensive risk report generation failed: {str(e)}")
            return {'error': str(e)}

    def export_risk_report(self, output_path: Union[str, Path]) -> None:
        """
        Export risk report to JSON file.

        Args:
            output_path: Path to save the report
        """
        try:
            risk_report = self.calculate_comprehensive_risk_report()

            if 'error' in risk_report:
                logger.error(
                    f"Cannot export risk report: {risk_report['error']}")
                return

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Export to JSON
            with open(output_path, 'w') as f:
                json.dump(risk_report, f, indent=2, default=str)

            logger.info(f"Risk report exported to {output_path}")

        except Exception as e:
            logger.error(f"Risk report export failed: {str(e)}")

    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get summary of key risk metrics.

        Returns:
            Dictionary with risk summary
        """
        try:
            # Get key metrics
            var_95 = self.calculate_var(0.95, 'historical')
            cvar_95 = self.calculate_cvar(0.95)
            drawdown = self.calculate_drawdown_metrics()
            volatility = self.calculate_volatility_metrics()

            # Compile summary
            summary = {
                'key_risk_metrics': {
                    'var_95_daily': var_95.get('var_daily', 0) if 'error' not in var_95 else 0,
                    'cvar_95_daily': cvar_95.get('cvar_daily', 0) if 'error' not in cvar_95 else 0,
                    'max_drawdown': drawdown.get('max_drawdown', 0) if 'error' not in drawdown else 0,
                    'annual_volatility': volatility.get('annual_volatility', 0) if 'error' not in volatility else 0
                },
                'risk_assessment': self._assess_risk_level(var_95, cvar_95, drawdown, volatility)
            }

            return summary

        except Exception as e:
            logger.error(f"Risk summary generation failed: {str(e)}")
            return {'error': str(e)}

    def _assess_risk_level(self, var_result: Dict, cvar_result: Dict,
                           drawdown_result: Dict, volatility_result: Dict) -> str:
        """
        Assess overall risk level based on metrics.

        Args:
            var_result: VaR calculation result
            cvar_result: CVaR calculation result
            drawdown_result: Drawdown calculation result
            volatility_result: Volatility calculation result

        Returns:
            Risk level assessment string
        """
        try:
            # Extract key metrics
            var_95 = var_result.get(
                'var_daily', 0) if 'error' not in var_result else 0
            cvar_95 = cvar_result.get(
                'cvar_daily', 0) if 'error' not in cvar_result else 0
            max_dd = drawdown_result.get(
                'max_drawdown', 0) if 'error' not in drawdown_result else 0
            ann_vol = volatility_result.get(
                'annual_volatility', 0) if 'error' not in volatility_result else 0

            # Risk scoring (simplified)
            risk_score = 0

            # VaR scoring
            if abs(var_95) < 0.01:  # < 1%
                risk_score += 1
            elif abs(var_95) < 0.02:  # 1-2%
                risk_score += 2
            else:  # > 2%
                risk_score += 3

            # Drawdown scoring
            if abs(max_dd) < 0.1:  # < 10%
                risk_score += 1
            elif abs(max_dd) < 0.2:  # 10-20%
                risk_score += 2
            else:  # > 20%
                risk_score += 3

            # Volatility scoring
            if ann_vol < 0.15:  # < 15%
                risk_score += 1
            elif ann_vol < 0.25:  # 15-25%
                risk_score += 2
            else:  # > 25%
                risk_score += 3

            # Risk level assessment
            if risk_score <= 4:
                return "LOW"
            elif risk_score <= 6:
                return "MODERATE"
            else:
                return "HIGH"

        except Exception as e:
            logger.warning(f"Risk assessment failed: {str(e)}")
            return "UNKNOWN"

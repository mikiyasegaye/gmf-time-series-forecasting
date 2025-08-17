"""
Performance analysis module for portfolio strategies.

This module provides comprehensive performance analysis including
risk metrics, attribution analysis, and performance reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for portfolio strategies.

    Provides detailed performance analysis including risk metrics,
    attribution analysis, and performance reporting capabilities.

    Attributes:
        returns_data (pd.DataFrame): Portfolio returns data
        benchmark_data (pd.DataFrame): Benchmark returns data
        analysis_results (Dict): Analysis results and metrics
        risk_metrics (Dict): Comprehensive risk metrics
    """

    def __init__(self, returns_data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None):
        """
        Initialize the Performance Analyzer.

        Args:
            returns_data: DataFrame with portfolio returns
            benchmark_data: DataFrame with benchmark returns (optional)
        """
        self.returns_data = returns_data.copy()
        self.benchmark_data = benchmark_data.copy() if benchmark_data is not None else None
        self.analysis_results = {}
        self.risk_metrics = {}

        # Validate inputs
        self._validate_inputs()

        logger.info(
            f"Initialized Performance Analyzer with {len(returns_data)} data points")

    def _validate_inputs(self) -> None:
        """
        Validate input parameters.

        Raises:
            ValueError: If inputs are invalid
        """
        if self.returns_data.empty:
            raise ValueError("Returns data cannot be empty")

        if self.benchmark_data is not None and self.benchmark_data.empty:
            raise ValueError("Benchmark data cannot be empty if provided")

        # Check for required columns
        required_cols = ['returns']
        missing_cols = [
            col for col in required_cols if col not in self.returns_data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in returns data: {missing_cols}")

    def analyze_performance(self, start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis.

        Args:
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)

        Returns:
            Dictionary with comprehensive analysis results

        Raises:
            RuntimeError: If analysis fails
        """
        try:
            # Filter data by date range
            analysis_data = self._filter_data_by_date(start_date, end_date)

            if len(analysis_data) == 0:
                raise ValueError(
                    "No data available for the specified date range")

            logger.info(
                f"Analyzing performance from {analysis_data.index[0]} to {analysis_data.index[-1]}")

            # Calculate basic performance metrics
            basic_metrics = self._calculate_basic_metrics(analysis_data)

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(analysis_data)

            # Calculate advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(analysis_data)

            # Calculate benchmark comparison if available
            benchmark_comparison = {}
            if self.benchmark_data is not None:
                benchmark_comparison = self._calculate_benchmark_comparison(
                    analysis_data)

            # Store analysis results
            self.analysis_results = {
                'analysis_period': {
                    'start_date': analysis_data.index[0].isoformat(),
                    'end_date': analysis_data.index[-1].isoformat(),
                    'total_days': len(analysis_data)
                },
                'basic_metrics': basic_metrics,
                'risk_metrics': risk_metrics,
                'advanced_metrics': advanced_metrics,
                'benchmark_comparison': benchmark_comparison,
                'analysis_date': datetime.now().isoformat()
            }

            # Store risk metrics separately for easy access
            self.risk_metrics = risk_metrics

            logger.info("Performance analysis completed successfully")
            return self.analysis_results

        except Exception as e:
            logger.error(f"Performance analysis failed: {str(e)}")
            raise RuntimeError(f"Performance analysis failed: {str(e)}")

    def _filter_data_by_date(self, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """
        Filter data by date range.

        Args:
            start_date: Start date string
            end_date: End date string

        Returns:
            Filtered DataFrame
        """
        filtered_data = self.returns_data.copy()

        if start_date:
            filtered_data = filtered_data[filtered_data.index >= start_date]

        if end_date:
            filtered_data = filtered_data[filtered_data.index <= end_date]

        return filtered_data

    def _calculate_basic_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate basic performance metrics.

        Args:
            data: Filtered returns data

        Returns:
            Dictionary with basic metrics
        """
        returns = data['returns']

        # Total return
        total_return = (1 + returns).prod() - 1

        # Annualized return
        days = len(returns)
        annual_return = (1 + total_return) ** (252 / days) - 1

        # Average daily return
        avg_daily_return = returns.mean()

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std(
        ) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0

        basic_metrics = {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'avg_daily_return': float(avg_daily_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio)
        }

        return basic_metrics

    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics.

        Args:
            data: Filtered returns data

        Returns:
            Dictionary with risk metrics
        """
        returns = data['returns']

        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()

        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = self.analysis_results.get('basic_metrics', {}).get(
            'annual_return', 0) / abs(max_drawdown) if max_drawdown != 0 else 0

        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Information ratio (if benchmark available)
        information_ratio = 0.0
        if self.benchmark_data is not None:
            benchmark_returns = self.benchmark_data[
                'returns'] if 'returns' in self.benchmark_data.columns else self.benchmark_data.iloc[:, 0]
            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean(
            ) * 252 / tracking_error if tracking_error > 0 else 0

        # Beta (if benchmark available)
        beta = 1.0
        if self.benchmark_data is not None:
            benchmark_returns = self.benchmark_data[
                'returns'] if 'returns' in self.benchmark_data.columns else self.benchmark_data.iloc[:, 0]
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

        # Alpha (if benchmark available)
        alpha = 0.0
        if self.benchmark_data is not None:
            benchmark_returns = self.benchmark_data[
                'returns'] if 'returns' in self.benchmark_data.columns else self.benchmark_data.iloc[:, 0]
            portfolio_return = returns.mean() * 252
            benchmark_return = benchmark_returns.mean() * 252
            alpha = portfolio_return - (beta * benchmark_return)

        risk_metrics = {
            'var_95': float(var_95),
            'var_99': float(var_99),
            'cvar_95': float(cvar_95),
            'cvar_99': float(cvar_99),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'information_ratio': float(information_ratio),
            'beta': float(beta),
            'alpha': float(alpha)
        }

        return risk_metrics

    def _calculate_advanced_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate advanced performance metrics.

        Args:
            data: Filtered returns data

        Returns:
            Dictionary with advanced metrics
        """
        returns = data['returns']

        # Rolling metrics
        window = min(30, len(returns) // 4)
        rolling_metrics = {}

        if window >= 5:
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            rolling_sharpe = (returns.rolling(
                window=window).mean() * 252) / rolling_vol
            rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)

            rolling_metrics = {
                'rolling_volatility_mean': float(rolling_vol.mean()),
                'rolling_volatility_std': float(rolling_vol.std()),
                'rolling_sharpe_mean': float(rolling_sharpe.mean()),
                'rolling_sharpe_std': float(rolling_sharpe.std())
            }

        # Win rate analysis
        positive_returns = returns > 0
        win_rate = positive_returns.mean()

        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_events(
            positive_returns, True)
        consecutive_losses = self._calculate_consecutive_events(
            positive_returns, False)

        # Return distribution analysis
        return_quintiles = returns.quantile([0.2, 0.4, 0.6, 0.8])

        # Volatility clustering
        volatility_clustering = self._calculate_volatility_clustering(returns)

        advanced_metrics = {
            'rolling_metrics': rolling_metrics,
            'win_rate': float(win_rate),
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'return_quintiles': return_quintiles.to_dict(),
            'volatility_clustering': volatility_clustering
        }

        return advanced_metrics

    def _calculate_consecutive_events(self, boolean_series: pd.Series, event_value: bool) -> Dict[str, Any]:
        """
        Calculate consecutive event statistics.

        Args:
            boolean_series: Boolean series
            event_value: Value to count consecutive occurrences for

        Returns:
            Dictionary with consecutive event statistics
        """
        consecutive_counts = []
        current_count = 0

        for value in boolean_series:
            if value == event_value:
                current_count += 1
            else:
                if current_count > 0:
                    consecutive_counts.append(current_count)
                    current_count = 0

        # Don't forget the last streak
        if current_count > 0:
            consecutive_counts.append(current_count)

        if not consecutive_counts:
            return {'max': 0, 'mean': 0, 'total_streaks': 0}

        return {
            'max': int(max(consecutive_counts)),
            'mean': float(np.mean(consecutive_counts)),
            'total_streaks': len(consecutive_counts)
        }

    def _calculate_volatility_clustering(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate volatility clustering metrics.

        Args:
            returns: Returns series

        Returns:
            Dictionary with volatility clustering metrics
        """
        # Calculate squared returns (proxy for volatility)
        squared_returns = returns ** 2

        # Autocorrelation of squared returns
        if len(returns) > 10:
            autocorr_1 = squared_returns.autocorr(lag=1)
            autocorr_5 = squared_returns.autocorr(lag=5)
            autocorr_10 = squared_returns.autocorr(lag=10)
        else:
            autocorr_1 = autocorr_5 = autocorr_10 = 0.0

        # Ljung-Box test statistic (simplified)
        n = len(returns)
        lb_stat = n * (n + 2) * (autocorr_1**2 + autocorr_5 **
                                 2 + autocorr_10**2) / (n - 1)

        return {
            'autocorr_1': float(autocorr_1) if not np.isnan(autocorr_1) else 0.0,
            'autocorr_5': float(autocorr_5) if not np.isnan(autocorr_5) else 0.0,
            'autocorr_10': float(autocorr_10) if not np.isnan(autocorr_10) else 0.0,
            'ljung_box_stat': float(lb_stat)
        }

    def _calculate_benchmark_comparison(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate benchmark comparison metrics.

        Args:
            data: Filtered returns data

        Returns:
            Dictionary with benchmark comparison metrics
        """
        if self.benchmark_data is None:
            return {}

        portfolio_returns = data['returns']
        benchmark_returns = self.benchmark_data['returns'] if 'returns' in self.benchmark_data.columns else self.benchmark_data.iloc[:, 0]

        # Align data by date
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned_data) == 0:
            return {'error': 'No aligned data available for benchmark comparison'}

        portfolio_returns = aligned_data['portfolio']
        benchmark_returns = aligned_data['benchmark']

        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns

        # Tracking error
        tracking_error = excess_returns.std() * np.sqrt(252)

        # Information ratio
        information_ratio = excess_returns.mean(
        ) * 252 / tracking_error if tracking_error > 0 else 0

        # Up/down capture ratios
        up_capture = self._calculate_capture_ratio(
            portfolio_returns, benchmark_returns, 'up')
        down_capture = self._calculate_capture_ratio(
            portfolio_returns, benchmark_returns, 'down')

        # Correlation
        correlation = portfolio_returns.corr(benchmark_returns)

        # Beta and Alpha
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

        portfolio_return = portfolio_returns.mean() * 252
        benchmark_return = benchmark_returns.mean() * 252
        alpha = portfolio_return - (beta * benchmark_return)

        # Outperformance analysis
        outperformance = (1 + portfolio_returns).prod() - \
            (1 + benchmark_returns).prod()

        comparison = {
            'tracking_error': float(tracking_error),
            'information_ratio': float(information_ratio),
            'up_capture': float(up_capture),
            'down_capture': float(down_capture),
            'correlation': float(correlation),
            'beta': float(beta),
            'alpha': float(alpha),
            'outperformance': float(outperformance)
        }

        return comparison

    def _calculate_capture_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                                 direction: str) -> float:
        """
        Calculate capture ratio for up or down markets.

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            direction: 'up' or 'down'

        Returns:
            Capture ratio
        """
        if direction == 'up':
            mask = benchmark_returns > 0
        else:
            mask = benchmark_returns < 0

        if not mask.any():
            return 0.0

        portfolio_captured = portfolio_returns[mask].sum()
        benchmark_captured = benchmark_returns[mask].sum()

        return portfolio_captured / benchmark_captured if benchmark_captured != 0 else 0.0

    def generate_performance_report(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a comprehensive performance report.

        Args:
            output_path: Path to save the report (optional)

        Returns:
            Report content as string
        """
        if not self.analysis_results:
            return "No performance analysis results available."

        # Generate report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PORTFOLIO PERFORMANCE ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(
            f"Analysis Period: {self.analysis_results['analysis_period']['start_date']} to {self.analysis_results['analysis_period']['end_date']}")
        report_lines.append(
            f"Total Days: {self.analysis_results['analysis_period']['total_days']}")
        report_lines.append("")

        # Basic metrics
        basic = self.analysis_results['basic_metrics']
        report_lines.append("BASIC PERFORMANCE METRICS:")
        report_lines.append("-" * 40)
        report_lines.append(
            f"Total Return: {basic['total_return']:.4f} ({basic['total_return']*100:.2f}%)")
        report_lines.append(
            f"Annual Return: {basic['annual_return']:.4f} ({basic['annual_return']*100:.2f}%)")
        report_lines.append(
            f"Volatility: {basic['volatility']:.4f} ({basic['volatility']*100:.2f}%)")
        report_lines.append(f"Sharpe Ratio: {basic['sharpe_ratio']:.4f}")
        report_lines.append(f"Sortino Ratio: {basic['sortino_ratio']:.4f}")
        report_lines.append("")

        # Risk metrics
        risk = self.analysis_results['risk_metrics']
        report_lines.append("RISK METRICS:")
        report_lines.append("-" * 40)
        report_lines.append(
            f"VaR (95%): {risk['var_95']:.4f} ({risk['var_95']*100:.2f}%)")
        report_lines.append(
            f"VaR (99%): {risk['var_99']:.4f} ({risk['var_99']*100:.2f}%)")
        report_lines.append(
            f"CVaR (95%): {risk['cvar_95']:.4f} ({risk['cvar_95']*100:.2f}%)")
        report_lines.append(
            f"Maximum Drawdown: {risk['max_drawdown']:.4f} ({risk['max_drawdown']*100:.2f}%)")
        report_lines.append(f"Calmar Ratio: {risk['calmar_ratio']:.4f}")
        report_lines.append(f"Skewness: {risk['skewness']:.4f}")
        report_lines.append(f"Kurtosis: {risk['kurtosis']:.4f}")
        report_lines.append("")

        # Benchmark comparison
        if self.analysis_results['benchmark_comparison']:
            benchmark = self.analysis_results['benchmark_comparison']
            report_lines.append("BENCHMARK COMPARISON:")
            report_lines.append("-" * 40)
            report_lines.append(
                f"Information Ratio: {benchmark['information_ratio']:.4f}")
            report_lines.append(f"Beta: {benchmark['beta']:.4f}")
            report_lines.append(f"Alpha: {benchmark['alpha']:.4f}")
            report_lines.append(f"Up Capture: {benchmark['up_capture']:.4f}")
            report_lines.append(
                f"Down Capture: {benchmark['down_capture']:.4f}")
            report_lines.append(
                f"Outperformance: {benchmark['outperformance']:.4f} ({benchmark['outperformance']*100:.2f}%)")
            report_lines.append("")

        # Advanced metrics
        advanced = self.analysis_results['advanced_metrics']
        report_lines.append("ADVANCED METRICS:")
        report_lines.append("-" * 40)
        report_lines.append(
            f"Win Rate: {advanced['win_rate']:.4f} ({advanced['win_rate']*100:.2f}%)")
        report_lines.append(
            f"Max Consecutive Wins: {advanced['consecutive_wins']['max']}")
        report_lines.append(
            f"Max Consecutive Losses: {advanced['consecutive_losses']['max']}")
        report_lines.append("")

        report_lines.append("=" * 80)

        report_content = "\n".join(report_lines)

        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(report_content)

            logger.info(f"Performance report saved to {output_path}")

        return report_content

    def plot_performance_analysis(self, show_risk_metrics: bool = True,
                                  show_benchmark: bool = True) -> None:
        """
        Plot comprehensive performance analysis.

        Args:
            show_risk_metrics: Whether to show risk metrics plots
            show_benchmark: Whether to show benchmark comparison plots

        Note: This method requires matplotlib to be available.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if not self.analysis_results:
                logger.warning("No analysis results available for plotting")
                return

            # Create subplots
            n_plots = 2 + (1 if show_risk_metrics else 0) + \
                (1 if show_benchmark else 0)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            plot_idx = 0

            # Plot 1: Cumulative Returns
            ax = axes[plot_idx]
            cumulative_returns = (1 + self.returns_data['returns']).cumprod()
            ax.plot(cumulative_returns.index,
                    cumulative_returns.values, linewidth=2, color='blue')
            ax.set_title('Cumulative Portfolio Performance')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.grid(True, alpha=0.3)
            plot_idx += 1

            # Plot 2: Rolling Volatility
            ax = axes[plot_idx]
            window = min(30, len(self.returns_data) // 4)
            if window >= 5:
                rolling_vol = self.returns_data['returns'].rolling(
                    window=window).std() * np.sqrt(252)
                ax.plot(rolling_vol.index, rolling_vol.values,
                        linewidth=2, color='red')
                ax.set_title(f'{window}-Day Rolling Volatility')
                ax.set_xlabel('Date')
                ax.set_ylabel('Annualized Volatility')
                ax.grid(True, alpha=0.3)
            plot_idx += 1

            # Plot 3: Risk Metrics (if requested)
            if show_risk_metrics and plot_idx < len(axes):
                ax = axes[plot_idx]
                risk_metrics = ['var_95', 'cvar_95', 'max_drawdown']
                risk_values = [self.risk_metrics.get(
                    metric, 0) for metric in risk_metrics]
                risk_labels = ['VaR (95%)', 'CVaR (95%)', 'Max Drawdown']

                bars = ax.bar(risk_labels, risk_values, color=[
                              'red', 'darkred', 'orange'])
                ax.set_title('Key Risk Metrics')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, value in zip(bars, risk_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.4f}', ha='center', va='bottom')

                plot_idx += 1

            # Plot 4: Benchmark Comparison (if requested)
            if show_benchmark and plot_idx < len(axes) and self.benchmark_data is not None:
                ax = axes[plot_idx]

                # Plot both portfolio and benchmark
                portfolio_cumulative = (
                    1 + self.returns_data['returns']).cumprod()
                benchmark_cumulative = (1 + self.benchmark_data['returns']).cumprod(
                ) if 'returns' in self.benchmark_data.columns else (1 + self.benchmark_data.iloc[:, 0]).cumprod()

                ax.plot(portfolio_cumulative.index, portfolio_cumulative.values,
                        label='Portfolio', linewidth=2, color='blue')
                ax.plot(benchmark_cumulative.index, benchmark_cumulative.values,
                        label='Benchmark', linewidth=2, color='red', alpha=0.7)
                ax.set_title('Portfolio vs Benchmark Performance')
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative Return')
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting performance analysis: {str(e)}")

    def export_analysis_results(self, output_path: Union[str, Path]) -> None:
        """
        Export analysis results to JSON file.

        Args:
            output_path: Path to save the results
        """
        if not self.analysis_results:
            logger.warning("No analysis results to export")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export to JSON
        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)

        logger.info(f"Analysis results exported to {output_path}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of performance analysis.

        Returns:
            Dictionary with performance summary
        """
        if not self.analysis_results:
            return {'error': 'No analysis results available'}

        summary = {
            'period': self.analysis_results['analysis_period'],
            'key_metrics': {
                'total_return': self.analysis_results['basic_metrics']['total_return'],
                'annual_return': self.analysis_results['basic_metrics']['annual_return'],
                'sharpe_ratio': self.analysis_results['basic_metrics']['sharpe_ratio'],
                'max_drawdown': self.analysis_results['risk_metrics']['max_drawdown'],
                'var_95': self.analysis_results['risk_metrics']['var_95']
            },
            'risk_profile': {
                'volatility': self.analysis_results['basic_metrics']['volatility'],
                'skewness': self.analysis_results['risk_metrics']['skewness'],
                'kurtosis': self.analysis_results['risk_metrics']['kurtosis'],
                'beta': self.analysis_results['risk_metrics']['beta']
            }
        }

        if self.analysis_results['benchmark_comparison']:
            summary['benchmark_comparison'] = {
                'information_ratio': self.analysis_results['benchmark_comparison']['information_ratio'],
                'outperformance': self.analysis_results['benchmark_comparison']['outperformance']
            }

        return summary

"""
Backtesting engine for portfolio strategies.

This module provides comprehensive backtesting capabilities for
portfolio strategies including performance tracking and analysis.
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


class BacktestEngine:
    """
    Advanced backtesting engine for portfolio strategies.

    Implements comprehensive backtesting with performance tracking,
    risk metrics calculation, and strategy validation.

    Attributes:
        strategy_weights (Dict): Strategy portfolio weights
        benchmark_weights (Dict): Benchmark portfolio weights
        returns_data (pd.DataFrame): Asset returns data
        backtest_results (Dict): Backtesting results and metrics
        rebalancing_frequency (str): Portfolio rebalancing frequency
    """

    def __init__(self, strategy_weights: Dict[str, float],
                 benchmark_weights: Dict[str, float],
                 returns_data: pd.DataFrame,
                 rebalancing_frequency: str = 'monthly'):
        """
        Initialize the Backtest Engine.

        Args:
            strategy_weights: Dictionary mapping assets to strategy weights
            benchmark_weights: Dictionary mapping assets to benchmark weights
            returns_data: DataFrame with asset returns (assets as columns)
            rebalancing_frequency: Portfolio rebalancing frequency
        """
        self.strategy_weights = strategy_weights.copy()
        self.benchmark_weights = benchmark_weights.copy()
        self.returns_data = returns_data.copy()
        self.rebalancing_frequency = rebalancing_frequency
        self.backtest_results = {}

        # Validate inputs
        self._validate_inputs()

        # Initialize performance tracking
        self._initialize_performance_tracking()

        logger.info(
            f"Initialized Backtest Engine with {len(strategy_weights)} strategy assets")

    def _validate_inputs(self) -> None:
        """
        Validate input parameters.

        Raises:
            ValueError: If inputs are invalid
        """
        # Check weights sum to 1
        strategy_sum = sum(self.strategy_weights.values())
        benchmark_sum = sum(self.benchmark_weights.values())

        if not np.isclose(strategy_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"Strategy weights must sum to 1.0, got {strategy_sum}")

        if not np.isclose(benchmark_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"Benchmark weights must sum to 1.0, got {benchmark_sum}")

        # Check all assets in weights exist in returns data
        strategy_assets = set(self.strategy_weights.keys())
        benchmark_assets = set(self.benchmark_weights.keys())
        available_assets = set(self.returns_data.columns)

        missing_strategy = strategy_assets - available_assets
        missing_benchmark = benchmark_assets - available_assets

        if missing_strategy:
            raise ValueError(
                f"Strategy assets not found in returns data: {missing_strategy}")

        if missing_benchmark:
            raise ValueError(
                f"Benchmark assets not found in returns data: {missing_benchmark}")

        # Validate rebalancing frequency
        valid_frequencies = ['daily', 'weekly',
                             'monthly', 'quarterly', 'yearly']
        if self.rebalancing_frequency not in valid_frequencies:
            raise ValueError(
                f"Invalid rebalancing frequency. Must be one of: {valid_frequencies}")

    def _initialize_performance_tracking(self) -> None:
        """
        Initialize performance tracking structures.
        """
        # Create portfolio returns series
        self.strategy_returns = pd.Series(
            index=self.returns_data.index, dtype=float)
        self.benchmark_returns = pd.Series(
            index=self.returns_data.index, dtype=float)

        # Initialize cumulative performance
        self.strategy_cumulative = pd.Series(
            index=self.returns_data.index, dtype=float)
        self.benchmark_cumulative = pd.Series(
            index=self.returns_data.index, dtype=float)

        # Initialize drawdown tracking
        self.strategy_drawdown = pd.Series(
            index=self.returns_data.index, dtype=float)
        self.benchmark_drawdown = pd.Series(
            index=self.returns_data.index, dtype=float)

        logger.info("Initialized performance tracking structures")

    def run_backtest(self, start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the backtest for the specified period.

        Args:
            start_date: Start date for backtesting (optional)
            end_date: End date for backtesting (optional)

        Returns:
            Dictionary with backtesting results

        Raises:
            RuntimeError: If backtesting fails
        """
        try:
            # Filter data by date range
            backtest_data = self._filter_data_by_date(start_date, end_date)

            if len(backtest_data) == 0:
                raise ValueError(
                    "No data available for the specified date range")

            logger.info(
                f"Running backtest from {backtest_data.index[0]} to {backtest_data.index[-1]}")

            # Calculate portfolio returns
            self._calculate_portfolio_returns(backtest_data)

            # Calculate cumulative performance
            self._calculate_cumulative_performance()

            # Calculate drawdowns
            self._calculate_drawdowns()

            # Generate performance metrics
            performance_metrics = self._calculate_performance_metrics()

            # Store results
            self.backtest_results = {
                'backtest_period': {
                    'start_date': backtest_data.index[0].isoformat(),
                    'end_date': backtest_data.index[-1].isoformat(),
                    'total_days': len(backtest_data)
                },
                'strategy_weights': self.strategy_weights,
                'benchmark_weights': self.benchmark_weights,
                'rebalancing_frequency': self.rebalancing_frequency,
                'performance_metrics': performance_metrics,
                'returns_data': {
                    'strategy_returns': self.strategy_returns.to_dict(),
                    'benchmark_returns': self.benchmark_returns.to_dict(),
                    'strategy_cumulative': self.strategy_cumulative.to_dict(),
                    'benchmark_cumulative': self.benchmark_cumulative.to_dict(),
                    'strategy_drawdown': self.strategy_drawdown.to_dict(),
                    'benchmark_drawdown': self.benchmark_drawdown.to_dict()
                },
                'backtest_date': datetime.now().isoformat()
            }

            logger.info("Backtest completed successfully")
            return self.backtest_results

        except Exception as e:
            logger.error(f"Backtesting failed: {str(e)}")
            raise RuntimeError(f"Backtesting failed: {str(e)}")

    def _filter_data_by_date(self, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """
        Filter returns data by date range.

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

    def _calculate_portfolio_returns(self, returns_data: pd.DataFrame) -> None:
        """
        Calculate portfolio returns for strategy and benchmark.

        Args:
            returns_data: Filtered returns data
        """
        # Calculate strategy returns
        strategy_returns = pd.Series(0.0, index=returns_data.index)
        for asset, weight in self.strategy_weights.items():
            if asset in returns_data.columns:
                strategy_returns += weight * returns_data[asset]

        # Calculate benchmark returns
        benchmark_returns = pd.Series(0.0, index=returns_data.index)
        for asset, weight in self.benchmark_weights.items():
            if asset in returns_data.columns:
                benchmark_returns += weight * returns_data[asset]

        # Store returns
        self.strategy_returns = strategy_returns
        self.benchmark_returns = benchmark_returns

        logger.info(
            f"Calculated portfolio returns for {len(returns_data)} periods")

    def _calculate_cumulative_performance(self) -> None:
        """
        Calculate cumulative performance for both portfolios.
        """
        # Strategy cumulative performance
        self.strategy_cumulative = (1 + self.strategy_returns).cumprod()

        # Benchmark cumulative performance
        self.benchmark_cumulative = (1 + self.benchmark_returns).cumprod()

        logger.info("Calculated cumulative performance")

    def _calculate_drawdowns(self) -> None:
        """
        Calculate drawdowns for both portfolios.
        """
        # Strategy drawdown
        strategy_peak = self.strategy_cumulative.expanding().max()
        self.strategy_drawdown = (
            self.strategy_cumulative - strategy_peak) / strategy_peak

        # Benchmark drawdown
        benchmark_peak = self.benchmark_cumulative.expanding().max()
        self.benchmark_drawdown = (
            self.benchmark_cumulative - benchmark_peak) / benchmark_peak

        logger.info("Calculated portfolio drawdowns")

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        # Basic return metrics
        strategy_total_return = self.strategy_cumulative.iloc[-1] - 1
        benchmark_total_return = self.benchmark_cumulative.iloc[-1] - 1

        # Annualized returns
        days = len(self.strategy_returns)
        strategy_annual_return = (
            1 + strategy_total_return) ** (252 / days) - 1
        benchmark_annual_return = (
            1 + benchmark_total_return) ** (252 / days) - 1

        # Volatility (annualized)
        strategy_volatility = self.strategy_returns.std() * np.sqrt(252)
        benchmark_volatility = self.benchmark_returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        strategy_sharpe = strategy_annual_return / \
            strategy_volatility if strategy_volatility > 0 else 0
        benchmark_sharpe = benchmark_annual_return / \
            benchmark_volatility if benchmark_volatility > 0 else 0

        # Maximum drawdown
        strategy_max_drawdown = self.strategy_drawdown.min()
        benchmark_max_drawdown = self.benchmark_drawdown.min()

        # VaR (95% confidence)
        strategy_var_95 = np.percentile(self.strategy_returns, 5)
        benchmark_var_95 = np.percentile(self.benchmark_returns, 5)

        # CVaR (95% confidence)
        strategy_cvar_95 = self.strategy_returns[self.strategy_returns <= strategy_var_95].mean(
        )
        benchmark_cvar_95 = self.benchmark_returns[self.benchmark_returns <= benchmark_var_95].mean(
        )

        # Outperformance metrics
        outperformance = strategy_total_return - benchmark_total_return
        outperformance_annual = strategy_annual_return - benchmark_annual_return

        # Volatility reduction
        volatility_reduction = benchmark_volatility - strategy_volatility

        # Sharpe ratio improvement
        sharpe_improvement = strategy_sharpe - benchmark_sharpe

        # Monthly performance analysis
        monthly_analysis = self._calculate_monthly_performance()

        # Win rate calculation
        win_rate = self._calculate_win_rate()

        metrics = {
            'strategy': {
                'total_return': float(strategy_total_return),
                'annual_return': float(strategy_annual_return),
                'volatility': float(strategy_volatility),
                'sharpe_ratio': float(strategy_sharpe),
                'max_drawdown': float(strategy_max_drawdown),
                'var_95': float(strategy_var_95),
                'cvar_95': float(strategy_cvar_95)
            },
            'benchmark': {
                'total_return': float(benchmark_total_return),
                'annual_return': float(benchmark_annual_return),
                'volatility': float(benchmark_volatility),
                'sharpe_ratio': float(benchmark_sharpe),
                'max_drawdown': float(benchmark_max_drawdown),
                'var_95': float(benchmark_var_95),
                'cvar_95': float(benchmark_cvar_95)
            },
            'comparison': {
                'outperformance': float(outperformance),
                'outperformance_annual': float(outperformance_annual),
                'volatility_reduction': float(volatility_reduction),
                'sharpe_improvement': float(sharpe_improvement),
                'volatility_reduction_pct': float(volatility_reduction / benchmark_volatility * 100) if benchmark_volatility > 0 else 0
            },
            'monthly_analysis': monthly_analysis,
            'win_rate': win_rate
        }

        return metrics

    def _calculate_monthly_performance(self) -> Dict[str, Any]:
        """
        Calculate monthly performance analysis.

        Returns:
            Dictionary with monthly performance metrics
        """
        try:
            # Resample to monthly frequency
            strategy_monthly = self.strategy_returns.resample(
                'M').apply(lambda x: (1 + x).prod() - 1)
            benchmark_monthly = self.benchmark_returns.resample(
                'M').apply(lambda x: (1 + x).prod() - 1)

            # Calculate monthly outperformance
            monthly_outperformance = strategy_monthly - benchmark_monthly

            # Monthly statistics
            monthly_stats = {
                'total_months': len(strategy_monthly),
                'positive_months_strategy': int((strategy_monthly > 0).sum()),
                'positive_months_benchmark': int((benchmark_monthly > 0).sum()),
                'positive_months_outperformance': int((monthly_outperformance > 0).sum()),
                'average_monthly_outperformance': float(monthly_outperformance.mean()),
                'monthly_outperformance_std': float(monthly_outperformance.std()),
                'best_month': {
                    'date': monthly_outperformance.idxmax().strftime('%Y-%m'),
                    'outperformance': float(monthly_outperformance.max())
                },
                'worst_month': {
                    'date': monthly_outperformance.idxmin().strftime('%Y-%m'),
                    'outperformance': float(monthly_outperformance.min())
                }
            }

            return monthly_stats

        except Exception as e:
            logger.warning(f"Monthly performance calculation failed: {str(e)}")
            return {'error': str(e)}

    def _calculate_win_rate(self) -> Dict[str, float]:
        """
        Calculate win rate metrics.

        Returns:
            Dictionary with win rate metrics
        """
        try:
            # Daily win rate
            daily_wins = (self.strategy_returns > self.benchmark_returns).sum()
            daily_total = len(self.strategy_returns)
            daily_win_rate = daily_wins / daily_total if daily_total > 0 else 0

            # Monthly win rate
            strategy_monthly = self.strategy_returns.resample(
                'M').apply(lambda x: (1 + x).prod() - 1)
            benchmark_monthly = self.benchmark_returns.resample(
                'M').apply(lambda x: (1 + x).prod() - 1)

            monthly_wins = (strategy_monthly > benchmark_monthly).sum()
            monthly_total = len(strategy_monthly)
            monthly_win_rate = monthly_wins / monthly_total if monthly_total > 0 else 0

            # Rolling win rate (30-day)
            window = min(30, len(self.strategy_returns) // 4)
            if window >= 5:
                rolling_wins = (self.strategy_returns > self.benchmark_returns).rolling(
                    window=window).mean()
                avg_rolling_win_rate = rolling_wins.mean()
            else:
                avg_rolling_win_rate = daily_win_rate

            win_rate_metrics = {
                'daily_win_rate': float(daily_win_rate * 100),
                'monthly_win_rate': float(monthly_win_rate * 100),
                'rolling_win_rate': float(avg_rolling_win_rate * 100)
            }

            return win_rate_metrics

        except Exception as e:
            logger.warning(f"Win rate calculation failed: {str(e)}")
            return {'error': str(e)}

    def get_backtest_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive backtest summary.

        Returns:
            Dictionary with backtest summary
        """
        if not self.backtest_results:
            return {'error': 'No backtest results available'}

        return self.backtest_results

    def plot_backtest_results(self, show_drawdowns: bool = True,
                              show_monthly_analysis: bool = True) -> None:
        """
        Plot comprehensive backtest results.

        Args:
            show_drawdowns: Whether to show drawdown plots
            show_monthly_analysis: Whether to show monthly analysis

        Note: This method requires matplotlib to be available.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if not self.backtest_results:
                logger.warning("No backtest results available for plotting")
                return

            # Create subplots
            n_plots = 2 + (1 if show_drawdowns else 0) + \
                (1 if show_monthly_analysis else 0)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            plot_idx = 0

            # Plot 1: Cumulative Returns
            ax = axes[plot_idx]
            ax.plot(self.strategy_cumulative.index, self.strategy_cumulative.values,
                    label='Strategy', linewidth=2, color='blue')
            ax.plot(self.benchmark_cumulative.index, self.benchmark_cumulative.values,
                    label='Benchmark', linewidth=2, color='red', alpha=0.7)
            ax.set_title('Cumulative Portfolio Performance')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

            # Plot 2: Rolling Returns (30-day)
            ax = axes[plot_idx]
            window = min(30, len(self.strategy_returns) // 4)
            if window >= 5:
                strategy_rolling = self.strategy_returns.rolling(
                    window=window).mean() * 252
                benchmark_rolling = self.benchmark_returns.rolling(
                    window=window).mean() * 252

                ax.plot(strategy_rolling.index, strategy_rolling.values,
                        label='Strategy', linewidth=2, color='blue')
                ax.plot(benchmark_rolling.index, benchmark_rolling.values,
                        label='Benchmark', linewidth=2, color='red', alpha=0.7)
                ax.set_title(f'{window}-Day Rolling Annualized Returns')
                ax.set_xlabel('Date')
                ax.set_ylabel('Annualized Return')
                ax.legend()
                ax.grid(True, alpha=0.3)
            plot_idx += 1

            # Plot 3: Drawdowns (if requested)
            if show_drawdowns and plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.fill_between(self.strategy_drawdown.index, self.strategy_drawdown.values, 0,
                                alpha=0.3, color='blue', label='Strategy')
                ax.fill_between(self.benchmark_drawdown.index, self.benchmark_drawdown.values, 0,
                                alpha=0.3, color='red', label='Benchmark')
                ax.set_title('Portfolio Drawdowns')
                ax.set_xlabel('Date')
                ax.set_ylabel('Drawdown')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1

            # Plot 4: Monthly Outperformance (if requested)
            if show_monthly_analysis and plot_idx < len(axes):
                ax = axes[plot_idx]
                strategy_monthly = self.strategy_returns.resample(
                    'M').apply(lambda x: (1 + x).prod() - 1)
                benchmark_monthly = self.benchmark_returns.resample(
                    'M').apply(lambda x: (1 + x).prod() - 1)
                monthly_outperformance = strategy_monthly - benchmark_monthly

                colors = ['green' if x >
                          0 else 'red' for x in monthly_outperformance]
                ax.bar(range(len(monthly_outperformance)), monthly_outperformance.values,
                       color=colors, alpha=0.7)
                ax.set_title('Monthly Strategy vs Benchmark Performance')
                ax.set_xlabel('Month')
                ax.set_ylabel('Outperformance')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting backtest results: {str(e)}")

    def export_backtest_results(self, output_path: Union[str, Path]) -> None:
        """
        Export backtest results to JSON file.

        Args:
            output_path: Path to save the results
        """
        if not self.backtest_results:
            logger.warning("No backtest results to export")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export to JSON
        with open(output_path, 'w') as f:
            json.dump(self.backtest_results, f, indent=2, default=str)

        logger.info(f"Backtest results exported to {output_path}")

    def get_performance_attribution(self) -> Dict[str, Any]:
        """
        Get performance attribution analysis.

        Returns:
            Dictionary with performance attribution metrics
        """
        if not self.backtest_results:
            return {'error': 'No backtest results available'}

        try:
            # Calculate asset contribution to strategy performance
            asset_contributions = {}
            for asset, weight in self.strategy_weights.items():
                if asset in self.returns_data.columns:
                    asset_returns = self.returns_data[asset]
                    asset_contribution = weight * asset_returns
                    asset_contributions[asset] = {
                        'weight': weight,
                        'total_contribution': float(asset_contribution.sum()),
                        'contribution_pct': float(asset_contribution.sum() / self.strategy_returns.sum() * 100) if self.strategy_returns.sum() != 0 else 0
                    }

            # Calculate sector/asset class contribution (if applicable)
            sector_analysis = self._analyze_sector_contribution()

            attribution = {
                'asset_contributions': asset_contributions,
                'sector_analysis': sector_analysis,
                'total_strategy_return': float(self.strategy_cumulative.iloc[-1] - 1),
                'total_benchmark_return': float(self.benchmark_cumulative.iloc[-1] - 1),
                'outperformance_decomposition': self._decompose_outperformance()
            }

            return attribution

        except Exception as e:
            logger.error(
                f"Performance attribution calculation failed: {str(e)}")
            return {'error': str(e)}

    def _analyze_sector_contribution(self) -> Dict[str, Any]:
        """
        Analyze sector contribution to performance.

        Returns:
            Dictionary with sector analysis
        """
        # This is a simplified sector analysis
        # In a real implementation, you would map assets to sectors

        # For now, group by asset characteristics
        tech_assets = ['TSLA', 'AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA']
        broad_market = ['SPY']
        fixed_income = ['BND']

        sector_contributions = {}

        # Tech sector contribution
        tech_weight = sum(self.strategy_weights.get(asset, 0)
                          for asset in tech_assets)
        if tech_weight > 0:
            tech_returns = sum(self.strategy_weights.get(asset, 0) * self.returns_data[asset]
                               for asset in tech_assets if asset in self.returns_data.columns)
            sector_contributions['technology'] = {
                'weight': float(tech_weight),
                'contribution': float(tech_returns.sum()),
                'contribution_pct': float(tech_returns.sum() / self.strategy_returns.sum() * 100) if self.strategy_returns.sum() != 0 else 0
            }

        # Broad market contribution
        broad_weight = sum(self.strategy_weights.get(asset, 0)
                           for asset in broad_market)
        if broad_weight > 0:
            broad_returns = sum(self.strategy_weights.get(asset, 0) * self.returns_data[asset]
                                for asset in broad_market if asset in self.returns_data.columns)
            sector_contributions['broad_market'] = {
                'weight': float(broad_weight),
                'contribution': float(broad_returns.sum()),
                'contribution_pct': float(broad_returns.sum() / self.strategy_returns.sum() * 100) if self.strategy_returns.sum() != 0 else 0
            }

        # Fixed income contribution
        fixed_weight = sum(self.strategy_weights.get(asset, 0)
                           for asset in fixed_income)
        if fixed_weight > 0:
            fixed_returns = sum(self.strategy_weights.get(asset, 0) * self.returns_data[asset]
                                for asset in fixed_income if asset in self.returns_data.columns)
            sector_contributions['fixed_income'] = {
                'weight': float(fixed_weight),
                'contribution': float(fixed_returns.sum()),
                'contribution_pct': float(fixed_returns.sum() / self.strategy_returns.sum() * 100) if self.strategy_returns.sum() != 0 else 0
            }

        return sector_contributions

    def _decompose_outperformance(self) -> Dict[str, float]:
        """
        Decompose outperformance into components.

        Returns:
            Dictionary with outperformance decomposition
        """
        try:
            # Calculate various outperformance components
            strategy_return = self.strategy_cumulative.iloc[-1] - 1
            benchmark_return = self.benchmark_cumulative.iloc[-1] - 1
            outperformance = strategy_return - benchmark_return

            # Decompose by return vs risk
            return_contribution = strategy_return - benchmark_return
            risk_contribution = (benchmark_return - self.strategy_returns.mean() * 252) - \
                (benchmark_return - self.benchmark_returns.mean() * 252)

            # Decompose by asset allocation vs stock selection
            # This is a simplified approach
            allocation_effect = 0.0
            selection_effect = outperformance - allocation_effect

            decomposition = {
                'total_outperformance': float(outperformance),
                'return_contribution': float(return_contribution),
                'risk_contribution': float(risk_contribution),
                'allocation_effect': float(allocation_effect),
                'selection_effect': float(selection_effect)
            }

            return decomposition

        except Exception as e:
            logger.warning(f"Outperformance decomposition failed: {str(e)}")
            return {'error': str(e)}

"""
Efficient frontier analysis and visualization module.

This module provides comprehensive efficient frontier analysis
including generation, visualization, and analysis capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import json
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class EfficientFrontier:
    """
    Efficient frontier analysis and visualization for portfolio optimization.

    Provides comprehensive analysis of the efficient frontier including
    generation, visualization, and risk-return analysis.

    Attributes:
        frontier_data (pd.DataFrame): Efficient frontier portfolios data
        portfolio_optimizer (PortfolioOptimizer): Portfolio optimizer instance
        analysis_results (Dict): Analysis results and insights
    """

    def __init__(self, portfolio_optimizer):
        """
        Initialize the Efficient Frontier analyzer.

        Args:
            portfolio_optimizer: PortfolioOptimizer instance
        """
        self.portfolio_optimizer = portfolio_optimizer
        self.frontier_data = pd.DataFrame()
        self.analysis_results = {}

        logger.info("Initialized Efficient Frontier analyzer")

    def generate_frontier(self, num_portfolios: int = 100,
                          return_range: Optional[Tuple[float, float]] = None,
                          save_results: bool = True) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios.

        Args:
            num_portfolios: Number of portfolios to generate
            return_range: Tuple of (min_return, max_return) for target returns
            save_results: Whether to save results to the instance

        Returns:
            DataFrame with efficient frontier portfolios
        """
        try:
            # Generate efficient frontier using portfolio optimizer
            frontier_df = self.portfolio_optimizer.generate_efficient_frontier(
                num_portfolios=num_portfolios,
                return_range=return_range
            )

            if save_results:
                self.frontier_data = frontier_df.copy()
                self._analyze_frontier()

            logger.info(
                f"Generated efficient frontier with {len(frontier_df)} portfolios")
            return frontier_df

        except Exception as e:
            logger.error(f"Efficient frontier generation failed: {str(e)}")
            raise RuntimeError(
                f"Efficient frontier generation failed: {str(e)}")

    def _analyze_frontier(self) -> None:
        """
        Analyze the generated efficient frontier.
        """
        if self.frontier_data.empty:
            return

        try:
            # Basic statistics
            analysis = {
                'total_portfolios': len(self.frontier_data),
                'return_range': {
                    'min': float(self.frontier_data['actual_return'].min()),
                    'max': float(self.frontier_data['actual_return'].max()),
                    'mean': float(self.frontier_data['actual_return'].mean())
                },
                'volatility_range': {
                    'min': float(self.frontier_data['volatility'].min()),
                    'max': float(self.frontier_data['volatility'].max()),
                    'mean': float(self.frontier_data['volatility'].mean())
                },
                'sharpe_range': {
                    'min': float(self.frontier_data['sharpe_ratio'].min()),
                    'max': float(self.frontier_data['sharpe_ratio'].max()),
                    'mean': float(self.frontier_data['sharpe_ratio'].mean())
                }
            }

            # Find optimal portfolios
            analysis['optimal_portfolios'] = {
                'max_sharpe': self._find_max_sharpe_portfolio(),
                'min_volatility': self._find_min_volatility_portfolio(),
                'max_return': self._find_max_return_portfolio()
            }

            # Risk-return analysis
            analysis['risk_return_analysis'] = self._analyze_risk_return_relationship()

            # Diversification analysis
            analysis['diversification_analysis'] = self._analyze_diversification()

            # Store analysis results
            self.analysis_results = analysis

            logger.info("Completed efficient frontier analysis")

        except Exception as e:
            logger.error(f"Frontier analysis failed: {str(e)}")

    def _find_max_sharpe_portfolio(self) -> Dict[str, Any]:
        """
        Find the portfolio with maximum Sharpe ratio.

        Returns:
            Dictionary with portfolio information
        """
        if self.frontier_data.empty:
            return {}

        max_sharpe_idx = self.frontier_data['sharpe_ratio'].idxmax()
        portfolio = self.frontier_data.loc[max_sharpe_idx]

        return {
            'index': int(max_sharpe_idx),
            'return': float(portfolio['actual_return']),
            'volatility': float(portfolio['volatility']),
            'sharpe_ratio': float(portfolio['sharpe_ratio']),
            'weights': self._extract_weights(portfolio)
        }

    def _find_min_volatility_portfolio(self) -> Dict[str, Any]:
        """
        Find the portfolio with minimum volatility.

        Returns:
            Dictionary with portfolio information
        """
        if self.frontier_data.empty:
            return {}

        min_vol_idx = self.frontier_data['volatility'].idxmin()
        portfolio = self.frontier_data.loc[min_vol_idx]

        return {
            'index': int(min_vol_idx),
            'return': float(portfolio['actual_return']),
            'volatility': float(portfolio['volatility']),
            'sharpe_ratio': float(portfolio['sharpe_ratio']),
            'weights': self._extract_weights(portfolio)
        }

    def _find_max_return_portfolio(self) -> Dict[str, Any]:
        """
        Find the portfolio with maximum return.

        Returns:
            Dictionary with portfolio information
        """
        if self.frontier_data.empty:
            return {}

        max_return_idx = self.frontier_data['actual_return'].idxmax()
        portfolio = self.frontier_data.loc[max_return_idx]

        return {
            'index': int(max_return_idx),
            'return': float(portfolio['actual_return']),
            'volatility': float(portfolio['volatility']),
            'sharpe_ratio': float(portfolio['sharpe_ratio']),
            'weights': self._extract_weights(portfolio)
        }

    def _extract_weights(self, portfolio: pd.Series) -> Dict[str, float]:
        """
        Extract asset weights from portfolio data.

        Args:
            portfolio: Portfolio series data

        Returns:
            Dictionary with asset weights
        """
        weights = {}
        for col in portfolio.index:
            if col.startswith('weight_'):
                asset = col.replace('weight_', '')
                weights[asset] = float(portfolio[col])

        return weights

    def _analyze_risk_return_relationship(self) -> Dict[str, Any]:
        """
        Analyze the risk-return relationship in the efficient frontier.

        Returns:
            Dictionary with risk-return analysis
        """
        if self.frontier_data.empty:
            return {}

        try:
            # Calculate correlation between return and volatility
            correlation = self.frontier_data['actual_return'].corr(
                self.frontier_data['volatility'])

            # Calculate slope of the efficient frontier
            # Use linear regression on the efficient part (sorted by return)
            sorted_data = self.frontier_data.sort_values('actual_return')

            # Calculate rolling slope (5-portfolio window)
            window_size = min(5, len(sorted_data) // 4)
            if window_size >= 2:
                slopes = []
                for i in range(window_size, len(sorted_data)):
                    window_data = sorted_data.iloc[i-window_size:i+1]
                    slope = np.polyfit(
                        window_data['volatility'], window_data['actual_return'], 1)[0]
                    slopes.append(slope)

                avg_slope = np.mean(slopes) if slopes else 0
            else:
                avg_slope = 0

            # Calculate risk-adjusted return efficiency
            # Sort by volatility and calculate cumulative return improvement
            vol_sorted = sorted_data.sort_values('volatility')
            cumulative_return = vol_sorted['actual_return'].cumsum()
            efficiency_score = cumulative_return.iloc[-1] / len(vol_sorted)

            analysis = {
                'return_volatility_correlation': float(correlation),
                'average_frontier_slope': float(avg_slope),
                'efficiency_score': float(efficiency_score),
                'risk_return_tradeoff': 'positive' if correlation > 0 else 'negative'
            }

            return analysis

        except Exception as e:
            logger.warning(f"Risk-return analysis failed: {str(e)}")
            return {'error': str(e)}

    def _analyze_diversification(self) -> Dict[str, Any]:
        """
        Analyze portfolio diversification across the efficient frontier.

        Returns:
            Dictionary with diversification analysis
        """
        if self.frontier_data.empty:
            return {}

        try:
            # Extract weight columns
            weight_cols = [
                col for col in self.frontier_data.columns if col.startswith('weight_')]

            if not weight_cols:
                return {'error': 'No weight data available'}

            # Calculate concentration metrics for each portfolio
            concentration_metrics = []

            for _, portfolio in self.frontier_data.iterrows():
                weights = [portfolio[col] for col in weight_cols]

                # Herfindahl-Hirschman Index (HHI)
                hhi = sum(w**2 for w in weights)

                # Effective number of assets
                effective_assets = 1 / hhi if hhi > 0 else 0

                # Maximum weight
                max_weight = max(weights)

                concentration_metrics.append({
                    'hhi': hhi,
                    'effective_assets': effective_assets,
                    'max_weight': max_weight
                })

            # Calculate summary statistics
            hhi_values = [m['hhi'] for m in concentration_metrics]
            effective_assets_values = [m['effective_assets']
                                       for m in concentration_metrics]
            max_weight_values = [m['max_weight']
                                 for m in concentration_metrics]

            analysis = {
                'concentration_metrics': {
                    'hhi': {
                        'mean': float(np.mean(hhi_values)),
                        'min': float(np.min(hhi_values)),
                        'max': float(np.max(hhi_values))
                    },
                    'effective_assets': {
                        'mean': float(np.mean(effective_assets_values)),
                        'min': float(np.min(effective_assets_values)),
                        'max': float(np.max(effective_assets_values))
                    },
                    'max_weight': {
                        'mean': float(np.mean(max_weight_values)),
                        'min': float(np.min(max_weight_values)),
                        'max': float(np.max(max_weight_values))
                    }
                },
                'diversification_trend': self._analyze_diversification_trend()
            }

            return analysis

        except Exception as e:
            logger.warning(f"Diversification analysis failed: {str(e)}")
            return {'error': str(e)}

    def _analyze_diversification_trend(self) -> Dict[str, Any]:
        """
        Analyze how diversification changes across the efficient frontier.

        Returns:
            Dictionary with diversification trend analysis
        """
        if self.frontier_data.empty:
            return {}

        try:
            # Sort by return to analyze trend
            sorted_data = self.frontier_data.sort_values('actual_return')

            # Calculate concentration metrics for each portfolio
            weight_cols = [
                col for col in sorted_data.columns if col.startswith('weight_')]

            hhi_trend = []
            effective_assets_trend = []

            for _, portfolio in sorted_data.iterrows():
                weights = [portfolio[col] for col in weight_cols]
                hhi = sum(w**2 for w in weights)
                effective_assets = 1 / hhi if hhi > 0 else 0

                hhi_trend.append(hhi)
                effective_assets_trend.append(effective_assets)

            # Calculate trend correlation
            returns = sorted_data['actual_return'].values
            hhi_correlation = np.corrcoef(returns, hhi_trend)[0, 1]
            effective_assets_correlation = np.corrcoef(
                returns, effective_assets_trend)[0, 1]

            # Determine trend direction
            def get_trend_direction(correlation):
                if abs(correlation) < 0.1:
                    return 'stable'
                elif correlation > 0:
                    return 'increasing'
                else:
                    return 'decreasing'

            analysis = {
                'hhi_trend': {
                    'correlation_with_return': float(hhi_correlation),
                    'direction': get_trend_direction(hhi_correlation),
                    'trend_strength': 'strong' if abs(hhi_correlation) > 0.5 else 'weak'
                },
                'effective_assets_trend': {
                    'correlation_with_return': float(effective_assets_correlation),
                    'direction': get_trend_direction(effective_assets_correlation),
                    'trend_strength': 'strong' if abs(effective_assets_correlation) > 0.5 else 'weak'
                }
            }

            return analysis

        except Exception as e:
            logger.warning(f"Diversification trend analysis failed: {str(e)}")
            return {'error': str(e)}

    def get_frontier_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of the efficient frontier.

        Returns:
            Dictionary with frontier summary
        """
        if self.frontier_data.empty:
            return {'error': 'No frontier data available'}

        summary = {
            'frontier_overview': {
                'total_portfolios': len(self.frontier_data),
                'generation_date': datetime.now().isoformat(),
                'assets': list(self.portfolio_optimizer.returns_data.columns)
            },
            'performance_summary': self.analysis_results.get('optimal_portfolios', {}),
            'risk_return_analysis': self.analysis_results.get('risk_return_analysis', {}),
            'diversification_analysis': self.analysis_results.get('diversification_analysis', {})
        }

        return summary

    def plot_efficient_frontier(self, show_optimal_portfolios: bool = True,
                                show_individual_assets: bool = True) -> None:
        """
        Plot the efficient frontier with optional features.

        Args:
            show_optimal_portfolios: Whether to highlight optimal portfolios
            show_individual_assets: Whether to show individual asset positions

        Note: This method requires matplotlib to be available.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if self.frontier_data.empty:
                logger.warning("No frontier data available for plotting")
                return

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot efficient frontier
            ax.scatter(self.frontier_data['volatility'],
                       self.frontier_data['actual_return'],
                       c=self.frontier_data['sharpe_ratio'],
                       cmap='viridis',
                       alpha=0.6,
                       s=30)

            # Add colorbar
            scatter = ax.scatter(self.frontier_data['volatility'],
                                 self.frontier_data['actual_return'],
                                 c=self.frontier_data['sharpe_ratio'],
                                 cmap='viridis')
            plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')

            # Highlight optimal portfolios
            if show_optimal_portfolios and 'optimal_portfolios' in self.analysis_results:
                opt_portfolios = self.analysis_results['optimal_portfolios']

                # Max Sharpe
                if 'max_sharpe' in opt_portfolios:
                    max_sharpe = opt_portfolios['max_sharpe']
                    ax.scatter(max_sharpe['volatility'], max_sharpe['return'],
                               color='red', s=200, marker='*',
                               label=f"Max Sharpe (SR: {max_sharpe['sharpe_ratio']:.3f})")

                # Min Volatility
                if 'min_volatility' in opt_portfolios:
                    min_vol = opt_portfolios['min_volatility']
                    ax.scatter(min_vol['volatility'], min_vol['return'],
                               color='green', s=200, marker='s',
                               label=f"Min Volatility (Vol: {min_vol['volatility']:.3f})")

                # Max Return
                if 'max_return' in opt_portfolios:
                    max_return = opt_portfolios['max_return']
                    ax.scatter(max_return['volatility'], max_return['return'],
                               color='orange', s=200, marker='^',
                               label=f"Max Return (Ret: {max_return['return']:.3f})")

            # Show individual assets
            if show_individual_assets:
                asset_vol = self.portfolio_optimizer.volatility
                asset_ret = self.portfolio_optimizer.expected_returns

                ax.scatter(asset_vol, asset_ret,
                           color='black', s=100, marker='o',
                           label='Individual Assets')

                # Add asset labels
                for asset in asset_vol.index:
                    ax.annotate(asset, (asset_vol[asset], asset_ret[asset]),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, alpha=0.7)

            # Customize plot
            ax.set_xlabel('Portfolio Volatility (Annualized)')
            ax.set_ylabel('Portfolio Return (Annualized)')
            ax.set_title('Efficient Frontier Analysis')
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting efficient frontier: {str(e)}")

    def export_frontier_data(self, output_path: Union[str, Path]) -> None:
        """
        Export efficient frontier data to CSV file.

        Args:
            output_path: Path to save the data
        """
        if self.frontier_data.empty:
            logger.warning("No frontier data to export")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export to CSV
        self.frontier_data.to_csv(output_path, index=False)

        logger.info(f"Efficient frontier data exported to {output_path}")

    def export_analysis_results(self, output_path: Union[str, Path]) -> None:
        """
        Export analysis results to JSON file.

        Args:
            output_path: Path to save the results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export to JSON
        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)

        logger.info(f"Analysis results exported to {output_path}")

    def get_portfolio_by_characteristics(self, target_return: Optional[float] = None,
                                         target_volatility: Optional[float] = None,
                                         target_sharpe: Optional[float] = None) -> pd.DataFrame:
        """
        Find portfolios matching specific characteristics.

        Args:
            target_return: Target return (within 5% tolerance)
            target_volatility: Target volatility (within 5% tolerance)
            target_sharpe: Target Sharpe ratio (within 5% tolerance)

        Returns:
            DataFrame with matching portfolios
        """
        if self.frontier_data.empty:
            return pd.DataFrame()

        # Create filter mask
        mask = pd.Series([True] * len(self.frontier_data),
                         index=self.frontier_data.index)

        if target_return is not None:
            tolerance = target_return * 0.05
            mask &= (self.frontier_data['actual_return'] >= target_return - tolerance) & \
                (self.frontier_data['actual_return']
                 <= target_return + tolerance)

        if target_volatility is not None:
            tolerance = target_volatility * 0.05
            mask &= (self.frontier_data['volatility'] >= target_volatility - tolerance) & \
                (self.frontier_data['volatility']
                 <= target_volatility + tolerance)

        if target_sharpe is not None:
            tolerance = target_sharpe * 0.05
            mask &= (self.frontier_data['sharpe_ratio'] >= target_sharpe - tolerance) & \
                (self.frontier_data['sharpe_ratio']
                 <= target_sharpe + tolerance)

        # Return filtered portfolios
        matching_portfolios = self.frontier_data[mask]

        logger.info(
            f"Found {len(matching_portfolios)} portfolios matching criteria")
        return matching_portfolios

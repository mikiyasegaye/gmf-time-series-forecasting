"""
Portfolio optimization module using Modern Portfolio Theory.

This module provides comprehensive portfolio optimization capabilities
including risk-return optimization, efficient frontier generation,
and portfolio allocation strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import json
from datetime import datetime
import warnings

# Scipy imports for optimization
try:
    from scipy.optimize import minimize
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn(
        "Scipy not available. Portfolio optimization functionality will be limited.")

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Advanced portfolio optimizer using Modern Portfolio Theory.

    Implements comprehensive portfolio optimization with multiple
    objective functions and constraints.

    Attributes:
        returns_data (pd.DataFrame): Asset returns data
        risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
        optimization_results (Dict): Results from optimization runs
        constraints (Dict): Optimization constraints
    """

    def __init__(self, returns_data: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize the Portfolio Optimizer.

        Args:
            returns_data: DataFrame with asset returns (assets as columns)
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "Scipy is required for portfolio optimization functionality")

        self.returns_data = returns_data.copy()
        self.risk_free_rate = risk_free_rate
        self.optimization_results = {}
        self.constraints = {}

        # Calculate basic statistics
        self._calculate_basic_stats()

        logger.info(
            f"Initialized Portfolio Optimizer with {len(returns_data.columns)} assets")

    def _calculate_basic_stats(self) -> None:
        """
        Calculate basic statistics for all assets.
        """
        # Expected returns (annualized)
        self.expected_returns = self.returns_data.mean() * 252

        # Covariance matrix (annualized)
        self.covariance_matrix = self.returns_data.cov() * 252

        # Correlation matrix
        self.correlation_matrix = self.returns_data.corr()

        # Volatility (annualized)
        self.volatility = self.returns_data.std() * np.sqrt(252)

        logger.info("Calculated basic portfolio statistics")

    def set_constraints(self, min_weight: float = 0.0, max_weight: float = 1.0,
                        target_return: Optional[float] = None,
                        target_volatility: Optional[float] = None) -> None:
        """
        Set optimization constraints.

        Args:
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            target_return: Target portfolio return (optional)
            target_volatility: Target portfolio volatility (optional)
        """
        self.constraints = {
            'min_weight': min_weight,
            'max_weight': max_weight,
            'target_return': target_return,
            'target_volatility': target_volatility
        }

        logger.info(
            f"Set optimization constraints: min_weight={min_weight}, max_weight={max_weight}")

    def optimize_max_sharpe(self, constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize portfolio for maximum Sharpe ratio.

        Args:
            constraints: Optional constraints override

        Returns:
            Dictionary with optimization results

        Raises:
            RuntimeError: If optimization fails
        """
        try:
            # Use provided constraints or default ones
            opt_constraints = constraints or self.constraints

            # Objective function: negative Sharpe ratio (minimize)
            def objective(weights):
                portfolio_return = np.sum(weights * self.expected_returns)
                portfolio_vol = np.sqrt(
                    np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
                sharpe_ratio = (portfolio_return -
                                self.risk_free_rate) / portfolio_vol
                return -sharpe_ratio  # Negative because we minimize

            # Constraints
            constraints_list = [
                # Weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]

            # Add target return constraint if specified
            if opt_constraints.get('target_return') is not None:
                constraints_list.append({
                    'type': 'eq',
                    'fun': lambda x: np.sum(x * self.expected_returns) - opt_constraints['target_return']
                })

            # Bounds for weights
            bounds = [(opt_constraints.get('min_weight', 0.0),
                      opt_constraints.get('max_weight', 1.0)) for _ in range(len(self.returns_data.columns))]

            # Initial guess (equal weights)
            initial_weights = np.array(
                [1.0 / len(self.returns_data.columns)] * len(self.returns_data.columns))

            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000}
            )

            if not result.success:
                raise RuntimeError(f"Optimization failed: {result.message}")

            # Calculate portfolio metrics
            optimal_weights = result.x
            portfolio_metrics = self._calculate_portfolio_metrics(
                optimal_weights)

            # Store results
            optimization_result = {
                'objective': 'max_sharpe',
                'weights': dict(zip(self.returns_data.columns, optimal_weights)),
                'metrics': portfolio_metrics,
                'optimization_success': result.success,
                'iterations': result.nit,
                'timestamp': datetime.now().isoformat()
            }

            self.optimization_results['max_sharpe'] = optimization_result

            logger.info(
                f"Max Sharpe optimization completed. Sharpe ratio: {portfolio_metrics['sharpe_ratio']:.4f}")
            return optimization_result

        except Exception as e:
            logger.error(f"Max Sharpe optimization failed: {str(e)}")
            raise RuntimeError(f"Max Sharpe optimization failed: {str(e)}")

    def optimize_min_volatility(self, constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize portfolio for minimum volatility.

        Args:
            constraints: Optional constraints override

        Returns:
            Dictionary with optimization results

        Raises:
            RuntimeError: If optimization fails
        """
        try:
            # Use provided constraints or default ones
            opt_constraints = constraints or self.constraints

            # Objective function: portfolio volatility
            def objective(weights):
                portfolio_vol = np.sqrt(
                    np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
                return portfolio_vol

            # Constraints
            constraints_list = [
                # Weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]

            # Add target return constraint if specified
            if opt_constraints.get('target_return') is not None:
                constraints_list.append({
                    'type': 'eq',
                    'fun': lambda x: np.sum(x * self.expected_returns) - opt_constraints['target_return']
                })

            # Bounds for weights
            bounds = [(opt_constraints.get('min_weight', 0.0),
                      opt_constraints.get('max_weight', 1.0)) for _ in range(len(self.returns_data.columns))]

            # Initial guess (equal weights)
            initial_weights = np.array(
                [1.0 / len(self.returns_data.columns)] * len(self.returns_data.columns))

            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000}
            )

            if not result.success:
                raise RuntimeError(f"Optimization failed: {result.message}")

            # Calculate portfolio metrics
            optimal_weights = result.x
            portfolio_metrics = self._calculate_portfolio_metrics(
                optimal_weights)

            # Store results
            optimization_result = {
                'objective': 'min_volatility',
                'weights': dict(zip(self.returns_data.columns, optimal_weights)),
                'metrics': portfolio_metrics,
                'optimization_success': result.success,
                'iterations': result.nit,
                'timestamp': datetime.now().isoformat()
            }

            self.optimization_results['min_volatility'] = optimization_result

            logger.info(
                f"Min volatility optimization completed. Volatility: {portfolio_metrics['volatility']:.4f}")
            return optimization_result

        except Exception as e:
            logger.error(f"Min volatility optimization failed: {str(e)}")
            raise RuntimeError(f"Min volatility optimization failed: {str(e)}")

    def optimize_target_return(self, target_return: float, constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize portfolio for a target return with minimum volatility.

        Args:
            target_return: Target annual return
            constraints: Optional constraints override

        Returns:
            Dictionary with optimization results

        Raises:
            RuntimeError: If optimization fails
        """
        try:
            # Use provided constraints or default ones
            opt_constraints = constraints or self.constraints.copy()
            opt_constraints['target_return'] = target_return

            # Use min volatility optimization with target return constraint
            return self.optimize_min_volatility(opt_constraints)

        except Exception as e:
            logger.error(f"Target return optimization failed: {str(e)}")
            raise RuntimeError(f"Target return optimization failed: {str(e)}")

    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio metrics.

        Args:
            weights: Portfolio weights array

        Returns:
            Dictionary with portfolio metrics
        """
        # Portfolio return
        portfolio_return = np.sum(weights * self.expected_returns)

        # Portfolio volatility
        portfolio_vol = np.sqrt(
            np.dot(weights.T, np.dot(self.covariance_matrix, weights)))

        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / \
            portfolio_vol if portfolio_vol > 0 else 0

        # Portfolio beta (assuming first asset as market proxy)
        if len(self.returns_data.columns) > 0:
            # First asset as market proxy
            market_returns = self.returns_data.iloc[:, 0]
            market_var = np.var(market_returns) * 252
            portfolio_cov = np.sum(
                weights * np.cov(self.returns_data.T, market_returns)[:-1, -1] * 252)
            portfolio_beta = portfolio_cov / market_var if market_var > 0 else 0
        else:
            portfolio_beta = 0

        # VaR (Value at Risk) - 95% confidence
        portfolio_returns = np.dot(self.returns_data, weights)
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)

        # CVaR (Conditional Value at Risk) - 95% confidence
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= np.percentile(
            portfolio_returns, 5)]) * np.sqrt(252)

        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        # Convert to pandas Series for expanding operations
        cumulative_series = pd.Series(cumulative_returns)
        running_max = cumulative_series.expanding().max()
        drawdown = (cumulative_series - running_max) / running_max
        max_drawdown = drawdown.min()

        # Diversification ratio
        weighted_vol = np.sum(weights * self.volatility)
        diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 0

        metrics = {
            'return': float(portfolio_return),
            'volatility': float(portfolio_vol),
            'sharpe_ratio': float(sharpe_ratio),
            'beta': float(portfolio_beta),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'max_drawdown': float(max_drawdown),
            'diversification_ratio': float(diversification_ratio)
        }

        return metrics

    def generate_efficient_frontier(self, num_portfolios: int = 100,
                                    return_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios.

        Args:
            num_portfolios: Number of portfolios to generate
            return_range: Tuple of (min_return, max_return) for target returns

        Returns:
            DataFrame with efficient frontier portfolios
        """
        try:
            # Determine return range if not provided
            if return_range is None:
                min_return = self.expected_returns.min()
                max_return = self.expected_returns.max()
                return_range = (min_return, max_return)

            # Generate target returns
            target_returns = np.linspace(
                return_range[0], return_range[1], num_portfolios)

            # Store efficient frontier portfolios
            efficient_frontier = []

            for target_return in target_returns:
                try:
                    # Optimize for this target return
                    result = self.optimize_target_return(target_return)

                    # Add to efficient frontier
                    portfolio_data = {
                        'target_return': target_return,
                        'actual_return': result['metrics']['return'],
                        'volatility': result['metrics']['volatility'],
                        'sharpe_ratio': result['metrics']['sharpe_ratio'],
                        'weights': result['weights']
                    }

                    efficient_frontier.append(portfolio_data)

                except Exception as e:
                    logger.warning(
                        f"Failed to optimize for target return {target_return:.4f}: {str(e)}")
                    continue

            # Create DataFrame
            frontier_df = pd.DataFrame(efficient_frontier)

            # Add weight columns
            for asset in self.returns_data.columns:
                frontier_df[f'weight_{asset}'] = [portfolio['weights'].get(
                    asset, 0.0) for portfolio in efficient_frontier]

            # Remove the weights dictionary column
            frontier_df = frontier_df.drop('weights', axis=1)

            logger.info(
                f"Generated efficient frontier with {len(frontier_df)} portfolios")
            return frontier_df

        except Exception as e:
            logger.error(f"Efficient frontier generation failed: {str(e)}")
            raise RuntimeError(
                f"Efficient frontier generation failed: {str(e)}")

    def get_portfolio_recommendations(self, risk_tolerance: str = 'moderate') -> Dict[str, Any]:
        """
        Get portfolio recommendations based on risk tolerance.

        Args:
            risk_tolerance: Risk tolerance level ('conservative', 'moderate', 'aggressive')

        Returns:
            Dictionary with portfolio recommendations
        """
        try:
            recommendations = {}

            # Get max Sharpe portfolio (balanced)
            if 'max_sharpe' in self.optimization_results:
                recommendations['balanced'] = self.optimization_results['max_sharpe']

            # Get min volatility portfolio (conservative)
            if 'min_volatility' in self.optimization_results:
                recommendations['conservative'] = self.optimization_results['min_volatility']

            # Generate aggressive portfolio (higher return target)
            if len(self.optimization_results) > 0:
                # Use 75th percentile of expected returns as aggressive target
                aggressive_return = np.percentile(self.expected_returns, 75)
                try:
                    aggressive_result = self.optimize_target_return(
                        aggressive_return)
                    recommendations['aggressive'] = aggressive_result
                except Exception as e:
                    logger.warning(
                        f"Could not generate aggressive portfolio: {str(e)}")

            # Map to risk tolerance
            risk_mapping = {
                'conservative': 'conservative',
                'moderate': 'balanced',
                'aggressive': 'aggressive'
            }

            selected_portfolio = recommendations.get(
                risk_mapping.get(risk_tolerance, 'moderate'))

            if selected_portfolio is None:
                # Fallback to first available portfolio
                selected_portfolio = list(recommendations.values())[
                    0] if recommendations else None

            result = {
                'risk_tolerance': risk_tolerance,
                'recommended_portfolio': selected_portfolio,
                'all_recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(
                f"Generated portfolio recommendations for {risk_tolerance} risk tolerance")
            return result

        except Exception as e:
            logger.error(
                f"Portfolio recommendations generation failed: {str(e)}")
            raise RuntimeError(
                f"Portfolio recommendations generation failed: {str(e)}")

    def calculate_portfolio_risk_decomposition(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate risk decomposition by asset.

        Args:
            weights: Portfolio weights array

        Returns:
            Dictionary with risk contribution by asset
        """
        try:
            portfolio_vol = np.sqrt(
                np.dot(weights.T, np.dot(self.covariance_matrix, weights)))

            # Calculate marginal contribution to risk
            marginal_contrib = np.dot(
                self.covariance_matrix, weights) / portfolio_vol

            # Calculate percentage contribution to risk
            risk_contrib = weights * marginal_contrib / portfolio_vol

            # Create risk decomposition dictionary
            risk_decomp = {}
            for i, asset in enumerate(self.returns_data.columns):
                risk_decomp[asset] = {
                    'weight': float(weights[i]),
                    'marginal_contribution': float(marginal_contrib[i]),
                    'risk_contribution': float(risk_contrib[i]),
                    'risk_contribution_pct': float(risk_contrib[i] * 100)
                }

            return risk_decomp

        except Exception as e:
            logger.error(f"Risk decomposition calculation failed: {str(e)}")
            raise RuntimeError(
                f"Risk decomposition calculation failed: {str(e)}")

    def save_optimization_results(self, filepath: Union[str, Path]) -> None:
        """
        Save optimization results to JSON file.

        Args:
            filepath: Path to save the results
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for JSON serialization
        export_data = {
            'export_date': datetime.now().isoformat(),
            'risk_free_rate': self.risk_free_rate,
            'constraints': self.constraints,
            'optimization_results': self.optimization_results,
            'basic_stats': {
                'expected_returns': self.expected_returns.to_dict(),
                'volatility': self.volatility.to_dict(),
                'correlation_matrix': self.correlation_matrix.to_dict()
            }
        }

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Optimization results saved to {filepath}")

    def load_optimization_results(self, filepath: Union[str, Path]) -> None:
        """
        Load optimization results from JSON file.

        Args:
            filepath: Path to the results file
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        # Load from JSON
        with open(filepath, 'r') as f:
            import_data = json.load(f)

        # Restore data
        self.risk_free_rate = import_data.get('risk_free_rate', 0.02)
        self.constraints = import_data.get('constraints', {})
        self.optimization_results = import_data.get('optimization_results', {})

        # Restore basic stats
        basic_stats = import_data.get('basic_stats', {})
        if 'expected_returns' in basic_stats:
            self.expected_returns = pd.Series(basic_stats['expected_returns'])
        if 'volatility' in basic_stats:
            self.volatility = pd.Series(basic_stats['volatility'])
        if 'correlation_matrix' in basic_stats:
            self.correlation_matrix = pd.DataFrame(
                basic_stats['correlation_matrix'])

        logger.info(f"Optimization results loaded from {filepath}")

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of all optimization results.

        Returns:
            Dictionary with optimization summary
        """
        summary = {
            'total_optimizations': len(self.optimization_results),
            'optimization_types': list(self.optimization_results.keys()),
            'risk_free_rate': self.risk_free_rate,
            'constraints': self.constraints,
            'assets': list(self.returns_data.columns),
            'data_points': len(self.returns_data)
        }

        # Add performance summary for each optimization
        performance_summary = {}
        for opt_type, results in self.optimization_results.items():
            metrics = results['metrics']
            performance_summary[opt_type] = {
                'return': metrics['return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio']
            }

        summary['performance_summary'] = performance_summary

        return summary

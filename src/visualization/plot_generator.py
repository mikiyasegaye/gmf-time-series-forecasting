"""
Plot generation utilities for financial analysis.

This module provides comprehensive plotting capabilities for
financial data, forecasts, and portfolio analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class PlotGenerator:
    """
    Comprehensive plot generator for financial analysis.

    Provides a wide range of plotting capabilities including
    time series plots, portfolio analysis, and risk metrics.

    Attributes:
        plot_style (str): Plotting style configuration
        color_palette (List[str]): Color palette for plots
        figure_size (Tuple[int, int]): Default figure size
    """

    def __init__(self, plot_style: str = 'default', figure_size: Tuple[int, int] = (12, 8)):
        """
        Initialize the Plot Generator.

        Args:
            plot_style: Plotting style ('default', 'seaborn', 'matplotlib')
            figure_size: Default figure size (width, height)
        """
        self.plot_style = plot_style
        self.figure_size = figure_size
        self.color_palette = self._initialize_color_palette()

        # Set plotting style
        self._set_plotting_style()

        logger.info(f"Initialized Plot Generator with style: {plot_style}")

    def _initialize_color_palette(self) -> List[str]:
        """
        Initialize color palette for plots.

        Returns:
            List of color codes
        """
        return [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf'   # Cyan
        ]

    def _set_plotting_style(self) -> None:
        """
        Set plotting style configuration.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if self.plot_style == 'seaborn':
                sns.set_style("whitegrid")
                sns.set_palette(self.color_palette)
            elif self.plot_style == 'matplotlib':
                plt.style.use('default')
            else:
                # Default style
                plt.style.use('default')
                plt.rcParams['figure.figsize'] = self.figure_size
                plt.rcParams['font.size'] = 10
                plt.rcParams['axes.grid'] = True
                plt.rcParams['grid.alpha'] = 0.3

        except ImportError:
            logger.warning(
                "Matplotlib/Seaborn not available. Plotting functionality will be limited.")

    def plot_time_series(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                         title: str = "Time Series Plot", show_volume: bool = False) -> None:
        """
        Plot time series data.

        Args:
            data: DataFrame with time series data
            columns: Columns to plot (default: all numeric columns)
            title: Plot title
            show_volume: Whether to show volume subplot
        """
        try:
            import matplotlib.pyplot as plt

            if data.empty:
                logger.warning("No data available for plotting")
                return

            # Select columns to plot
            if columns is None:
                numeric_cols = data.select_dtypes(
                    include=[np.number]).columns.tolist()
                columns = numeric_cols[:5]  # Limit to 5 columns

            # Create subplots
            n_plots = 1 + \
                (1 if show_volume and 'Volume' in data.columns else 0)
            fig, axes = plt.subplots(
                n_plots, 1, figsize=self.figure_size, sharex=True)

            if n_plots == 1:
                axes = [axes]

            # Main time series plot
            ax_main = axes[0]
            for i, col in enumerate(columns):
                if col in data.columns:
                    color = self.color_palette[i % len(self.color_palette)]
                    ax_main.plot(
                        data.index, data[col], label=col, color=color, linewidth=1.5)

            ax_main.set_title(title)
            ax_main.set_ylabel('Value')
            ax_main.legend()
            ax_main.grid(True, alpha=0.3)

            # Volume subplot
            if show_volume and 'Volume' in data.columns:
                ax_volume = axes[1]
                ax_volume.bar(
                    data.index, data['Volume'], alpha=0.7, color='gray')
                ax_volume.set_ylabel('Volume')
                ax_volume.grid(True, alpha=0.3)

            # Format x-axis
            if isinstance(data.index, pd.DatetimeIndex):
                ax_main.xaxis.set_major_locator(
                    plt.matplotlib.dates.YearLocator())
                ax_main.xaxis.set_major_formatter(
                    plt.matplotlib.dates.DateFormatter('%Y'))

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Time series plotting failed: {str(e)}")

    def plot_returns_distribution(self, returns: pd.Series, title: str = "Returns Distribution") -> None:
        """
        Plot returns distribution with statistical information.

        Args:
            returns: Series with returns data
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if returns.empty:
                logger.warning("No returns data available for plotting")
                return

            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Histogram with KDE
            ax1.hist(returns, bins=50, density=True,
                     alpha=0.7, color=self.color_palette[0])
            sns.kdeplot(returns, ax=ax1,
                        color=self.color_palette[1], linewidth=2)
            ax1.set_title(f"{title} - Histogram")
            ax1.set_xlabel('Returns')
            ax1.set_ylabel('Density')
            ax1.grid(True, alpha=0.3)

            # Q-Q plot
            from scipy import stats
            stats.probplot(returns.dropna(), dist="norm", plot=ax2)
            ax2.set_title(f"{title} - Q-Q Plot")
            ax2.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f"Mean: {returns.mean():.4f}\nStd: {returns.std():.4f}\nSkew: {returns.skew():.4f}\nKurt: {returns.kurtosis():.4f}"
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning(
                "Matplotlib/Seaborn/Scipy not available for plotting")
        except Exception as e:
            logger.error(f"Returns distribution plotting failed: {str(e)}")

    def plot_portfolio_performance(self, portfolio_data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None,
                                   title: str = "Portfolio Performance") -> None:
        """
        Plot comprehensive portfolio performance analysis.

        Args:
            portfolio_data: DataFrame with portfolio data
            benchmark_data: Optional benchmark data for comparison
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt

            if portfolio_data.empty:
                logger.warning("No portfolio data available for plotting")
                return

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            # Plot 1: Cumulative Returns
            ax1 = axes[0]
            if 'cumulative_returns' in portfolio_data.columns:
                ax1.plot(portfolio_data.index, portfolio_data['cumulative_returns'],
                         label='Portfolio', color=self.color_palette[0], linewidth=2)

            if benchmark_data is not None and 'cumulative_returns' in benchmark_data.columns:
                ax1.plot(benchmark_data.index, benchmark_data['cumulative_returns'],
                         label='Benchmark', color=self.color_palette[1], linewidth=2, alpha=0.7)

            ax1.set_title('Cumulative Returns')
            ax1.set_ylabel('Cumulative Return')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Rolling Volatility
            ax2 = axes[1]
            if 'rolling_volatility' in portfolio_data.columns:
                ax2.plot(portfolio_data.index, portfolio_data['rolling_volatility'],
                         color=self.color_palette[2], linewidth=2)
                ax2.set_title('Rolling Volatility (30-day)')
                ax2.set_ylabel('Volatility')
                ax2.grid(True, alpha=0.3)

            # Plot 3: Drawdown
            ax3 = axes[2]
            if 'drawdown' in portfolio_data.columns:
                ax3.fill_between(portfolio_data.index, portfolio_data['drawdown'], 0,
                                 alpha=0.3, color=self.color_palette[3])
                ax3.set_title('Portfolio Drawdown')
                ax3.set_ylabel('Drawdown')
                ax3.grid(True, alpha=0.3)

            # Plot 4: Rolling Sharpe Ratio
            ax4 = axes[3]
            if 'rolling_sharpe' in portfolio_data.columns:
                ax4.plot(portfolio_data.index, portfolio_data['rolling_sharpe'],
                         color=self.color_palette[4], linewidth=2)
                ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax4.set_title('Rolling Sharpe Ratio (30-day)')
                ax4.set_ylabel('Sharpe Ratio')
                ax4.grid(True, alpha=0.3)

            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Portfolio performance plotting failed: {str(e)}")

    def plot_efficient_frontier(self, frontier_data: pd.DataFrame, title: str = "Efficient Frontier") -> None:
        """
        Plot efficient frontier analysis.

        Args:
            frontier_data: DataFrame with efficient frontier data
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt

            if frontier_data.empty:
                logger.warning("No frontier data available for plotting")
                return

            # Create the plot
            fig, ax = plt.subplots(figsize=self.figure_size)

            # Plot efficient frontier
            if 'volatility' in frontier_data.columns and 'return' in frontier_data.columns:
                scatter = ax.scatter(frontier_data['volatility'], frontier_data['return'],
                                     c=frontier_data.get('sharpe_ratio', 0), cmap='viridis',
                                     alpha=0.6, s=30)

                # Add colorbar
                plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')

                # Highlight optimal portfolios
                if 'sharpe_ratio' in frontier_data.columns:
                    max_sharpe_idx = frontier_data['sharpe_ratio'].idxmax()
                    max_sharpe_point = frontier_data.loc[max_sharpe_idx]
                    ax.scatter(max_sharpe_point['volatility'], max_sharpe_point['return'],
                               color='red', s=200, marker='*',
                               label=f"Max Sharpe (SR: {max_sharpe_point['sharpe_ratio']:.3f})")

                if 'volatility' in frontier_data.columns:
                    min_vol_idx = frontier_data['volatility'].idxmin()
                    min_vol_point = frontier_data.loc[min_vol_idx]
                    ax.scatter(min_vol_point['volatility'], min_vol_point['return'],
                               color='green', s=200, marker='s',
                               label=f"Min Volatility (Vol: {min_vol_point['volatility']:.3f})")

                ax.set_xlabel('Portfolio Volatility (Annualized)')
                ax.set_ylabel('Portfolio Return (Annualized)')
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Efficient frontier plotting failed: {str(e)}")

    def plot_risk_metrics(self, risk_data: Dict[str, Any], title: str = "Risk Metrics Analysis") -> None:
        """
        Plot comprehensive risk metrics.

        Args:
            risk_data: Dictionary with risk metrics data
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt

            if not risk_data:
                logger.warning("No risk data available for plotting")
                return

            # Create subplots
            n_metrics = len(risk_data)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            plot_idx = 0

            # Plot 1: VaR at different confidence levels
            if 'var_analysis' in risk_data:
                ax1 = axes[plot_idx]
                var_data = risk_data['var_analysis']
                confidence_levels = list(var_data.keys())
                var_values = [var_data[level]['var_daily']
                              for level in confidence_levels]

                bars = ax1.bar(confidence_levels, var_values,
                               color=self.color_palette[:len(confidence_levels)])
                ax1.set_title('Value at Risk (VaR)')
                ax1.set_ylabel('VaR (Daily)')
                ax1.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, value in zip(bars, var_values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                             f'{value:.4f}', ha='center', va='bottom')

                plot_idx += 1

            # Plot 2: Volatility analysis
            if 'volatility_analysis' in risk_data:
                ax2 = axes[plot_idx]
                vol_data = risk_data['volatility_analysis']

                if 'rolling_volatility' in vol_data:
                    rolling_vols = vol_data['rolling_volatility']
                    for period, metrics in rolling_vols.items():
                        ax2.bar(period, metrics['mean'], label=f'{period} Mean',
                                color=self.color_palette[plot_idx % len(self.color_palette)])

                ax2.set_title('Rolling Volatility Analysis')
                ax2.set_ylabel('Volatility')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                plot_idx += 1

            # Plot 3: Drawdown analysis
            if 'drawdown_analysis' in risk_data:
                ax3 = axes[plot_idx]
                dd_data = risk_data['drawdown_analysis']

                if 'recovery_analysis' in dd_data:
                    recovery = dd_data['recovery_analysis']
                    metrics = ['avg_recovery_days', 'max_recovery_days']
                    values = [recovery.get(metric, 0) for metric in metrics]

                    bars = ax3.bar(
                        metrics, values, color=self.color_palette[plot_idx % len(self.color_palette)])
                    ax3.set_title('Recovery Time Analysis')
                    ax3.set_ylabel('Days')
                    ax3.grid(True, alpha=0.3)

                    # Add value labels
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height,
                                 f'{value:.0f}', ha='center', va='bottom')

                plot_idx += 1

            # Plot 4: Tail risk analysis
            if 'tail_risk_analysis' in risk_data:
                ax4 = axes[plot_idx]
                tail_data = risk_data['tail_risk_analysis']

                if 'tail_risk_metrics' in tail_data:
                    tail_metrics = tail_data['tail_risk_metrics']
                    confidence_levels = list(tail_metrics.keys())
                    cvar_values = [tail_metrics[level]['cvar']
                                   for level in confidence_levels]

                    bars = ax4.bar(confidence_levels, cvar_values,
                                   color=self.color_palette[plot_idx % len(self.color_palette)])
                    ax4.set_title('Conditional Value at Risk (CVaR)')
                    ax4.set_ylabel('CVaR (Daily)')
                    ax4.grid(True, alpha=0.3)

                    # Add value labels
                    for bar, value in zip(bars, cvar_values):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height,
                                 f'{value:.4f}', ha='center', va='bottom')

            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Risk metrics plotting failed: {str(e)}")

    def plot_correlation_matrix(self, correlation_matrix: pd.DataFrame, title: str = "Correlation Matrix") -> None:
        """
        Plot correlation matrix heatmap.

        Args:
            correlation_matrix: DataFrame with correlation matrix
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if correlation_matrix.empty:
                logger.warning("No correlation data available for plotting")
                return

            # Create the plot
            fig, ax = plt.subplots(figsize=self.figure_size)

            # Create heatmap
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

            ax.set_title(title)
            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            logger.error(f"Correlation matrix plotting failed: {str(e)}")

    def plot_rolling_metrics(self, data: pd.DataFrame, metric: str, window: int = 30,
                             title: str = "Rolling Metrics") -> None:
        """
        Plot rolling metrics over time.

        Args:
            data: DataFrame with time series data
            metric: Metric column to plot
            window: Rolling window size
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt

            if data.empty or metric not in data.columns:
                logger.warning(
                    f"No data available for plotting rolling {metric}")
                return

            # Calculate rolling metric
            rolling_metric = data[metric].rolling(window=window).mean()

            # Create the plot
            fig, ax = plt.subplots(figsize=self.figure_size)

            # Plot original and rolling metric
            ax.plot(data.index, data[metric], alpha=0.5, label=f'Original {metric}',
                    color=self.color_palette[0])
            ax.plot(data.index, rolling_metric, label=f'{window}-day Rolling {metric}',
                    color=self.color_palette[1], linewidth=2)

            ax.set_title(f"{title} - {metric}")
            ax.set_xlabel('Date')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Rolling metrics plotting failed: {str(e)}")

    def save_plot(self, filename: str, dpi: int = 300, format: str = 'png') -> None:
        """
        Save the current plot to file.

        Args:
            filename: Output filename
            dpi: DPI for the saved image
            format: Output format ('png', 'pdf', 'svg', 'jpg')
        """
        try:
            import matplotlib.pyplot as plt

            plt.savefig(filename, dpi=dpi, format=format, bbox_inches='tight')
            logger.info(f"Plot saved to {filename}")

        except ImportError:
            logger.warning("Matplotlib not available for saving plots")
        except Exception as e:
            logger.error(f"Plot saving failed: {str(e)}")

    def create_dashboard_layout(self, n_plots: int, layout: str = 'grid') -> Tuple[Any, List]:
        """
        Create a dashboard layout for multiple plots.

        Args:
            n_plots: Number of plots to include
            layout: Layout type ('grid', 'vertical', 'horizontal')

        Returns:
            Tuple of (figure, axes_list)
        """
        try:
            import matplotlib.pyplot as plt

            if layout == 'grid':
                # Calculate grid dimensions
                cols = int(np.ceil(np.sqrt(n_plots)))
                rows = int(np.ceil(n_plots / cols))

                fig, axes = plt.subplots(
                    rows, cols, figsize=(cols * 6, rows * 4))

                # Flatten axes if needed
                if n_plots == 1:
                    axes = [axes]
                elif rows == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()

                # Hide unused subplots
                for i in range(n_plots, len(axes)):
                    axes[i].set_visible(False)

            elif layout == 'vertical':
                fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
                if n_plots == 1:
                    axes = [axes]

            elif layout == 'horizontal':
                fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 8))
                if n_plots == 1:
                    axes = [axes]

            else:
                raise ValueError(f"Unknown layout: {layout}")

            return fig, axes

        except ImportError:
            logger.warning(
                "Matplotlib not available for creating dashboard layout")
            return None, []
        except Exception as e:
            logger.error(f"Dashboard layout creation failed: {str(e)}")
            return None, []

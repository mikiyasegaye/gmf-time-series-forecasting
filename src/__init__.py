"""
GMF Time Series Forecasting Package

A comprehensive financial forecasting and portfolio optimization system
built with modern Python practices and enterprise-grade architecture.

Author: GMF Investment Team
Version: 2.0.0
"""

# Core data processing modules
from .data import DataProcessor, DataLoader

# Forecasting model modules
from .models import LSTMForecaster, ARIMAForecaster, ModelEvaluator

# Portfolio optimization modules
from .portfolio import PortfolioOptimizer, EfficientFrontier

# Backtesting and performance analysis
from .backtesting import BacktestEngine, PerformanceAnalyzer

# Risk management and utilities
from .utils import RiskMetrics, ValidationUtils

# Visualization and dashboard
from .visualization import PlotGenerator, DashboardCreator
from .dashboard import GMFDashboard

# Package metadata
__version__ = "2.0.0"
__author__ = "GMF Investment Team"
__email__ = "investments@gmf.com"

# Public API
__all__ = [
    # Data
    "DataProcessor", "DataLoader",

    # Models
    "LSTMForecaster", "ARIMAForecaster", "ModelEvaluator",

    # Portfolio
    "PortfolioOptimizer", "EfficientFrontier",

    # Backtesting
    "BacktestEngine", "PerformanceAnalyzer",

    # Utilities
    "RiskMetrics", "ValidationUtils",

    # Visualization
    "PlotGenerator", "DashboardCreator",

    # Dashboard
    "GMFDashboard"
]

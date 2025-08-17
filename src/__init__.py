"""
GMF Time Series Forecasting System

A comprehensive, production-ready time series forecasting and portfolio optimization
system designed for financial markets analysis and investment decision support.

Author: GMF Investment Team
Version: 2.0.0
"""

# Core data processing modules
from .data import DataProcessor, DataLoader
from .data.real_time_streamer import RealTimeDataStreamer, FinancialDataStreamer

# Forecasting models
from .models import LSTMForecaster, ARIMAForecaster, ModelEvaluator
from .models.shap_explainer import SHAPExplainer, ModelExplainabilityManager

# Portfolio management
from .portfolio import PortfolioOptimizer, EfficientFrontier

# Backtesting system
from .backtesting import BacktestEngine, PerformanceAnalyzer

# Risk management and utilities
from .utils import RiskMetrics, ValidationUtils

# Visualization system
from .visualization import PlotGenerator, DashboardCreator

# Advanced analytics
from .analytics.advanced_analytics import AdvancedAnalyticsEngine, MarketRegimeDetector, SentimentAnalyzer

# Reporting system
from .reporting.automated_reporter import AutomatedReporter, ReportTemplate

# Dashboard
from .dashboard import GMFDashboard

# Demo modules
from .demos import run_advanced_features_demo

# Package metadata
__version__ = "2.0.0"
__author__ = "GMF Investment Team"
__email__ = "investments@gmf.com"

# Export all public classes and functions
__all__ = [
    # Data processing
    'DataProcessor', 'DataLoader', 'RealTimeDataStreamer', 'FinancialDataStreamer',

    # Forecasting models
    'LSTMForecaster', 'ARIMAForecaster', 'ModelEvaluator',
    'SHAPExplainer', 'ModelExplainabilityManager',

    # Portfolio management
    'PortfolioOptimizer', 'EfficientFrontier',

    # Backtesting
    'BacktestEngine', 'PerformanceAnalyzer',

    # Risk management
    'RiskMetrics', 'ValidationUtils',

    # Visualization
    'PlotGenerator', 'DashboardCreator',

    # Advanced analytics
    'AdvancedAnalyticsEngine', 'MarketRegimeDetector', 'SentimentAnalyzer',

    # Reporting
    'AutomatedReporter', 'ReportTemplate',

    # Dashboard
    'GMFDashboard',

    # Demo modules
    'run_advanced_features_demo'
]

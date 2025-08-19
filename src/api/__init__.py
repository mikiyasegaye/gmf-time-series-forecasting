"""
GMF Time Series Forecasting API

This module provides REST API endpoints for the GMF forecasting system.
"""

from .main import app
from .endpoints import *

__all__ = ['app']
__version__ = "3.0.0"
__author__ = "GMF Investment Team"

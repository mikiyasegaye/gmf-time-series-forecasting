"""
API endpoints for GMF Time Series Forecasting system.
"""

from .main import app
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json

# Import GMF modules
from ..models.lstm_forecaster import LSTMForecaster
from ..models.arima_forecaster import ARIMAForecaster
from ..portfolio.portfolio_optimizer import PortfolioOptimizer
from ..analytics.advanced_analytics import AdvancedAnalyticsEngine
from ..reporting.automated_reporter import AutomatedReporter
from ..data.data_loader import DataLoader
from ..data.data_processor import DataProcessor

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1")

# Initialize core components
try:
    data_loader = DataLoader()
    data_processor = DataProcessor()
    analytics_engine = AdvancedAnalyticsEngine()
    reporter = AutomatedReporter("api_reports")
except Exception as e:
    logger.error(f"Failed to initialize core components: {e}")
    data_loader = None
    data_processor = None
    analytics_engine = None
    reporter = None

# Forecasting endpoints


@router.post("/forecasting/lstm")
async def forecast_lstm(
    request: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Generate LSTM forecast for given data."""
    try:
        if not data_loader or not data_processor:
            raise HTTPException(
                status_code=500, detail="System not initialized")

        # Extract parameters
        symbol = request.get("symbol", "TSLA")
        forecast_days = request.get("forecast_days", 30)
        lookback_days = request.get("lookback_days", 252)

        # Load and process data
        data = data_loader.load_stock_data(symbol, lookback_days)
        processed_data = data_processor.process_data(data)

        # Create and train LSTM model
        forecaster = LSTMForecaster()
        forecaster.train(processed_data)

        # Generate forecast
        forecast = forecaster.forecast(forecast_days)

        return {
            "symbol": symbol,
            "forecast_days": forecast_days,
            "forecast": forecast.tolist(),
            "model_info": {
                "type": "LSTM",
                "accuracy": forecaster.get_accuracy(),
                "training_date": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"LSTM forecasting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecasting/arima")
async def forecast_arima(
    request: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Generate ARIMA forecast for given data."""
    try:
        if not data_loader or not data_processor:
            raise HTTPException(
                status_code=500, detail="System not initialized")

        # Extract parameters
        symbol = request.get("symbol", "TSLA")
        forecast_days = request.get("forecast_days", 30)
        lookback_days = request.get("lookback_days", 252)

        # Load and process data
        data = data_loader.load_stock_data(symbol, lookback_days)
        processed_data = data_processor.process_data(data)

        # Create and train ARIMA model
        forecaster = ARIMAForecaster()
        forecaster.train(processed_data)

        # Generate forecast
        forecast = forecaster.forecast(forecast_days)

        return {
            "symbol": symbol,
            "forecast_days": forecast_days,
            "forecast": forecast.tolist(),
            "model_info": {
                "type": "ARIMA",
                "parameters": forecaster.get_model_params(),
                "training_date": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"ARIMA forecasting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Portfolio optimization endpoints


@router.post("/portfolio/optimize")
async def optimize_portfolio(
    request: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Optimize portfolio using Modern Portfolio Theory."""
    try:
        if not data_loader or not data_processor:
            raise HTTPException(
                status_code=500, detail="System not initialized")

        # Extract parameters
        symbols = request.get("symbols", ["AAPL", "GOOGL", "MSFT", "TSLA"])
        target_return = request.get("target_return", None)
        risk_tolerance = request.get("risk_tolerance", "moderate")

        # Load data for all symbols
        portfolio_data = {}
        for symbol in symbols:
            data = data_loader.load_stock_data(symbol, 252)
            processed_data = data_processor.process_data(data)
            portfolio_data[symbol] = processed_data

        # Create portfolio optimizer
        optimizer = PortfolioOptimizer()

        # Optimize portfolio
        if target_return:
            optimal_weights = optimizer.optimize_with_target_return(
                portfolio_data, target_return
            )
        else:
            optimal_weights = optimizer.optimize_risk_parity(portfolio_data)

        # Calculate portfolio metrics
        portfolio_metrics = optimizer.calculate_portfolio_metrics(
            portfolio_data, optimal_weights
        )

        return {
            "symbols": symbols,
            "optimal_weights": optimal_weights,
            "portfolio_metrics": portfolio_metrics,
            "optimization_date": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Portfolio optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/efficient-frontier")
async def get_efficient_frontier(
    symbols: str = Query("AAPL,GOOGL,MSFT,TSLA"),
    num_portfolios: int = Query(100, ge=10, le=1000)
) -> Dict[str, Any]:
    """Generate efficient frontier for given symbols."""
    try:
        if not data_loader or not data_processor:
            raise HTTPException(
                status_code=500, detail="System not initialized")

        # Parse symbols
        symbol_list = [s.strip() for s in symbols.split(",")]

        # Load data for all symbols
        portfolio_data = {}
        for symbol in symbol_list:
            data = data_loader.load_stock_data(symbol, 252)
            processed_data = data_processor.process_data(data)
            portfolio_data[symbol] = processed_data

        # Create portfolio optimizer
        optimizer = PortfolioOptimizer()

        # Generate efficient frontier
        frontier_data = optimizer.generate_efficient_frontier(
            portfolio_data, num_portfolios
        )

        return {
            "symbols": symbol_list,
            "efficient_frontier": frontier_data,
            "generation_date": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Efficient frontier error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints


@router.post("/analytics/market-regimes")
async def analyze_market_regimes(
    request: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Analyze market regimes for given data."""
    try:
        if not analytics_engine:
            raise HTTPException(
                status_code=500, detail="Analytics engine not initialized")

        # Extract parameters
        symbol = request.get("symbol", "SPY")
        lookback_days = request.get("lookback_days", 252)

        # Load data
        data = data_loader.load_stock_data(symbol, lookback_days)
        processed_data = data_processor.process_data(data)

        # Run market regime analysis
        analysis = analytics_engine.run_comprehensive_analysis(processed_data)

        return {
            "symbol": symbol,
            "analysis": analysis,
            "analysis_date": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Market regime analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/sentiment")
async def analyze_sentiment(
    request: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Analyze market sentiment for given data."""
    try:
        if not analytics_engine:
            raise HTTPException(
                status_code=500, detail="Analytics engine not initialized")

        # Extract parameters
        symbol = request.get("symbol", "SPY")
        lookback_days = request.get("lookback_days", 252)

        # Load data
        data = data_loader.load_stock_data(symbol, lookback_days)
        processed_data = data_processor.process_data(data)

        # Run sentiment analysis
        sentiment = analytics_engine.sentiment_analyzer.analyze_market_sentiment(
            processed_data
        )

        return {
            "symbol": symbol,
            "sentiment_analysis": sentiment,
            "analysis_date": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Reporting endpoints


@router.post("/reports/portfolio")
async def generate_portfolio_report(
    request: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Generate portfolio performance report."""
    try:
        if not reporter:
            raise HTTPException(
                status_code=500, detail="Reporter not initialized")

        # Extract parameters
        portfolio_data = request.get("portfolio_data", {})
        report_type = request.get("report_type", "performance")

        # Generate report
        if report_type == "performance":
            report_path = reporter.generate_portfolio_report(portfolio_data)
        elif report_type == "risk":
            report_path = reporter.generate_risk_report(portfolio_data)
        else:
            report_path = reporter.generate_comprehensive_report(
                portfolio_data)

        return {
            "report_type": report_type,
            "report_path": str(report_path),
            "generation_date": datetime.now().isoformat(),
            "download_url": f"/api/v1/reports/download/{report_path.name}"
        }

    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/download/{filename}")
async def download_report(filename: str) -> Dict[str, Any]:
    """Download generated report."""
    try:
        if not reporter:
            raise HTTPException(
                status_code=500, detail="Reporter not initialized")

        # This would typically return a file response
        # For now, return file info
        return {
            "filename": filename,
            "status": "available",
            "message": "Report download endpoint - implement file serving logic"
        }

    except Exception as e:
        logger.error(f"Report download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data endpoints


@router.get("/data/symbols")
async def get_available_symbols() -> Dict[str, Any]:
    """Get list of available symbols."""
    try:
        # Return available symbols
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA",
                   "AMZN", "META", "NVDA", "SPY", "BND"]

        return {
            "available_symbols": symbols,
            "total_count": len(symbols),
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Symbols retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    days: int = Query(252, ge=1, le=2520),
    include_indicators: bool = Query(True)
) -> Dict[str, Any]:
    """Get historical data for a symbol."""
    try:
        if not data_loader or not data_processor:
            raise HTTPException(
                status_code=500, detail="System not initialized")

        # Load data
        data = data_loader.load_stock_data(symbol, days)

        if include_indicators:
            data = data_processor.process_data(data)

        # Convert to JSON-serializable format
        data_dict = data.to_dict(orient="records")

        return {
            "symbol": symbol,
            "days": days,
            "data_points": len(data_dict),
            "data": data_dict,
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Historical data retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System endpoints


@router.get("/system/status")
async def get_system_status() -> Dict[str, Any]:
    """Get system status and health information."""
    try:
        status = {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "version": "3.0.0",
            "components": {
                "data_loader": "operational" if data_loader else "error",
                "data_processor": "operational" if data_processor else "error",
                "analytics_engine": "operational" if analytics_engine else "error",
                "reporter": "operational" if reporter else "error"
            },
            "environment": "development"
        }

        return status

    except Exception as e:
        logger.error(f"System status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include router in main app
app.include_router(router)

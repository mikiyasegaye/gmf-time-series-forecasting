"""
Data processing and preprocessing module for financial time series data.

This module provides robust data cleaning, validation, and preprocessing
capabilities for financial data analysis and forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from datetime import datetime, timezone
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    A robust data processor for financial time series data.

    Handles data loading, cleaning, validation, and preprocessing
    for portfolio analysis and forecasting models.

    Attributes:
        data_path (Path): Path to the data directory
        processed_path (Path): Path to store processed data
        assets (List[str]): List of asset symbols to process
    """

    def __init__(self, data_path: Union[str, Path], assets: Optional[List[str]] = None):
        """
        Initialize the DataProcessor.

        Args:
            data_path: Path to the raw data directory
            assets: List of asset symbols to process. If None, uses default assets.
        """
        self.data_path = Path(data_path)
        self.processed_path = self.data_path / "processed"
        self.processed_path.mkdir(exist_ok=True)

        # Default assets if none specified
        if assets is None:
            self.assets = [
                "TSLA", "AAPL", "AMZN", "GOOG", "META",
                "MSFT", "NVDA", "SPY", "BND"
            ]
        else:
            self.assets = assets

        # Validate data path exists
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data path {self.data_path} does not exist")

        logger.info(f"Initialized DataProcessor for {len(self.assets)} assets")

    def load_asset_data(self, asset: str) -> pd.DataFrame:
        """
        Load historical data for a specific asset.

        Args:
            asset: Asset symbol (e.g., 'TSLA')

        Returns:
            DataFrame with OHLCV data

        Raises:
            FileNotFoundError: If asset data file doesn't exist
            ValueError: If data format is invalid
        """
        try:
            # Try to load from processed data first
            processed_file = self.processed_path / f"{asset}_processed.csv"
            if processed_file.exists():
                df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
                logger.info(
                    f"Loaded processed data for {asset}: {len(df)} records")
                return df

            # Load from raw data
            raw_file = self.data_path / "raw" / f"{asset}_data.csv"
            if not raw_file.exists():
                raise FileNotFoundError(f"No data file found for {asset}")

            df = pd.read_csv(raw_file)

            # Validate required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [
                col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns for {asset}: {missing_cols}")

            # Clean and process the data
            df = self._clean_asset_data(df, asset)

            # Save processed data
            df.to_csv(processed_file)
            logger.info(
                f"Processed and saved data for {asset}: {len(df)} records")

            return df

        except Exception as e:
            logger.error(f"Error loading data for {asset}: {str(e)}")
            raise

    def _clean_asset_data(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Clean and preprocess asset data.

        Args:
            df: Raw asset data DataFrame
            asset: Asset symbol for logging

        Returns:
            Cleaned DataFrame
        """
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'])

        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates(subset=['Date'])
        if len(df) < initial_rows:
            logger.warning(
                f"Removed {initial_rows - len(df)} duplicate rows for {asset}")

        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)

        # Handle missing values
        df = self._handle_missing_values(df, asset)

        # Calculate returns
        df['Returns'] = df['Close'].pct_change()

        # Calculate log returns
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Calculate volatility (rolling 30-day)
        df['Volatility_30d'] = df['Returns'].rolling(
            window=30).std() * np.sqrt(252)

        # Set date as index
        df.set_index('Date', inplace=True)

        # Remove rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        if len(df) < initial_rows:
            logger.warning(
                f"Removed {initial_rows - len(df)} rows with NaN values for {asset}")

        return df

    def _handle_missing_values(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Handle missing values in asset data.

        Args:
            df: Asset data DataFrame
            asset: Asset symbol for logging

        Returns:
            DataFrame with handled missing values
        """
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(
                f"Missing values in {asset}: {missing_counts.to_dict()}")

            # Forward fill for OHLCV data
            price_cols = ['Open', 'High', 'Low', 'Close']
            df[price_cols] = df[price_cols].fillna(method='ffill')

            # Fill volume with 0 (no trading)
            df['Volume'] = df['Volume'].fillna(0)

            # Check remaining missing values
            remaining_missing = df.isnull().sum().sum()
            if remaining_missing > 0:
                logger.warning(
                    f"Remaining missing values in {asset}: {remaining_missing}")

        return df

    def load_analyst_ratings(self) -> pd.DataFrame:
        """
        Load and process analyst ratings data.

        Returns:
            DataFrame with processed analyst ratings

        Raises:
            FileNotFoundError: If analyst ratings file doesn't exist
        """
        try:
            ratings_file = self.data_path / "raw" / "raw_analyst_ratings.csv"
            if not ratings_file.exists():
                raise FileNotFoundError("Analyst ratings file not found")

            logger.info("Loading analyst ratings data...")
            df = pd.read_csv(ratings_file)

            # Basic cleaning
            df = self._clean_analyst_ratings(df)

            # Save processed ratings
            processed_ratings = self.processed_path / "analyst_ratings_processed.csv"
            df.to_csv(processed_ratings, index=False)

            logger.info(f"Processed analyst ratings: {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error loading analyst ratings: {str(e)}")
            raise

    def _clean_analyst_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process analyst ratings data.

        Args:
            df: Raw analyst ratings DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Convert date columns
        date_cols = ['date', 'published_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Remove rows with invalid dates
        initial_rows = len(df)
        df = df.dropna(subset=['date'])
        if len(df) < initial_rows:
            logger.warning(
                f"Removed {initial_rows - len(df)} rows with invalid dates")

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        return df

    def calculate_risk_metrics(self, asset: str) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Dictionary of risk metrics
        """
        try:
            df = self.load_asset_data(asset)

            # Calculate risk metrics
            returns = df['Returns'].dropna()

            metrics = {
                'asset': asset,
                'mean_return': returns.mean() * 252,  # Annualized
                'volatility': returns.std() * np.sqrt(252),  # Annualized
                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                'var_95': np.percentile(returns, 5),
                'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
                'max_drawdown': self._calculate_max_drawdown(df['Close']),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'data_points': len(returns)
            }

            logger.info(f"Calculated risk metrics for {asset}")
            return metrics

        except Exception as e:
            logger.error(
                f"Error calculating risk metrics for {asset}: {str(e)}")
            raise

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        Calculate maximum drawdown from peak.

        Args:
            prices: Price series

        Returns:
            Maximum drawdown as a percentage
        """
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()

    def generate_correlation_matrix(self) -> pd.DataFrame:
        """
        Generate correlation matrix for all assets.

        Returns:
            Correlation matrix DataFrame
        """
        try:
            # Load all asset data
            asset_data = {}
            for asset in self.assets:
                try:
                    df = self.load_asset_data(asset)
                    asset_data[asset] = df['Returns'].dropna()
                except Exception as e:
                    logger.warning(
                        f"Could not load data for {asset}: {str(e)}")
                    continue

            # Align data by date
            aligned_data = pd.DataFrame(asset_data)
            aligned_data = aligned_data.dropna()

            # Calculate correlation matrix
            corr_matrix = aligned_data.corr()

            # Save correlation matrix
            corr_file = self.processed_path / "correlation_matrix.csv"
            corr_matrix.to_csv(corr_file)

            logger.info(
                f"Generated correlation matrix for {len(corr_matrix)} assets")
            return corr_matrix

        except Exception as e:
            logger.error(f"Error generating correlation matrix: {str(e)}")
            raise

    def get_data_summary(self) -> Dict[str, Dict]:
        """
        Get comprehensive summary of all asset data.

        Returns:
            Dictionary with asset summaries
        """
        summary = {}

        for asset in self.assets:
            try:
                df = self.load_asset_data(asset)
                risk_metrics = self.calculate_risk_metrics(asset)

                summary[asset] = {
                    'data_points': len(df),
                    'date_range': f"{df.index.min().date()} to {df.index.max().date()}",
                    'risk_metrics': risk_metrics
                }

            except Exception as e:
                logger.warning(f"Could not summarize {asset}: {str(e)}")
                summary[asset] = {'error': str(e)}

        return summary

    def validate_data_quality(self) -> Dict[str, bool]:
        """
        Validate data quality for all assets.

        Returns:
            Dictionary with validation results
        """
        validation_results = {}

        for asset in self.assets:
            try:
                df = self.load_asset_data(asset)

                # Check data quality
                quality_checks = {
                    'has_data': len(df) > 0,
                    'no_missing_prices': df[['Open', 'High', 'Low', 'Close']].isnull().sum().sum() == 0,
                    'positive_prices': (df[['Open', 'High', 'Low', 'Close']] > 0).all().all(),
                    'logical_ohlc': (df['High'] >= df['Low']).all() and (df['High'] >= df['Close']).all() and (df['High'] >= df['Open']).all(),
                    'sufficient_history': len(df) >= 252,  # At least 1 year
                    'stationary_returns': self._test_stationarity(df['Returns'].dropna())
                }

                validation_results[asset] = quality_checks

            except Exception as e:
                logger.warning(f"Could not validate {asset}: {str(e)}")
                validation_results[asset] = {'error': str(e)}

        return validation_results

    def _test_stationarity(self, returns: pd.Series, significance_level: float = 0.05) -> bool:
        """
        Test if returns series is stationary using ADF test.

        Args:
            returns: Returns series
            significance_level: Significance level for the test

        Returns:
            True if stationary, False otherwise
        """
        try:
            from statsmodels.tsa.stattools import adfuller

            if len(returns) < 10:  # Need minimum observations
                return False

            result = adfuller(returns.dropna())
            p_value = result[1]

            return p_value < significance_level

        except ImportError:
            logger.warning(
                "statsmodels not available, skipping stationarity test")
            return True  # Assume stationary if can't test
        except Exception as e:
            logger.warning(f"Error in stationarity test: {str(e)}")
            return True  # Assume stationary if test fails

    def process_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean price data.

        Args:
            data: Raw price data DataFrame

        Returns:
            Processed DataFrame with clean data
        """
        try:
            # Make a copy to avoid modifying original
            processed_data = data.copy()

            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in processed_data.columns for col in required_cols):
                # If we don't have the required columns, return as-is
                logger.warning(
                    "Required columns not found, returning data as-is")
                return processed_data

            # Remove any rows with negative or zero prices
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                processed_data = processed_data[processed_data[col] > 0]

            # Ensure High >= Low, High >= Open, High >= Close
            processed_data = processed_data[
                (processed_data['High'] >= processed_data['Low']) &
                (processed_data['High'] >= processed_data['Open']) &
                (processed_data['High'] >= processed_data['Close'])
            ]

            # Remove any rows with missing values
            processed_data = processed_data.dropna()

            # Calculate returns if we have enough data
            if len(processed_data) > 1:
                processed_data['Returns'] = processed_data['Close'].pct_change()
                processed_data['Log_Returns'] = np.log(
                    processed_data['Close'] / processed_data['Close'].shift(1))

            logger.info(
                f"Processed price data: {len(processed_data)} clean records")
            return processed_data

        except Exception as e:
            logger.error(f"Error processing price data: {str(e)}")
            # Return original data if processing fails
            return data

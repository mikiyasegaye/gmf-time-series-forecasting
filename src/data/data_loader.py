"""
Data loading and caching module for financial time series data.

This module provides efficient data loading with caching capabilities
to optimize performance for repeated data access.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Efficient data loader with caching for financial time series data.
    
    Provides optimized data loading with intelligent caching to
    reduce I/O operations and improve performance.
    
    Attributes:
        cache_dir (Path): Directory for storing cached data
        cache_ttl (timedelta): Time-to-live for cached data
        data_processor (DataProcessor): Data processor instance
    """
    
    def __init__(self, data_processor, cache_dir: Optional[Union[str, Path]] = None, 
                 cache_ttl_hours: int = 24):
        """
        Initialize the DataLoader.
        
        Args:
            data_processor: DataProcessor instance for data processing
            cache_dir: Directory for caching (defaults to data/processed/cache)
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.data_processor = data_processor
        
        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = self.data_processor.processed_path / "cache"
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(exist_ok=True)
        
        # Set cache TTL
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        logger.info(f"Initialized DataLoader with cache TTL: {cache_ttl_hours} hours")
    
    def get_asset_data(self, asset: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Get asset data with optional caching.
        
        Args:
            asset: Asset symbol
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with asset data
        """
        if use_cache:
            cached_data = self._get_cached_data(asset)
            if cached_data is not None:
                logger.info(f"Loaded cached data for {asset}")
                return cached_data
        
        # Load fresh data
        data = self.data_processor.load_asset_data(asset)
        
        # Cache the data
        if use_cache:
            self._cache_data(asset, data)
        
        return data
    
    def get_multiple_assets(self, assets: List[str], use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple assets efficiently.
        
        Args:
            assets: List of asset symbols
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping asset symbols to DataFrames
        """
        asset_data = {}
        
        for asset in assets:
            try:
                asset_data[asset] = self.get_asset_data(asset, use_cache)
            except Exception as e:
                logger.warning(f"Could not load data for {asset}: {str(e)}")
                continue
        
        return asset_data
    
    def get_aligned_returns(self, assets: List[str], start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get aligned returns data for multiple assets.
        
        Args:
            assets: List of asset symbols
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            
        Returns:
            DataFrame with aligned returns data
        """
        # Load all asset data
        asset_data = self.get_multiple_assets(assets)
        
        # Extract returns and align by date
        returns_data = {}
        for asset, df in asset_data.items():
            if 'Returns' in df.columns:
                returns = df['Returns'].dropna()
                if start_date:
                    returns = returns[returns.index >= start_date]
                if end_date:
                    returns = returns[returns.index <= end_date]
                returns_data[asset] = returns
        
        # Create aligned DataFrame
        aligned_returns = pd.DataFrame(returns_data)
        aligned_returns = aligned_returns.dropna()
        
        logger.info(f"Generated aligned returns for {len(aligned_returns.columns)} assets, {len(aligned_returns)} observations")
        return aligned_returns
    
    def get_portfolio_data(self, portfolio_weights: Dict[str, float], 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get portfolio-level data based on weights.
        
        Args:
            portfolio_weights: Dictionary mapping assets to weights
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            
        Returns:
            DataFrame with portfolio returns and cumulative performance
        """
        # Get aligned returns for portfolio assets
        assets = list(portfolio_weights.keys())
        returns_data = self.get_aligned_returns(assets, start_date, end_date)
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns_data.index)
        for asset, weight in portfolio_weights.items():
            if asset in returns_data.columns:
                portfolio_returns += weight * returns_data[asset]
        
        # Calculate cumulative performance
        portfolio_data = pd.DataFrame({
            'Returns': portfolio_returns,
            'Cumulative_Returns': (1 + portfolio_returns).cumprod(),
            'Drawdown': self._calculate_drawdown(portfolio_returns)
        })
        
        logger.info(f"Generated portfolio data for {len(portfolio_weights)} assets")
        return portfolio_data
    
    def get_risk_metrics_batch(self, assets: List[str]) -> pd.DataFrame:
        """
        Get risk metrics for multiple assets in batch.
        
        Args:
            assets: List of asset symbols
            
        Returns:
            DataFrame with risk metrics for all assets
        """
        metrics_list = []
        
        for asset in assets:
            try:
                metrics = self.data_processor.calculate_risk_metrics(asset)
                metrics_list.append(metrics)
            except Exception as e:
                logger.warning(f"Could not calculate metrics for {asset}: {str(e)}")
                continue
        
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.set_index('asset', inplace=True)
            return metrics_df
        else:
            return pd.DataFrame()
    
    def get_market_data_summary(self) -> Dict[str, Union[int, str, pd.DataFrame]]:
        """
        Get comprehensive market data summary.
        
        Returns:
            Dictionary with market summary information
        """
        summary = {
            'total_assets': len(self.data_processor.assets),
            'data_quality': self.data_processor.validate_data_quality(),
            'risk_metrics': self.get_risk_metrics_batch(self.data_processor.assets),
            'correlation_matrix': self.data_processor.generate_correlation_matrix(),
            'last_updated': datetime.now().isoformat()
        }
        
        return summary
    
    def _get_cached_data(self, asset: str) -> Optional[pd.DataFrame]:
        """
        Get cached data for an asset if available and valid.
        
        Args:
            asset: Asset symbol
            
        Returns:
            Cached DataFrame if valid, None otherwise
        """
        cache_file = self.cache_dir / f"{asset}_cache.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is still valid
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age > self.cache_ttl:
                logger.info(f"Cache expired for {asset}, removing old cache")
                cache_file.unlink()
                return None
            
            # Load cached data
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Validate cached data
            if self._validate_cached_data(cached_data):
                return cached_data
            else:
                logger.warning(f"Invalid cached data for {asset}, removing cache")
                cache_file.unlink()
                return None
                
        except Exception as e:
            logger.warning(f"Error loading cache for {asset}: {str(e)}")
            if cache_file.exists():
                cache_file.unlink()
            return None
    
    def _cache_data(self, asset: str, data: pd.DataFrame) -> None:
        """
        Cache data for an asset.
        
        Args:
            asset: Asset symbol
            data: DataFrame to cache
        """
        try:
            cache_file = self.cache_dir / f"{asset}_cache.pkl"
            
            # Create cache entry with metadata
            cache_entry = {
                'data': data,
                'timestamp': datetime.now(),
                'asset': asset,
                'checksum': self._calculate_checksum(data)
            }
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_entry, f)
            
            logger.debug(f"Cached data for {asset}")
            
        except Exception as e:
            logger.warning(f"Error caching data for {asset}: {str(e)}")
    
    def _validate_cached_data(self, cached_data: Dict) -> bool:
        """
        Validate cached data integrity.
        
        Args:
            cached_data: Cached data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required keys
            required_keys = ['data', 'timestamp', 'asset', 'checksum']
            if not all(key in cached_data for key in required_keys):
                return False
            
            # Check data type
            if not isinstance(cached_data['data'], pd.DataFrame):
                return False
            
            # Check checksum
            expected_checksum = self._calculate_checksum(cached_data['data'])
            if cached_data['checksum'] != expected_checksum:
                return False
            
            # Check data is not empty
            if len(cached_data['data']) == 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """
        Calculate checksum for data validation.
        
        Args:
            data: DataFrame to checksum
            
        Returns:
            MD5 checksum string
        """
        # Convert to string representation for checksum
        data_str = data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _calculate_drawdown(self, returns: pd.Series) -> pd.Series:
        """
        Calculate drawdown series from returns.
        
        Args:
            returns: Returns series
            
        Returns:
            Drawdown series
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    def clear_cache(self, asset: Optional[str] = None) -> None:
        """
        Clear cache for specific asset or all assets.
        
        Args:
            asset: Asset symbol to clear cache for, or None for all
        """
        if asset:
            cache_file = self.cache_dir / f"{asset}_cache.pkl"
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Cleared cache for {asset}")
        else:
            # Clear all cache files
            cache_files = list(self.cache_dir.glob("*_cache.pkl"))
            for cache_file in cache_files:
                cache_file.unlink()
            logger.info(f"Cleared cache for all assets ({len(cache_files)} files)")
    
    def get_cache_info(self) -> Dict[str, Union[int, List[str]]]:
        """
        Get information about cached data.
        
        Returns:
            Dictionary with cache information
        """
        cache_files = list(self.cache_dir.glob("*_cache.pkl"))
        
        cache_info = {
            'total_cached_assets': len(cache_files),
            'cached_assets': [f.stem.replace('_cache', '') for f in cache_files],
            'cache_size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
            'oldest_cache': min(f.stat().st_mtime for f in cache_files) if cache_files else None,
            'newest_cache': max(f.stat().st_mtime for f in cache_files) if cache_files else None
        }
        
        return cache_info

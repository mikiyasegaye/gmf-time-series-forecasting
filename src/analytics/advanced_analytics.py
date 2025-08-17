#!/usr/bin/env python3
"""
Advanced Analytics Engine

This module provides advanced financial analytics capabilities for the GMF Time Series
Forecasting system, including sentiment analysis, market regime detection, and
advanced statistical modeling.

Author: GMF Investment Team
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Advanced market regime detection using multiple methodologies."""

    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        self.regime_history = []

    def detect_volatility_regimes(self, returns: pd.Series, num_regimes: int = 3) -> Dict[str, Any]:
        """Detect volatility regimes using rolling volatility analysis."""
        try:
            # Calculate rolling volatility
            rolling_vol = returns.rolling(
                window=self.lookback_window).std() * np.sqrt(252)
            clean_vol = rolling_vol.dropna()

            # Use K-means clustering to identify regimes
            scaler = StandardScaler()
            vol_scaled = scaler.fit_transform(clean_vol.values.reshape(-1, 1))

            kmeans = KMeans(n_clusters=num_regimes, random_state=42)
            regime_labels = kmeans.fit_predict(vol_scaled)

            # Calculate regime statistics
            regime_stats = {}
            for i in range(num_regimes):
                regime_mask = regime_labels == i
                regime_vol = clean_vol[regime_mask]

                regime_stats[f'regime_{i}'] = {
                    'volatility': regime_vol.mean(),
                    'duration': regime_mask.sum(),
                    'percentage': (regime_mask.sum() / len(clean_vol)) * 100
                }

            # Identify current regime
            current_vol = rolling_vol.iloc[-1]
            current_regime = kmeans.predict(
                scaler.transform([[current_vol]]))[0]

            return {
                'regime_labels': regime_labels,
                'regime_stats': regime_stats,
                'current_regime': current_regime,
                'method': 'volatility_clustering'
            }

        except Exception as e:
            logger.error(f"Error detecting volatility regimes: {str(e)}")
            return {}


class SentimentAnalyzer:
    """Financial sentiment analysis using multiple data sources."""

    def __init__(self):
        self.sentiment_history = []

    def analyze_market_sentiment(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market sentiment using technical indicators."""
        try:
            if market_data.empty:
                return {'sentiment_score': 0, 'sentiment_label': 'neutral', 'confidence': 0}

            sentiment_indicators = {}

            # RSI sentiment
            if 'RSI' in market_data.columns:
                rsi = market_data['RSI'].iloc[-1]
                if rsi > 70:
                    rsi_sentiment = -0.5  # Overbought
                elif rsi < 30:
                    rsi_sentiment = 0.5   # Oversold
                else:
                    rsi_sentiment = 0
                sentiment_indicators['RSI'] = rsi_sentiment

            # MACD sentiment
            if 'MACD' in market_data.columns and 'MACD_Signal' in market_data.columns:
                macd = market_data['MACD'].iloc[-1]
                macd_signal = market_data['MACD_Signal'].iloc[-1]

                if macd > macd_signal:
                    macd_sentiment = 0.3  # Bullish crossover
                else:
                    macd_sentiment = -0.3  # Bearish crossover
                sentiment_indicators['MACD'] = macd_sentiment

            # Calculate composite sentiment score
            if sentiment_indicators:
                composite_score = np.mean(list(sentiment_indicators.values()))
                confidence = len(sentiment_indicators) / 4
            else:
                composite_score = 0
                confidence = 0

            # Determine sentiment label
            if composite_score > 0.2:
                sentiment_label = 'bullish'
            elif composite_score < -0.2:
                sentiment_label = 'bearish'
            else:
                sentiment_label = 'neutral'

            return {
                'sentiment_score': composite_score,
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'indicators': sentiment_indicators,
                'method': 'technical_indicators'
            }

        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {str(e)}")
            return {'sentiment_score': 0, 'sentiment_label': 'neutral', 'confidence': 0}


class AdvancedAnalyticsEngine:
    """Main engine coordinating all advanced analytics capabilities."""

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        logger.info("Advanced analytics engine initialized")

    def run_comprehensive_analysis(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive market analysis using all available methods."""
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'analysis_methods': []
            }

            # Market regime analysis
            if 'Returns' in market_data.columns:
                regime_analysis = self.regime_detector.detect_volatility_regimes(
                    market_data['Returns']
                )
                if regime_analysis:
                    results['volatility_regimes'] = regime_analysis
                    results['analysis_methods'].append(
                        'volatility_regime_detection')

            # Market sentiment analysis
            sentiment_analysis = self.sentiment_analyzer.analyze_market_sentiment(
                market_data)
            if sentiment_analysis:
                results['market_sentiment'] = sentiment_analysis
                results['analysis_methods'].append('market_sentiment_analysis')

            # Analysis summary
            results['summary'] = self._generate_analysis_summary(results)

            logger.info(
                f"Comprehensive analysis completed with {len(results['analysis_methods'])} methods")
            return results

        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of analysis results."""
        summary = {
            'total_analyses': len(results.get('analysis_methods', [])),
            'key_findings': [],
            'risk_assessment': 'LOW',
            'market_outlook': 'NEUTRAL'
        }

        # Key findings
        if 'volatility_regimes' in results:
            current_regime = results['volatility_regimes'].get(
                'current_regime', 'unknown')
            summary['key_findings'].append(
                f"Current volatility regime: {current_regime}")

        if 'market_sentiment' in results:
            sentiment = results['market_sentiment'].get(
                'sentiment_label', 'unknown')
            summary['key_findings'].append(f"Market sentiment: {sentiment}")

        # Market outlook
        if 'market_sentiment' in results:
            sentiment = results['market_sentiment'].get(
                'sentiment_label', 'neutral')
            if sentiment == 'bullish':
                summary['market_outlook'] = 'BULLISH'
            elif sentiment == 'bearish':
                summary['market_outlook'] = 'BEARISH'

        return summary

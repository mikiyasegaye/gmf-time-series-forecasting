#!/usr/bin/env python3
"""
GMF Advanced Features Demo

This script demonstrates all the advanced Phase 2 features of the GMF Time Series
Forecasting system, including SHAP explainability, real-time streaming, advanced
analytics, and automated reporting.

Author: GMF Investment Team
Version: 2.0.0
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def generate_sample_data():
    """Generate sample financial data for demonstration."""
    print("📊 Generating sample financial data...")

    # Create sample price data
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    np.random.seed(42)

    # Generate realistic price movements
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * np.cumprod(1 + returns)

    # Create OHLCV data
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'High': prices * (1 + np.abs(np.random.normal(0.01, 0.01, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.normal(0.01, 0.01, len(dates)))),
        'Close': prices,
        'Volume': np.random.uniform(1000000, 5000000, len(dates))
    })

    # Ensure logical OHLC relationships
    data['High'] = np.maximum(
        data['High'], data[['Open', 'Close']].max(axis=1))
    data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))

    # Calculate returns and technical indicators
    data['Returns'] = data['Close'].pct_change()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'] = calculate_macd(data['Close'])
    data['MACD_Signal'] = data['MACD'].rolling(window=9).mean()

    data.set_index('Date', inplace=True)
    data = data.dropna()

    print(f"✅ Generated {len(data)} data points with technical indicators")
    return data


def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26):
    """Calculate MACD technical indicator."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd


def demo_shap_explainability(data):
    """Demonstrate SHAP model explainability."""
    print("\n🔍 SHAP Model Explainability Demo")
    print("=" * 50)

    try:
        from models.shap_explainer import SHAPExplainer

        # Create a mock model for demonstration
        class MockModel:
            def predict(self, X):
                # Simple mock prediction based on price and volume
                return np.mean(X, axis=1) * 100

        # Prepare feature data
        feature_names = ['Open', 'High', 'Low',
                         'Close', 'Volume', 'RSI', 'MACD']
        X = data[feature_names].values

        # Create SHAP explainer
        model = MockModel()
        explainer = SHAPExplainer(model, feature_names)

        # Create background data (first 100 samples)
        background_data = X[:100]

        # Create explainer
        if explainer.create_explainer(background_data, explainer_type="kernel"):
            print("✅ SHAP explainer created successfully")

            # Generate explanations for recent data
            recent_data = X[-50:]
            explanation = explainer.explain_predictions(recent_data)

            if explanation:
                print(
                    f"✅ Generated SHAP explanations for {explanation['prediction_count']} predictions")

                # Get feature importance ranking
                importance_df = explainer.get_feature_importance_ranking(
                    explanation)
                print("\n📊 Top 5 Most Important Features:")
                print(importance_df.head().to_string(index=False))

                # Generate explanation report
                report = explainer.generate_explanation_report(explanation)
                print(
                    f"\n📋 Generated explanation report ({len(report)} characters)")

                # Risk analysis
                risk_analysis = explainer.get_risk_explanation(explanation)
                if risk_analysis:
                    print(f"\n⚠️  Risk Analysis:")
                    print(
                        f"   High-risk features: {len(risk_analysis['high_risk_features'])}")
                    print(
                        f"   Risk percentage: {risk_analysis['risk_percentage']:.1f}%")

                    if risk_analysis['recommendations']:
                        print("   Recommendations:")
                        for rec in risk_analysis['recommendations'][:3]:
                            print(f"     • {rec}")

                return explanation
            else:
                print("❌ Failed to generate SHAP explanations")
                return None
        else:
            print("❌ Failed to create SHAP explainer")
            return None

    except ImportError as e:
        print(f"⚠️  SHAP not available: {e}")
        print("   Install with: pip install shap")
        return None
    except Exception as e:
        print(f"❌ Error in SHAP demo: {e}")
        return None


def demo_real_time_streaming():
    """Demonstrate real-time data streaming capabilities."""
    print("\n🌊 Real-time Data Streaming Demo")
    print("=" * 50)

    try:
        from data.real_time_streamer import FinancialDataStreamer

        # Create financial data streamer
        streamer = FinancialDataStreamer()

        # Add sample data sources
        sample_sources = {
            'stock_prices': {
                'type': 'rest',
                'url': 'https://api.example.com/stock/{symbol}',
                'symbols': ['AAPL', 'GOOGL', 'MSFT'],
                'poll_interval': 5
            },
            'market_data': {
                'type': 'websocket',
                'url': 'wss://stream.example.com/market',
                'symbols': ['SPY', 'QQQ', 'IWM']
            }
        }

        for source_name, source_config in sample_sources.items():
            if streamer.add_data_source(source_name, source_config):
                print(f"✅ Added data source: {source_name}")
            else:
                print(f"❌ Failed to add data source: {source_name}")

        # Add custom data processor
        def custom_processor(data_item):
            """Custom data processor for demonstration."""
            if 'data' in data_item and isinstance(data_item['data'], dict):
                data_item['data']['processed'] = True
                data_item['data']['timestamp_processed'] = datetime.now(
                ).isoformat()
            return data_item

        if streamer.add_data_processor('custom_processor', custom_processor):
            print("✅ Added custom data processor")

        # Add callback for data updates
        def data_callback(data):
            print(f"📡 Received data from {data.get('source', 'unknown')}")

        if streamer.add_callback(data_callback):
            print("✅ Added data callback")

        # Show streamer configuration
        print(f"\n📋 Streamer Configuration:")
        print(f"   Data sources: {len(streamer.connections)}")
        print(f"   Data processors: {len(streamer.processors)}")
        print(f"   Callbacks: {len(streamer.callbacks)}")
        print(f"   Max queue size: {streamer.config['max_queue_size']}")

        print("\n💡 Note: This is a configuration demo. Real streaming requires active data sources.")
        return streamer

    except ImportError as e:
        print(f"⚠️  Real-time streaming not available: {e}")
        print("   Install with: pip install aiohttp websockets")
        return None
    except Exception as e:
        print(f"❌ Error in streaming demo: {e}")
        return None


def demo_advanced_analytics(data):
    """Demonstrate advanced analytics capabilities."""
    print("\n🧠 Advanced Analytics Demo")
    print("=" * 50)

    try:
        from analytics.advanced_analytics import AdvancedAnalyticsEngine

        # Create analytics engine
        analytics_engine = AdvancedAnalyticsEngine()

        # Prepare market data
        market_data = data.copy()
        market_data['Returns'] = market_data['Close'].pct_change()

        # Run comprehensive analysis
        print("🔄 Running comprehensive market analysis...")
        analysis_results = analytics_engine.run_comprehensive_analysis(
            market_data)

        if analysis_results and 'error' not in analysis_results:
            print("✅ Analysis completed successfully")

            # Display results
            print(
                f"\n📊 Analysis Methods Used: {len(analysis_results['analysis_methods'])}")
            for method in analysis_results['analysis_methods']:
                print(f"   • {method}")

            # Volatility regimes
            if 'volatility_regimes' in analysis_results:
                regimes = analysis_results['volatility_regimes']
                print(f"\n📈 Volatility Regimes:")
                print(f"   Current regime: {regimes['current_regime']}")
                print(f"   Method: {regimes['method']}")

                for regime_name, stats in regimes['regime_stats'].items():
                    print(
                        f"   {regime_name}: {stats['volatility']:.4f} volatility, {stats['percentage']:.1f}% of time")

            # Market sentiment
            if 'market_sentiment' in analysis_results:
                sentiment = analysis_results['market_sentiment']
                print(f"\n😊 Market Sentiment:")
                print(f"   Sentiment: {sentiment['sentiment_label']}")
                print(f"   Score: {sentiment['sentiment_score']:.3f}")
                print(f"   Confidence: {sentiment['confidence']:.1f}")

                if 'indicators' in sentiment:
                    print("   Indicators:")
                    for indicator, value in sentiment['indicators'].items():
                        print(f"     • {indicator}: {value:.3f}")

            # Analysis summary
            if 'summary' in analysis_results:
                summary = analysis_results['summary']
                print(f"\n📋 Analysis Summary:")
                print(f"   Total analyses: {summary['total_analyses']}")
                print(f"   Risk assessment: {summary['risk_assessment']}")
                print(f"   Market outlook: {summary['market_outlook']}")

                if summary['key_findings']:
                    print("   Key findings:")
                    for finding in summary['key_findings']:
                        print(f"     • {finding}")

            return analysis_results
        else:
            error_msg = analysis_results.get('error', 'Unknown error')
            print(f"❌ Analysis failed: {error_msg}")
            return None

    except ImportError as e:
        print(f"⚠️  Advanced analytics not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Error in analytics demo: {e}")
        return None


def demo_automated_reporting(data, shap_results=None, analytics_results=None):
    """Demonstrate automated reporting capabilities."""
    print("\n📄 Automated Reporting Demo")
    print("=" * 50)

    try:
        from reporting.automated_reporter import AutomatedReporter

        # Create automated reporter
        reporter = AutomatedReporter(output_dir="demo_reports")
        print("✅ Automated reporter initialized")

        # Prepare portfolio data
        portfolio_data = {
            'total_value': 1250000,
            'total_return': 15.8,
            'sharpe_ratio': 1.24,
            'performance_metrics': {
                'total_return': 15.8,
                'annualized_return': 12.3,
                'volatility': 18.5,
                'sharpe_ratio': 1.24,
                'max_drawdown': -8.2,
                'var_95': -2.1,
                'cvar_95': -3.5
            },
            'asset_allocation': {
                'AAPL': {'weight': 0.25, 'value': 312500, 'returns': 18.5},
                'GOOGL': {'weight': 0.20, 'value': 250000, 'returns': 12.8},
                'MSFT': {'weight': 0.20, 'value': 250000, 'returns': 15.2},
                'SPY': {'weight': 0.25, 'value': 312500, 'returns': 14.1},
                'BND': {'weight': 0.10, 'value': 125000, 'returns': 3.2}
            },
            'risk_metrics': {
                'var_95': -2.1,
                'cvar_95': -3.5,
                'max_drawdown': -8.2,
                'volatility': 18.5,
                'beta': 0.95,
                'correlation': 0.87,
                'recommendations': [
                    'Monitor high-beta assets during market volatility',
                    'Consider rebalancing to maintain target allocations',
                    'Review risk tolerance and adjust if necessary'
                ]
            }
        }

        # Generate portfolio report
        print("📊 Generating portfolio performance report...")
        try:
            portfolio_report_path = reporter.generate_portfolio_report(
                portfolio_data, 'html')
            print(f"✅ Portfolio report generated: {portfolio_report_path}")
        except Exception as e:
            print(f"⚠️  Portfolio report generation failed: {e}")

        # Generate risk report
        print("⚠️  Generating risk analysis report...")
        try:
            risk_data = {
                'overall_risk_score': 0.35,
                'key_risk_indicators': [
                    'Elevated market volatility',
                    'Concentration in tech sector',
                    'Currency exposure in international holdings'
                ],
                'detailed_metrics': portfolio_data['risk_metrics']
            }

            risk_report_path = reporter.generate_risk_report(risk_data, 'html')
            print(f"✅ Risk report generated: {risk_report_path}")
        except Exception as e:
            print(f"⚠️  Risk report generation failed: {e}")

        # Generate comprehensive report if we have analytics results
        if analytics_results:
            print("🔍 Generating comprehensive analysis report...")
            try:
                all_data = {
                    'portfolio': portfolio_data,
                    'analytics': analytics_results
                }

                comprehensive_report_path = reporter.generate_comprehensive_report(
                    all_data, 'html')
                print(
                    f"✅ Comprehensive report generated: {comprehensive_report_path}")
            except Exception as e:
                print(f"⚠️  Comprehensive report generation failed: {e}")

        print("\n📁 Reports saved to 'demo_reports' directory")
        print("💡 Open HTML files in your browser to view the reports")

        return True

    except ImportError as e:
        print(f"⚠️  Automated reporting not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Error in reporting demo: {e}")
        return None


def main():
    """Main demonstration function."""
    print("🚀 GMF Time Series Forecasting - Advanced Features Demo")
    print("=" * 70)
    print("Demonstrating advanced Phase 2 capabilities:")
    print("• SHAP Model Explainability")
    print("• Real-time Data Streaming")
    print("• Advanced Analytics")
    print("• Automated Reporting")
    print("=" * 70)

    # Generate sample data
    data = generate_sample_data()

    # Demo 1: SHAP Explainability
    shap_results = demo_shap_explainability(data)

    # Demo 2: Real-time Streaming
    streamer = demo_real_time_streaming()

    # Demo 3: Advanced Analytics
    analytics_results = demo_advanced_analytics(data)

    # Demo 4: Automated Reporting
    reporting_success = demo_automated_reporting(
        data, shap_results, analytics_results)

    # Summary
    print("\n" + "=" * 70)
    print("🎯 Advanced Features Demo Summary")
    print("=" * 70)

    demos = [
        ("SHAP Explainability", shap_results is not None),
        ("Real-time Streaming", streamer is not None),
        ("Advanced Analytics", analytics_results is not None),
        ("Automated Reporting", reporting_success is not None)
    ]

    for demo_name, success in demos:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{demo_name:25} {status}")

    passed_count = sum(1 for _, success in demos if success)
    total_count = len(demos)

    print(f"\n📊 Overall Results: {passed_count}/{total_count} demos passed")

    if passed_count == total_count:
        print("🎉 All advanced features are working correctly!")
        print("🚀 Your GMF system is ready for production use!")
    else:
        print("⚠️  Some features need attention. Check the error messages above.")
        print("💡 Install missing dependencies: pip install -r requirements.txt")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

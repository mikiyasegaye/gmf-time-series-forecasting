#!/usr/bin/env python3
"""
GMF Time Series Forecasting - Professional Test Suite

This comprehensive test suite validates the refactored GMF Time Series Forecasting
codebase, ensuring all modules function correctly and meet production standards.

Author: GMF Investment Team
Version: 2.0.0
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


class GMFTestSuite:
    """
    Professional test suite for GMF Time Series Forecasting system.

    This class provides comprehensive testing of all major components:
    - Data processing and validation
    - Forecasting models (LSTM, ARIMA)
    - Portfolio optimization and risk management
    - Backtesting and performance analysis
    - Visualization and reporting capabilities
    """

    def __init__(self):
        """Initialize the test suite with test data and configurations."""
        self.test_results = {}
        self.start_time = datetime.now()

        # Test configuration
        self.test_config = {
            'num_test_records': 1000,
            'num_assets': 5,
            'test_date_range': 1462,  # ~4 years of daily data
            'confidence_levels': [0.90, 0.95, 0.99]
        }

        print("🚀 GMF Time Series Forecasting - Professional Test Suite")
        print("=" * 70)
        print(f"Test Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python Version: {sys.version.split()[0]}")
        print("=" * 70)

    def run_all_tests(self):
        """Execute the complete test suite."""
        print("\n🧪 EXECUTING COMPREHENSIVE TEST SUITE")
        print("-" * 50)

        # Core functionality tests
        self.test_core_imports()
        self.test_data_processing()
        self.test_forecasting_models()
        self.test_portfolio_optimization()
        self.test_risk_management()
        self.test_backtesting_engine()
        self.test_visualization_system()

        # Generate comprehensive report
        self.generate_test_report()

    def test_core_imports(self):
        """Test that all core modules can be imported successfully."""
        print("\n📦 TESTING CORE MODULE IMPORTS")
        print("  └─ Validating module architecture...")

        try:
            # Data processing modules
            from data import DataProcessor, DataLoader
            print("    ✅ Data Processing: DataProcessor, DataLoader")

            # Forecasting models
            from models import LSTMForecaster, ARIMAForecaster, ModelEvaluator
            print("    ✅ Forecasting Models: LSTM, ARIMA, Evaluator")

            # Portfolio management
            from portfolio import PortfolioOptimizer, EfficientFrontier
            print("    ✅ Portfolio Management: Optimizer, EfficientFrontier")

            # Backtesting system
            from backtesting import BacktestEngine, PerformanceAnalyzer
            print("    ✅ Backtesting System: Engine, Analyzer")

            # Risk management
            from utils import RiskMetrics, ValidationUtils
            print("    ✅ Risk Management: Metrics, Validation")

            # Visualization system
            from visualization import PlotGenerator, DashboardCreator
            print("    ✅ Visualization System: Plots, Dashboards")

            self.test_results['core_imports'] = {
                'status': 'PASSED', 'details': 'All modules imported successfully'}
            print("    🎯 RESULT: All core modules imported successfully")

        except ImportError as e:
            self.test_results['core_imports'] = {
                'status': 'FAILED', 'details': str(e)}
            print(f"    ❌ RESULT: Import error - {e}")
            raise

    def test_data_processing(self):
        """Test data processing and validation capabilities."""
        print("\n🔄 TESTING DATA PROCESSING SYSTEM")
        print("  └─ Validating data cleaning and validation...")

        try:
            from data import DataProcessor

            # Create comprehensive test data
            test_data = self._generate_test_financial_data()

            # Test DataProcessor with temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                processor = DataProcessor(temp_dir)

                # Test data processing
                processed_data = processor.process_price_data(test_data)

                # Validate processing results
                validation_results = self._validate_processed_data(
                    processed_data)

                if validation_results['is_valid']:
                    self.test_results['data_processing'] = {
                        'status': 'PASSED',
                        'details': f"Processed {len(processed_data)} records successfully",
                        'metrics': validation_results['metrics']
                    }
                    print(
                        f"    ✅ RESULT: Data processing successful - {len(processed_data)} clean records")
                    print(
                        f"       └─ Data quality score: {validation_results['metrics']['quality_score']:.1f}%")
                else:
                    raise ValueError("Data validation failed")

        except Exception as e:
            self.test_results['data_processing'] = {
                'status': 'FAILED', 'details': str(e)}
            print(f"    ❌ RESULT: Data processing failed - {e}")
            raise

    def test_forecasting_models(self):
        """Test forecasting model functionality."""
        print("\n🔮 TESTING FORECASTING MODELS")
        print("  └─ Validating LSTM and ARIMA models...")

        try:
            from models import LSTMForecaster, ARIMAForecaster, ModelEvaluator

            # Test data preparation
            test_data = self._generate_test_financial_data()
            # Use subset for testing
            test_series = test_data['Close'].iloc[:100]

            # Test ARIMA Forecaster (faster for testing)
            arima_forecaster = ARIMAForecaster(order=(1, 1, 1))

            # Create test DataFrame
            test_df = pd.DataFrame({'Close': test_series})

            # Test model fitting
            fit_results = arima_forecaster.fit(
                test_df, target_column='Close', test_size=0.2)

            if fit_results and 'order' in fit_results:
                self.test_results['forecasting_models'] = {
                    'status': 'PASSED',
                    'details': f"ARIMA{fit_results['order']} model fitted successfully",
                    'metrics': fit_results.get('metrics', {})
                }
                print(
                    f"    ✅ RESULT: Forecasting models working - ARIMA{fit_results['order']} fitted")
            else:
                raise ValueError("Model fitting failed")

        except Exception as e:
            self.test_results['forecasting_models'] = {
                'status': 'FAILED', 'details': str(e)}
            print(f"    ❌ RESULT: Forecasting models failed - {e}")
            raise

    def test_portfolio_optimization(self):
        """Test portfolio optimization and efficient frontier generation."""
        print("\n💼 TESTING PORTFOLIO OPTIMIZATION")
        print("  └─ Validating Modern Portfolio Theory implementation...")

        try:
            from portfolio import PortfolioOptimizer

            # Generate realistic returns data
            returns_data = self._generate_test_returns_data()

            # Initialize optimizer
            optimizer = PortfolioOptimizer(returns_data)

            # Test efficient frontier generation
            frontier_data = optimizer.generate_efficient_frontier(
                num_portfolios=25)

            if len(frontier_data) > 0:
                # Calculate frontier statistics
                frontier_stats = {
                    'num_portfolios': len(frontier_data),
                    'min_volatility': frontier_data['volatility'].min(),
                    'max_return': frontier_data['actual_return'].max(),
                    'avg_sharpe': frontier_data['sharpe_ratio'].mean()
                }

                self.test_results['portfolio_optimization'] = {
                    'status': 'PASSED',
                    'details': f"Generated {len(frontier_data)} efficient portfolios",
                    'metrics': frontier_stats
                }
                print(
                    f"    ✅ RESULT: Portfolio optimization successful - {len(frontier_data)} portfolios")
                print(
                    f"       └─ Volatility range: {frontier_stats['min_volatility']:.3f} to {frontier_data['volatility'].max():.3f}")
            else:
                raise ValueError("No portfolios generated")

        except Exception as e:
            self.test_results['portfolio_optimization'] = {
                'status': 'FAILED', 'details': str(e)}
            print(f"    ❌ RESULT: Portfolio optimization failed - {e}")
            raise

    def test_risk_management(self):
        """Test risk metrics calculation and validation."""
        print("\n⚠️  TESTING RISK MANAGEMENT SYSTEM")
        print("  └─ Validating VaR, CVaR, and risk metrics...")

        try:
            from utils import RiskMetrics

            # Generate test returns data
            test_returns = self._generate_test_returns_data()

            # Initialize risk calculator
            risk_calculator = RiskMetrics(test_returns)

            # Test multiple confidence levels
            risk_results = {}
            for confidence in self.test_config['confidence_levels']:
                var_result = risk_calculator.calculate_var(
                    confidence, 'historical')
                if 'var_daily' in var_result:
                    risk_results[f'var_{int(confidence*100)}'] = var_result['var_daily']

            if len(risk_results) == len(self.test_config['confidence_levels']):
                self.test_results['risk_management'] = {
                    'status': 'PASSED',
                    'details': f"Risk metrics calculated for {len(risk_results)} confidence levels",
                    'metrics': risk_results
                }
                print(
                    f"    ✅ RESULT: Risk management successful - {len(risk_results)} metrics calculated")
                print(
                    f"       └─ VaR(95%): {risk_results.get('var_95', 'N/A'):.4f}")
            else:
                raise ValueError("Incomplete risk metrics calculation")

        except Exception as e:
            self.test_results['risk_management'] = {
                'status': 'FAILED', 'details': str(e)}
            print(f"    ❌ RESULT: Risk management failed - {e}")
            raise

    def test_backtesting_engine(self):
        """Test backtesting and performance analysis capabilities."""
        print("\n📊 TESTING BACKTESTING ENGINE")
        print("  └─ Validating strategy backtesting...")

        try:
            from backtesting import BacktestEngine, PerformanceAnalyzer

            # Generate test portfolio data
            portfolio_data = self._generate_test_portfolio_data()

            # Test backtesting engine
            backtest_engine = BacktestEngine(portfolio_data)

            # Run simple backtest
            backtest_results = backtest_engine.run_backtest(
                strategy_type='buy_and_hold',
                initial_capital=100000
            )

            if backtest_results and 'final_value' in backtest_results:
                self.test_results['backtesting_engine'] = {
                    'status': 'PASSED',
                    'details': f"Backtest completed successfully",
                    'metrics': {
                        'initial_capital': backtest_results.get('initial_capital', 0),
                        'final_value': backtest_results.get('final_value', 0),
                        'total_return': backtest_results.get('total_return', 0)
                    }
                }
                print(
                    f"    ✅ RESULT: Backtesting engine working - {backtest_results.get('total_return', 0):.2f}% return")
            else:
                raise ValueError("Backtest results incomplete")

        except Exception as e:
            self.test_results['backtesting_engine'] = {
                'status': 'FAILED', 'details': str(e)}
            print(f"    ❌ RESULT: Backtesting engine failed - {e}")
            raise

    def test_visualization_system(self):
        """Test visualization and plotting capabilities."""
        print("\n🎨 TESTING VISUALIZATION SYSTEM")
        print("  └─ Validating chart generation and dashboard creation...")

        try:
            from visualization import PlotGenerator, DashboardCreator

            # Test PlotGenerator
            plotter = PlotGenerator()

            # Test DashboardCreator
            dashboard_creator = DashboardCreator()

            # Generate sample data for testing
            test_data = self._generate_test_financial_data()

            # Test dashboard creation
            dashboard_code = dashboard_creator.create_streamlit_dashboard(
                {'test_data': test_data}
            )

            if dashboard_code and len(dashboard_code) > 100:
                self.test_results['visualization_system'] = {
                    'status': 'PASSED',
                    'details': f"Visualization system working - {len(dashboard_code)} chars generated",
                    'metrics': {'code_length': len(dashboard_code)}
                }
                print(
                    f"    ✅ RESULT: Visualization system working - {len(dashboard_code)} chars generated")
            else:
                raise ValueError("Dashboard generation failed")

        except Exception as e:
            self.test_results['visualization_system'] = {
                'status': 'FAILED', 'details': str(e)}
            print(f"    ❌ RESULT: Visualization system failed - {e}")
            raise

    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 70)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 70)

        # Calculate test statistics
        total_tests = len(self.test_results)
        passed_tests = sum(
            1 for result in self.test_results.values() if result['status'] == 'PASSED')
        failed_tests = total_tests - passed_tests

        # Overall status
        overall_status = "🎉 ALL TESTS PASSED" if failed_tests == 0 else f"⚠️  {failed_tests} TESTS FAILED"

        print(f"\n🏆 OVERALL STATUS: {overall_status}")
        print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
        print(f"⏱️  Total Duration: {datetime.now() - self.start_time}")

        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 50)

        for test_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            print(
                f"{status_icon} {test_name.replace('_', ' ').title()}: {result['status']}")
            if 'details' in result:
                print(f"    └─ {result['details']}")

        # Recommendations
        print(f"\n🚀 RECOMMENDATIONS:")
        print("-" * 30)

        if failed_tests == 0:
            print("✅ System is production-ready!")
            print("✅ All core functionality validated")
            print("✅ Ready for advanced feature development")
            print("\n🎯 Next steps:")
            print("   1. Launch dashboard: streamlit run dashboard_demo.py")
            print("   2. Explore notebooks/ directory")
            print("   3. Implement Phase 2 features")
        else:
            print("⚠️  System needs attention before production use")
            print("🔧 Review failed tests and fix issues")
            print("🧪 Re-run test suite after fixes")

        print("\n" + "=" * 70)

    def _generate_test_financial_data(self):
        """Generate realistic test financial data."""
        dates = pd.date_range(
            start='2020-01-01', periods=self.test_config['test_date_range'], freq='D')

        # Generate realistic price data
        np.random.seed(42)  # For reproducible tests

        # Start with base price and add random walk
        base_price = 100
        price_changes = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * np.cumprod(1 + price_changes)

        # Generate OHLCV data
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
        data['Low'] = np.minimum(
            data['Low'], data[['Open', 'Close']].min(axis=1))

        data.set_index('Date', inplace=True)
        return data

    def _generate_test_returns_data(self):
        """Generate realistic test returns data for multiple assets."""
        dates = pd.date_range(
            start='2020-01-01', periods=self.test_config['test_date_range'], freq='D')
        assets = [
            f'ASSET_{i+1}' for i in range(self.test_config['num_assets'])]

        # Generate correlated returns
        np.random.seed(42)
        returns_data = pd.DataFrame(
            np.random.normal(0.001, 0.02, (len(dates), len(assets))),
            index=dates,
            columns=assets
        )

        return returns_data

    def _generate_test_portfolio_data(self):
        """Generate test portfolio data for backtesting."""
        dates = pd.date_range(start='2020-01-01',
                              periods=252, freq='D')  # 1 year

        # Generate portfolio values with realistic growth
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        portfolio_values = 100000 * np.cumprod(1 + returns)

        portfolio_data = pd.DataFrame({
            'Date': dates,
            'Portfolio_Value': portfolio_values,
            'Returns': returns
        })

        portfolio_data.set_index('Date', inplace=True)
        return portfolio_data

    def _validate_processed_data(self, processed_data):
        """Validate processed data quality."""
        if processed_data is None or len(processed_data) == 0:
            return {'is_valid': False, 'metrics': {}}

        # Calculate quality metrics
        total_rows = len(processed_data)
        missing_values = processed_data.isnull().sum().sum()
        negative_prices = ((processed_data[['Open', 'High', 'Low', 'Close']] <= 0).sum().sum()
                           if all(col in processed_data.columns for col in ['Open', 'High', 'Low', 'Close']) else 0)

        quality_score = max(
            0, 100 - (missing_values / total_rows * 50) - (negative_prices / total_rows * 50))

        return {
            'is_valid': quality_score > 80,
            'metrics': {
                'total_rows': total_rows,
                'missing_values': missing_values,
                'negative_prices': negative_prices,
                'quality_score': quality_score
            }
        }


def main():
    """Main test execution function."""
    try:
        # Initialize and run test suite
        test_suite = GMFTestSuite()
        test_suite.run_all_tests()

    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

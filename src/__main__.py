#!/usr/bin/env python3
"""
GMF Time Series Forecasting - Main Package Entry Point

This module provides the main entry point when running the package directly.
It allows users to run: python -m src

Usage:
    python -m src                    # Launch dashboard
    python -m src --dashboard        # Launch dashboard
    python -m src --test             # Run tests
    python -m src --help             # Show help
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main entry point for the GMF package."""
    parser = argparse.ArgumentParser(
        description="GMF Time Series Forecasting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src                    # Launch dashboard
    python -m src --dashboard        # Launch dashboard
    python -m src --test             # Run test suite
    python -m src --help             # Show this help
        """
    )

    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Launch the interactive dashboard'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run the comprehensive test suite'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='GMF Time Series Forecasting v2.0.0'
    )

    args = parser.parse_args()

    # Default behavior: launch dashboard
    if not args.test:
        launch_dashboard()
    else:
        run_tests()


def launch_dashboard():
    """Launch the GMF Dashboard."""
    try:
        from dashboard import GMFDashboard

        print("🚀 Launching GMF Time Series Forecasting Dashboard...")
        print("📊 Version: 2.0.0")
        print("=" * 50)

        # Initialize and run dashboard
        dashboard = GMFDashboard()
        dashboard.run_dashboard()

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Dashboard Error: {e}")
        sys.exit(1)


def run_tests():
    """Run the GMF test suite."""
    try:
        from tests.test_gmf_system import GMFTestSuite

        print("🧪 Running GMF Time Series Forecasting Test Suite...")
        print("📊 Version: 2.0.0")
        print("=" * 50)

        # Initialize and run test suite
        test_suite = GMFTestSuite()
        test_suite.run_all_tests()

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Test Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

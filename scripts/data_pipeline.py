#!/usr/bin/env python3
"""
Data Pipeline Script

Automates the data loading, cleaning, and preprocessing steps from Task 1.
Useful for refreshing data or rerunning the preprocessing workflow.

Usage:
    python scripts/data_pipeline.py [--refresh-all]
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


def load_and_clean_data():
    """Load and clean all financial data"""
    print("📊 Loading financial data...")
    # TODO: Extract data loading logic from Task 1 notebook
    # This will load TSLA, AAPL, AMZN, GOOG, META, MSFT, NVDA, SPY, BND
    pass


def fetch_latest_data():
    """Fetch latest data from YFinance"""
    print("🔄 Fetching latest market data...")
    # TODO: Implement YFinance fetching for SPY and BND only
    pass


def main():
    parser = argparse.ArgumentParser(description="Run data pipeline")
    parser.add_argument(
        "--refresh-all",
        action="store_true",
        help="Refresh all data from sources"
    )

    args = parser.parse_args()

    print("🔧 Starting data pipeline...")

    if args.refresh_all:
        fetch_latest_data()

    load_and_clean_data()

    print("✅ Data pipeline completed!")


if __name__ == "__main__":
    main()

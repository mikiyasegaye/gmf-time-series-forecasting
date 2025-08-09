# GMF Time Series Forecasting Challenge

**Guide Me in Finance (GMF) Investments** - Time series forecasting project for portfolio management optimization through market trend prediction, volatility analysis, and data-driven asset allocation strategies.

## 📊 Project Overview

This project implements advanced time series forecasting models to enhance portfolio management strategies for GMF Investments. The goal is to predict market trends, optimize asset allocation, and enhance portfolio performance while minimizing risks and capitalizing on market opportunities.

## 🎯 Business Objective

GMF Investments leverages cutting-edge technology and data-driven insights to provide clients with tailored investment strategies. By integrating advanced time series forecasting models, GMF aims to:

- Predict market trends and volatility
- Optimize asset allocation
- Enhance portfolio performance
- Minimize risks while capitalizing on market opportunities

## 📈 Assets Analyzed

**Primary Assets:**

- **TSLA (Tesla)**: High-growth, high-risk stock in consumer discretionary sector
- **SPY (S&P 500 ETF)**: Broad U.S. market exposure with moderate risk
- **BND (Vanguard Total Bond Market ETF)**: Stability and income with low risk

**Additional Tech Stocks for Sector Analysis:**

- **AAPL (Apple)**, **AMZN (Amazon)**, **GOOG (Google)**, **META (Meta)**, **MSFT (Microsoft)**, **NVDA (NVIDIA)**

**Total: 9 assets + 1.4M analyst ratings for comprehensive analysis**

## 🏗️ Project Structure

```
gmf-time-series-forecasting/
├── data/
│   ├── raw/                    # Historical financial data
│   │   ├── yfinance_data/      # Stock price data (TSLA, AAPL, AMZN, GOOG, META, MSFT, NVDA)
│   │   └── raw_analyst_ratings.csv  # 1.4M analyst ratings
│   └── processed/              # Cleaned and processed data (9 assets + risk metrics)
│       ├── *_processed.csv     # Cleaned price data for all assets
│       ├── *_returns.csv       # Daily returns for all assets
│       ├── all_risk_metrics.csv # Comprehensive risk analysis
│       └── correlation_matrix.csv # Asset correlation matrix
├── notebooks/                  # Jupyter notebooks for analysis
│   └── 01_data_preprocessing_and_exploration.ipynb  # ✅ Task 1 Complete
├── scripts/                    # Automation scripts
│   ├── .gitkeep               # Directory tracking
│   └── data_pipeline.py       # Task 1 data processing automation
├── src/                        # Python package structure
│   ├── __init__.py            # Main package
│   ├── data/                  # Data processing modules
│   ├── models/                # Model development modules (ready for Task 2)
│   ├── utils/                 # Utility functions
│   └── visualization/         # Plotting and visualization
├── models/                     # Model storage (ready for Task 2)
│   └── .gitkeep               # Directory tracking
├── tests/                      # Unit tests
│   └── __init__.py            # Test package
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                 # Git ignore rules
```

## ✅ Task Progress

### Task 1: Preprocess and Explore the Data ✅ **COMPLETED**

**Comprehensive Data Utilization:**

- ✅ **All YFinance files**: TSLA, AAPL, AMZN, GOOG, META, MSFT, NVDA, SPY, BND (9 assets)
- ✅ **Analyst ratings**: 1.4M records with sentiment analysis and mentions tracking
- ✅ **Timezone consistency**: Fixed timezone conflicts between data sources
- ✅ **Robust datetime parsing**: Handled mixed timezone formats in analyst data
- ✅ **Enterprise-grade error handling**: Production-ready data processing

**Analysis Components:**

- ✅ **Data loading and cleaning** with timezone normalization
- ✅ **Comprehensive EDA** with tech sector vs target assets comparison
- ✅ **Stationarity testing** (ADF tests) - all return series confirmed stationary
- ✅ **Volatility analysis** across all sectors with rolling calculations
- ✅ **Risk metrics calculation** (Sharpe, VaR, drawdown, skewness, kurtosis)
- ✅ **Correlation analysis** with comprehensive heatmap for all assets
- ✅ **Outlier analysis** and extreme returns identification
- ✅ **Analyst sentiment integration** with mentions over time
- ✅ **Risk-return visualization** for all assets

**Key Findings:**

- **TSLA**: Highest volatility among all tech stocks, strong growth potential
- **Tech sector**: High correlation (0.6-0.8 range) indicating sector momentum
- **SPY**: Excellent diversification benefits with moderate risk
- **BND**: Low correlation with tech stocks, provides portfolio stability
- **NVDA**: Highest returns among tech stocks in the analysis period
- All return series are stationary (perfect for ARIMA models)
- Data quality is excellent across all 9 assets (2015-2025)

### Task 2: Develop Time Series Forecasting Models 🚀 **READY TO START**

**Objectives:**

- **ARIMA/SARIMA models**: Classical statistical forecasting with auto-parameter selection
- **LSTM models**: Deep learning approach for complex pattern recognition
- **Enhanced features**: Incorporate sector correlations and analyst sentiment
- **Model comparison**: Performance evaluation using MAE, RMSE, MAPE
- **Parameter optimization**: Grid search and auto_arima for best configurations
- **Chronological splits**: 2015-2023 train, 2024-2025 test (time-aware validation)

### Task 3: Forecast Future Market Trends 📋 **PENDING**

**Objectives:**

- Generate 6-12 month forecasts
- Visualize forecasts with confidence intervals
- Analyze trend patterns and volatility
- Identify market opportunities and risks

### Task 4: Optimize Portfolio Based on Forecast 📋 **PENDING**

**Objectives:**

- Implement Modern Portfolio Theory (MPT)
- Generate Efficient Frontier
- Identify Maximum Sharpe Ratio Portfolio
- Identify Minimum Volatility Portfolio
- Recommend optimal portfolio weights

### Task 5: Strategy Backtesting 📋 **PENDING**

**Objectives:**

- Implement backtesting framework
- Compare strategy vs benchmark (60% SPY / 40% BND)
- Calculate performance metrics
- Validate model-driven approach

## 🛠️ Technical Stack

- **Python 3.12** with virtual environment
- **Data Analysis**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn
- **Financial Data**: yfinance
- **Time Series**: statsmodels, pmdarima
- **Machine Learning**: scikit-learn, tensorflow, keras
- **Development**: jupyter, git

## 📦 Installation

```bash
# Clone repository
git clone <repository-url>
cd gmf-time-series-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Launch Jupyter
jupyter notebook notebooks/

# Run Task 1 (completed)
# Open: notebooks/01_data_preprocessing_and_exploration.ipynb
```

## 📊 Data Sources

- **YFinance API**: Real-time financial data for SPY and BND
- **Historical CSV Files**: TSLA, AAPL, AMZN, GOOG, META, MSFT, NVDA (2015-2025)
- **Analyst Ratings**: 1.4M records with headlines, publishers, and sentiment data
- **Market Data**: OHLCV, volume, dividends, stock splits
- **Comprehensive Coverage**: 9 assets with timezone-normalized, production-ready data

## 🎯 Key Insights from Task 1

1. **Comprehensive Asset Analysis**:

   - **9 assets analyzed**: Full tech sector + diversified portfolio components
   - **Tech sector correlation**: 0.6-0.8 range showing strong sector momentum
   - **TSLA**: Highest volatility but strong growth potential among all assets
   - **SPY**: Excellent diversification with moderate risk profile
   - **BND**: Low correlation with tech stocks, ideal for portfolio stability

2. **Data Quality Excellence**:

   - **All return series stationary**: Perfect foundation for ARIMA/SARIMA models
   - **No missing data issues**: Robust preprocessing with timezone consistency
   - **1.4M analyst records**: Rich sentiment data for enhanced forecasting
   - **Production-ready**: Enterprise-grade error handling and data validation

3. **Advanced Risk Insights**:

   - **TSLA**: Maximum drawdown -73.63%, highest reward/risk among tech stocks
   - **NVDA**: Highest returns in tech sector during analysis period
   - **Correlation patterns**: Clear sector clustering for portfolio optimization
   - **Volatility analysis**: 30-day rolling patterns reveal market cycles

4. **Enhanced Features for Task 2**:
   - **Sector momentum signals**: Tech stock correlations for ensemble models
   - **Sentiment indicators**: Analyst mentions and news impact analysis
   - **Market regime detection**: Volatility patterns for adaptive modeling

## 📈 Next Steps

1. **Task 2**: Develop forecasting models (ARIMA + LSTM)
2. **Task 3**: Generate future market predictions
3. **Task 4**: Optimize portfolio allocation
4. **Task 5**: Backtest strategy performance

## 🤝 Contributing

This project follows best practices for data science projects:

- Modular code structure
- Comprehensive documentation
- Version control with git
- Reproducible analysis

## 📝 License

This project is part of the 10 Academy Artificial Intelligence Mastery program.

---

**Status**: Task 1 Complete ✅ | Task 2 Ready 🚀 | Tasks 3-5 Pending 📋

**Task 1 Achievement**: ⭐ **Enterprise-Grade Data Analysis** with full utilization of all 9 assets + 1.4M analyst records

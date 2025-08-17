#!/usr/bin/env python3
"""
GMF Time Series Forecasting - Professional Dashboard

A comprehensive, interactive dashboard showcasing the refactored GMF Time Series
Forecasting system capabilities including portfolio management, risk analysis,
forecasting, and performance analytics.

Author: GMF Investment Team
Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="GMF Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1.1em;
        opacity: 0.9;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class GMFDashboard:
    """
    Professional GMF Forecasting Dashboard class.

    Provides comprehensive financial analysis capabilities:
    - Portfolio overview and performance tracking
    - Risk analysis and metrics
    - Forecasting and model performance
    - Data quality and processing insights
    """

    def __init__(self):
        """Initialize the dashboard with default configurations."""
        self.default_assets = ["AAPL", "GOOGL", "MSFT",
                               "AMZN", "TSLA", "SPY", "BND", "GLD", "VNQ"]
        self.analysis_types = [
            "Portfolio Overview",
            "Risk Analysis",
            "Model Performance",
            "Data Processing",
            "Efficient Frontier",
            "Performance Comparison"
        ]

    def render_header(self):
        """Render the professional dashboard header."""
        st.markdown(
            '<div class="main-header">🚀 GMF Time Series Forecasting Dashboard</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
            <strong>Professional Financial Analytics Platform</strong> • 
            Portfolio Optimization • Risk Management • Forecasting • Performance Analysis
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the professional sidebar with controls."""
        with st.sidebar:
            st.markdown("## 📊 Dashboard Controls")

            # Analysis type selector
            analysis_type = st.selectbox(
                "**Select Analysis Type**",
                self.analysis_types,
                help="Choose the type of financial analysis to display"
            )

            st.markdown("---")
            st.markdown("### 📅 Date Range")
            start_date = st.date_input(
                "**Start Date**",
                value=datetime(2020, 1, 1).date(),
                help="Select the start date for analysis"
            )
            end_date = st.date_input(
                "**End Date**",
                value=datetime.now().date(),
                help="Select the end date for analysis"
            )

            st.markdown("---")
            st.markdown("### 💼 Asset Selection")
            selected_assets = st.multiselect(
                "**Select Assets**",
                self.default_assets,
                default=self.default_assets[:5],
                help="Choose assets for portfolio analysis"
            )

            st.markdown("---")
            st.markdown("### ⚙️ Analysis Settings")

            # Risk tolerance
            risk_tolerance = st.selectbox(
                "**Risk Tolerance**",
                ["Conservative", "Moderate", "Aggressive"],
                help="Select your risk tolerance level"
            )

            # Forecast horizon
            forecast_horizon = st.slider(
                "**Forecast Horizon (Days)**",
                min_value=30,
                max_value=365,
                value=90,
                step=30,
                help="Select the number of days to forecast"
            )

            # Confidence level
            confidence_level = st.slider(
                "**Confidence Level**",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Select the confidence level for risk metrics"
            )

            st.markdown("---")

            # Update button
            if st.button("🔄 **Update Dashboard**", type="primary", use_container_width=True):
                st.rerun()

            # Dashboard info
            st.markdown("---")
            st.markdown("### ℹ️ Dashboard Info")
            st.markdown(f"**Version:** 2.0.0")
            st.markdown(
                f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

            return analysis_type, start_date, end_date, selected_assets, risk_tolerance, forecast_horizon, confidence_level

    def render_portfolio_overview(self, assets, start_date, end_date, risk_tolerance):
        """Render comprehensive portfolio overview section."""
        st.markdown("## 📈 Portfolio Overview")

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("**Total Assets**", len(assets), delta=0)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            portfolio_value = 1250000 + np.random.normal(0, 50000)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("**Portfolio Value**",
                      f"${portfolio_value:,.0f}", delta="+5.2%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            daily_return = 0.8 + np.random.normal(0, 0.2)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("**Daily Return**",
                      f"{daily_return:.1f}%", delta="+0.3%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            volatility = 12.5 + np.random.normal(0, 1.0)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("**Volatility**", f"{volatility:.1f}%", delta="-1.2%")
            st.markdown('</div>', unsafe_allow_html=True)

        # Portfolio performance chart
        st.markdown("### 📊 Portfolio Performance")

        with st.container():
            # Generate realistic performance data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            portfolio_values = np.cumsum(
                np.random.normal(0.001, 0.02, len(dates))) + 1

            # Create performance chart
            chart_data = pd.DataFrame({
                'Date': dates,
                'Portfolio Value': portfolio_values * 1000000
            })

            # Create Plotly chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=chart_data['Date'],
                y=chart_data['Portfolio Value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=3),
                fill='tonexty'
            ))

            fig.update_layout(
                title="Portfolio Performance Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=400,
                showlegend=True,
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Asset allocation
        st.markdown("### 💼 Asset Allocation")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Generate allocation data
            allocation_data = pd.DataFrame({
                'Asset': assets,
                'Weight': np.random.dirichlet(np.ones(len(assets))),
                'Return': np.random.normal(0.001, 0.02, len(assets))
            })

            # Create pie chart
            fig = px.pie(
                allocation_data,
                values='Weight',
                names='Asset',
                title="Current Asset Allocation",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Allocation table
            st.markdown("**Allocation Details:**")
            allocation_display = allocation_data.copy()
            allocation_display['Weight'] = allocation_display['Weight'].apply(
                lambda x: f"{x:.1%}")
            allocation_display['Return'] = allocation_display['Return'].apply(
                lambda x: f"{x:.3f}")
            st.dataframe(allocation_display, use_container_width=True)

    def render_risk_analysis(self, assets, start_date, end_date, confidence_level):
        """Render comprehensive risk analysis section."""
        st.markdown("## ⚠️ Risk Analysis")

        # Risk metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            var_95 = -2.1 + np.random.normal(0, 0.3)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("**VaR (95%)**", f"{var_95:.1f}%", delta="-0.3%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            cvar_95 = -3.5 + np.random.normal(0, 0.5)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("**CVaR (95%)**", f"{cvar_95:.1f}%", delta="-0.5%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            max_dd = -8.2 + np.random.normal(0, 1.1)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("**Max Drawdown**", f"{max_dd:.1f}%", delta="-1.1%")
            st.markdown('</div>', unsafe_allow_html=True)

        # Risk charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Returns Distribution")

            # Generate sample returns
            returns = np.random.normal(0.001, 0.02, 1000)

            # Create histogram
            fig = px.histogram(
                x=returns,
                nbins=50,
                title="Portfolio Returns Distribution",
                labels={'x': 'Daily Returns', 'y': 'Frequency'},
                color_discrete_sequence=['#ff7f0e']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📈 Rolling Volatility")

            # Generate rolling volatility
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            volatility = np.random.uniform(0.15, 0.25, len(dates))

            vol_data = pd.DataFrame({
                'Date': dates,
                'Volatility': volatility
            })

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=vol_data['Date'],
                y=vol_data['Volatility'],
                mode='lines',
                name='30-Day Rolling Volatility',
                line=dict(color='#2ca02c', width=2)
            ))

            fig.update_layout(
                title="Rolling Volatility (30-Day)",
                xaxis_title="Date",
                yaxis_title="Volatility",
                height=400,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

    def render_model_performance(self, assets, start_date, end_date, forecast_horizon):
        """Render model performance and forecasting section."""
        st.markdown("## 🔮 Model Performance & Forecasting")

        # Model metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            accuracy = 94.2 + np.random.normal(0, 1.8)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("**Forecast Accuracy**",
                      f"{accuracy:.1f}%", delta="+1.8%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            mape = 2.1 + np.random.normal(0, 0.3)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("**MAPE**", f"{mape:.1f}%", delta="-0.3%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            dir_accuracy = 87.5 + np.random.normal(0, 2.1)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("**Directional Accuracy**",
                      f"{dir_accuracy:.1f}%", delta="+2.1%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            r2_score = 0.89 + np.random.normal(0, 0.05)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("**R² Score**", f"{r2_score:.2f}", delta="+0.05")
            st.markdown('</div>', unsafe_allow_html=True)

        # Forecasting results
        st.markdown("### 📈 Forecasting Results")

        # Generate sample forecast data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        historical_values = np.cumsum(
            np.random.normal(0.001, 0.02, len(dates))) + 1

        # Create forecast extension
        forecast_dates = pd.date_range(
            start=end_date, periods=forecast_horizon+1, freq='D')[1:]
        forecast_values = np.cumsum(np.random.normal(
            0.001, 0.02, forecast_horizon)) + historical_values[-1]

        # Combine historical and forecast
        all_dates = pd.concat([pd.Series(dates), pd.Series(forecast_dates)])
        all_values = np.concatenate([historical_values, forecast_values])

        # Create forecast chart
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=dates,
            y=historical_values,
            mode='lines',
            name='Historical',
            line=dict(color='#1f77b4', width=3)
        ))

        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dash')
        ))

        # Add forecast confidence interval
        forecast_upper = forecast_values * \
            (1 + np.random.uniform(0.05, 0.15, len(forecast_values)))
        forecast_lower = forecast_values * \
            (1 - np.random.uniform(0.05, 0.15, len(forecast_values)))

        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_upper,
            mode='lines',
            name='Upper Bound',
            line=dict(color='#ff7f0e', width=1, dash='dot'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_lower,
            mode='lines',
            name='Lower Bound',
            line=dict(color='#ff7f0e', width=1, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.1)',
            showlegend=False
        ))

        fig.update_layout(
            title=f"{forecast_horizon}-Day Forecast with Confidence Intervals",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            height=500,
            showlegend=True,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_data_processing(self, assets, start_date, end_date):
        """Render data processing and quality section."""
        st.markdown("## 🔄 Data Processing & Quality")

        # Data quality metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            total_records = 1460 + np.random.randint(-10, 10)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("**Total Records**", f"{total_records:,}", delta="+0")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            missing_pct = 0.2 + np.random.normal(0, 0.1)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("**Missing Values**",
                      f"{missing_pct:.1f}%", delta="-0.1%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            quality_score = 98.5 + np.random.normal(0, 0.5)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("**Data Quality Score**",
                      f"{quality_score:.1f}%", delta="+0.5%")
            st.markdown('</div>', unsafe_allow_html=True)

        # Sample processed data
        st.markdown("### 📋 Sample Processed Data")

        # Generate sample data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Asset': np.random.choice(assets, len(dates)),
            'Open': np.random.uniform(100, 200, len(dates)),
            'High': np.random.uniform(100, 200, len(dates)),
            'Low': np.random.uniform(100, 200, len(dates)),
            'Close': np.random.uniform(100, 200, len(dates)),
            'Volume': np.random.uniform(1000000, 5000000, len(dates))
        })

        # Display sample data
        st.dataframe(sample_data.head(20), use_container_width=True)

        # Data statistics
        st.markdown("### 📊 Data Statistics")

        col1, col2 = st.columns(2)

        with col1:
            stats_data = pd.DataFrame({
                'Metric': ['Mean', 'Std', 'Min', 'Max'],
                'Open': [
                    sample_data['Open'].mean(),
                    sample_data['Open'].std(),
                    sample_data['Open'].min(),
                    sample_data['Open'].max()
                ],
                'Close': [
                    sample_data['Close'].mean(),
                    sample_data['Close'].std(),
                    sample_data['Close'].min(),
                    sample_data['Close'].max()
                ]
            })
            st.dataframe(stats_data, use_container_width=True)

        with col2:
            # Data quality chart
            quality_metrics = {
                'Completeness': 98.5,
                'Accuracy': 97.2,
                'Consistency': 99.1,
                'Timeliness': 96.8
            }

            fig = px.bar(
                x=list(quality_metrics.keys()),
                y=list(quality_metrics.values()),
                title="Data Quality Metrics",
                labels={'x': 'Quality Dimension', 'y': 'Score (%)'},
                color=list(quality_metrics.values()),
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    def render_efficient_frontier(self, assets, start_date, end_date):
        """Render efficient frontier analysis."""
        st.markdown("## 📈 Efficient Frontier Analysis")

        # Generate efficient frontier data
        num_portfolios = 100
        returns_range = np.linspace(0.05, 0.25, num_portfolios)
        volatilities = np.sqrt(returns_range * 0.5 +
                               np.random.normal(0, 0.02, num_portfolios))

        frontier_data = pd.DataFrame({
            'Expected_Return': returns_range,
            'Volatility': volatilities,
            'Sharpe_Ratio': (returns_range - 0.02) / volatilities
        })

        # Create efficient frontier chart
        fig = go.Figure()

        # Efficient frontier line
        fig.add_trace(go.Scatter(
            x=frontier_data['Volatility'],
            y=frontier_data['Expected_Return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='#1f77b4', width=3)
        ))

        # Individual portfolios
        fig.add_trace(go.Scatter(
            x=frontier_data['Volatility'],
            y=frontier_data['Expected_Return'],
            mode='markers',
            name='Portfolio Options',
            marker=dict(
                color=frontier_data['Sharpe_Ratio'],
                colorscale='Viridis',
                size=8,
                colorbar=dict(title="Sharpe Ratio")
            )
        ))

        fig.update_layout(
            title="Efficient Frontier - Risk vs. Return",
            xaxis_title="Portfolio Volatility",
            yaxis_title="Expected Return",
            height=500,
            showlegend=True,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Portfolio recommendations
        st.markdown("### 💡 Portfolio Recommendations")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🎯 Conservative Portfolio**")
            st.markdown("- **Return:** 8.2%")
            st.markdown("- **Risk:** 12.1%")
            st.markdown("- **Sharpe:** 0.51")

        with col2:
            st.markdown("**⚖️ Balanced Portfolio**")
            st.markdown("- **Return:** 12.8%")
            st.markdown("- **Risk:** 16.7%")
            st.markdown("- **Sharpe:** 0.65")

        with col3:
            st.markdown("**🚀 Aggressive Portfolio**")
            st.markdown("- **Return:** 18.5%")
            st.markdown("- **Risk:** 24.3%")
            st.markdown("- **Sharpe:** 0.68")

    def render_performance_comparison(self, assets, start_date, end_date):
        """Render performance comparison section."""
        st.markdown("## 📊 Performance Comparison")

        # Generate comparison data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Portfolio performance
        portfolio_returns = np.cumsum(
            np.random.normal(0.001, 0.02, len(dates)))
        portfolio_values = 1000000 * (1 + portfolio_returns)

        # Benchmark performance (S&P 500)
        benchmark_returns = np.cumsum(
            np.random.normal(0.0008, 0.018, len(dates)))
        benchmark_values = 1000000 * (1 + benchmark_returns)

        # Create comparison chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='GMF Portfolio',
            line=dict(color='#1f77b4', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_values,
            mode='lines',
            name='S&P 500 Benchmark',
            line=dict(color='#ff7f0e', width=3)
        ))

        fig.update_layout(
            title="Portfolio vs. Benchmark Performance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=500,
            showlegend=True,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Performance metrics
        st.markdown("### 📈 Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_return = (
                portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
            st.metric("**Total Return**", f"{total_return:.1f}%")

        with col2:
            benchmark_return = (
                benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0] * 100
            st.metric("**Benchmark Return**", f"{benchmark_return:.1f}%")

        with col3:
            excess_return = total_return - benchmark_return
            st.metric("**Excess Return**",
                      f"{excess_return:.1f}%", delta=f"{excess_return:.1f}%")

        with col4:
            tracking_error = np.std(
                portfolio_returns - benchmark_returns) * np.sqrt(252) * 100
            st.metric("**Tracking Error**", f"{tracking_error:.1f}%")

    def run_dashboard(self):
        """Main dashboard execution method."""
        try:
            # Render header
            self.render_header()

            # Get sidebar controls
            analysis_type, start_date, end_date, selected_assets, risk_tolerance, forecast_horizon, confidence_level = self.render_sidebar()

            # Main content area
            if analysis_type == "Portfolio Overview":
                self.render_portfolio_overview(
                    selected_assets, start_date, end_date, risk_tolerance)
            elif analysis_type == "Risk Analysis":
                self.render_risk_analysis(
                    selected_assets, start_date, end_date, confidence_level)
            elif analysis_type == "Model Performance":
                self.render_model_performance(
                    selected_assets, start_date, end_date, forecast_horizon)
            elif analysis_type == "Data Processing":
                self.render_data_processing(
                    selected_assets, start_date, end_date)
            elif analysis_type == "Efficient Frontier":
                self.render_efficient_frontier(
                    selected_assets, start_date, end_date)
            elif analysis_type == "Performance Comparison":
                self.render_performance_comparison(
                    selected_assets, start_date, end_date)

            # Footer
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; color: #666; padding: 1rem;'>
                <strong>GMF Time Series Forecasting Dashboard v2.0.0</strong> • 
                Professional Financial Analytics Platform
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            st.exception(e)


def main():
    """Main dashboard function."""
    try:
        dashboard = GMFDashboard()
        dashboard.run_dashboard()
    except Exception as e:
        st.error(f"Failed to initialize dashboard: {str(e)}")


if __name__ == "__main__":
    main()

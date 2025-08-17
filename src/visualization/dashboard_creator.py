"""
Interactive dashboard creator for financial analysis.

This module provides comprehensive dashboard creation capabilities
for financial data visualization and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class DashboardCreator:
    """
    Interactive dashboard creator for financial analysis.

    Provides comprehensive dashboard creation capabilities including
    Streamlit-based dashboards and interactive visualizations.

    Attributes:
        dashboard_config (Dict): Dashboard configuration settings
        available_widgets (List[str]): List of available dashboard widgets
        theme_config (Dict): Theme and styling configuration
    """

    def __init__(self, theme: str = 'light'):
        """
        Initialize the Dashboard Creator.

        Args:
            theme: Dashboard theme ('light', 'dark', 'custom')
        """
        self.theme = theme
        self.dashboard_config = self._initialize_dashboard_config()
        self.available_widgets = self._get_available_widgets()
        self.theme_config = self._initialize_theme_config()

        logger.info(
            "Initialized Dashboard Creator with theme: {}".format(theme))

    def _initialize_dashboard_config(self) -> Dict[str, Any]:
        """
        Initialize dashboard configuration.

        Returns:
            Dictionary with dashboard configuration
        """
        config = {
            'page_title': 'GMF Time Series Forecasting Dashboard',
            'page_icon': '📊',
            'layout': 'wide',
            'initial_sidebar_state': 'expanded',
            'max_width': 1200,
            'show_toolbar': True,
            'show_navbar': True,
            'refresh_interval': 300,  # 5 minutes
            'enable_export': True,
            'enable_print': True
        }

        return config

    def _get_available_widgets(self) -> List[str]:
        """
        Get list of available dashboard widgets.

        Returns:
            List of available widget types
        """
        return [
            'metrics_card',
            'line_chart',
            'bar_chart',
            'scatter_plot',
            'heatmap',
            'table',
            'gauge',
            'progress_bar',
            'pie_chart',
            'histogram',
            'box_plot',
            'area_chart'
        ]

    def _initialize_theme_config(self) -> Dict[str, Any]:
        """
        Initialize theme configuration.

        Returns:
            Dictionary with theme settings
        """
        if self.theme == 'light':
            theme_config = {
                'primary_color': '#1f77b4',
                'secondary_color': '#ff7f0e',
                'background_color': '#ffffff',
                'text_color': '#000000',
                'accent_color': '#2ca02c',
                'border_color': '#e0e0e0',
                'chart_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            }
        elif self.theme == 'dark':
            theme_config = {
                'primary_color': '#4CAF50',
                'secondary_color': '#FF9800',
                'background_color': '#1a1a1a',
                'text_color': '#ffffff',
                'accent_color': '#2196F3',
                'border_color': '#404040',
                'chart_colors': ['#4CAF50', '#FF9800', '#2196F3', '#F44336', '#9C27B0']
            }
        else:
            # Custom theme
            theme_config = {
                'primary_color': '#6366f1',
                'secondary_color': '#f59e0b',
                'background_color': '#f8fafc',
                'text_color': '#1e293b',
                'accent_color': '#10b981',
                'border_color': '#e2e8f0',
                'chart_colors': ['#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6']
            }

        return theme_config

    def create_streamlit_dashboard(self, data_sources: Dict[str, pd.DataFrame],
                                   dashboard_layout: str = 'standard') -> str:
        """
        Create a Streamlit-based dashboard.

        Args:
            data_sources: Dictionary mapping names to DataFrames
            dashboard_layout: Layout type ('standard', 'compact', 'fullscreen')

        Returns:
            Python code string for the Streamlit dashboard
        """
        try:
            dashboard_code = self._generate_streamlit_code(
                data_sources, dashboard_layout)
            return dashboard_code

        except Exception as e:
            logger.error(
                "Streamlit dashboard creation failed: {}".format(str(e)))
            return "# Error creating dashboard: {}".format(str(e))

    def _generate_streamlit_code(self, data_sources: Dict[str, pd.DataFrame],
                                 layout: str) -> str:
        """
        Generate Streamlit dashboard code.

        Args:
            data_sources: Dictionary mapping names to DataFrames
            layout: Dashboard layout type

        Returns:
            Python code string for the Streamlit dashboard
        """
        # Header
        code_lines = [
            "import streamlit as st",
            "import pandas as pd",
            "import numpy as np",
            "import plotly.express as px",
            "import plotly.graph_objects as go",
            "from plotly.subplots import make_subplots",
            "import datetime",
            "",
            "# Page configuration",
            "st.set_page_config(",
            "    page_title='{}',".format(self.dashboard_config['page_title']),
            "    page_icon='{}',".format(self.dashboard_config['page_icon']),
            "    layout='{}',".format(self.dashboard_config['layout']),
            "    initial_sidebar_state='{}'".format(
                self.dashboard_config['initial_sidebar_state']),
            ")",
            "",
            "# Custom CSS for styling",
            self._generate_custom_css(),
            "",
            "# Main dashboard function",
            "def main():",
            "    # Header",
            "    st.title('🚀 GMF Time Series Forecasting Dashboard')",
            "    st.markdown('---')",
            "",
            "    # Sidebar controls",
            "    with st.sidebar:",
            "        st.header('📊 Dashboard Controls')",
            "        ",
            "        # Date range selector",
            "        st.subheader('📅 Date Range')",
            "        start_date = st.date_input('Start Date', value=datetime.date(2020, 1, 1))",
            "        end_date = st.date_input('End Date', value=datetime.date.today())",
            "        ",
            "        # Asset selector",
            "        st.subheader('💼 Asset Selection')",
            "        available_assets = ['All Assets'] + list(data_sources.keys())",
            "        selected_assets = st.multiselect('Select Assets', available_assets, default=['All Assets'])",
            "        ",
            "        # Analysis type selector",
            "        st.subheader('🔍 Analysis Type')",
            "        analysis_type = st.selectbox(",
            "            'Select Analysis',",
            "            ['Portfolio Overview', 'Risk Analysis', 'Performance Metrics', 'Forecasting']",
            "        )",
            "        ",
            "        # Update button",
            "        if st.button('🔄 Update Dashboard'):",
            "            st.rerun()",
            "",
            "    # Main content area",
            "    if analysis_type == 'Portfolio Overview':",
            "        display_portfolio_overview(data_sources, start_date, end_date, selected_assets)",
            "    elif analysis_type == 'Risk Analysis':",
            "        display_risk_analysis(data_sources, start_date, end_date, selected_assets)",
            "    elif analysis_type == 'Performance Metrics':",
            "        display_performance_metrics(data_sources, start_date, end_date, selected_assets)",
            "    elif analysis_type == 'Forecasting':",
            "        display_forecasting_analysis(data_sources, start_date, end_date, selected_assets)",
            "",
            "# Portfolio overview function",
            "def display_portfolio_overview(data_sources, start_date, end_date, selected_assets):",
            "    st.header('📈 Portfolio Overview')",
            "    ",
            "    # Key metrics row",
            "    col1, col2, col3, col4 = st.columns(4)",
            "    ",
            "    with col1:",
            "        st.metric('Total Assets', len(data_sources), delta=0)",
            "    with col2:",
            "        st.metric('Portfolio Value', '$1,000,000', delta='+5.2%')",
            "    with col3:",
            "        st.metric('Daily Return', '0.8%', delta='+0.3%')",
            "    with col4:",
            "        st.metric('Volatility', '12.5%', delta='-1.2%')",
            "    ",
            "    # Portfolio performance chart",
            "    st.subheader('📊 Portfolio Performance')",
            "    ",
            "    # Create sample performance data",
            "    dates = pd.date_range(start=start_date, end=end_date, freq='D')",
            "    portfolio_values = np.random.cumsum(np.random.normal(0.001, 0.02, len(dates))) + 1",
            "    ",
            "    fig = go.Figure()",
            "    fig.add_trace(go.Scatter(",
            "        x=dates, y=portfolio_values,",
            "        mode='lines',",
            "        name='Portfolio Value',",
            "        line=dict(color='#1f77b4', width=2)",
            "    ))",
            "    ",
            "    fig.update_layout(",
            "        title='Portfolio Value Over Time',",
            "        xaxis_title='Date',",
            "        yaxis_title='Portfolio Value ($)',",
            "        height=400",
            "        showlegend=True",
            "    )",
            "    ",
            "    st.plotly_chart(fig, use_container_width=True)",
            "",
            "# Risk analysis function",
            "def display_risk_analysis(data_sources, start_date, end_date, selected_assets):",
            "    st.header('⚠️ Risk Analysis')",
            "    ",
            "    # Risk metrics",
            "    col1, col2, col3 = st.columns(3)",
            "    ",
            "    with col1:",
            "        st.metric('VaR (95%)', '-2.1%', delta='-0.3%')",
            "    with col2:",
            "        st.metric('CVaR (95%)', '-3.5%', delta='-0.5%')",
            "    with col3:",
            "        st.metric('Max Drawdown', '-8.2%', delta='-1.1%')",
            "    ",
            "    # Risk charts",
            "    col1, col2 = st.columns(2)",
            "    ",
            "    with col1:",
            "        st.subheader('📊 Returns Distribution')",
            "        # Create sample returns data",
            "        returns = np.random.normal(0.001, 0.02, 1000)",
            "        fig = px.histogram(returns, nbins=50, title='Daily Returns Distribution')",
            "        st.plotly_chart(fig, use_container_width=True)",
            "    ",
            "    with col2:",
            "        st.subheader('📈 Rolling Volatility')",
            "        dates = pd.date_range(start=start_date, end=end_date, freq='D')",
            "        volatility = np.random.uniform(0.15, 0.25, len(dates))",
            "        ",
            "        fig = go.Figure()",
            "        fig.add_trace(go.Scatter(",
            "            x=dates, y=volatility,",
            "            mode='lines',",
            "            name='30-Day Rolling Volatility',",
            "            line=dict(color='#ff7f0e', width=2)",
            "        ))",
            "        ",
            "        fig.update_layout(",
            "            title='Rolling Volatility',",
            "            xaxis_title='Date',",
            "            yaxis_title='Volatility',",
            "            height=300",
            "        )",
            "        ",
            "        st.plotly_chart(fig, use_container_width=True)",
            "",
            "# Performance metrics function",
            "def display_performance_metrics(data_sources, start_date, end_date, selected_assets):",
            "    st.header('📊 Performance Metrics')",
            "    ",
            "    # Performance metrics",
            "    col1, col2, col3, col4 = st.columns(4)",
            "    ",
            "    with col1:",
            "        st.metric('Total Return', '15.8%', delta='+2.1%')",
            "    with col2:",
            "        st.metric('Annualized Return', '12.3%', delta='+1.5%')",
            "    with col3:",
            "        st.metric('Sharpe Ratio', '1.45', delta='+0.12')",
            "    with col4:",
            "        st.metric('Sortino Ratio', '2.01', delta='+0.18')",
            "    ",
            "    # Performance comparison chart",
            "    st.subheader('📈 Performance Comparison')",
            "    ",
            "    dates = pd.date_range(start=start_date, end=end_date, freq='D')",
            "    portfolio_cumulative = np.random.cumsum(np.random.normal(0.001, 0.02, len(dates))) + 1",
            "    benchmark_cumulative = np.random.cumsum(np.random.normal(0.0008, 0.018, len(dates))) + 1",
            "    ",
            "    fig = go.Figure()",
            "    fig.add_trace(go.Scatter(",
            "        x=dates, y=portfolio_cumulative,",
            "        mode='lines',",
            "        name='Portfolio',",
            "        line=dict(color='#1f77b4', width=2)",
            "    ))",
            "    fig.add_trace(go.Scatter(",
            "        x=dates, y=benchmark_cumulative,",
            "        mode='lines',",
            "        name='Benchmark',",
            "        line=dict(color='#ff7f0e', width=2)",
            "    ))",
            "    ",
            "    fig.update_layout(",
            "        title='Portfolio vs Benchmark Performance',",
            "        xaxis_title='Date',",
            "        yaxis_title='Cumulative Return',",
            "        height=400,",
            "        showlegend=True",
            "    )",
            "    ",
            "    st.plotly_chart(fig, use_container_width=True)",
            "",
            "# Forecasting analysis function",
            "def display_forecasting_analysis(data_sources, start_date, end_date, selected_assets):",
            "    st.header('🔮 Forecasting Analysis')",
            "    ",
            "    # Forecasting controls",
            "    col1, col2 = st.columns(2)",
            "    ",
            "    with col1:",
            "        forecast_horizon = st.selectbox('Forecast Horizon', [30, 60, 90, 180, 365])",
            "    with col2:",
            "        confidence_level = st.selectbox('Confidence Level', [0.80, 0.90, 0.95, 0.99])",
            "    ",
            "    # Generate forecast",
            "    if st.button('🚀 Generate Forecast'):",
            "        st.success('Forecast generated for {} days with {}% confidence!'.format(forecast_horizon, confidence_level*100))",
            "        ",
            "        # Create sample forecast data",
            "        forecast_dates = pd.date_range(start=end_date, periods=forecast_horizon+1, freq='D')[1:]",
            "        forecast_values = np.random.cumsum(np.random.normal(0.001, 0.02, forecast_horizon)) + 1",
            "        ",
            "        # Historical data for context",
            "        historical_dates = pd.date_range(start=start_date, end=end_date, freq='D')",
            "        historical_values = np.random.cumsum(np.random.normal(0.001, 0.02, len(historical_dates))) + 1",
            "        ",
            "        fig = go.Figure()",
            "        ",
            "        # Historical data",
            "        fig.add_trace(go.Scatter(",
            "            x=historical_dates, y=historical_values,",
            "            mode='lines',",
            "            name='Historical',",
            "            line=dict(color='#1f77b4', width=2)",
            "        ))",
            "        ",
            "        # Forecast data",
            "        fig.add_trace(go.Scatter(",
            "            x=forecast_dates, y=forecast_values,",
            "            mode='lines',",
            "            name='Forecast',",
            "            line=dict(color='#ff7f0e', width=2, dash='dash')",
            "        ))",
            "        ",
            "        fig.update_layout(",
            "            title='{}-Day Forecast'.format(forecast_horizon),",
            "            xaxis_title='Date',",
            "            yaxis_title='Portfolio Value',",
            "            height=400,",
            "            showlegend=True",
            "        )",
            "        ",
            "        st.plotly_chart(fig, use_container_width=True)",
            "    ",
            "    # Model performance metrics",
            "    st.subheader('📊 Model Performance')",
            "    ",
            "    col1, col2, col3 = st.columns(3)",
            "    ",
            "    with col1:",
            "        st.metric('Forecast Accuracy', '94.2%', delta='+1.8%')",
            "    with col2:",
            "        st.metric('MAPE', '2.1%', delta='-0.3%')",
            "    with col3:",
            "        st.metric('Directional Accuracy', '87.5%', delta='+2.1%')",
            "",
            "# Data sources (placeholder - replace with actual data)",
            "data_sources = {",
        ]

        # Add data sources
        for name, df in data_sources.items():
            code_lines.append(
                "    '{}': pd.DataFrame(),  # Replace with actual {} data".format(name, name))

        code_lines.extend([
            "}",
            "",
            "# Run the dashboard",
            "if __name__ == '__main__':",
            "    main()",
            ""
        ])

        return "\n".join(code_lines)

    def _generate_custom_css(self) -> str:
        """
        Generate custom CSS for the dashboard.

        Returns:
            CSS code string
        """
        css_code = [
            "st.markdown(",
            "    '''",
            "    <style>",
            "    .main-header {",
            "        font-size: 2.5rem;",
            "        font-weight: bold;",
            "        color: #1f77b4;",
            "        text-align: center;",
            "        margin-bottom: 2rem;",
            "    }",
            "    .metric-card {",
            "        background-color: #f8f9fa;",
            "        padding: 1rem;",
            "        border-radius: 0.5rem;",
            "        border-left: 4px solid #1f77b4;",
            "    }",
            "    .sidebar .sidebar-content {",
            "        background-color: #f8f9fa;",
            "    }",
            "    </style>",
            "    ''',",
            "    unsafe_allow_html=True",
            ")"
        ]

        return "\n".join(css_code)

    def create_plotly_dashboard(self, data_sources: Dict[str, pd.DataFrame],
                                dashboard_type: str = 'portfolio') -> str:
        """
        Create a Plotly-based dashboard.

        Args:
            data_sources: Dictionary mapping names to DataFrames
            dashboard_type: Type of dashboard ('portfolio', 'risk', 'performance')

        Returns:
            Python code string for the Plotly dashboard
        """
        try:
            dashboard_code = self._generate_plotly_code(
                data_sources, dashboard_type)
            return dashboard_code

        except Exception as e:
            logger.error("Plotly dashboard creation failed: {}".format(str(e)))
            return "# Error creating Plotly dashboard: {}".format(str(e))

    def _generate_plotly_code(self, data_sources: Dict[str, pd.DataFrame],
                              dashboard_type: str) -> str:
        """
        Generate Plotly dashboard code.

        Args:
            data_sources: Dictionary mapping names to DataFrames
            dashboard_type: Type of dashboard

        Returns:
            Python code string for the Plotly dashboard
        """
        code_lines = [
            "import plotly.graph_objects as go",
            "import plotly.express as px",
            "from plotly.subplots import make_subplots",
            "import pandas as pd",
            "import numpy as np",
            "from datetime import datetime, timedelta",
            "",
            "def create_{}_dashboard(data_sources):".format(dashboard_type),
            "    \"\"\"",
            "    Create a {} dashboard using Plotly.".format(dashboard_type),
            "    \"\"\"",
            "    ",
            "    # Create subplots",
            "    if dashboard_type == 'portfolio':",
            "        fig = make_subplots(",
            "            rows=2, cols=2,",
            "            subplot_titles=('Portfolio Performance', 'Asset Allocation', 'Risk Metrics', 'Correlation Matrix'),",
            "            specs=[[{\"type\": \"scatter\"}, {\"type\": \"pie\"}],",
            "                   [{\"type\": \"bar\"}, {\"type\": \"heatmap\"}]]",
            "        )",
            "    elif dashboard_type == 'risk':",
            "        fig = make_subplots(",
            "            rows=2, cols=2,",
            "            subplot_titles=('VaR Analysis', 'Volatility Trends', 'Drawdown Analysis', 'Tail Risk'),",
            "            specs=[[{\"type\": \"bar\"}, {\"type\": \"scatter\"}],",
            "                   [{\"type\": \"scatter\"}, {\"type\": \"histogram\"}]]",
            "        )",
            "    else:  # performance",
            "        fig = make_subplots(",
            "            rows=2, cols=2,",
            "            subplot_titles=('Returns Distribution', 'Performance Comparison', 'Rolling Metrics', 'Risk-Adjusted Returns'),",
            "            specs=[[{\"type\": \"histogram\"}, {\"type\": \"scatter\"}],",
            "                   [{\"type\": \"scatter\"}, {\"type\": \"scatter\"}]]",
            "        )",
            "    ",
            "    # Add traces based on dashboard type",
            "    if dashboard_type == 'portfolio':",
            "        # Portfolio performance line chart",
            "        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')",
            "        portfolio_values = np.random.cumsum(np.random.normal(0.001, 0.02, len(dates))) + 1",
            "        ",
            "        fig.add_trace(",
            "            go.Scatter(x=dates, y=portfolio_values, name='Portfolio', line=dict(color='#1f77b4')),",
            "            row=1, col=1",
            "        )",
            "        ",
            "        # Asset allocation pie chart",
            "        assets = list(data_sources.keys())",
            "        allocations = np.random.dirichlet(np.ones(len(assets)))",
            "        ",
            "        fig.add_trace(",
            "            go.Pie(labels=assets, values=allocations, name='Allocation'),",
            "            row=1, col=2",
            "        )",
            "        ",
            "        # Risk metrics bar chart",
            "        risk_metrics = ['VaR', 'CVaR', 'Max DD', 'Volatility']",
            "        risk_values = [0.02, 0.035, 0.082, 0.125]",
            "        ",
            "        fig.add_trace(",
            "            go.Bar(x=risk_metrics, y=risk_values, name='Risk Metrics', marker_color='#ff7f0e'),",
            "            row=2, col=1",
            "        )",
            "        ",
            "        # Correlation matrix heatmap",
            "        n_assets = len(assets)",
            "        correlation_matrix = np.random.uniform(-0.8, 0.8, (n_assets, n_assets))",
            "        np.fill_diagonal(correlation_matrix, 1.0)",
            "        ",
            "        fig.add_trace(",
            "            go.Heatmap(z=correlation_matrix, x=assets, y=assets, colorscale='RdBu'),",
            "            row=2, col=2",
            "        )",
            "    ",
            "    # Update layout",
            "    fig.update_layout(",
            "        title='{} Dashboard'.format(dashboard_type.title()),",
            "        height=800,",
            "        showlegend=True,",
            "        template='plotly_white'",
            "    )",
            "    ",
            "    return fig",
            "",
            "# Example usage",
            "if __name__ == '__main__':",
            "    # Create sample data sources",
            "    sample_data = {",
        ]

        # Add sample data sources
        for name in data_sources.keys():
            code_lines.append(
                "        '{}': pd.DataFrame(np.random.randn(100, 5)),".format(name))

        code_lines.extend([
            "    }",
            "",
            "    # Create and show dashboard",
            "    dashboard = create_{}_dashboard(sample_data)".format(
                dashboard_type),
            "    dashboard.show()",
            ""
        ])

        return "\n".join(code_lines)

    def create_html_dashboard(self, data_sources: Dict[str, pd.DataFrame],
                              dashboard_config: Dict[str, Any]) -> str:
        """
        Create an HTML-based dashboard.

        Args:
            data_sources: Dictionary mapping names to DataFrames
            dashboard_config: Dashboard configuration

        Returns:
            HTML code string for the dashboard
        """
        try:
            html_code = self._generate_html_code(
                data_sources, dashboard_config)
            return html_code

        except Exception as e:
            logger.error("HTML dashboard creation failed: {}".format(str(e)))
            return "<!-- Error creating HTML dashboard: {} -->".format(str(e))

    def _generate_html_code(self, data_sources: Dict[str, pd.DataFrame],
                            config: Dict[str, Any]) -> str:
        """
        Generate HTML dashboard code.

        Args:
            data_sources: Dictionary mapping names to DataFrames
            config: Dashboard configuration

        Returns:
            HTML code string
        """
        html_code = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "    <meta charset='UTF-8'>",
            "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "    <title>{}</title>".format(config.get('title',
                                           'GMF Dashboard')),
            "    <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
            "    <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>",
            "    <style>",
            "        body {",
            "            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;",
            "            margin: 0;",
            "            padding: 20px;",
            "            background-color: #f5f5f5;",
            "        }",
            "        .dashboard-container {",
            "            max-width: 1200px;",
            "            margin: 0 auto;",
            "            background-color: white;",
            "            border-radius: 10px;",
            "            box-shadow: 0 2px 10px rgba(0,0,0,0.1);",
            "            padding: 20px;",
            "        }",
            "        .header {",
            "            text-align: center;",
            "            margin-bottom: 30px;",
            "            color: #333;",
            "        }",
            "        .metrics-row {",
            "            display: grid;",
            "            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));",
            "            gap: 20px;",
            "            margin-bottom: 30px;",
            "        }",
            "        .metric-card {",
            "            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);",
            "            color: white;",
            "            padding: 20px;",
            "            border-radius: 10px;",
            "            text-align: center;",
            "        }",
            "        .metric-value {",
            "            font-size: 2em;",
            "            font-weight: bold;",
            "            margin-bottom: 5px;",
            "        }",
            "        .metric-label {",
            "            font-size: 0.9em;",
            "            opacity: 0.9;",
            "        }",
            "        .charts-row {",
            "            display: grid;",
            "            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));",
            "            gap: 20px;",
            "            margin-bottom: 30px;",
            "        }",
            "        .chart-container {",
            "            background-color: #f8f9fa;",
            "            border-radius: 10px;",
            "            padding: 20px;",
            "            min-height: 400px;",
            "        }",
            "    </style>",
            "</head>",
            "<body>",
            "    <div class='dashboard-container'>",
            "        <div class='header'>",
            "            <h1>🚀 GMF Time Series Forecasting Dashboard</h1>",
            "            <p>Comprehensive financial analysis and portfolio management</p>",
            "        </div>",
            "        ",
            "        <div class='metrics-row'>",
            "            <div class='metric-card'>",
            "                <div class='metric-value'>$1,250,000</div>",
            "                <div class='metric-label'>Portfolio Value</div>",
            "            </div>",
            "            <div class='metric-card'>",
            "                <div class='metric-value'>+15.8%</div>",
            "                <div class='metric-label'>Total Return</div>",
            "            </div>",
            "            <div class='metric-card'>",
            "                <div class='metric-value'>1.45</div>",
            "                <div class='metric-label'>Sharpe Ratio</div>",
            "            </div>",
            "            <div class='metric-card'>",
            "                <div class='metric-value'>12.5%</div>",
            "                <div class='metric-label'>Volatility</div>",
            "            </div>",
            "        </div>",
            "        ",
            "        <div class='charts-row'>",
            "            <div class='chart-container'>",
            "                <h3>Portfolio Performance</h3>",
            "                <div id='performance-chart'></div>",
            "            </div>",
            "            <div class='chart-container'>",
            "                <h3>Asset Allocation</h3>",
            "                <div id='allocation-chart'></div>",
            "            </div>",
            "        </div>",
            "        ",
            "        <div class='charts-row'>",
            "            <div class='chart-container'>",
            "                <h3>Risk Metrics</h3>",
            "                <div id='risk-chart'></div>",
            "            </div>",
            "            <div class='chart-container'>",
            "                <h3>Correlation Matrix</h3>",
            "                <div id='correlation-chart'></div>",
            "            </div>",
            "        </div>",
            "    </div>",
            "    ",
            "    <script>",
            "        // Sample data for charts",
            "        const dates = Array.from({length: 100}, (_, i) => new Date(2020, 0, i + 1));",
            "        const portfolioValues = Array.from({length: 100}, (_, i) => 1000000 + i * 2500 + Math.random() * 50000);",
            "        ",
            "        // Performance chart",
            "        const performanceTrace = {",
            "            x: dates,",
            "            y: portfolioValues,",
            "            type: 'scatter',",
            "            mode: 'lines',",
            "            name: 'Portfolio Value',",
            "            line: {color: '#667eea', width: 2}",
            "        };",
            "        ",
            "        Plotly.newPlot('performance-chart', [performanceTrace], {",
            "            title: 'Portfolio Performance Over Time',",
            "            xaxis: {title: 'Date'},",
            "            yaxis: {title: 'Portfolio Value ($)'},",
            "            height: 350",
            "        });",
            "        ",
            "        // Asset allocation chart",
            "        const allocationData = [",
            "            {values: [30, 25, 20, 15, 10], labels: ['Stocks', 'Bonds', 'Real Estate', 'Commodities', 'Cash'],",
            "             type: 'pie', marker: {colors: ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']}}",
            "        ];",
            "        ",
            "        Plotly.newPlot('allocation-chart', allocationData, {",
            "            title: 'Portfolio Asset Allocation',",
            "            height: 350",
            "        });",
            "        ",
            "        // Risk metrics chart",
            "        const riskData = [",
            "            {x: ['VaR (95%)', 'CVaR (95%)', 'Max Drawdown', 'Volatility'],",
            "             y: [2.1, 3.5, 8.2, 12.5],",
            "             type: 'bar',",
            "             marker: {color: '#f093fb'}}",
            "        ];",
            "        ",
            "        Plotly.newPlot('risk-chart', riskData, {",
            "            title: 'Key Risk Metrics (%)',",
            "            yaxis: {title: 'Percentage (%)'},",
            "            height: 350",
            "        });",
            "        ",
            "        // Correlation matrix chart",
            "        const correlationData = [",
            "            {z: [[1, 0.3, 0.1, -0.2], [0.3, 1, 0.4, 0.1], [0.1, 0.4, 1, 0.6], [-0.2, 0.1, 0.6, 1]],",
            "             x: ['Stocks', 'Bonds', 'Real Estate', 'Commodities'],",
            "             y: ['Stocks', 'Bonds', 'Real Estate', 'Commodities'],",
            "             type: 'heatmap',",
            "             colorscale: 'RdBu'}",
            "        ];",
            "        ",
            "        Plotly.newPlot('correlation-chart', correlationData, {",
            "            title: 'Asset Correlation Matrix',",
            "            height: 350",
            "        });",
            "    </script>",
            "</body>",
            "</html>"
        ]

        return "\n".join(html_code)

    def export_dashboard(self, dashboard_code: str, output_path: str,
                         format_type: str = 'python') -> None:
        """
        Export dashboard code to file.

        Args:
            dashboard_code: Dashboard code string
            output_path: Output file path
            format_type: Format type ('python', 'html', 'streamlit')
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(dashboard_code)

            logger.info("Dashboard exported to {}".format(output_path))

        except Exception as e:
            logger.error("Dashboard export failed: {}".format(str(e)))

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get summary of dashboard creation capabilities.

        Returns:
            Dictionary with dashboard summary
        """
        summary = {
            'available_formats': ['Streamlit', 'Plotly', 'HTML'],
            'available_widgets': self.available_widgets,
            'theme_config': self.theme_config,
            'dashboard_config': self.dashboard_config,
            'creation_methods': [
                'create_streamlit_dashboard',
                'create_plotly_dashboard',
                'create_html_dashboard'
            ]
        }

        return summary

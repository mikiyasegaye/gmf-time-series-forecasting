#!/usr/bin/env python3
"""
Automated Reporting System

This module provides automated report generation capabilities for the GMF Time Series
Forecasting system, including portfolio performance, risk analysis, and forecasting reports.

Author: GMF Investment Team
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
import yaml
import os
from datetime import datetime, timedelta
from pathlib import Path
import jinja2
import warnings

# Try to import optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")

try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    warnings.warn(
        "WeasyPrint not available. Install with: pip install weasyprint")

logger = logging.getLogger(__name__)


class ReportTemplate:
    """Base class for report templates."""

    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize report template.

        Args:
            template_path: Path to template file
        """
        self.template_path = template_path
        self.template_engine = None

        # Initialize template_content with default
        self.template_content = self._get_default_template()

        if template_path and Path(template_path).exists():
            self._load_template()

    def _load_template(self):
        """Load template from file."""
        try:
            if self.template_path and os.path.exists(self.template_path):
                with open(self.template_path, 'r') as f:
                    self.template_content = f.read()
                logger.info(f"Loaded template from {self.template_path}")
            else:
                self.template_content = self._get_default_template()
                logger.info("Using default template")
        except Exception as e:
            logger.error(f"Failed to load template: {str(e)}")
            self.template_content = self._get_default_template()

        # Ensure template_content is always set
        if not hasattr(self, 'template_content') or not self.template_content:
            self.template_content = self._get_default_template()

    def _get_default_template(self) -> str:
        """Get default HTML template."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #1f77b4; color: white; padding: 20px; text-align: center; }
                .section { margin: 20px 0; padding: 15px; border-left: 4px solid #1f77b4; }
                .metric { display: inline-block; margin: 10px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #1f77b4; }
                .metric-label { font-size: 14px; color: #666; }
                table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f8f9fa; }
                .chart { text-align: center; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ title }}</h1>
                <p>Generated on {{ generation_time }}</p>
            </div>
            
            {% for section in sections %}
            <div class="section">
                <h2>{{ section.title }}</h2>
                {{ section.content | safe }}
            </div>
            {% endfor %}
        </body>
        </html>
        """

    def render(self, data: Dict[str, Any]) -> str:
        """
        Render template with data.

        Args:
            data: Data to render in template

        Returns:
            Rendered HTML string
        """
        try:
            template = jinja2.Template(self.template_content)
            return template.render(**data)
        except Exception as e:
            logger.error(f"Failed to render template: {str(e)}")
            return f"<h1>Error rendering report: {str(e)}</h1>"


class AutomatedReporter:
    """
    Automated report generator for GMF forecasting system.

    Features:
    - Portfolio performance reports
    - Risk analysis reports
    - Forecasting accuracy reports
    - Custom report templates
    - Multiple output formats (HTML, PDF, JSON)
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize automated reporter.

        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize report templates
        self.templates = {}
        self._load_default_templates()

        # Report configuration
        self.report_config = {
            'include_charts': True,
            'include_metrics': True,
            'include_recommendations': True,
            'chart_format': 'png',
            'dpi': 300
        }

        logger.info(
            f"Automated reporter initialized with output directory: {output_dir}")

    def _load_default_templates(self):
        """Load default report templates."""
        # Portfolio Performance Template
        self.templates['portfolio_performance'] = ReportTemplate()

        # Risk Analysis Template
        self.templates['risk_analysis'] = ReportTemplate()

        # Forecasting Report Template
        self.templates['forecasting_report'] = ReportTemplate()

        # Executive Summary Template
        self.templates['executive_summary'] = ReportTemplate()

    def generate_portfolio_report(self, portfolio_data: Dict[str, Any],
                                  output_format: str = 'html') -> str:
        """
        Generate portfolio performance report.

        Args:
            portfolio_data: Portfolio performance data
            output_format: Output format ('html', 'pdf', 'json')

        Returns:
            Path to generated report
        """
        try:
            # Generate report content
            report_content = self._generate_portfolio_content(portfolio_data)

            # Generate report file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_report_{timestamp}"

            if output_format == 'html':
                return self._save_html_report(report_content, filename)
            elif output_format == 'pdf':
                return self._save_pdf_report(report_content, filename)
            elif output_format == 'json':
                return self._save_json_report(portfolio_data, filename)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

        except Exception as e:
            logger.error(f"Failed to generate portfolio report: {str(e)}")
            raise

    def generate_risk_report(self, risk_data: Dict[str, Any],
                             output_format: str = 'html') -> str:
        """
        Generate risk analysis report.

        Args:
            risk_data: Risk analysis data
            output_format: Output format ('html', 'pdf', 'json')

        Returns:
            Path to generated report
        """
        try:
            # Generate report content
            report_content = self._generate_risk_content(risk_data)

            # Generate report file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"risk_report_{timestamp}"

            if output_format == 'html':
                return self._save_html_report(report_content, filename)
            elif output_format == 'pdf':
                return self._save_pdf_report(report_content, filename)
            elif output_format == 'json':
                return self._save_json_report(risk_data, filename)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

        except Exception as e:
            logger.error(f"Failed to generate risk report: {str(e)}")
            raise

    def generate_forecasting_report(self, forecast_data: Dict[str, Any],
                                    output_format: str = 'html') -> str:
        """
        Generate forecasting accuracy report.

        Args:
            forecast_data: Forecasting performance data
            output_format: Output format ('html', 'pdf', 'json')

        Returns:
            Path to generated report
        """
        try:
            # Generate report content
            report_content = self._generate_forecasting_content(forecast_data)

            # Generate report file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forecasting_report_{timestamp}"

            if output_format == 'html':
                return self._save_html_report(report_content, filename)
            elif output_format == 'pdf':
                return self._save_pdf_report(report_content, filename)
            elif output_format == 'json':
                return self._save_json_report(forecast_data, filename)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

        except Exception as e:
            logger.error(f"Failed to generate forecasting report: {str(e)}")
            raise

    def generate_executive_summary(self, all_data: Dict[str, Any],
                                   output_format: str = 'html') -> str:
        """
        Generate executive summary report.

        Args:
            all_data: Combined data from all reports
            output_format: Output format ('html', 'pdf', 'json')

        Returns:
            Path to generated report
        """
        try:
            # Generate report content
            report_content = self._generate_executive_content(all_data)

            # Generate report file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"executive_summary_{timestamp}"

            if output_format == 'html':
                return self._save_html_report(report_content, filename)
            elif output_format == 'pdf':
                return self._save_pdf_report(report_content, filename)
            elif output_format == 'json':
                return self._save_json_report(all_data, filename)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

        except Exception as e:
            logger.error(f"Failed to generate executive summary: {str(e)}")
            raise

    def _generate_portfolio_content(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate portfolio report content."""
        content = {
            'title': 'Portfolio Performance Report',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sections': []
        }

        # Portfolio Overview Section
        overview_section = {
            'title': 'Portfolio Overview',
            'content': self._generate_portfolio_overview(portfolio_data)
        }
        content['sections'].append(overview_section)

        # Performance Metrics Section
        if 'performance_metrics' in portfolio_data:
            metrics_section = {
                'title': 'Performance Metrics',
                'content': self._generate_performance_metrics(portfolio_data['performance_metrics'])
            }
            content['sections'].append(metrics_section)

        # Asset Allocation Section
        if 'asset_allocation' in portfolio_data:
            allocation_section = {
                'title': 'Asset Allocation',
                'content': self._generate_asset_allocation(portfolio_data['asset_allocation'])
            }
            content['sections'].append(allocation_section)

        # Risk Metrics Section
        if 'risk_metrics' in portfolio_data:
            risk_section = {
                'title': 'Risk Analysis',
                'content': self._generate_risk_metrics(portfolio_data['risk_metrics'])
            }
            content['sections'].append(risk_section)

        return content

    def _generate_portfolio_overview(self, portfolio_data: Dict[str, Any]) -> str:
        """Generate portfolio overview HTML."""
        html = "<div class='overview'>"

        # Key metrics
        if 'total_value' in portfolio_data:
            html += f"""
            <div class='metric'>
                <div class='metric-value'>${portfolio_data['total_value']:,.2f}</div>
                <div class='metric-label'>Total Portfolio Value</div>
            </div>
            """

        if 'total_return' in portfolio_data:
            html += f"""
            <div class='metric'>
                <div class='metric-value'>{portfolio_data['total_return']:.2f}%</div>
                <div class='metric-label'>Total Return</div>
            </div>
            """

        if 'sharpe_ratio' in portfolio_data:
            html += f"""
            <div class='metric'>
                <div class='metric-value'>{portfolio_data['sharpe_ratio']:.3f}</div>
                <div class='metric-label'>Sharpe Ratio</div>
            </div>
            """

        html += "</div>"
        return html

    def _generate_performance_metrics(self, metrics: Dict[str, Any]) -> str:
        """Generate performance metrics HTML."""
        html = "<table><tr><th>Metric</th><th>Value</th><th>Benchmark</th><th>Outperformance</th></tr>"

        metric_mappings = {
            'total_return': 'Total Return',
            'annualized_return': 'Annualized Return',
            'volatility': 'Volatility',
            'sharpe_ratio': 'Sharpe Ratio',
            'max_drawdown': 'Maximum Drawdown',
            'var_95': 'VaR (95%)',
            'cvar_95': 'CVaR (95%)'
        }

        for metric_key, metric_name in metric_mappings.items():
            if metric_key in metrics:
                value = metrics[metric_key]
                benchmark = metrics.get(f'{metric_key}_benchmark', 'N/A')
                outperformance = metrics.get(
                    f'{metric_key}_outperformance', 'N/A')

                html += f"<tr><td>{metric_name}</td><td>{value}</td><td>{benchmark}</td><td>{outperformance}</td></tr>"

        html += "</table>"
        return html

    def _generate_asset_allocation(self, allocation: Dict[str, Any]) -> str:
        """Generate asset allocation HTML."""
        if isinstance(allocation, dict):
            html = "<table><tr><th>Asset</th><th>Weight</th><th>Value</th><th>Return</th></tr>"

            for asset, details in allocation.items():
                if isinstance(details, dict):
                    weight = details.get('weight', 0)
                    value = details.get('value', 0)
                    returns = details.get('returns', 0)

                    html += f"<tr><td>{asset}</td><td>{weight:.1%}</td><td>${value:,.2f}</td><td>{returns:.2f}%</td></tr>"

            html += "</table>"
        else:
            html = "<p>Asset allocation data not available</p>"

        return html

    def _generate_risk_metrics(self, risk_data: Dict[str, Any]) -> str:
        """Generate risk metrics HTML."""
        html = "<div class='risk-metrics'>"

        # Risk metrics table
        html += "<table><tr><th>Risk Metric</th><th>Value</th><th>Risk Level</th></tr>"

        risk_mappings = {
            'var_95': 'VaR (95%)',
            'cvar_95': 'CVaR (95%)',
            'max_drawdown': 'Maximum Drawdown',
            'volatility': 'Volatility',
            'beta': 'Beta',
            'correlation': 'Correlation'
        }

        for risk_key, risk_name in risk_mappings.items():
            if risk_key in risk_data:
                value = risk_data[risk_key]
                risk_level = self._get_risk_level(risk_key, value)

                html += f"<tr><td>{risk_name}</td><td>{value}</td><td>{risk_level}</td></tr>"

        html += "</table>"

        # Risk recommendations
        if 'recommendations' in risk_data:
            html += "<h3>Risk Recommendations</h3><ul>"
            for rec in risk_data['recommendations']:
                html += f"<li>{rec}</li>"
            html += "</ul>"

        html += "</div>"
        return html

    def _get_risk_level(self, metric: str, value: float) -> str:
        """Determine risk level for a metric."""
        if metric == 'var_95' or metric == 'cvar_95':
            if abs(value) < 0.02:
                return "<span style='color: green;'>LOW</span>"
            elif abs(value) < 0.05:
                return "<span style='color: orange;'>MEDIUM</span>"
            else:
                return "<span style='color: red;'>HIGH</span>"
        elif metric == 'max_drawdown':
            if abs(value) < 0.10:
                return "<span style='color: green;'>LOW</span>"
            elif abs(value) < 0.20:
                return "<span style='color: orange;'>MEDIUM</span>"
            else:
                return "<span style='color: red;'>HIGH</span>"
        else:
            return "<span style='color: blue;'>NORMAL</span>"

    def _generate_risk_content(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk report content."""
        content = {
            'title': 'Risk Analysis Report',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sections': []
        }

        # Risk Overview Section
        overview_section = {
            'title': 'Risk Overview',
            'content': self._generate_risk_overview(risk_data)
        }
        content['sections'].append(overview_section)

        # Detailed Risk Metrics Section
        if 'detailed_metrics' in risk_data:
            metrics_section = {
                'title': 'Detailed Risk Metrics',
                'content': self._generate_risk_metrics(risk_data['detailed_metrics'])
            }
            content['sections'].append(metrics_section)

        return content

    def _generate_risk_overview(self, risk_data: Dict[str, Any]) -> str:
        """Generate risk overview HTML."""
        html = "<div class='risk-overview'>"

        # Overall risk score
        if 'overall_risk_score' in risk_data:
            score = risk_data['overall_risk_score']
            risk_level = self._get_overall_risk_level(score)

            html += f"""
            <div class='metric'>
                <div class='metric-value'>{score:.2f}</div>
                <div class='metric-label'>Overall Risk Score</div>
                <div class='risk-level'>{risk_level}</div>
            </div>
            """

        # Key risk indicators
        if 'key_risk_indicators' in risk_data:
            html += "<h3>Key Risk Indicators</h3><ul>"
            for indicator in risk_data['key_risk_indicators']:
                html += f"<li>{indicator}</li>"
            html += "</ul>"

        html += "</div>"
        return html

    def _get_overall_risk_level(self, score: float) -> str:
        """Get overall risk level description."""
        if score < 0.3:
            return "<span style='color: green;'>LOW RISK</span>"
        elif score < 0.7:
            return "<span style='color: orange;'>MEDIUM RISK</span>"
        else:
            return "<span style='color: red;'>HIGH RISK</span>"

    def _generate_forecasting_content(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate forecasting report content."""
        content = {
            'title': 'Forecasting Accuracy Report',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sections': []
        }

        # Model Performance Section
        if 'model_performance' in forecast_data:
            performance_section = {
                'title': 'Model Performance',
                'content': self._generate_model_performance(forecast_data['model_performance'])
            }
            content['sections'].append(performance_section)

        # Accuracy Metrics Section
        if 'accuracy_metrics' in forecast_data:
            accuracy_section = {
                'title': 'Accuracy Metrics',
                'content': self._generate_accuracy_metrics(forecast_data['accuracy_metrics'])
            }
            content['sections'].append(accuracy_section)

        # Forecast Analysis Section
        if 'forecast_analysis' in forecast_data:
            forecast_section = {
                'title': 'Forecast Analysis',
                'content': self._generate_forecast_analysis(forecast_data['forecast_analysis'])
            }
            content['sections'].append(forecast_section)

        return content

    def _generate_executive_content(self, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary content."""
        content = {
            'title': 'Executive Summary Report',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sections': []
        }

        # Executive Summary Section
        summary_section = {
            'title': 'Executive Summary',
            'content': self._generate_executive_summary_content(all_data)
        }
        content['sections'].append(summary_section)

        # Key Performance Indicators Section
        kpi_section = {
            'title': 'Key Performance Indicators',
            'content': self._generate_kpi_summary(all_data)
        }
        content['sections'].append(kpi_section)

        # Recommendations Section
        recommendations_section = {
            'title': 'Strategic Recommendations',
            'content': self._generate_strategic_recommendations(all_data)
        }
        content['sections'].append(recommendations_section)

        return content

    def _save_html_report(self, content: Dict[str, Any], filename: str) -> str:
        """Save HTML report to file."""
        try:
            # Render template
            template = self.templates['portfolio_performance']
            html_content = template.render(content)

            # Save file
            filepath = self.output_dir / f"{filename}.html"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"HTML report saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save HTML report: {str(e)}")
            raise

    def _save_pdf_report(self, content: Dict[str, Any], filename: str) -> str:
        """Save PDF report to file."""
        if not WEASYPRINT_AVAILABLE:
            raise ImportError(
                "WeasyPrint not available. Install with: pip install weasyprint")

        try:
            # Generate HTML first
            template = self.templates['portfolio_performance']
            html_content = template.render(content)

            # Convert to PDF
            filepath = self.output_dir / f"{filename}.pdf"
            HTML(string=html_content).write_pdf(filepath)

            logger.info(f"PDF report saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save PDF report: {str(e)}")
            raise

    def _save_json_report(self, data: Dict[str, Any], filename: str) -> str:
        """Save JSON report to file."""
        try:
            filepath = self.output_dir / f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"JSON report saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save JSON report: {str(e)}")
            raise

    def generate_comprehensive_report(self, all_data: Dict[str, Any],
                                      output_format: str = 'html') -> str:
        """
        Generate comprehensive report combining all report types.

        Args:
            all_data: Combined data from all report types
            output_format: Output format ('html', 'pdf', 'json')

        Returns:
            Path to generated report
        """
        try:
            # Generate comprehensive content
            content = {
                'title': 'GMF Comprehensive Financial Report',
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sections': []
            }

            # Add all sections
            if 'portfolio' in all_data:
                portfolio_section = {
                    'title': 'Portfolio Performance',
                    'content': self._generate_portfolio_overview(all_data['portfolio'])
                }
                content['sections'].append(portfolio_section)

            if 'risk' in all_data:
                risk_section = {
                    'title': 'Risk Analysis',
                    'content': self._generate_risk_overview(all_data['risk'])
                }
                content['sections'].append(risk_section)

            if 'forecasting' in all_data:
                forecast_section = {
                    'title': 'Forecasting Performance',
                    'content': self._generate_model_performance(all_data['forecasting'])
                }
                content['sections'].append(forecast_section)

            # Generate report file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_report_{timestamp}"

            if output_format == 'html':
                return self._save_html_report(content, filename)
            elif output_format == 'pdf':
                return self._save_pdf_report(content, filename)
            elif output_format == 'json':
                return self._save_json_report(all_data, filename)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {str(e)}")
            raise

    def _generate_model_performance(self, performance_data: Dict[str, Any]) -> str:
        """Generate model performance HTML."""
        html = "<div class='model-performance'>"

        # Performance metrics table
        html += "<table><tr><th>Model</th><th>Accuracy</th><th>MAPE</th><th>R² Score</th></tr>"

        if 'models' in performance_data:
            for model_name, model_data in performance_data['models'].items():
                accuracy = model_data.get('accuracy', 'N/A')
                mape = model_data.get('mape', 'N/A')
                r2 = model_data.get('r2_score', 'N/A')

                html += f"<tr><td>{model_name}</td><td>{accuracy}</td><td>{mape}</td><td>{r2}</td></tr>"

        html += "</table></div>"
        return html

    def _generate_accuracy_metrics(self, accuracy_data: Dict[str, Any]) -> str:
        """Generate accuracy metrics HTML."""
        html = "<div class='accuracy-metrics'>"

        # Accuracy overview
        if 'overall_accuracy' in accuracy_data:
            html += f"<h3>Overall Accuracy: {accuracy_data['overall_accuracy']:.2f}%</h3>"

        # Detailed metrics
        if 'detailed_metrics' in accuracy_data:
            html += "<table><tr><th>Metric</th><th>Value</th></tr>"

            for metric_name, value in accuracy_data['detailed_metrics'].items():
                html += f"<tr><td>{metric_name}</td><td>{value}</td></tr>"

            html += "</table>"

        html += "</div>"
        return html

    def _generate_forecast_analysis(self, forecast_data: Dict[str, Any]) -> str:
        """Generate forecast analysis HTML."""
        html = "<div class='forecast-analysis'>"

        # Forecast summary
        if 'forecast_summary' in forecast_data:
            summary = forecast_data['forecast_summary']
            html += f"<h3>Forecast Summary</h3>"
            html += f"<p>Horizon: {summary.get('horizon', 'N/A')}</p>"
            html += f"<p>Confidence Level: {summary.get('confidence_level', 'N/A')}</p>"

        # Forecast metrics
        if 'forecast_metrics' in forecast_data:
            html += "<table><tr><th>Metric</th><th>Value</th></tr>"

            for metric_name, value in forecast_data['forecast_metrics'].items():
                html += f"<tr><td>{metric_name}</td><td>{value}</td></tr>"

            html += "</table>"

        html += "</div>"
        return html

    def _generate_executive_summary_content(self, all_data: Dict[str, Any]) -> str:
        """Generate executive summary content HTML."""
        html = "<div class='executive-summary'>"

        # High-level summary
        html += "<h3>Executive Summary</h3>"
        html += "<p>This report provides a comprehensive overview of the GMF Time Series Forecasting system performance.</p>"

        # Key highlights
        html += "<h4>Key Highlights</h4><ul>"

        if 'portfolio' in all_data and 'total_return' in all_data['portfolio']:
            html += f"<li>Portfolio Total Return: {all_data['portfolio']['total_return']:.2f}%</li>"

        if 'risk' in all_data and 'overall_risk_score' in all_data['risk']:
            risk_score = all_data['risk']['overall_risk_score']
            html += f"<li>Overall Risk Score: {risk_score:.2f}</li>"

        if 'forecasting' in all_data and 'overall_accuracy' in all_data['forecasting']:
            accuracy = all_data['forecasting']['overall_accuracy']
            html += f"<li>Forecasting Accuracy: {accuracy:.2f}%</li>"

        html += "</ul></div>"
        return html

    def _generate_kpi_summary(self, all_data: Dict[str, Any]) -> str:
        """Generate KPI summary HTML."""
        html = "<div class='kpi-summary'>"

        # KPI table
        html += "<table><tr><th>KPI Category</th><th>Metric</th><th>Value</th><th>Status</th></tr>"

        # Portfolio KPIs
        if 'portfolio' in all_data:
            portfolio = all_data['portfolio']
            if 'sharpe_ratio' in portfolio:
                sharpe = portfolio['sharpe_ratio']
                status = "✅" if sharpe > 1.0 else "⚠️" if sharpe > 0.5 else "❌"
                html += f"<tr><td>Portfolio</td><td>Sharpe Ratio</td><td>{sharpe:.3f}</td><td>{status}</td></tr>"

        # Risk KPIs
        if 'risk' in all_data:
            risk = all_data['risk']
            if 'overall_risk_score' in risk:
                risk_score = risk['overall_risk_score']
                status = "✅" if risk_score < 0.3 else "⚠️" if risk_score < 0.7 else "❌"
                html += f"<tr><td>Risk</td><td>Overall Risk Score</td><td>{risk_score:.2f}</td><td>{status}</td></tr>"

        # Forecasting KPIs
        if 'forecasting' in all_data:
            forecasting = all_data['forecasting']
            if 'overall_accuracy' in forecasting:
                accuracy = forecasting['overall_accuracy']
                status = "✅" if accuracy > 90 else "⚠️" if accuracy > 80 else "❌"
                html += f"<tr><td>Forecasting</td><td>Overall Accuracy</td><td>{accuracy:.2f}%</td><td>{status}</td></tr>"

        html += "</table></div>"
        return html

    def _generate_strategic_recommendations(self, all_data: Dict[str, Any]) -> str:
        """Generate strategic recommendations HTML."""
        html = "<div class='strategic-recommendations'>"

        html += "<h3>Strategic Recommendations</h3><ul>"

        # Portfolio recommendations
        if 'portfolio' in all_data:
            portfolio = all_data['portfolio']
            if 'sharpe_ratio' in portfolio and portfolio['sharpe_ratio'] < 1.0:
                html += "<li>Consider portfolio rebalancing to improve risk-adjusted returns</li>"
            if 'volatility' in portfolio and portfolio['volatility'] > 0.20:
                html += "<li>Implement risk management strategies to reduce portfolio volatility</li>"

        # Risk recommendations
        if 'risk' in all_data:
            risk = all_data['risk']
            if 'overall_risk_score' in risk and risk['overall_risk_score'] > 0.7:
                html += "<li>Conduct comprehensive risk assessment and implement mitigation strategies</li>"

        # Forecasting recommendations
        if 'forecasting' in all_data:
            forecasting = all_data['forecasting']
            if 'overall_accuracy' in forecasting and forecasting['overall_accuracy'] < 85:
                html += "<li>Review and enhance forecasting models to improve accuracy</li>"

        # General recommendations
        html += "<li>Continue monitoring key performance indicators</li>"
        html += "<li>Regular review of investment strategies</li>"
        html += "<li>Maintain robust risk management framework</li>"

        html += "</ul></div>"
        return html

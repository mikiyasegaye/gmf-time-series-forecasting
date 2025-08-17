#!/usr/bin/env python3
"""
SHAP Model Explainability Module

This module provides comprehensive model explainability using SHAP (SHapley Additive exPlanations)
for the GMF Time Series Forecasting system. It helps understand how the LSTM models make predictions.

Author: GMF Investment Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings
from datetime import datetime, timedelta

# Try to import SHAP, but handle gracefully if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based model explainability for time series forecasting models.

    Provides comprehensive insights into:
    - Feature importance rankings
    - Individual prediction explanations
    - Model behavior analysis
    - Risk assessment explanations
    """

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained forecasting model (LSTM, ARIMA, etc.)
            feature_names: Names of input features
        """
        self.model = model
        self.feature_names = feature_names or [
            f"feature_{i}" for i in range(100)]
        self.explainer = None
        self.shap_values = None

        if not SHAP_AVAILABLE:
            logger.warning(
                "SHAP not available. Some functionality will be limited.")

    def create_explainer(self, background_data: np.ndarray, explainer_type: str = "deep") -> bool:
        """
        Create SHAP explainer for the model.

        Args:
            background_data: Background dataset for explainer
            explainer_type: Type of explainer ("deep", "tree", "linear", "kernel")

        Returns:
            bool: True if explainer created successfully
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP not available. Cannot create explainer.")
            return False

        try:
            if explainer_type == "deep" and hasattr(self.model, 'predict'):
                # For deep learning models like LSTM
                self.explainer = shap.DeepExplainer(
                    self.model, background_data)
                logger.info("Deep SHAP explainer created successfully")
                return True

            elif explainer_type == "tree" and hasattr(self.model, 'feature_importances_'):
                # For tree-based models
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Tree SHAP explainer created successfully")
                return True

            elif explainer_type == "linear" and hasattr(self.model, 'coef_'):
                # For linear models
                self.explainer = shap.LinearExplainer(
                    self.model, background_data)
                logger.info("Linear SHAP explainer created successfully")
                return True

            else:
                # Fallback to kernel explainer
                self.explainer = shap.KernelExplainer(
                    self.model.predict, background_data)
                logger.info("Kernel SHAP explainer created successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {str(e)}")
            return False

    def explain_predictions(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Generate SHAP explanations for predictions.

        Args:
            data: Input data for explanation

        Returns:
            Dict containing SHAP values and explanations
        """
        if not self.explainer:
            logger.error(
                "SHAP explainer not initialized. Call create_explainer first.")
            return {}

        try:
            # Generate SHAP values
            self.shap_values = self.explainer.shap_values(data)

            # Handle different output formats
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[0]  # Take first output

            # Calculate feature importance
            feature_importance = np.abs(self.shap_values).mean(axis=0)

            # Create explanation summary
            explanation = {
                'shap_values': self.shap_values,
                'feature_importance': feature_importance,
                'feature_names': self.feature_names[:len(feature_importance)],
                'prediction_count': len(data),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(
                f"Generated SHAP explanations for {len(data)} predictions")
            return explanation

        except Exception as e:
            logger.error(f"Failed to generate SHAP explanations: {str(e)}")
            return {}

    def get_feature_importance_ranking(self, explanation: Dict[str, Any]) -> pd.DataFrame:
        """
        Get ranked feature importance from SHAP explanations.

        Args:
            explanation: SHAP explanation dictionary

        Returns:
            DataFrame with ranked feature importance
        """
        if not explanation or 'feature_importance' not in explanation:
            return pd.DataFrame()

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': explanation['feature_names'],
            'importance': explanation['feature_importance'],
            'rank': range(1, len(explanation['feature_importance']) + 1)
        })

        # Sort by importance (descending)
        importance_df = importance_df.sort_values(
            'importance', ascending=False).reset_index(drop=True)
        importance_df['rank'] = range(1, len(importance_df) + 1)

        return importance_df

    def plot_feature_importance(self, explanation: Dict[str, Any], top_n: int = 20,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance from SHAP explanations.

        Args:
            explanation: SHAP explanation dictionary
            top_n: Number of top features to display
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        if not explanation or 'feature_importance' not in explanation:
            logger.error("No explanation data available for plotting")
            return plt.figure()

        # Get top features
        importance_df = self.get_feature_importance_ranking(explanation)
        top_features = importance_df.head(top_n)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'])

        # Customize plot
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('SHAP Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importance (SHAP)')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            ax.text(importance + 0.001, i, f'{importance:.4f}',
                    va='center', fontsize=10)

        # Invert y-axis for better readability
        ax.invert_yaxis()

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")

        return fig

    def plot_individual_explanation(self, explanation: Dict[str, Any],
                                    sample_idx: int = 0, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot individual prediction explanation.

        Args:
            explanation: SHAP explanation dictionary
            sample_idx: Index of sample to explain
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        if not explanation or 'shap_values' not in explanation:
            logger.error("No SHAP values available for individual explanation")
            return plt.figure()

        # Get SHAP values for specific sample
        sample_shap = explanation['shap_values'][sample_idx]
        feature_names = explanation['feature_names'][:len(sample_shap)]

        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Sort features by absolute SHAP value
        sorted_indices = np.argsort(np.abs(sample_shap))
        sorted_shap = sample_shap[sorted_indices]
        sorted_features = [feature_names[i] for i in sorted_indices]

        # Create waterfall plot
        y_pos = np.arange(len(sorted_features))
        colors = ['red' if x < 0 else 'blue' for x in sorted_shap]

        bars = ax.barh(y_pos, sorted_shap, color=colors, alpha=0.7)

        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title(
            f'Individual Prediction Explanation - Sample {sample_idx}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, shap_val) in enumerate(zip(bars, sorted_shap)):
            ax.text(shap_val + (0.001 if shap_val >= 0 else -0.001), i,
                    f'{shap_val:.4f}', va='center', fontsize=9)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Individual explanation plot saved to {save_path}")

        return fig

    def generate_explanation_report(self, explanation: Dict[str, Any],
                                    output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive explanation report.

        Args:
            explanation: SHAP explanation dictionary
            output_path: Optional path to save the report

        Returns:
            Report content as string
        """
        if not explanation:
            return "No explanation data available."

        # Get feature importance ranking
        importance_df = self.get_feature_importance_ranking(explanation)

        # Generate report
        report = []
        report.append("=" * 60)
        report.append("SHAP MODEL EXPLANABILITY REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {explanation.get('timestamp', 'Unknown')}")
        report.append(
            f"Predictions Analyzed: {explanation.get('prediction_count', 0)}")
        report.append("")

        # Top features
        report.append("TOP 10 MOST IMPORTANT FEATURES:")
        report.append("-" * 40)
        for _, row in importance_df.head(10).iterrows():
            report.append(
                f"{row['rank']:2d}. {row['feature']:<25} {row['importance']:.6f}")
        report.append("")

        # Feature importance statistics
        report.append("FEATURE IMPORTANCE STATISTICS:")
        report.append("-" * 40)
        report.append(f"Total Features: {len(importance_df)}")
        report.append(
            f"Mean Importance: {importance_df['importance'].mean():.6f}")
        report.append(
            f"Std Importance: {importance_df['importance'].std():.6f}")
        report.append(
            f"Max Importance: {importance_df['importance'].max():.6f}")
        report.append(
            f"Min Importance: {importance_df['importance'].min():.6f}")
        report.append("")

        # Model insights
        report.append("MODEL INSIGHTS:")
        report.append("-" * 40)

        # Identify key features
        high_importance = importance_df[importance_df['importance']
                                        > importance_df['importance'].quantile(0.8)]
        report.append(
            f"High Importance Features (>80th percentile): {len(high_importance)}")

        # Feature categories
        technical_features = [f for f in importance_df['feature'] if any(
            x in f.lower() for x in ['ma', 'rsi', 'bollinger', 'macd'])]
        price_features = [f for f in importance_df['feature'] if any(
            x in f.lower() for x in ['open', 'high', 'low', 'close', 'volume'])]

        report.append(f"Technical Indicators: {len(technical_features)}")
        report.append(f"Price/Volume Features: {len(price_features)}")

        report.append("")
        report.append("=" * 60)

        report_content = "\n".join(report)

        # Save report if path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(report_content)
                logger.info(f"Explanation report saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save report: {str(e)}")

        return report_content

    def get_risk_explanation(self, explanation: Dict[str, Any],
                             risk_threshold: float = 0.01) -> Dict[str, Any]:
        """
        Analyze model risk based on SHAP explanations.

        Args:
            explanation: SHAP explanation dictionary
            risk_threshold: Threshold for high-risk features

        Returns:
            Risk analysis dictionary
        """
        if not explanation or 'feature_importance' not in explanation:
            return {}

        # Identify high-risk features
        high_risk_features = []
        for feature, importance in zip(explanation['feature_names'], explanation['feature_importance']):
            if importance > risk_threshold:
                high_risk_features.append({
                    'feature': feature,
                    'importance': importance,
                    'risk_level': 'HIGH' if importance > risk_threshold * 2 else 'MEDIUM'
                })

        # Calculate risk metrics
        total_importance = sum(explanation['feature_importance'])
        risk_importance = sum([f['importance'] for f in high_risk_features])
        risk_percentage = (risk_importance / total_importance) * \
            100 if total_importance > 0 else 0

        risk_analysis = {
            'high_risk_features': high_risk_features,
            'total_risk_importance': risk_importance,
            'risk_percentage': risk_percentage,
            'risk_threshold': risk_threshold,
            'recommendations': self._generate_risk_recommendations(high_risk_features)
        }

        return risk_analysis

    def _generate_risk_recommendations(self, high_risk_features: List[Dict]) -> List[str]:
        """Generate risk mitigation recommendations."""
        recommendations = []

        if not high_risk_features:
            recommendations.append(
                "✅ Model shows low risk with well-distributed feature importance")
            return recommendations

        recommendations.append("⚠️  High-risk features detected. Consider:")

        for feature in high_risk_features:
            if 'volume' in feature['feature'].lower():
                recommendations.append(
                    f"  • Monitor {feature['feature']} for unusual patterns")
            elif 'technical' in feature['feature'].lower():
                recommendations.append(
                    f"  • Validate {feature['feature']} calculation logic")
            else:
                recommendations.append(
                    f"  • Review {feature['feature']} data quality and relevance")

        recommendations.append("  • Implement additional validation checks")
        recommendations.append(
            "  • Consider feature engineering to reduce dependency")

        return recommendations


class ModelExplainabilityManager:
    """
    High-level manager for model explainability across multiple models.
    """

    def __init__(self):
        """Initialize the explainability manager."""
        self.explainers = {}
        self.explanations = {}
        self.reports = {}

    def add_model(self, model_name: str, model, feature_names: Optional[List[str]] = None):
        """Add a model for explainability analysis."""
        self.explainers[model_name] = SHAPExplainer(model, feature_names)
        logger.info(f"Added model '{model_name}' for explainability analysis")

    def explain_all_models(self, data: np.ndarray, background_data: np.ndarray) -> Dict[str, Any]:
        """Generate explanations for all registered models."""
        results = {}

        for model_name, explainer in self.explainers.items():
            try:
                # Create explainer
                if explainer.create_explainer(background_data):
                    # Generate explanations
                    explanation = explainer.explain_predictions(data)
                    if explanation:
                        results[model_name] = explanation
                        self.explanations[model_name] = explanation
                        logger.info(
                            f"Generated explanations for model '{model_name}'")
                    else:
                        logger.warning(
                            f"Failed to generate explanations for model '{model_name}'")
                else:
                    logger.error(
                        f"Failed to create explainer for model '{model_name}'")
            except Exception as e:
                logger.error(
                    f"Error explaining model '{model_name}': {str(e)}")

        return results

    def generate_comparative_report(self, output_path: Optional[str] = None) -> str:
        """Generate comparative report across all models."""
        if not self.explanations:
            return "No explanations available for comparison."

        report = []
        report.append("=" * 80)
        report.append("COMPARATIVE MODEL EXPLAINABILITY REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Models Analyzed: {len(self.explanations)}")
        report.append("")

        # Compare feature importance across models
        for model_name, explanation in self.explanations.items():
            report.append(f"MODEL: {model_name.upper()}")
            report.append("-" * 40)

            importance_df = self.explainers[model_name].get_feature_importance_ranking(
                explanation)
            top_features = importance_df.head(5)

            for _, row in top_features.iterrows():
                report.append(
                    f"  {row['rank']}. {row['feature']:<20} {row['importance']:.6f}")

            report.append("")

        # Model comparison insights
        report.append("MODEL COMPARISON INSIGHTS:")
        report.append("-" * 40)

        # Find common important features
        all_top_features = []
        for explanation in self.explanations.values():
            importance_df = self.explainers[list(self.explanations.keys())[
                0]].get_feature_importance_ranking(explanation)
            all_top_features.extend(importance_df.head(10)['feature'].tolist())

        # Count feature occurrences
        feature_counts = {}
        for feature in all_top_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

        # Show commonly important features
        common_features = [f for f, c in feature_counts.items() if c > 1]
        if common_features:
            report.append("Commonly Important Features Across Models:")
            for feature in common_features[:5]:
                report.append(
                    f"  • {feature} (appears in {feature_counts[feature]} models)")
        else:
            report.append(
                "No common important features detected across models")

        report.append("")
        report.append("=" * 80)

        report_content = "\n".join(report)

        # Save report if path provided
        if output_path:
            try:
                with open(output_path, 'w') as output_path:
                    output_path.write(report_content)
                logger.info(f"Comparative report saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save comparative report: {str(e)}")

        return report_content

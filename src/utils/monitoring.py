"""
Monitoring and logging utilities for GMF Time Series Forecasting system.
"""

import logging
import time
import functools
import traceback
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Configure logging


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """Setup comprehensive logging configuration."""

    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console handler
            logging.FileHandler(
                log_file) if log_file else logging.NullHandler()
        ]
    )

    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


class PerformanceMonitor:
    """Monitor performance metrics for the application."""

    def __init__(self):
        self.metrics = {
            "api_calls": 0,
            "total_response_time": 0.0,
            "errors": 0,
            "start_time": time.time()
        }
        self.endpoint_times = {}
        self.error_log = []

    def record_api_call(self, endpoint: str, response_time: float, success: bool = True):
        """Record API call metrics."""
        self.metrics["api_calls"] += 1
        self.metrics["total_response_time"] += response_time

        if not success:
            self.metrics["errors"] += 1

        # Record endpoint-specific timing
        if endpoint not in self.endpoint_times:
            self.endpoint_times[endpoint] = []
        self.endpoint_times[endpoint].append(response_time)

    def record_error(self, endpoint: str, error: Exception, request_data: Dict[str, Any] = None):
        """Record error details."""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "request_data": request_data
        }
        self.error_log.append(error_info)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        uptime = time.time() - self.metrics["start_time"]
        avg_response_time = (
            self.metrics["total_response_time"] / self.metrics["api_calls"]
            if self.metrics["api_calls"] > 0 else 0
        )

        # Calculate endpoint averages
        endpoint_averages = {}
        for endpoint, times in self.endpoint_times.items():
            if times:
                endpoint_averages[endpoint] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "call_count": len(times)
                }

        return {
            "uptime_seconds": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "total_api_calls": self.metrics["api_calls"],
            "total_errors": self.metrics["errors"],
            "error_rate": (self.metrics["errors"] / self.metrics["api_calls"] * 100) if self.metrics["api_calls"] > 0 else 0,
            "average_response_time": avg_response_time,
            "endpoint_performance": endpoint_averages,
            "last_error": self.error_log[-1] if self.error_log else None
        }

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {
            "api_calls": 0,
            "total_response_time": 0.0,
            "errors": 0,
            "start_time": time.time()
        }
        self.endpoint_times = {}
        self.error_log = []


class HealthChecker:
    """Health check utilities for the system."""

    def __init__(self):
        self.health_checks = {}
        self.last_check = {}

    def add_health_check(self, name: str, check_func: Callable[[], bool]):
        """Add a health check function."""
        self.health_checks[name] = check_func

    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_status = "healthy"

        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                is_healthy = check_func()
                check_time = time.time() - start_time

                results[name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "response_time": check_time,
                    "timestamp": datetime.now().isoformat()
                }

                if not is_healthy:
                    overall_status = "unhealthy"

            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                overall_status = "unhealthy"

            self.last_check[name] = datetime.now()

        return {
            "overall_status": overall_status,
            "checks": results,
            "timestamp": datetime.now().isoformat()
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information."""
        health_checks = self.run_health_checks()

        # Add system information
        import psutil
        try:
            system_info = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "process_count": len(psutil.pids())
            }
        except ImportError:
            system_info = {"error": "psutil not available"}

        return {
            **health_checks,
            "system_resources": system_info
        }


class MetricsCollector:
    """Collect and store various metrics."""

    def __init__(self, storage_file: Optional[str] = None):
        self.storage_file = storage_file
        self.metrics_history = []
        self.max_history_size = 1000

    def collect_metric(self, metric_name: str, value: Any, tags: Dict[str, str] = None):
        """Collect a metric."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "metric": metric_name,
            "value": value,
            "tags": tags or {}
        }

        self.metrics_history.append(metric)

        # Maintain history size
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]

        # Save to file if specified
        if self.storage_file:
            self._save_metrics()

    def get_metric_history(self, metric_name: str, hours: int = 24) -> list:
        """Get metric history for a specific metric."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            m for m in self.metrics_history
            if m["metric"] == metric_name and
            datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        if not self.metrics_history:
            return {"message": "No metrics collected yet"}

        # Group by metric name
        metric_groups = {}
        for metric in self.metrics_history:
            name = metric["metric"]
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(metric)

        # Calculate summaries
        summary = {}
        for name, metrics in metric_groups.items():
            values = [m["value"]
                      for m in metrics if isinstance(m["value"], (int, float))]
            if values:
                summary[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "latest": metrics[-1]["value"],
                    "latest_timestamp": metrics[-1]["timestamp"]
                }

        return summary

    def _save_metrics(self):
        """Save metrics to file."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save metrics: {e}")


# Global instances
performance_monitor = PerformanceMonitor()
health_checker = HealthChecker()
metrics_collector = MetricsCollector("logs/metrics.json")

# Decorator for monitoring API endpoints


def monitor_endpoint(endpoint_name: str):
    """Decorator to monitor endpoint performance."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                response_time = time.time() - start_time
                performance_monitor.record_api_call(
                    endpoint_name, response_time, success=True)
                return result
            except Exception as e:
                response_time = time.time() - start_time
                performance_monitor.record_api_call(
                    endpoint_name, response_time, success=False)
                performance_monitor.record_error(endpoint_name, e, kwargs)
                raise
        return wrapper
    return decorator

# Initialize basic health checks


def initialize_health_checks():
    """Initialize basic health checks."""

    def check_data_loader():
        """Check if data loader is working."""
        try:
            from ..data.data_loader import DataLoader
            loader = DataLoader()
            return True
        except Exception:
            return False

    def check_models():
        """Check if models can be loaded."""
        try:
            from ..models.lstm_forecaster import LSTMForecaster
            from ..models.arima_forecaster import ARIMAForecaster
            return True
        except Exception:
            return False

    def check_analytics():
        """Check if analytics engine is working."""
        try:
            from ..analytics.advanced_analytics import AdvancedAnalyticsEngine
            engine = AdvancedAnalyticsEngine()
            return True
        except Exception:
            return False

    # Add health checks
    health_checker.add_health_check("data_loader", check_data_loader)
    health_checker.add_health_check("models", check_models)
    health_checker.add_health_check("analytics", check_analytics)


# Initialize when module is imported
initialize_health_checks()

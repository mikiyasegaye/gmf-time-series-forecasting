#!/usr/bin/env python3
"""
GMF Time Series Forecasting - Production Features Demo

This script demonstrates the production-ready features of the GMF system:
- Docker containerization and deployment
- FastAPI REST API with comprehensive endpoints
- Real-time monitoring and health checks
- Performance optimization and scaling
"""

import sys
import os
import time
import requests
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.append('src')


def print_header(title: str, width: int = 80):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print(f"\n{'-' * width}")
    print(f" {title}")
    print("-" * width)


def check_docker_installation() -> bool:
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(['docker', '--version'],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Docker installed: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Docker not working: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Docker not installed")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Docker check timed out")
        return False


def check_docker_compose() -> bool:
    """Check if Docker Compose is available."""
    try:
        result = subprocess.run(['docker-compose', '--version'],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Docker Compose available: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Docker Compose not working: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Docker Compose not installed")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Docker Compose check timed out")
        return False


def build_docker_image() -> bool:
    """Build the Docker image for the GMF application."""
    print_section("Building Docker Image")

    try:
        print("🔨 Building GMF Docker image...")
        result = subprocess.run([
            'docker', 'build', '-t', 'gmf-forecasting:latest', '.'
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("✅ Docker image built successfully")
            return True
        else:
            print(f"❌ Docker build failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Docker build timed out")
        return False
    except Exception as e:
        print(f"❌ Docker build error: {e}")
        return False


def start_docker_container() -> bool:
    """Start the Docker container using docker-compose."""
    print_section("Starting Docker Container")

    try:
        print("🚀 Starting GMF container with docker-compose...")
        result = subprocess.run([
            'docker-compose', 'up', '-d', 'gmf-app'
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ Docker container started successfully")
            return True
        else:
            print(f"❌ Docker container start failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Docker container start timed out")
        return False
    except Exception as e:
        print(f"❌ Docker container start error: {e}")
        return False


def wait_for_api_startup(max_wait: int = 60) -> bool:
    """Wait for the API to start up and become available."""
    print_section("Waiting for API Startup")

    print(f"⏳ Waiting up to {max_wait} seconds for API to start...")

    for i in range(max_wait):
        try:
            response = requests.get('http://localhost:8000/health', timeout=5)
            if response.status_code == 200:
                print(f"✅ API is ready after {i+1} seconds")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(1)
        if (i + 1) % 10 == 0:
            print(f"   Still waiting... ({i+1}/{max_wait}s)")

    print("❌ API failed to start within timeout")
    return False


def test_api_endpoints() -> Dict[str, bool]:
    """Test various API endpoints."""
    print_section("Testing API Endpoints")

    endpoints = {
        "Health Check": "/health",
        "Root": "/",
        "API Info": "/api/info",
        "Available Symbols": "/api/v1/data/symbols",
        "System Status": "/api/v1/system/status"
    }

    results = {}
    base_url = "http://localhost:8000"

    for name, endpoint in endpoints.items():
        try:
            print(f"🔍 Testing {name}...")
            response = requests.get(f"{base_url}{endpoint}", timeout=10)

            if response.status_code == 200:
                print(f"✅ {name}: Success")
                results[name] = True
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                results[name] = False

        except requests.exceptions.RequestException as e:
            print(f"❌ {name}: Connection error - {e}")
            results[name] = False

    return results


def test_forecasting_endpoints() -> Dict[str, bool]:
    """Test forecasting API endpoints."""
    print_section("Testing Forecasting Endpoints")

    endpoints = {
        "LSTM Forecasting": {
            "url": "/api/v1/forecasting/lstm",
            "data": {"symbol": "TSLA", "forecast_days": 5, "lookback_days": 30}
        },
        "ARIMA Forecasting": {
            "url": "/api/v1/forecasting/arima",
            "data": {"symbol": "TSLA", "forecast_days": 5, "lookback_days": 30}
        }
    }

    results = {}
    base_url = "http://localhost:8000"

    for name, config in endpoints.items():
        try:
            print(f"🔍 Testing {name}...")
            response = requests.post(
                f"{base_url}{config['url']}",
                json=config['data'],
                timeout=30
            )

            if response.status_code == 200:
                print(f"✅ {name}: Success")
                results[name] = True
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                if response.text:
                    print(f"   Error: {response.text[:200]}...")
                results[name] = False

        except requests.exceptions.RequestException as e:
            print(f"❌ {name}: Connection error - {e}")
            results[name] = False

    return results


def test_portfolio_endpoints() -> Dict[str, bool]:
    """Test portfolio optimization API endpoints."""
    print_section("Testing Portfolio Endpoints")

    endpoints = {
        "Portfolio Optimization": {
            "url": "/api/v1/portfolio/optimize",
            "data": {"symbols": ["AAPL", "GOOGL", "MSFT"], "risk_tolerance": "moderate"}
        },
        "Efficient Frontier": {
            "url": "/api/v1/portfolio/efficient-frontier",
            "params": {"symbols": "AAPL,GOOGL,MSFT", "num_portfolios": 50}
        }
    }

    results = {}
    base_url = "http://localhost:8000"

    for name, config in endpoints.items():
        try:
            print(f"🔍 Testing {name}...")

            if "data" in config:
                response = requests.post(
                    f"{base_url}{config['url']}",
                    json=config['data'],
                    timeout=30
                )
            else:
                response = requests.get(
                    f"{base_url}{config['url']}",
                    params=config['params'],
                    timeout=30
                )

            if response.status_code == 200:
                print(f"✅ {name}: Success")
                results[name] = True
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                if response.text:
                    print(f"   Error: {response.text[:200]}...")
                results[name] = False

        except requests.exceptions.RequestException as e:
            print(f"❌ {name}: Connection error - {e}")
            results[name] = False

    return results


def test_analytics_endpoints() -> Dict[str, bool]:
    """Test analytics API endpoints."""
    print_section("Testing Analytics Endpoints")

    endpoints = {
        "Market Regime Analysis": {
            "url": "/api/v1/analytics/market-regimes",
            "data": {"symbol": "SPY", "lookback_days": 252}
        },
        "Sentiment Analysis": {
            "url": "/api/v1/analytics/sentiment",
            "data": {"symbol": "SPY", "lookback_days": 252}
        }
    }

    results = {}
    base_url = "http://localhost:8000"

    for name, config in endpoints.items():
        try:
            print(f"🔍 Testing {name}...")
            response = requests.post(
                f"{base_url}{config['url']}",
                json=config['data'],
                timeout=30
            )

            if response.status_code == 200:
                print(f"✅ {name}: Success")
                results[name] = True
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                if response.text:
                    print(f"   Error: {response.text[:200]}...")
                results[name] = False

        except requests.exceptions.RequestException as e:
            print(f"❌ {name}: Connection error - {e}")
            results[name] = False

    return results


def test_reporting_endpoints() -> Dict[str, bool]:
    """Test reporting API endpoints."""
    print_section("Testing Reporting Endpoints")

    endpoints = {
        "Portfolio Report Generation": {
            "url": "/api/v1/reports/portfolio",
            "data": {"portfolio_data": {}, "report_type": "performance"}
        }
    }

    results = {}
    base_url = "http://localhost:8000"

    for name, config in endpoints.items():
        try:
            print(f"🔍 Testing {name}...")
            response = requests.post(
                f"{base_url}{config['url']}",
                json=config['data'],
                timeout=30
            )

            if response.status_code == 200:
                print(f"✅ {name}: Success")
                results[name] = True
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                if response.text:
                    print(f"   Error: {response.text[:200]}...")
                results[name] = False

        except requests.exceptions.RequestException as e:
            print(f"❌ {name}: Connection error - {e}")
            results[name] = False

    return results


def check_container_logs() -> None:
    """Check container logs for any errors."""
    print_section("Checking Container Logs")

    try:
        print("📋 Checking container logs...")
        result = subprocess.run([
            'docker-compose', 'logs', 'gmf-app', '--tail=20'
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            logs = result.stdout
            if "error" in logs.lower() or "exception" in logs.lower():
                print("⚠️  Found errors in logs:")
                print(logs)
            else:
                print("✅ No errors found in recent logs")
        else:
            print(f"❌ Failed to get logs: {result.stderr}")

    except Exception as e:
        print(f"❌ Error checking logs: {e}")


def stop_docker_container() -> None:
    """Stop the Docker container."""
    print_section("Stopping Docker Container")

    try:
        print("🛑 Stopping GMF container...")
        result = subprocess.run([
            'docker-compose', 'down'
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ Docker container stopped successfully")
        else:
            print(f"❌ Failed to stop container: {result.stderr}")

    except Exception as e:
        print(f"❌ Error stopping container: {e}")


def generate_performance_report(all_results: Dict[str, Dict[str, bool]]) -> None:
    """Generate a performance report."""
    print_section("Production Features Demo Results")

    total_tests = 0
    passed_tests = 0

    print("📊 Test Results Summary:")
    print("=" * 60)

    for category, results in all_results.items():
        if results:
            category_total = len(results)
            category_passed = sum(1 for result in results.values() if result)
            total_tests += category_total
            passed_tests += category_passed

            print(f"\n{category}:")
            for test_name, result in results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  {status} {test_name}")

            print(
                f"  Category Result: {category_passed}/{category_total} passed")

    print("\n" + "=" * 60)
    print(f"Overall Result: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! Production features are working correctly.")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Most tests passed. Some features need attention.")
    else:
        print("❌ Many tests failed. Production features need significant work.")


def main():
    """Main demo function."""
    print_header(
        "GMF Time Series Forecasting - Production Features Demo")

    print("This demo will test the production-ready features:")
    print("• Docker containerization")
    print("• FastAPI REST API")
    print("• Monitoring and logging")
    print("• Performance optimization")

    # Check Docker installation
    print_section("Docker Environment Check")
    docker_available = check_docker_installation()
    docker_compose_available = check_docker_compose()

    if not docker_available or not docker_compose_available:
        print("\n❌ Docker environment not ready. Please install Docker and Docker Compose.")
        print("   Visit: https://docs.docker.com/get-docker/")
        return

    # Build and start container
    if not build_docker_image():
        print("\n❌ Failed to build Docker image. Check the Dockerfile and requirements.")
        return

    if not start_docker_container():
        print("\n❌ Failed to start Docker container. Check docker-compose.yml.")
        return

    # Wait for API startup
    if not wait_for_api_startup():
        print("\n❌ API failed to start. Checking container logs...")
        check_container_logs()
        stop_docker_container()
        return

    # Test all endpoints
    all_results = {}

    # Basic API endpoints
    all_results["Basic API"] = test_api_endpoints()

    # Forecasting endpoints
    all_results["Forecasting"] = test_forecasting_endpoints()

    # Portfolio endpoints
    all_results["Portfolio"] = test_portfolio_endpoints()

    # Analytics endpoints
    all_results["Analytics"] = test_analytics_endpoints()

    # Reporting endpoints
    all_results["Reporting"] = test_reporting_endpoints()

    # Check logs for any issues
    check_container_logs()

    # Generate performance report
    generate_performance_report(all_results)

    # Cleanup
    print_section("Cleanup")
    print("🧹 Cleaning up Docker resources...")
    stop_docker_container()

    print("\n" + "=" * 80)
    print("Production Features Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

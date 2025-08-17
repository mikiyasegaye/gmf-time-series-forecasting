# GMF Time Series Forecasting - Makefile
# Provides common development tasks and shortcuts

.PHONY: help install test dashboard lint type-check check all clean

# Default target
help:
	@echo "🚀 GMF Time Series Forecasting - Development Commands"
	@echo "=================================================="
	@echo ""
	@echo "📦 Setup Commands:"
	@echo "  install      Install project dependencies"
	@echo "  setup        Full development setup"
	@echo ""
	@echo "🧪 Testing Commands:"
	@echo "  test         Run the test suite"
	@echo "  check        Run all code quality checks"
	@echo ""
	@echo "🎨 Development Commands:"
	@echo "  dashboard    Launch the interactive dashboard"
	@echo "  lint         Run code linting"
	@echo "  type-check   Run type checking"
	@echo ""
	@echo "🔧 Utility Commands:"
	@echo "  all          Run full development workflow"
	@echo "  clean        Clean up temporary files"
	@echo "  help         Show this help message"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# Run tests
test:
	@echo "🧪 Running test suite..."
	python -m src --test

# Launch dashboard
dashboard:
	@echo "🚀 Launching dashboard..."
	python -m src --dashboard

# Run linting
lint:
	@echo "🔍 Running code linting..."
	flake8 src/ tests/

# Run type checking
type-check:
	@echo "🔍 Running type checking..."
	mypy src/

# Run all checks
check: lint type-check test
	@echo "✅ All checks completed!"

# Full development setup
setup: install check
	@echo "🎉 Development environment ready!"

# Run all development tasks
all: setup dashboard

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "✅ Cleanup completed!"

# Development workflow
dev: clean install check
	@echo "🚀 Development environment ready!"
	@echo "💡 Run 'make dashboard' to launch the dashboard"

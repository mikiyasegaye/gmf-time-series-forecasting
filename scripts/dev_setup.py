#!/usr/bin/env python3
"""
GMF Development Setup Script

This script provides common development tasks and shortcuts.
It's designed for developers working on the GMF system.

Usage:
    python scripts/dev_setup.py --help
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def install_dependencies():
    """Install project dependencies."""
    return run_command("pip install -r requirements.txt", "Installing dependencies")


def run_tests():
    """Run the test suite."""
    return run_command("python -m src --test", "Running test suite")


def launch_dashboard():
    """Launch the dashboard."""
    return run_command("python -m src --dashboard", "Launching dashboard")


def run_linting():
    """Run code linting."""
    return run_command("flake8 src/ tests/", "Running code linting")


def run_type_checking():
    """Run type checking."""
    return run_command("mypy src/", "Running type checking")


def run_all_checks():
    """Run all code quality checks."""
    print("🔍 Running all code quality checks...")

    checks = [
        ("Linting", run_linting),
        ("Type Checking", run_type_checking),
        ("Tests", run_tests)
    ]

    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))

    print("\n📊 Code Quality Check Results:")
    print("=" * 40)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:15} {status}")

    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All checks passed! Code is ready for production.")
    else:
        print("\n⚠️  Some checks failed. Please fix issues before proceeding.")

    return all_passed


def main():
    """Main development setup function."""
    parser = argparse.ArgumentParser(
        description="GMF Development Setup and Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/dev_setup.py --install     # Install dependencies
    python scripts/dev_setup.py --test        # Run tests
    python scripts/dev_setup.py --dashboard   # Launch dashboard
    python scripts/dev_setup.py --check       # Run all quality checks
    python scripts/dev_setup.py --all         # Full development setup
        """
    )

    parser.add_argument(
        '--install',
        action='store_true',
        help='Install project dependencies'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run the test suite'
    )

    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Launch the dashboard'
    )

    parser.add_argument(
        '--lint',
        action='store_true',
        help='Run code linting'
    )

    parser.add_argument(
        '--type-check',
        action='store_true',
        help='Run type checking'
    )

    parser.add_argument(
        '--check',
        action='store_true',
        help='Run all code quality checks'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Full development setup (install + all checks)'
    )

    args = parser.parse_args()

    if args.all:
        print("🚀 Full Development Setup")
        print("=" * 30)
        install_dependencies()
        run_all_checks()
    elif args.install:
        install_dependencies()
    elif args.test:
        run_tests()
    elif args.dashboard:
        launch_dashboard()
    elif args.lint:
        run_linting()
    elif args.type_check:
        run_type_checking()
    elif args.check:
        run_all_checks()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

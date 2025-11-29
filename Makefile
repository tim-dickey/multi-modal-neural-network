# Makefile for multi-modal-neural-network project
# Usage: make <target>

.PHONY: help install install-dev test test-unit test-integration test-cov lint format clean pre-commit

# Default target
help:
	@echo "Available targets:"
	@echo "  install         - Install production dependencies"
	@echo "  install-dev     - Install development dependencies"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration- Run integration tests only"
	@echo "  test-cov        - Run tests with coverage report"
	@echo "  test-fast       - Run tests in parallel (faster)"
	@echo "  lint            - Run linting checks"
	@echo "  format          - Format code with ruff"
	@echo "  pre-commit      - Run pre-commit hooks"
	@echo "  clean           - Clean up temporary files"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-xdist pre-commit ruff bandit safetensors
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/ --ignore=tests/test_integration.py -v

test-integration:
	pytest tests/test_integration.py -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=80

test-fast:
	pytest tests/ --ignore=tests/test_integration.py -n auto -q

# Linting and formatting
lint:
	ruff check src/ tests/
	bandit -c pyproject.toml -r src/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# Pre-commit hooks
pre-commit:
	pre-commit run --all-files

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "coverage.xml" -delete 2>/dev/null || true

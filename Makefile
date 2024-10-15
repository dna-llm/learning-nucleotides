.PHONY: check_uv install check test update help

check_uv: # install `uv` if not installed
	@if ! command -v uv > /dev/null 2>&1; then \
		echo "uv is not installed, installing now..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@uv self update

install: check_uv ## Install the virtual environment and  pre-commit hooks
	@echo "📦 Creating virtual environment"
	@uv venv --python=3.11 --seed
	@uv pip install packaging torch
	@uv sync --all-extras --no-build-isolation-package flash-attn
	@echo "🛠️ Installing developer tools..."
	@uv run pre-commit install

requirements: check_uv
	@echo "Making requirements.txt and requirements-cpu.txt"
	@uv pip compile pyproject.toml -o requirements.txt

check: ## Run code quality tools
	@echo "⚡️ Linting code: Running ruff"
	@uv run ruff check .
	@echo "🧹 Checking code: Running pre-commit"
	@uv run pre-commit run --all-files

test: ## Test the code with pytest
	@echo "✅ Testing code: Running pytest"
	@uv run pytest

update: ## Update pre-commit hooks
	@echo "⚙️ Updating dependencies and pre-commit hooks"
	@uv lock --upgrade
	@uv run pre-commit autoupdate

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

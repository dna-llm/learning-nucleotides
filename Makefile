.PHONY: check_uv install requirements check test update help

check_uv: # install `uv` if not installed
	@if ! command -v uv > /dev/null 2>&1; then \
		echo "uv is not installed, installing now..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@uv self update

install: check_uv
	echo "📦 Creating virtual environment and installing dependencies" && \
	uv venv --python=3.11 --seed && \
	. .venv/bin/activate && \
	uv pip install packaging torch && \
	uv sync --all-extras --no-build-isolation-package flash-attn --locked
	@uv run pre-commit install

requirements: check_uv
	@echo "Exporting dependencies to requirements.txt..."
	@uv export --output-file requirements.txt --no-hashes

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
	$(MAKE) requirements
	@uv run pre-commit autoupdate

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

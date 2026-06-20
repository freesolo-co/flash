# Flash developer tasks. Run `make` (or `make help`) to list them.
# Everything runs from the repo root and assumes `uv` is installed:
# https://docs.astral.sh/uv/getting-started/installation/

# Default the test suite to offline mode so nothing reaches provider/network APIs.
# Override per-invocation (e.g. `make test FLASH_SKIP_NET=0`) when you really mean to.
export FLASH_SKIP_NET ?= 1

.DEFAULT_GOAL := help
.PHONY: help setup test test-live lint lint-fix fmt fmt-check typecheck check server cli clean

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | sort | \
		awk -F':.*## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install deps (server extra + dev tools) into the uv venv
	uv sync --extra server --dev

test: ## Run the offline CPU test suite (FLASH_SKIP_NET=1)
	uv run pytest

test-live: ## Run live tests that hit real provider APIs (needs creds)
	FLASH_LIVE=1 FLASH_SKIP_NET=0 uv run pytest -m live

lint: ## Ruff lint check
	uv run ruff check .

lint-fix: ## Ruff lint check with autofix
	uv run ruff check --fix .

fmt: ## Format the code with ruff
	uv run ruff format .

fmt-check: ## Check formatting without writing changes
	uv run ruff format --check .

typecheck: ## Run mypy over the package
	uv run mypy flash

check: lint test ## Lint + offline tests (what CI runs)

server: ## Run the control plane locally (operator-side)
	uv run flash-server

cli: ## Show the slm client CLI help
	uv run slm --help

clean: ## Remove tool caches (pytest/ruff/mypy/pyc)
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

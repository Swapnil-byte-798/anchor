# =============================================================================
# ANCHOR — every command a reviewer needs. Run all of them from the repo root.
#
# The 60-second path:   make install && make demo
#
# Nothing here needs an API key or a network connection. The measurement
# targets (eval, ablate, taxonomy, judge-check, thresholds, gate, readme) are
# the point of the project: every number that ends up in README.md or in
# eval/thresholds.yaml is written there by one of them, never typed by hand.
# =============================================================================

.DEFAULT_GOAL := help
.PHONY: help install test lint typecheck fmt ingest serve up down ask agent \
        golden eval ablate taxonomy judge-check thresholds gate readme \
        readme-check demo clean

# Override on the command line:  make ask Q="What is the SEV-2 response time?"
Q ?= How long are gateway request logs retained?
AQ ?= What was the average p95 latency across 2025?
# Prefer the project venv if one exists, then python3 — macOS ships no bare
# `python`, so defaulting to it breaks every target here on a fresh clone.
# Override with:  make eval PY=/path/to/python
PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python \
              || command -v python3 \
              || command -v python)

# Same problem for the console script: `anchor` is only on PATH once the venv is
# activated, so every measurement target failed on a fresh clone without it.
ANCHOR ?= $(shell [ -x .venv/bin/anchor ] && echo .venv/bin/anchor \
                  || command -v anchor \
                  || echo anchor)

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---- setup ------------------------------------------------------------------

install: ## Editable install with the dev extras
	$(PY) -m pip install -e ".[dev]"

# ---- quality ----------------------------------------------------------------

test: ## Run the test suite
	$(PY) -m pytest

lint: ## ruff check (E,F,I,UP,B)
	$(PY) -m ruff check src tests scripts

typecheck: ## mypy over the package
	$(PY) -m mypy src

fmt: ## ruff format + autofix imports
	$(PY) -m ruff check --fix src tests scripts
	$(PY) -m ruff format src tests scripts

# ---- service ----------------------------------------------------------------

ingest: ## Build data/index.db and data/vectors.npy from corpus/
	$(ANCHOR) ingest

serve: ## Run the API on :8000 with reload
	uvicorn anchor.api:create_app --factory --reload --host 0.0.0.0 --port 8000

up: ## docker compose up (builds, ingests if needed, serves :8000)
	mkdir -p data
	docker compose up --build -d

down: ## docker compose down
	docker compose down

ask: ## One grounded answer:  make ask Q="..."
	$(ANCHOR) ask "$(Q)"

agent: ## One bounded agent run:  make agent AQ="..."
	$(ANCHOR) agent "$(AQ)"

# ---- the measurement layer --------------------------------------------------

golden: ## Regenerate eval/golden/qa.jsonl from the raw corpus text
	$(PY) scripts/build_golden.py

eval: ## Run the golden set; writes eval/reports/
	$(ANCHOR) eval

ablate: ## 4 strategies x 6 chunk configs -> eval/reports/ablation.md
	$(ANCHOR) ablate

taxonomy: ## Classify the worst cases -> eval/reports/failures.md
	$(ANCHOR) taxonomy

judge-check: ## Judge vs 30 human labels (agreement + kappa) -> eval/reports/
	$(ANCHOR) judge-check

thresholds: ## GENERATE eval/thresholds.yaml from a real run (never hand-typed)
	$(ANCHOR) thresholds

gate: ## THE REGRESSION GATE — fail if a metric breaches eval/thresholds.yaml
	$(ANCHOR) gate

readme: ## Rewrite the README metric table from the latest run
	$(PY) scripts/update_readme.py --write

readme-check: ## Fail if the committed README table is stale
	$(PY) scripts/update_readme.py --check

# ---- the reviewer path ------------------------------------------------------

demo: ## ingest + one answer + the full eval. The 60-second tour.
	$(MAKE) ingest
	$(MAKE) ask
	$(MAKE) eval

clean: ## Remove the index, caches and generated report JSON
	rm -rf data
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -f eval/reports/*.json eval/reports/*.jsonl
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +

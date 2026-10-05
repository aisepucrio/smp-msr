# Pipeline driver for the Reddit corpus + practitioner survey study.
# Run `make help` for the target list.

PY      := python3
CORPUS  := $(shell ls -t data/*_corpus.csv 2>/dev/null | head -1)
MOCK    := data/mock/mock_corpus.csv
ROBERTA := data/roberta_sentiment.jsonl

.DEFAULT_GOAL := help
.PHONY: help install smoke verify corpus figures catalog roberta survey lint clean clean-mock all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	$(PY) -m pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Verification (no real data needed)
# ---------------------------------------------------------------------------
smoke: ## End-to-end run of both analysis scripts against synthetic data
	@mkdir -p data/mock
	$(PY) scripts/generate_mock_data.py --out $(MOCK) --roberta --survey
	$(PY) scripts/analyze_corpus.py \
		--corpus $(MOCK) \
		--out data/mock/figures \
		--roberta data/mock/mock_roberta_sentiment.jsonl \
		--curated-catalog data/mock/_absent.csv \
		--full-catalog data/mock/_absent.csv \
		--metrics-csv data/mock/per_metric.csv
	$(PY) scripts/analyze_survey.py \
		--csv data/mock/mock_survey_results.csv \
		--out data/mock/figures
	@echo "smoke test OK (synthetic data: results are meaningless by design)"

verify: ## Byte-compile + lint every script
	$(PY) -m py_compile scripts/*.py
	$(PY) -m pyflakes scripts/*.py
	@echo "verify OK"

# ---------------------------------------------------------------------------
# Real pipeline (needs the dumps; see README)
# ---------------------------------------------------------------------------
download: ## Fetch the five subreddit dumps (~206 MB submissions, ~1.5 GB with comments)
	$(PY) scripts/download_offline.py --comments

corpus: ## Build data/<timestamp>_corpus.csv from the dumps
	@test -n "$(CORPUS)" || { echo "no data/*_corpus.csv yet — run 'make download corpus'"; exit 1; }
	@echo "using corpus: $(CORPUS)"

figures: ## Recompute every paper number and figure from the real corpus
	$(PY) scripts/analyze_corpus.py
	@echo "figures -> docs/figures/"

catalog: ## LLM open extraction of the metric catalog (needs llama-cpp-python)
	$(PY) scripts/llm_catalog.py
	$(PY) scripts/consolidate_catalog.py

roberta: ## Score the framing-matched docs with the second sentiment model
	$(PY) scripts/score_roberta.py

survey: ## Recompute the survey numbers and figures
	$(PY) scripts/analyze_survey.py

all: figures survey ## Regenerate everything that is derived from data/

lint: verify ## Alias for verify

# ---------------------------------------------------------------------------
clean-mock: ## Remove synthetic smoke-test data ONLY (never touches the real corpus)
	rm -rf data/mock

clean: ## Remove derived outputs (figures + generated CSVs); keeps the corpus
	rm -rf docs/figures data/metricas_analise_sentimento.csv

# ---------------------------------------------------------------------------
# Optional: Overleaf figure (only works inside an Overleaf working copy, which
# is gitignored and therefore absent from a plain clone).
# ---------------------------------------------------------------------------
overleaf/workflow.pdf: overleaf/workflow.svg
	rsvg-convert -f pdf -o overleaf/workflow.pdf overleaf/workflow.svg

.PHONY: clean-mock workflow
workflow: overleaf/workflow.pdf ## Rebuild the Overleaf study-design figure

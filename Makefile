# Reproducible entry points for both analysis pipelines.
#
#   make setup     install dependencies
#   make test      run the test suite (no API key needed)
#   make example   run both pipelines end to end on synthetic data (no API key)
#   make lexicon   Pipeline A on your own data
#   make llm       Pipeline B on your own data (requires ANTHROPIC_API_KEY)
#   make all       both pipelines on your own data
#   make clean     delete generated results

PYTHON := python3
RUN := PYTHONPATH=src $(PYTHON) -m adviceaudit

# Override on the command line, e.g.  make lexicon INPUT=data/raw/mydata.csv
INPUT   ?= data/raw/responses.csv
RESULTS ?= results

.PHONY: setup test example lexicon llm all figures clean help

help:
	@grep -E '^#   ' $(MAKEFILE_LIST) | sed 's/^#   //'

setup:
	$(PYTHON) -m pip install -r requirements.txt

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests -q

# --- Demonstration on synthetic data, no credentials required --------------
example:
	$(PYTHON) scripts/make_example_data.py
	$(RUN) count  --input data/example/responses_example.csv \
	              --output $(RESULTS)/example/counts.csv
	$(RUN) fisher --input $(RESULTS)/example/counts.csv \
	              --output $(RESULTS)/example/fisher_results.csv \
	              --excel-output $(RESULTS)/example/fisher_results.xlsx
	$(RUN) annotate --input data/example/responses_example.csv \
	                --output $(RESULTS)/example/annotated.csv --mock
	$(RUN) ordinal  --input $(RESULTS)/example/annotated.csv \
	                --output $(RESULTS)/example/ordinal_results.csv \
	                --distribution-output $(RESULTS)/example/score_distribution.csv \
	                --excel-output $(RESULTS)/example/ordinal_results.xlsx
	$(RUN) figures  --input $(RESULTS)/example/annotated.csv \
	                --output-dir $(RESULTS)/example/figures

# --- Pipeline A: lexicon keyword counts -> Fisher's exact ------------------
lexicon:
	$(RUN) count  --input $(INPUT) --output $(RESULTS)/lexicon/counts.csv
	$(RUN) fisher --input $(RESULTS)/lexicon/counts.csv \
	              --output $(RESULTS)/lexicon/fisher_results.csv \
	              --excel-output $(RESULTS)/lexicon/fisher_results.xlsx

# --- Pipeline B: LLM ordinal annotation -> Mann-Whitney U ------------------
llm:
	$(RUN) annotate --input $(INPUT) --output $(RESULTS)/llm/annotated.csv
	$(RUN) ordinal  --input $(RESULTS)/llm/annotated.csv \
	                --output $(RESULTS)/llm/ordinal_results.csv \
	                --distribution-output $(RESULTS)/llm/score_distribution.csv \
	                --excel-output $(RESULTS)/llm/ordinal_results.xlsx

figures:
	$(RUN) figures --input $(RESULTS)/llm/annotated.csv \
	               --output-dir $(RESULTS)/llm/figures

all: lexicon llm figures

clean:
	rm -rf $(RESULTS)/lexicon $(RESULTS)/llm $(RESULTS)/example
	@echo "Removed generated results. The annotation cache under \
$(RESULTS)/cache was kept; delete it manually to force re-annotation."

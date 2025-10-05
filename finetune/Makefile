SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.DEFAULT_GOAL := help

help:
	@echo "Targets: venv, install, train, infer, eval, clean"
	@echo "Examples:"
	@echo "  make install"
	@echo "  make train CFG=configs/causal-lm-lora-4bit.yaml"
	@echo "  make infer MODEL=/path/to/base MODEL ADAPTER=adapters/my-run PROMPT='Hello'"
	@echo "  make eval MODEL=/path/to/base MODEL ADAPTER=adapters/my-run FILE=data/samples/eval.jsonl"

venv:
	python3 -m venv $(VENV)

install: venv
	"$(PIP)" install -U pip
	"$(PIP)" install -r requirements.txt

train:
	@if [ -z "$(CFG)" ]; then echo "Usage: make train CFG=configs/<file>.yaml"; exit 1; fi
	"$(PY)" scripts/train.py --config "$(CFG)"

infer:
	@if [ -z "$(MODEL)" ]; then echo "Usage: make infer MODEL=/path/to/base [ADAPTER=path] PROMPT='...'"; exit 1; fi
	"$(PY)" scripts/infer.py --base_model_path "$(MODEL)" $(if $(ADAPTER),--adapter_path "$(ADAPTER)",) $(if $(PROMPT),--prompt "$(PROMPT)",) $(if $(PROMPT_FILE),--prompt_file "$(PROMPT_FILE)",)

# FILE is eval dataset jsonl path
# MODEL and optional ADAPTER as above
eval:
	@if [ -z "$(MODEL)" ] || [ -z "$(FILE)" ]; then echo "Usage: make eval MODEL=/path/to/base [ADAPTER=path] FILE=data/samples/eval.jsonl"; exit 1; fi
	"$(PY)" scripts/eval_perplexity.py --base_model_path "$(MODEL)" $(if $(ADAPTER),--adapter_path "$(ADAPTER)",) --eval_file "$(FILE)"

clean:
	rm -rf outputs/.tmp __pycache__

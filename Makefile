PYTHON ?= python

init:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt

ingest:
	$(PYTHON) scripts/ingest.py

parse:
	$(PYTHON) scripts/parse_pdfs.py

index:
	$(PYTHON) scripts/build_index.py

dataset:
	$(PYTHON) scripts/build_dataset.py
	
dataset_small:
	$(PYTHON) scripts/build_dataset_pdf_small.py

finetune:
	$(PYTHON) scripts/finetune.py

train:
	bash scripts/run_train.sh

eval:
	$(PYTHON) scripts/evaluate.py

serve:
	bash scripts/serve_api.sh

query:
	$(PYTHON) scripts/query.py

test:
	pytest -q

.PHONY: init ingest parse index dataset dataset_small finetune train eval serve query test

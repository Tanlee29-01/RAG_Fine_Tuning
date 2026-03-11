PYTHON ?= python

init:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt

parse:
	$(PYTHON) scripts/parse_pdfs.py

index:
	$(PYTHON) scripts/build_index.py

dataset:
	$(PYTHON) scripts/build_dataset.py
	
dataset_small:
	$(PYTHON) scripts/build_dataset_pdf_small.py

train:
	bash scripts/run_train.sh

eval:
	bash scripts/run_eval.sh

serve:
	bash scripts/serve_api.sh

test:
	pytest -q

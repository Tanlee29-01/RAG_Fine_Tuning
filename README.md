# {{ project_name }}

Production-oriented template for **AI + RAG + Finetuning** workflows.

## Features

- PDF ingestion and parsing pipeline
- Cleaning, section splitting, metadata building
- Hybrid retrieval (dense + {% if use_bm25 %}BM25{% else %}optional lexical{% endif %})
- QLoRA/LoRA training pipeline for `{{ base_model }}`
- Evaluation for retrieval, generation, schema validation
- FastAPI service for chat and document operations
- Config-driven project layout

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
make init
```

## Suggested flow

1. Put PDFs into `data/raw/`
2. Parse and clean documents
3. Build chunks and indexes
4. Create SFT / eval datasets
5. Train adapters
6. Run evaluation
7. Serve API

## Generate this project from Copier

```bash
pip install copier
copier copy <path-or-git-url-to-template> {{ project_slug }}
```

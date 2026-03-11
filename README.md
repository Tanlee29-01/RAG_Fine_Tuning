# RAG Fine-Tuning

Production-oriented project for **Retrieval-Augmented Generation (RAG)** combined with **QLoRA fine-tuning** of large language models.

## Architecture

```
RAG_Fine_Tuning/
├── configs/                # YAML configuration files
│   ├── app.yaml            # Application and API settings
│   ├── model.yaml          # Embedding and generator model settings
│   ├── rag.yaml            # Retrieval and context-building settings
│   ├── train.yaml          # Fine-tuning hyperparameters
│   ├── eval.yaml           # Evaluation metrics and judge model
│   └── schema.yaml         # JSON schema file paths
├── data/
│   ├── raw/                # Input PDF documents
│   ├── datasets/           # Processed JSONL datasets (sft_train, sft_val, eval_qa)
│   ├── embeddings/         # Cached embeddings
│   └── artifacts/          # Build logs and error records
├── schemas/                # JSON Schema definitions for data validation
├── scripts/                # Operational entry points
│   ├── ingest.py           # Parse and clean PDFs
│   ├── build_index.py      # Build FAISS vector index
│   ├── build_dataset.py    # Generate SFT training data via LLM
│   ├── query.py            # Interactive retrieval demo
│   ├── finetune.py         # Run QLoRA fine-tuning
│   ├── evaluate.py         # Run evaluation benchmark
│   ├── run_train.sh        # Shell wrapper for training
│   ├── run_eval.sh         # Shell wrapper for evaluation
│   └── serve_api.sh        # Start the FastAPI server
├── src/                    # Main Python package
│   ├── ingestion/          # PDF loading and parsing
│   ├── preprocessing/      # Text cleaning and section splitting
│   ├── chunking/           # Document chunking strategies
│   ├── embeddings/         # Embedding model interface
│   ├── indexing/           # Index pipeline (BM25 + vector store)
│   ├── retrieval/          # Hybrid retrieval, reranking, context building
│   ├── generation/         # LLM generation, prompt building, response validation
│   ├── finetuning/         # QLoRA training pipeline and dataset builder
│   ├── evaluation/         # Retrieval and generation benchmarking
│   ├── api/                # FastAPI application and routers
│   └── utils/              # Config loading, logging, metrics, I/O helpers
├── tests/                  # Unit and integration tests (organized by subsystem)
├── .env.example            # Environment variable template
├── Makefile                # Developer workflow shortcuts
├── docker-compose.yml      # Container orchestration
├── pyproject.toml          # Project metadata and build config
└── requirements.txt        # Python dependencies
```

## Features

- **Document ingestion**: Load and parse PDF documents with metadata extraction
- **Preprocessing**: Text cleaning, normalization, and section splitting
- **Hybrid retrieval**: Dense vector search (FAISS) + BM25 lexical search
- **Reranking**: Cross-encoder reranking for improved precision
- **QLoRA fine-tuning**: Fine-tune Qwen2.5 or other causal LLMs on custom Q&A data
- **Structured generation**: JSON-schema-validated answers with citations
- **Evaluation**: Recall@k, MRR, and LLM-judge-based generation quality metrics
- **FastAPI service**: REST endpoints for chat, document management, and training

## Quickstart

### 1. Setup

```bash
git clone https://github.com/Tanlee29-01/RAG_Fine_Tuning.git
cd RAG_Fine_Tuning
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys and model settings
```

### 2. Ingest documents

```bash
# Place PDF files in data/raw/
make ingest      # Parse and clean documents
make index       # Build FAISS vector index
```

### 3. Query interactively

```bash
make query
# Type a question and see retrieved chunks
```

### 4. Create training dataset

```bash
# Requires GITHUB_TOKEN or OPENAI_API_KEY in .env
make dataset
```

### 5. Fine-tune the model

```bash
# Requires GENERATOR_MODEL and either LOCAL_MODEL_DIR or ALLOW_HF_DOWNLOAD=1 in .env
make finetune
```

### 6. Evaluate

```bash
make eval
```

### 7. Serve the API

```bash
make serve
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## Key Workflows

### RAG Pipeline

```
data/raw/*.pdf
  └─ ingestion (pdf_loader → parser)
  └─ preprocessing (cleaner → section_splitter)
  └─ chunking (RecursiveCharacterTextSplitter)
  └─ embeddings (sentence-transformers)
  └─ indexing (FAISS + BM25)
  └─ retrieval (hybrid → reranker → context_builder)
  └─ generation (prompt_builder → LLM → response_validator)
  └─ API response with answer + citations
```

### Fine-Tuning Pipeline

```
data/raw/*.pdf
  └─ build_dataset.py (LLM-generated Q&A pairs)
  └─ data/datasets/sft_train.jsonl
  └─ finetuning/train_qlora.py (QLoRA + SFTTrainer)
  └─ models/adapters/<adapter-name>/
  └─ (optional) merge_adapter.py → models/merged/
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/chat/` | Ask a question (RAG pipeline) |
| GET | `/documents/` | List indexed documents |
| POST | `/train/` | Trigger fine-tuning job |

### Example chat request

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

```json
{
  "answer": "The document covers ...",
  "citations": [
    {
      "doc_id": "my-document",
      "chunk_id": "my-document_p5_c2",
      "quote": "exact quote from the document",
      "page": 5
    }
  ]
}
```

## Configuration

All parameters are configured through YAML files in `configs/` and environment variables in `.env`.

| Config file | Purpose |
|-------------|---------|
| `configs/app.yaml` | App name, environment, log level, API host/port |
| `configs/model.yaml` | Embedding model, generator model, quantization |
| `configs/rag.yaml` | Retrieval top-k, hybrid search, reranker, context limits |
| `configs/train.yaml` | Training epochs, batch size, LoRA rank/alpha/dropout |
| `configs/eval.yaml` | Recall@k values, MRR, judge model |

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Description |
|----------|-------------|
| `EMBEDDING_MODEL` | HuggingFace embedding model name |
| `GENERATOR_MODEL` | HuggingFace generator model ID |
| `LOCAL_MODEL_DIR` | Local path to downloaded model (optional) |
| `ALLOW_HF_DOWNLOAD` | Set to `1` to auto-download models from HuggingFace |
| `HF_TOKEN` | HuggingFace access token |
| `OPENAI_API_KEY` | OpenAI API key (for evaluation judge) |
| `GITHUB_TOKEN` | GitHub token (for GitHub Models inference) |

## Development

```bash
make test          # Run pytest
make ingest        # Parse PDFs
make index         # Build index
make dataset       # Create training data
make finetune      # Fine-tune model
make eval          # Evaluate
make serve         # Start API server
make query         # Interactive query demo
```

## Docker

```bash
docker compose up
# API available at http://localhost:8000
```

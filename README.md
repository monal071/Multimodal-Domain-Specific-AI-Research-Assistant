# Domain-Specific AI Research Assistant

A domain-specific research assistant implementing a high-performance Retrieval-Augmented Generation (RAG) pipeline to query scientific literature. The system runs fully locally — embeddings, reranking, and generation all run on your machine with no external API calls.

---

## Key Features

1. **Intelligent PDF Ingestion**
   - Uses **Docling** for structured layout extraction, table parsing, and visual structure mapping.
   - Implements a custom heading-aware text chunker with conditional sliding-window overlaps.

2. **Advanced Hybrid Retrieval**
   - **Dense Retrieval**: Semantic search powered by `BAAI/bge-large-en-v1.5` embeddings stored in **ChromaDB**.
   - **Sparse Retrieval**: Keyword matching powered by the **BM25Okapi** algorithm.
   - **Rank Fusion**: Combines dense and sparse candidates using **Reciprocal Rank Fusion (RRF)**.

3. **Two-Stage Retrieval & Optimization**
   - **Query Rewriting (HyDE)**: Uses the base model to transform conversational queries into precise academic search terms before retrieval.
   - **Reranking**: Scores and filters top results using a cross-encoder (`BAAI/bge-reranker-v2-m3`).
   - **Context Expansion**: Optional adjacent-chunk lookup using document structural indices.

4. **Local Generation via Ollama**
   - Answer generation using `qwen3:8b` (or any Ollama model) hosted locally.
   - Real-time token streaming to the web interface.
   - Conversational memory window to resolve multi-turn references (e.g., "that", "this method").

---

## Directory Structure

```text
├── DATA/                           # Excluded from version control
│   ├── raw data/
│   │   └── papers/                 # Place source PDF research papers here
│   ├── PARSED DATA/                # JSONL outputs from ingestion pipeline
│   └── CHROMADB/                   # ChromaDB persistent vector store
├── src/
│   ├── config.py                   # Centralized settings (paths, models, thresholds)
│   ├── schema.py                   # Shared data classes (RAGResult, RetrievedChunk)
│   ├── pipeline_01_ingest.py       # PDF parser & heading-aware chunker
│   ├── pipeline_02_embed.py        # BGE embedding generator & ChromaDB indexer
│   ├── rag_engine.py               # Core RAG Engine (Hybrid Search + Reranker + Ollama)
│   └── app.py                      # Gradio web UI
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Prerequisites
- **Python 3.11+**
- **NVIDIA GPU** with CUDA (recommended) — CPU fallback is supported
- **Ollama** installed and running: https://ollama.com

### 2. Pull the Base Model
```bash
ollama pull qwen3:8b
```

### 3. Create Environment & Install Dependencies
```bash
conda create -n rag-assistant python=3.11 -y
conda activate rag-assistant
pip install -r requirements.txt
```

---

## Pipeline Execution

### Step 1: Ingest PDFs
Place your research papers (`.pdf`) into `DATA/raw data/papers/`, then run:
```bash
python src/pipeline_01_ingest.py
```
Parses PDFs with Docling, chunks text with heading-aware splitter, and writes `.jsonl` to `DATA/PARSED DATA/`.

### Step 2: Build Vector Index
```bash
python src/pipeline_02_embed.py
```
Generates BGE embeddings on CUDA, stores vectors in ChromaDB (`DATA/CHROMADB/`), and builds the BM25 corpus.

### Step 3: Run the Web UI
```bash
python src/app.py
```
Opens the Gradio chat interface at `http://localhost:7861`.

### Step 3 (alternative): CLI Mode
```bash
python src/rag_engine.py
```
Interactive terminal. Commands: `exit` | `clear` | prefix `norewrite` to skip HyDE query rewriting.

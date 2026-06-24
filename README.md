# Multimodal Domain-Specific AI Research Assistant (RAG + LoRA Fine-Tuning)

A domain-specific research assistant implementing a high-performance Retrieval-Augmented Generation (RAG) pipeline to query scientific literature. The system runs local models for text embeddings, cross-encoder reranking, and generation with 4-bit quantization.

---

## Key Features

1. **Intelligent PDF Ingestion**:
   - Uses **Docling** for structured layout extraction, table parsing, and visual structure mapping.
   - Splits documents into page-based batches to manage memory overhead for long papers.
   - Implements a custom heading-aware text chunker with conditional sliding-window overlaps.

2. **Advanced Hybrid Retrieval**:
   - **Dense Retrieval**: Semantic search powered by `BAAI/bge-large-en-v1.5` embeddings and a **FAISS IVFFlat** index.
   - **Sparse Retrieval**: Keyword matching powered by the **BM25Okapi** algorithm.
   - **Rank Fusion**: Combines dense and sparse candidates using **Reciprocal Rank Fusion (RRF)**.

3. **Two-Stage Retrieval & Optimization**:
   - **Query Rewriting**: Uses the generation model to transform conversational queries into precise academic search queries.
   - **Reranking**: Scores and filters top results using a cross-encoder model (`BAAI/bge-reranker-v2-m3`).
   - **Context Expansion**: Optional adjacent-chunk lookup using document structural indices.

4. **Quantized Local Generation**:
   - Implements a conversational answer engine using **DeepSeek-R1-Distill-Qwen-8B** quantized to 4-bit precision (`bitsandbytes` `nf4`).
   - Streams generation directly to the console in real-time using HuggingFace `TextStreamer`.
   - Conversational memory window to resolve multi-turn context (e.g., "that", "this method").
   - VRAM-optimized design (Reranker mapped to CPU to prevent out-of-memory errors on 8GB GPUs).

---

## Directory Structure

```text
├── DATA/                               # Excluded from version control
│   ├── raw data/
│   │   └── papers/                     # Put source PDF research papers here
│   ├── PARSED DATA/                    # JSONL outputs from ingestion
│   ├── INDEX/                          # FAISS index and metadata pickle files
│   └── faiss_index/                    # Alternative/legacy FAISS indexes
├── models/                             # Local weights folder (e.g., DeepSeek-R1-Qwen)
├── src/
│   ├── config.py                       # Centralized global configurations and paths
│   ├── models.py                       # Shared data classes (RAGResult, RetrievedChunk)
│   ├── ingestion.py                    # PDF parser & custom chunker
│   ├── index0.py                       # Embedding generator & FAISS index builder
│   └── index.py                        # Conversational query engine with hybrid retrieval
├── .gitignore                          # Exclusions for large models, indexes, and environments
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## Setup & Installation

### 1. Prerequisites
- **Python 3.11** is recommended for library compatibility (especially PyTorch and CUDA bindings).
- **NVIDIA GPU** with CUDA support and at least 8GB of VRAM.

### 2. Environment Setup
Create and activate a virtual environment (using Conda is recommended):
```bash
conda create -n env1 python=3.11 -y
conda activate env1
```

### 3. Install Dependencies
Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Pipeline Execution

### Step 1: Ingestion (PDF to Chunks)
Place your target research papers (`.pdf`) into the `DATA/raw data/papers` directory, then execute the ingestion script:
```bash
python src/ingestion.py
```
This splits and parses the PDFs into structured markdown, processes tables, chunks the text, and writes `.jsonl` files into `DATA/PARSED DATA`.

### Step 2: Index Building
Generate embeddings for the parsed document chunks and build the search index:
```bash
python src/index0.py
```
This script runs BGE embedding on CUDA, creates a FAISS index, builds a BM25 lookup database, and saves the retrieval assets to `DATA/INDEX/`.

### Step 3: Run the Query Assistant CLI
Launch the interactive CLI interface to chat with your document repository:
```bash
python src/index.py
```
- Type your question at the `>>>` prompt.
- Prefix your question with `norewrite ` to bypass the DeepSeek query-rewriting step.
- Type `clear` to reset conversational memory.
- Type `exit` to exit the session.

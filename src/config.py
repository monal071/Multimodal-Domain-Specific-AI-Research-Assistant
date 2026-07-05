from pathlib import Path

BASE_DIR = Path(r"D:\projects\Multimodal Domain-Specific AI Research Assistant (RAG + LoRA Fine-Tuning)")

# ── Data Directories ──────────────────────────────────────────────────────────
PDF_DIR    = BASE_DIR / "DATA" / "raw data" / "papers"
PARSED_DIR = BASE_DIR / "DATA" / "PARSED DATA"
INDEX_DIR  = BASE_DIR / "DATA" / "INDEX"
# MODEL_PATH is no longer needed — Ollama manages the model file

# ── Models ────────────────────────────────────────────────────────────────────
EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# ── Indexing Settings ─────────────────────────────────────────────────────────
EMBED_DIM   = 1024
EMBED_BATCH = 16
NPROBE      = 32

# ── Query Engine Settings ─────────────────────────────────────────────────────
EMBED_DEVICE  = "cpu"
RERANK_DEVICE = "cpu"

FAISS_TOP_K    = 50
RERANK_TOP_N   = 4
CONTEXT_WINDOW = 0

# ── Ollama Settings ──────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"   # default Ollama address
OLLAMA_MODEL = "deepseek-r1:8b"           # model tag as shown in `ollama list`
OLLAMA_TIMEOUT = 300                      # seconds; raise if you use a big model

MAX_NEW_TOKENS = 2048
MAX_HISTORY    = 4

# ── Fine-Tuning Settings ──────────────────────────────────────────────────────
FT_DATASET_PATH = BASE_DIR / "DATA" / "finetune" / "synthetic_qa_dataset.json"
FT_OUTPUT_DIR   = BASE_DIR / "DATA" / "models" / "lora-adapter"
FT_MODEL_ID     = "meta-llama/Meta-Llama-3-8B-Instruct"
FT_BATCH_SIZE   = 2
FT_EPOCHS       = 3

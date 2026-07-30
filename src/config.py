from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data Directories ──────────────────────────────────────────────────────────
PDF_DIR    = BASE_DIR / "DATA" / "raw data" / "papers"
PARSED_DIR = BASE_DIR / "DATA" / "PARSED DATA"
CHROMA_DIR       = BASE_DIR / "DATA" / "CHROMADB"
CHROMA_COLLECTION = "research_papers"

# ── Models ────────────────────────────────────────────────────────────────────
EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# ── Indexing Settings ─────────────────────────────────────────────────────────
EMBED_DIM   = 1024
EMBED_BATCH = 16
CHROMA_BATCH = 500   # upsert batch size for ChromaDB

# ── Query Engine Settings ─────────────────────────────────────────────────────
EMBED_DEVICE  = "cuda"
RERANK_DEVICE = "cuda"

RETRIEVAL_TOP_K = 50   # candidates fetched before reranking
RERANK_TOP_N    = 4
CONTEXT_WINDOW  = 0

# ── Ollama Settings ──────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"   # default Ollama address
OLLAMA_MODEL = "qwen3:8b"                 # model tag as shown in `ollama list`
OLLAMA_TIMEOUT = 300                      # seconds; raise if you use a big model

MAX_NEW_TOKENS = 2048
MAX_HISTORY    = 4

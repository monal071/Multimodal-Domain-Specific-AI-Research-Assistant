from pathlib import Path

BASE_DIR = Path(r"D:\projects\Multimodal Domain-Specific AI Research Assistant (RAG + LoRA Fine-Tuning)")

# ── Data Directories ──────────────────────────────────────────────────────────
PDF_DIR    = BASE_DIR / "DATA" / "raw data" / "papers"
PARSED_DIR = BASE_DIR / "DATA" / "PARSED DATA"
INDEX_DIR        = BASE_DIR / "DATA" / "INDEX"   # kept for legacy reference
CHROMA_DIR       = BASE_DIR / "DATA" / "CHROMADB"
CHROMA_COLLECTION = "research_papers"
# MODEL_PATH is no longer needed — Ollama manages the model file

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
OLLAMA_MODEL = "qwen2.5:7b-instruct"      # model tag as shown in `ollama list`
OLLAMA_TIMEOUT = 300                      # seconds; raise if you use a big model

MAX_NEW_TOKENS = 2048
MAX_HISTORY    = 4

# ── Fine-Tuning Settings (Unsloth LoRA) ──────────────────────────────────────
# Base model — use an unsloth/* variant for 4-bit pre-quantized weights.
# Options:
#   "unsloth/Qwen2.5-7B-Instruct"                ← ACTIVE (strong reasoning & fast RAG)
#   "unsloth/Meta-Llama-3.1-8B-Instruct"         (strong general instruction following)
#   "unsloth/mistral-7b-instruct-v0.3"           (lightweight)
FT_MODEL_ID     = "unsloth/Qwen2.5-7B-Instruct"

# Dataset paths
FT_DATASET_PATH = BASE_DIR / "DATA" / "finetune" / "synthetic_qa_dataset.jsonl"
FT_OUTPUT_DIR   = BASE_DIR / "DATA" / "models" / "lora-adapter"
FT_GGUF_DIR     = BASE_DIR / "DATA" / "models" / "gguf"

# LoRA hyperparameters (Unsloth-optimised defaults)
FT_LORA_R           = 16          # rank; 8 for lighter, 32 for higher capacity
FT_LORA_ALPHA       = 32          # typically 2 × r
FT_LORA_DROPOUT     = 0           # 0 is Unsloth-recommended
FT_TARGET_MODULES   = [           # layers to apply LoRA to
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training hyperparameters
FT_MAX_SEQ_LEN      = 2048        # max token length per sample
FT_LOAD_IN_4BIT     = True        # 4-bit QLoRA (halves VRAM requirement)
FT_BATCH_SIZE       = 2           # per-device batch size
FT_GRAD_ACCUM       = 4           # effective batch = FT_BATCH_SIZE × FT_GRAD_ACCUM = 8
FT_EPOCHS           = 3
FT_LR               = 2e-4
FT_WARMUP_STEPS     = 10
FT_WEIGHT_DECAY     = 0.01
FT_LR_SCHEDULER     = "linear"

# Output options
FT_EXPORT_GGUF      = True        # export GGUF so it can be re-imported into Ollama as deepseek-r1:8b
FT_GGUF_QUANT       = "q4_k_m"   # quantisation method for GGUF export

# Dataset generation settings (pipeline_03a_gen_dataset.py)
# Generation backend priority: groq → ollama → template
FT_GEN_MIN_CHARS    = 300         # skip chunks shorter than this
FT_GEN_MAX_CHUNKS   = 5000        # cap to avoid runaway generation
FT_GEN_USE_OLLAMA   = True        # fallback if Groq is unavailable
FT_GEN_QUESTIONS_PER_CHUNK = 1    # number of QA pairs to generate per chunk

# ── Groq API Settings (dataset generation) ───────────────────────────────────
# Get a free API key at https://console.groq.com
# Key is read from the GROQ_API_KEY environment variable (set in .env)
GROQ_API_KEY_ENV    = "GROQ_API_KEY"   # env-var name (do NOT hardcode the key here)
GROQ_MODEL          = "llama-3.3-70b-versatile"  # or: "deepseek-r1-distill-llama-70b"
GROQ_BASE_URL       = "https://api.groq.com/openai/v1"
GROQ_TIMEOUT        = 60           # seconds per API call
GROQ_MAX_TOKENS     = 512          # max tokens in generated answer
GROQ_TEMPERATURE    = 0.7

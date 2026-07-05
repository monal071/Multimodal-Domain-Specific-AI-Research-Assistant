"""
RAG Pipeline — Part 1: Index Builder
  - Reads all .jsonl files from PARSED DATA directory
  - Encodes chunks with BGE-M3 (best for research text)
  - Builds a FAISS IVFFlat index with cosine similarity
  - Saves index + metadata to disk
"""

import json
import pickle
import time
import numpy as np
from pathlib import Path
import os
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import PARSED_DIR, INDEX_DIR, EMBED_MODEL, EMBED_DIM, EMBED_BATCH, NPROBE, EMBED_DEVICE

# FAISS: IVFFlat — fast approximate search
# nlist = number of Voronoi cells; sqrt(N) is a good heuristic
# We'll set it after counting total chunks

INDEX_DIR.mkdir(exist_ok=True, parents=True)

# ── load all chunks ────────────────────────────────────────────────────────────
print("Loading chunks from JSONL files …")
all_chunks: list[dict] = []

for jsonl_path in sorted(PARSED_DIR.glob("*.jsonl")):
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_chunks.append(json.loads(line))

print(f"  Loaded {len(all_chunks):,} chunks from {len(list(PARSED_DIR.glob('*.jsonl')))} files\n")

if not all_chunks:
    raise RuntimeError("No chunks found — run the PDF chunker first.")

# ── embed ──────────────────────────────────────────────────────────────────────
print(f"Loading embedding model: {EMBED_MODEL} …")
embedder = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)

# BGE models work best with a query prefix; for indexing use passage prefix
texts = [
    f"passage: {c['text']}" for c in all_chunks
]

print(f"Embedding {len(texts):,} passages (batch={EMBED_BATCH}) …")
t0 = time.time()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
embeddings = embedder.encode(
    texts,
    batch_size=EMBED_BATCH,
    normalize_embeddings=True,       # cosine sim = dot product after L2 norm
    show_progress_bar=True,
    convert_to_numpy=True,
).astype("float32")

print(f"  Done in {time.time()-t0:.1f}s  |  shape: {embeddings.shape}\n")

# ── build FAISS index ──────────────────────────────────────────────────────────
N      = len(all_chunks)
nlist  = max(64, min(4096, int(N ** 0.5)))   # sqrt heuristic, clamped

print(f"Building FAISS IVFFlat index  (N={N:,}, nlist={nlist}, dim={EMBED_DIM}) …")

# Inner product on L2-normed vectors == cosine similarity
quantizer = faiss.IndexFlatIP(EMBED_DIM)
index     = faiss.IndexIVFFlat(quantizer, EMBED_DIM, nlist, faiss.METRIC_INNER_PRODUCT)

# IVF must be trained before adding vectors
print("  Training index …")
index.train(embeddings)
print("  Adding vectors …")
index.add(embeddings)
index.nprobe = NPROBE

print(f"  Index total vectors: {index.ntotal:,}\n")

# ── save to disk ───────────────────────────────────────────────────────────────
index_path    = INDEX_DIR / "faiss.index"
metadata_path = INDEX_DIR / "metadata.pkl"

faiss.write_index(index, str(index_path))
print(f"FAISS index saved → {index_path}")

# Store only what we need for retrieval (no full text duplication of embeddings)
metadata = [
    {
        "chunk_id":     c["chunk_id"],
        "doc_id":       c["doc_id"],
        "source_file":  c["source_file"],
        "chunk_index":  c["chunk_index"],
        "section_path": c.get("section_path", []),
        "heading":      c.get("heading"),
        "text":         c["text"],          # kept for context assembly
        "has_table":    c.get("has_table", False),
        "has_formula":  c.get("has_formula", False),
        "has_code":     c.get("has_code", False),
        "prev_chunk_id": c.get("prev_chunk_id"),
        "next_chunk_id": c.get("next_chunk_id"),
    }
    for c in all_chunks
]

with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)
print(f"Metadata saved    → {metadata_path}")

# Also build a chunk_id → index lookup for neighbour expansion
id_to_idx = {c["chunk_id"]: i for i, c in enumerate(metadata)}
with open(INDEX_DIR / "id_to_idx.pkl", "wb") as f:
    pickle.dump(id_to_idx, f)
print(f"ID→idx map saved  → {INDEX_DIR / 'id_to_idx.pkl'}\n")

print("Index build complete ✓")

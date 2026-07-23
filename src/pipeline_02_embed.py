"""
RAG Pipeline - Part 2: ChromaDB Index Builder
  - Reads all .jsonl files from PARSED DATA directory
  - Encodes chunks with BGE-Large (best for research text)
  - Upserts embeddings + metadata into a ChromaDB persistent collection
  - Idempotent: re-running skips already-indexed chunk IDs
"""

import json
import time
import os
import numpy as np
from pathlib import Path

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    PARSED_DIR, CHROMA_DIR, CHROMA_COLLECTION,
    EMBED_MODEL, EMBED_DIM, EMBED_BATCH, CHROMA_BATCH,
)

# Auto-detect device
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {_DEVICE}")

# -- setup ChromaDB -----------------------------------------------------------
CHROMA_DIR.mkdir(exist_ok=True, parents=True)

print(f"Opening ChromaDB at: {CHROMA_DIR}")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},   # cosine similarity via HNSW
)
print(f"  Collection '{CHROMA_COLLECTION}'  |  vectors already stored: {collection.count():,}\n")

# -- load all chunks ----------------------------------------------------------
print("Loading chunks from JSONL files ...")
all_chunks = []

for jsonl_path in sorted(PARSED_DIR.glob("*.jsonl")):
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_chunks.append(json.loads(line))

n_files = len(list(PARSED_DIR.glob("*.jsonl")))
print(f"  Loaded {len(all_chunks):,} chunks from {n_files} files\n")

if not all_chunks:
    raise RuntimeError("No chunks found. Run pipeline_01_ingest.py first.")

# -- check which chunks are already indexed -----------------------------------
print("Checking for already-indexed chunks ...")
existing = collection.get(include=[])
existing_ids = set(existing["ids"])
n_new = len(all_chunks) - len(existing_ids)
print(f"  Already in DB: {len(existing_ids):,}  |  New to index: {n_new:,}\n")

new_chunks = [c for c in all_chunks if c["chunk_id"] not in existing_ids]

if not new_chunks:
    print("All chunks already indexed. Nothing to do.")
    exit(0)

# -- embed --------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print(f"Loading embedding model: {EMBED_MODEL} ...")
embedder = SentenceTransformer(EMBED_MODEL, device=_DEVICE)

texts = ["passage: " + c["text"] for c in new_chunks]

print(f"Embedding {len(texts):,} new passages (batch={EMBED_BATCH}) ...")
t0 = time.time()

embeddings = embedder.encode(
    texts,
    batch_size=EMBED_BATCH,
    normalize_embeddings=True,       # cosine sim = dot product after L2 norm
    show_progress_bar=True,
    convert_to_numpy=True,
).astype("float32")

print(f"  Done in {time.time()-t0:.1f}s  |  shape: {embeddings.shape}\n")

# -- upsert into ChromaDB -----------------------------------------------------
print(f"Upserting {len(new_chunks):,} chunks into ChromaDB (batch={CHROMA_BATCH}) ...")
t0 = time.time()

for batch_start in tqdm(range(0, len(new_chunks), CHROMA_BATCH)):
    batch_end        = batch_start + CHROMA_BATCH
    batch_chunks     = new_chunks[batch_start:batch_end]
    batch_embeddings = embeddings[batch_start:batch_end]

    ids       = [c["chunk_id"] for c in batch_chunks]
    documents = [c["text"]     for c in batch_chunks]
    metadatas = []
    for c in batch_chunks:
        section_list = c.get("section_path") or []
        metadatas.append({
            "doc_id":        c["doc_id"],
            "source_file":   c["source_file"],
            "chunk_index":   int(c["chunk_index"]),
            "heading":       c.get("heading") or "",
            "section_path":  " > ".join(section_list),
            "has_table":     bool(c.get("has_table",   False)),
            "has_formula":   bool(c.get("has_formula", False)),
            "has_code":      bool(c.get("has_code",    False)),
            "prev_chunk_id": c.get("prev_chunk_id") or "",
            "next_chunk_id": c.get("next_chunk_id") or "",
        })

    collection.upsert(
        ids=ids,
        embeddings=batch_embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

print(f"\n  Done in {time.time()-t0:.1f}s")
print(f"  Total vectors in collection: {collection.count():,}\n")
print("ChromaDB index build complete.")

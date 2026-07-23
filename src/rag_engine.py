"""
RAG Pipeline — Query Engine with Hybrid Retrieval
  - BM25 + ChromaDB with Reciprocal Rank Fusion
  - BGE embeddings on CUDA/CPU, DeepSeek via Ollama (llama.cpp)
  - Reranker, query rewriting (HyDE), conversation memory
"""

import json
import os
import string
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    CHROMA_DIR, CHROMA_COLLECTION,
    EMBED_MODEL, RERANK_MODEL, EMBED_DIM,
    EMBED_DEVICE, RERANK_DEVICE, RETRIEVAL_TOP_K, RERANK_TOP_N, CONTEXT_WINDOW,
    MAX_NEW_TOKENS, MAX_HISTORY,
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
)
from schema import RetrievedChunk, RAGResult


# ══════════════════════════════════════════════════════════════════════════════
#  BM25 INDEX
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation + split. BM25 needs real words not subwords."""
    text   = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if len(t) > 1]


class BM25Index:
    def __init__(self, metadata: list[dict]):
        corpus      = [_tokenize(m["text"]) for m in metadata]
        self._bm25  = BM25Okapi(corpus)
        self._meta  = metadata

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores   = self._bm25.get_scores(tokens)
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(idx, float(scores[idx])) for idx in top_idxs]


# ══════════════════════════════════════════════════════════════════════════════
#  RECIPROCAL RANK FUSION
# ══════════════════════════════════════════════════════════════════════════════

def _chroma_result_to_chunk(chunk_id: str, document: str, metadata: dict, score: float = 0.0) -> RetrievedChunk:
    section_path = [s for s in metadata.get("section_path", "").split(" > ") if s]
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=metadata["doc_id"],
        source_file=metadata["source_file"],
        section_path=section_path,
        heading=metadata.get("heading") or None,
        text=document,
        score=score,
        has_table=bool(metadata.get("has_table", False)),
        has_formula=bool(metadata.get("has_formula", False)),
    )


def reciprocal_rank_fusion(
    chroma_results: list[RetrievedChunk],
    bm25_results:   list[tuple[int, float]],
    bm25_corpus:    list[dict],
    top_n:          int,
    k:              int = 60,
) -> list[RetrievedChunk]:
    """
    Merge ChromaDB semantic + BM25 keyword ranked lists.
    RRF score = 1/(rank+k) summed across retrievers.
    Chunks appearing high in both lists win.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, RetrievedChunk] = {}

    # score semantic (ChromaDB) results
    for rank, chunk in enumerate(chroma_results):
        cid = chunk.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rank + k)
        chunk_map[cid]  = chunk

    # score BM25 results
    for rank, (meta_idx, _) in enumerate(bm25_results):
        m   = bm25_corpus[meta_idx]
        cid = m["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rank + k)
        if cid not in chunk_map:
            section_path = [s for s in m.get("section_path", "").split(" > ") if s]
            chunk_map[cid] = RetrievedChunk(
                chunk_id=cid,
                doc_id=m["doc_id"],
                source_file=m["source_file"],
                section_path=section_path,
                heading=m.get("heading") or None,
                text=m["text"],
                score=0.0,
                has_table=bool(m.get("has_table", False)),
                has_formula=bool(m.get("has_formula", False)),
            )

    # sort by RRF score
    ranked_ids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)
    results    = []
    for cid in ranked_ids[:top_n]:
        c       = chunk_map[cid]
        c.score = rrf_scores[cid]
        results.append(c)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  RAG ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class RAGEngine:
    def __init__(self):
        print("=" * 60)
        print("Loading RAG Engine ...")
        print("=" * 60)

        # -- ChromaDB ---------------------------------------------------------
        t0 = time.time()
        chroma_client   = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = chroma_client.get_collection(name=CHROMA_COLLECTION)
        n_total         = self.collection.count()
        print(f"[1/5] ChromaDB collection  ({n_total:,} vectors)  {time.time()-t0:.1f}s")

        # -- Build BM25 corpus from ChromaDB (load all docs + metadata) -------
        t0 = time.time()
        all_data = self.collection.get(include=["documents", "metadatas"])
        self._bm25_corpus: list[dict] = [
            {
                "chunk_id":      cid,
                "text":          doc,
                "doc_id":        meta["doc_id"],
                "source_file":   meta["source_file"],
                "heading":       meta.get("heading") or None,
                "section_path":  meta.get("section_path", ""),
                "has_table":     meta.get("has_table",   False),
                "has_formula":   meta.get("has_formula", False),
                "has_code":      meta.get("has_code",    False),
                "prev_chunk_id": meta.get("prev_chunk_id") or None,
                "next_chunk_id": meta.get("next_chunk_id") or None,
                "chunk_index":   meta.get("chunk_index", 0),
            }
            for cid, doc, meta in zip(
                all_data["ids"],
                all_data["documents"],
                all_data["metadatas"],
            )
        ]
        self._id_to_corpus_idx: dict[str, int] = {
            c["chunk_id"]: i for i, c in enumerate(self._bm25_corpus)
        }
        self._bm25 = BM25Index(self._bm25_corpus)
        print(f"[2/5] BM25 index           ({len(self._bm25_corpus):,} docs)  {time.time()-t0:.1f}s")


        # -- Embedder (auto device: CUDA if available, else CPU) -------------
        import torch
        _cuda_ok = torch.cuda.is_available()
        _embed_device  = EMBED_DEVICE  if (_cuda_ok or EMBED_DEVICE  == "cpu") else "cpu"
        _rerank_device = RERANK_DEVICE if (_cuda_ok or RERANK_DEVICE == "cpu") else "cpu"
        if not _cuda_ok:
            print("  [INFO] No CUDA detected — running embedder & reranker on CPU")
        t0 = time.time()
        self.embedder = SentenceTransformer(EMBED_MODEL, device=_embed_device)
        print(f"[3/5] Embedder          ({EMBED_MODEL} on {_embed_device})  {time.time()-t0:.1f}s")

        # ── Reranker (auto device) ────────────────────────────────
        self.reranker = None
        if RERANK_MODEL:
            t0 = time.time()
            self.reranker = CrossEncoder(RERANK_MODEL, device=_rerank_device, max_length=512)
            print(f"[3b]  Reranker           ({RERANK_MODEL} on {_rerank_device})  {time.time()-t0:.1f}s")

        # ── Ollama health-check (non-fatal for evaluation mode) ──
        t0 = time.time()
        print(f"[4/5] Connecting to Ollama ({OLLAMA_URL}) model={OLLAMA_MODEL} ...")
        self._ollama_available = False
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            if not any(OLLAMA_MODEL in m for m in available):
                print(f"  WARNING: '{OLLAMA_MODEL}' not found in Ollama.")
                print(f"  Run:  ollama pull {OLLAMA_MODEL}")
                print(f"  Available models: {available}")
            else:
                self._ollama_available = True
                print(f"      OK — model ready  {time.time()-t0:.1f}s")
        except requests.exceptions.ConnectionError:
            print(f"  WARNING: Cannot reach Ollama at {OLLAMA_URL} — answer generation disabled.", file=sys.stderr)
            print(f"  Retrieval pipeline still available. Start Ollama to enable answers.", file=sys.stderr)

        # ── Conversation memory ──────────────────────────────────
        self._history: list[dict] = []   # {"question": ..., "answer": ...}

        print("=" * 60)
        print("Ready\n")

    # ──────────────────────────────────────────────────────────────
    
    #  QUERY REWRITING
    # ──────────────────────────────────────────────────────────────

    def _generate_hyde_document(self, question: str) -> str:
        """
        Use Ollama to generate a hypothetical answer to the user's question.
        This provides a rich semantic document to embed for FAISS retrieval.
        """
        prompt = (
            f"Please write a short, highly technical, and factual academic passage "
            f"that directly answers the following question. Do not include any filler "
            f"or conversational text. Just output the passage itself.\n\n"
            f"Question: {question}\n\nPassage:"
        )
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model":  OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 150,
                        "temperature": 0.3,
                    },
                },
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            hyde_doc = resp.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"  [hyde] Ollama error: {e} — using original query")
            return question

        # strip DeepSeek <think> tags if present
        if "<think>" in hyde_doc:
            hyde_doc = hyde_doc.split("</think>")[-1].strip()

        if not hyde_doc or len(hyde_doc) < 10:
            return question

        print(f"        HyDE snippet: '{hyde_doc[:80]}...'", flush=True)
        return hyde_doc

    # ──────────────────────────────────────────────────────────────
    #  RETRIEVAL
    # ──────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.embedder.encode(
            "query: " + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        return vec.reshape(1, -1)

    def _chroma_search(self, query_vec: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        """Semantic search via ChromaDB HNSW index."""
        results = self.collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for chunk_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance = 1 - cosine_similarity; convert to similarity
            score = 1.0 - float(distance)
            chunks.append(_chroma_result_to_chunk(chunk_id, document, metadata, score))
        return chunks

    def _hybrid_search(self, query: str, query_vec: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        """ChromaDB semantic + BM25 keyword, fused with RRF."""
        chroma_results = self._chroma_search(query_vec, top_k)
        bm25_results   = self._bm25.search(query, top_k)
        return reciprocal_rank_fusion(chroma_results, bm25_results, self._bm25_corpus, top_n=top_k)

    def _rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not self.reranker or not chunks:
            return chunks[:top_n]
        pairs  = [(query, c.text) for c in chunks]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_n]

    def _expand_context(self, chunks: list[RetrievedChunk], window: int) -> list[RetrievedChunk]:
        """Expand each chunk by fetching its prev/next neighbours from ChromaDB."""
        if window == 0:
            return chunks
        seen: set[str]         = set()
        expanded_ids: list[str] = []

        for c in chunks:
            base_idx = self._id_to_corpus_idx.get(c.chunk_id)
            if base_idx is None:
                continue
            for offset in range(-window, window + 1):
                n = base_idx + offset
                if 0 <= n < len(self._bm25_corpus):
                    neighbour = self._bm25_corpus[n]
                    if neighbour["doc_id"] == c.doc_id:
                        nid = neighbour["chunk_id"]
                        if nid not in seen:
                            seen.add(nid)
                            expanded_ids.append(nid)

        if not expanded_ids:
            return chunks

        # Fetch the actual documents from ChromaDB by ID
        fetched = self.collection.get(
            ids=expanded_ids,
            include=["documents", "metadatas"],
        )
        result = [
            _chroma_result_to_chunk(cid, doc, meta)
            for cid, doc, meta in zip(
                fetched["ids"],
                fetched["documents"],
                fetched["metadatas"],
            )
        ]
        # Keep original sort order (by chunk_index within doc)
        result.sort(key=lambda c: (c.doc_id, c.chunk_id))
        return result

    # ──────────────────────────────────────────────────────────────
    #  CONTEXT + PROMPT
    # ──────────────────────────────────────────────────────────────

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        by_doc: dict[str, list[RetrievedChunk]] = {}
        for c in chunks:
            by_doc.setdefault(c.doc_id, []).append(c)

        parts = []
        for doc_chunks in by_doc.values():
            doc_chunks.sort(key=lambda c: c.chunk_id)
            parts.append(f"### Source: {doc_chunks[0].source_file}")
            for c in doc_chunks:
                breadcrumb = " > ".join(c.section_path) if c.section_path else "-"
                parts.append(f"[{breadcrumb}]")
                parts.append(c.text)
                parts.append("")
        return "\n".join(parts).strip()

    def _build_history_str(self) -> str:
        """Format recent conversation turns for the prompt."""
        if not self._history:
            return ""
        lines = ["=== CONVERSATION HISTORY ==="]
        for turn in self._history[-MAX_HISTORY:]:
            lines.append(f"Q: {turn['question']}")
            lines.append(f"A: {turn['answer'][:300]}...")   # truncate long answers
            lines.append("")
        return "\n".join(lines)

    def _build_prompt(self, query: str, context: str) -> str:
        history_str = self._build_history_str()

        system = textwrap.dedent("""\
            You are a domain-specific research assistant with access to excerpts from scientific papers.
            Answer the user's question using ONLY the provided context.
            - Be precise and cite the source file and page when relevant.
            - If the context does not contain enough information, say so explicitly.
            - For technical claims, reference key terms from the source.
            - Structure long answers with bullet points or numbered steps.
            - Use conversation history to resolve references like "that", "it", "this approach".
        """)

        user_parts = []
        if history_str:
            user_parts.append(history_str)
        user_parts.append("=== CONTEXT ===")
        user_parts.append(context)
        user_parts.append(f"\n=== QUESTION ===\n{query}")
        user_parts.append("\n=== ANSWER ===")

        # Plain-text prompt — Ollama's /api/generate accepts raw text.
        # The model’s system instruction is prepended as a clear header.
        return f"### SYSTEM\n{system}\n\n### USER\n" + "\n".join(user_parts) + "\n\n### ASSISTANT\n"

    # ──────────────────────────────────────────────────────────────
    #  GENERATION
    # ──────────────────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        """
        Stream tokens from Ollama and print them as they arrive,
        mimicking the old TextStreamer behaviour.
        """
        answer_parts: list[str] = []
        try:
            with requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model":  OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": MAX_NEW_TOKENS,
                        "temperature": 0.0,
                        "repeat_penalty": 1.05,
                    },
                },
                timeout=OLLAMA_TIMEOUT,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    chunk = json.loads(raw_line)
                    token = chunk.get("response", "")
                    if token:
                        print(token, end="", flush=True)
                        answer_parts.append(token)
                    if chunk.get("done"):
                        break
        except requests.exceptions.RequestException as e:
            return f"[Ollama error: {e}]"

        print()  # newline after streamed output
        answer = "".join(answer_parts).strip()

        # strip DeepSeek-R1 chain-of-thought safely
        if "<think>" in answer:
            if "</think>" in answer:
                answer = answer.split("</think>")[-1].strip()
            else:
                answer = "[Generation cut off during thinking. Please increase MAX_NEW_TOKENS.]"

        return answer

    # ──────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def query(
        self,
        question:       str,
        top_k:          int  = RETRIEVAL_TOP_K,
        top_n:          int  = RERANK_TOP_N,
        rewrite_query:  bool = True,
        expand_context: bool = False,
        verbose:        bool = False,
    ) -> RAGResult:
        latency = {}

        # 1. Generating HyDE document
        print("  [1/5] Generating HyDE document ...", flush=True)
        t0 = time.time()
        search_query = self._generate_hyde_document(question) if rewrite_query else question
        latency["hyde"] = time.time() - t0

        # 2. Embed (rewritten) query
        print("  [2/5] Embedding query ...", flush=True)
        t0 = time.time()
        q_vec = self._embed_query(search_query)
        latency["embed"] = time.time() - t0
        print(f"        done ({latency['embed']*1000:.0f}ms)", flush=True)

        # 3. Hybrid retrieval
        print("  [3/5] Hybrid search (ChromaDB + BM25) ...", flush=True)
        t0 = time.time()
        candidates = self._hybrid_search(search_query, q_vec, top_k)
        latency["retrieval"] = time.time() - t0
        print(f"        {len(candidates)} candidates ({latency['retrieval']*1000:.0f}ms)", flush=True)

        # 4. Rerank
        print("  [4/5] Reranking ...", flush=True)
        t0 = time.time()
        ranked = self._rerank(question, candidates, top_n)  # rerank on ORIGINAL question
        latency["rerank"] = time.time() - t0
        print(f"        {len(ranked)} kept ({latency['rerank']*1000:.0f}ms)", flush=True)

        # 5. Context expansion (optional)
        context_chunks = (
            self._expand_context(ranked, CONTEXT_WINDOW)
            if expand_context and CONTEXT_WINDOW > 0
            else ranked
        )

        context = self._build_context(context_chunks)

        # 6. Generate
        print("  [5/5] Generating answer ...", flush=True)
        t0 = time.time()
        prompt = self._build_prompt(question, context)
        answer = self._generate(prompt)
        latency["generate"] = time.time() - t0
        print(f"        done ({latency['generate']:.1f}s)", flush=True)

        # 7. Save to conversation memory
        self._history.append({"question": question, "answer": answer})
        if len(self._history) > MAX_HISTORY:
            self._history.pop(0)

        return RAGResult(query=question, answer=answer, chunks=ranked, latency=latency)

    def clear_history(self):
        self._history = []
        print("Conversation history cleared.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def print_result(result: RAGResult):
    print("\n" + "=" * 70)
    print(f"QUERY: {result.query}")
    print("=" * 70)
    print(f"\nANSWER:\n{result.answer}\n")
    print("-" * 70)
    print("SOURCES:")
    for i, c in enumerate(result.chunks, 1):
        section = " > ".join(c.section_path) if c.section_path else "-"
        rscore  = f"  rerank={c.rerank_score:.3f}" if c.rerank_score is not None else ""
        print(f"  [{i}] {c.source_file}  | {section}")
        print(f"       rrf={c.score:.4f}{rscore}")
    print("-" * 70)
    total = sum(result.latency.values())
    lat   = "  ".join(f"{k}={v*1000:.0f}ms" for k, v in result.latency.items())
    print(f"LATENCY: {lat} | total={total:.2f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    engine = RAGEngine()
    print("Commands: 'exit' | 'clear' (reset memory) | 'norewrite' prefix to skip query rewriting\n")

    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            continue
        if q.lower() == "exit":
            break
        if q.lower() == "clear":
            engine.clear_history()
            continue

        rewrite = True
        if q.lower().startswith("norewrite "):
            q       = q[10:].strip()
            rewrite = False

        result = engine.query(q, rewrite_query=rewrite)
        print_result(result)
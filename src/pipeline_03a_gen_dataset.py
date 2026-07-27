"""
pipeline_03a_gen_dataset.py
───────────────────────────
RAG-aware synthetic dataset generator for Unsloth fine-tuning.

Reads all parsed JSONL chunk files, groups consecutive chunks from the same
paper (simulating live RAG retrieval), and generates structured Q&A pairs in
the OpenAI chat format.

The assistant response always follows the 4-section template:
  1. Core Contribution
  2. Architectural & Mathematical Mechanics
  3. Empirical Results
  4. Citations

======== PARALLEL MULTI-KEY SUPPORT ========
Supply multiple Groq (or Cerebras) API keys to multiply throughput N×:

  In .env:
    GROQ_API_KEY=gsk_key1
    GROQ_API_KEY_2=gsk_key2
    GROQ_API_KEY_3=gsk_key3
    GROQ_API_KEY_4=gsk_key4

Each key runs in its own thread. 4 keys = 4× speed.

======== FAST MODEL OPTIONS ========
  --model llama-3.1-8b-instant   # ~2s/call (vs ~10s for 70B) — recommended for bulk
  --model llama-3.3-70b-versatile # default — higher quality, slower
  --model gemma2-9b-it            # good middle ground on Groq

======== CEREBRAS (FASTEST FREE INFERENCE — 2100 tok/s) ========
  --backend cerebras
  Set CEREBRAS_API_KEY in .env. Sign up free at cloud.cerebras.ai
  Models: llama-3.3-70b (default), llama3.1-8b

======== RECOMMENDED FAST RUN ========
  python -X utf8 src/pipeline_03a_gen_dataset.py \\
    --model llama-3.1-8b-instant \\
    --workers 4 \\
    --max-rows 1000

  With 4 Groq keys + 8B model → ~1000 rows in ~10 minutes.

Generation backend priority (auto fallback):
  1. Groq (multi-key parallel)  — high-quality, free
  2. Cerebras                   — fastest free inference
  3. Ollama (local)             — local LLM fallback
  4. Template-based             — offline, no LLM

Usage:
  python -X utf8 src/pipeline_03a_gen_dataset.py
  python -X utf8 src/pipeline_03a_gen_dataset.py --dry-run
  python -X utf8 src/pipeline_03a_gen_dataset.py --backend cerebras
  python -X utf8 src/pipeline_03a_gen_dataset.py --model llama-3.1-8b-instant --workers 4
  python -X utf8 src/pipeline_03a_gen_dataset.py --max-rows 1000
  python -X utf8 src/pipeline_03a_gen_dataset.py --max 300 --model llama-3.1-8b-instant
"""

import argparse
import json
import os
import queue
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Ensure Unicode content in chunks (e.g. math symbols) prints cleanly on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── path bootstrap ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# Load .env before importing config so env-vars are available
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from config import (
    PARSED_DIR, FT_DATASET_PATH,
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    FT_GEN_MIN_CHARS, FT_GEN_MAX_CHUNKS,
    FT_GEN_USE_OLLAMA,
    GROQ_API_KEY_ENV, GROQ_MODEL, GROQ_BASE_URL,
    GROQ_TIMEOUT, GROQ_MAX_TOKENS, GROQ_TEMPERATURE,
)

# ── Cerebras defaults ──────────────────────────────────────────────────────────
CEREBRAS_BASE_URL    = "https://api.cerebras.ai/v1"
CEREBRAS_MODEL       = "llama3.1-8b"   # fast model; use "llama-3.3-70b" for higher quality
CEREBRAS_API_KEY_ENV = "CEREBRAS_API_KEY"

# ── prompt constants ───────────────────────────────────────────────────────────

_GENERATOR_SYSTEM_PROMPT = """\
You are an expert AI/ML researcher who creates high-quality fine-tuning \
datasets for deep learning research assistants.

You will be given 2-4 consecutive excerpts from the same research paper \
(labelled [Chunk 1], [Chunk 2], etc.). Your job is to:
  1. Write ONE specific, technical question a researcher would ask about \
the content of these chunks.
  2. Write a thorough answer that STRICTLY follows this 4-section Markdown \
structure:

### Core Contribution
A direct 1-2 sentence summary of what the paper or excerpt actually solves \
or proposes.

### Architectural & Mathematical Mechanics
How it works - use LaTeX for any equations (e.g. $\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}$) \
and bullet points for architectural layers or steps. Ground every claim in \
the provided chunks.

### Empirical Results
Brief statement of baseline comparisons and reported metrics. \
If the chunks do not contain this information, write exactly: \
"The provided context does not include specific empirical results."

### Citations
Bullet-point list linking each major claim to its source chunk, \
e.g. "- Adaptive learning rates: [Chunk 2]".

Rules:
- The question must be specific and non-trivial (not "What is this about?").
- Every claim in the answer MUST be grounded in the provided chunks.
- Do NOT invent information not present in the chunks.
- All four headings are REQUIRED in the answer.

Respond ONLY with valid JSON in this exact format (no markdown fences):
{
  "question": "...",
  "answer": "..."
}"""

_INFERENCE_SYSTEM_PROMPT = (
    "You are a deep learning research assistant. "
    "Answer the user's query using only the provided context. "
    "Structure your response with the following Markdown headings: "
    "### Core Contribution, "
    "### Architectural & Mathematical Mechanics, "
    "### Empirical Results, "
    "### Citations."
)

_REQUIRED_HEADINGS = [
    "### Core Contribution",
    "### Architectural & Mathematical Mechanics",
    "### Empirical Results",
    "### Citations",
]

_TEMPLATE_QUESTIONS = [
    "What key method or mechanism is described in these research excerpts?",
    "What problem do these passages address and what solution is proposed?",
    "Explain the main technical contribution described in these excerpts.",
    "What are the key findings or results discussed in these passages?",
    "How does the approach in these excerpts differ from prior work?",
    "Describe the mathematical formulation presented in these passages.",
    "What architectural design decisions are highlighted in these excerpts?",
]

_HEADING_NOISE = re.compile(
    r"^(abstract|introduction|related work|conclusion|references|"
    r"acknowledgements?|appendix|table of contents)$",
    re.IGNORECASE,
)

_FORMULA_HEAVY = re.compile(r"[=\u2211\u220f\u222b\u2202\u2207\u2248\u2264\u2265\u2208\u2286\u2287]")


# ── context block & row builders ───────────────────────────────────────────────

def build_context_block(chunks: list[dict]) -> str:
    lines = ["<context>"]
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "").strip().replace("\n", " ")[:800]
        lines.append(f"[Chunk {i}: {text}]")
    lines.append("</context>")
    return "\n".join(lines)


def build_chat_row(context_block: str, question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": _INFERENCE_SYSTEM_PROMPT},
            {"role": "user",      "content": f"{context_block}\n\nQuery: {question}"},
            {"role": "assistant", "content": answer},
        ]
    }


# ── response validation ────────────────────────────────────────────────────────

def _parse_llm_response(raw: str) -> Optional[dict]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",           "", raw, flags=re.MULTILINE)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        question = data.get("question", "").strip()
        answer   = data.get("answer",   "").strip()
        if not question or not answer or len(answer) < 80:
            return None
        if not all(h in answer for h in _REQUIRED_HEADINGS):
            return None
        return {"question": question, "answer": answer}
    except json.JSONDecodeError:
        pass
    return None


# ── template fallback ──────────────────────────────────────────────────────────

def make_template_row(chunks: list[dict]) -> Optional[dict]:
    first = chunks[0]
    heading = str(first.get("heading") or "")
    if _HEADING_NOISE.match(heading):
        return None
    combined = " ".join(c.get("text", "")[:600] for c in chunks).strip()
    if _FORMULA_HEAVY.search(combined) and len(combined) < 400:
        return None

    question      = random.choice(_TEMPLATE_QUESTIONS)
    context_block = build_context_block(chunks)
    chunk_refs    = "\n".join(f"- Content from [Chunk {i}]" for i in range(1, len(chunks) + 1))

    answer = (
        "### Core Contribution\n"
        f"{combined[:300].strip()}...\n\n"
        "### Architectural & Mathematical Mechanics\n"
        "The provided excerpts describe the following:\n\n"
        f"- {combined[300:600].strip()}\n\n"
        "### Empirical Results\n"
        "The provided context does not include specific empirical results.\n\n"
        f"### Citations\n{chunk_refs}"
    )
    return build_chat_row(context_block, question, answer)


# ── Groq backend ───────────────────────────────────────────────────────────────

def _collect_groq_keys() -> list[str]:
    """
    Collect all Groq API keys from environment variables.
    Looks for GROQ_API_KEY, GROQ_API_KEY_2, ... GROQ_API_KEY_10.
    Returns a deduplicated list of non-empty keys.
    """
    keys = []
    candidates = [GROQ_API_KEY_ENV] + [f"{GROQ_API_KEY_ENV}_{i}" for i in range(2, 11)]
    seen = set()
    for env_var in candidates:
        k = os.environ.get(env_var, "").strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)
    return keys


def _collect_cerebras_keys() -> list[str]:
    """
    Collect all Cerebras API keys from environment variables.
    Looks for CEREBRAS_API_KEY, CEREBRAS_API_KEY_2, ... CEREBRAS_API_KEY_10.
    Returns a deduplicated list of non-empty keys.
    """
    keys = []
    candidates = [CEREBRAS_API_KEY_ENV] + [f"{CEREBRAS_API_KEY_ENV}_{i}" for i in range(2, 11)]
    seen = set()
    for env_var in candidates:
        k = os.environ.get(env_var, "").strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)
    return keys


class GroqBackend:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model   = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        self.url = f"{GROQ_BASE_URL}/chat/completions"

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            r = requests.get(f"{GROQ_BASE_URL}/models", headers=self.headers, timeout=8)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, chunks: list[dict], retries: int = 3) -> Optional[dict]:
        context_block = build_context_block(chunks)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _GENERATOR_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Research paper chunks to analyse:\n\n{context_block}"},
            ],
            "max_tokens":  GROQ_MAX_TOKENS,
            "temperature": GROQ_TEMPERATURE,
        }
        for attempt in range(retries):
            try:
                resp = requests.post(self.url, headers=self.headers, json=payload, timeout=GROQ_TIMEOUT)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("retry-after", 15))
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                raw    = resp.json()["choices"][0]["message"]["content"]
                parsed = _parse_llm_response(raw)
                if parsed:
                    return build_chat_row(context_block, parsed["question"], parsed["answer"])
            except (requests.RequestException, KeyError, IndexError) as exc:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"\n  [groq warn] {exc}", flush=True)
        return None


# ── Cerebras backend ───────────────────────────────────────────────────────────

class CerebrasBackend:
    """
    Cerebras Cloud — up to 2100 tok/s, free tier.
    API is OpenAI-compatible. Sign up at cloud.cerebras.ai
    """
    def __init__(self, model: str = CEREBRAS_MODEL):
        self.model   = model
        self.api_key = os.environ.get(CEREBRAS_API_KEY_ENV, "")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        self.url = f"{CEREBRAS_BASE_URL}/chat/completions"

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            r = requests.get(f"{CEREBRAS_BASE_URL}/models", headers=self.headers, timeout=8)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, chunks: list[dict], retries: int = 3) -> Optional[dict]:
        context_block = build_context_block(chunks)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _GENERATOR_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Research paper chunks to analyse:\n\n{context_block}"},
            ],
            "max_tokens":  GROQ_MAX_TOKENS,
            "temperature": GROQ_TEMPERATURE,
        }
        for attempt in range(retries):
            try:
                resp = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("retry-after", 5))
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                raw    = resp.json()["choices"][0]["message"]["content"]
                parsed = _parse_llm_response(raw)
                if parsed:
                    return build_chat_row(context_block, parsed["question"], parsed["answer"])
            except (requests.RequestException, KeyError, IndexError) as exc:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"\n  [cerebras warn] {exc}", flush=True)
        return None


# ── Ollama backend ─────────────────────────────────────────────────────────────

class OllamaBackend:
    def is_available(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, chunks: list[dict], retries: int = 2) -> Optional[dict]:
        context_block = build_context_block(chunks)
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": _GENERATOR_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Research paper chunks to analyse:\n\n{context_block}"},
            ],
            "stream":  False,
            "options": {"temperature": 0.7, "num_predict": 768},
        }
        for attempt in range(retries + 1):
            try:
                resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
                resp.raise_for_status()
                raw    = resp.json()["message"]["content"]
                parsed = _parse_llm_response(raw)
                if parsed:
                    return build_chat_row(context_block, parsed["question"], parsed["answer"])
            except (requests.RequestException, KeyError) as exc:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                else:
                    print(f"\n  [ollama warn] {exc}", flush=True)
        return None


# ── chunk loading & grouping ───────────────────────────────────────────────────

def load_chunks(min_chars: int, max_chunks: int) -> list:
    chunks = []
    jsonl_files = sorted(PARSED_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"[error] No JSONL files found in {PARSED_DIR}")
        sys.exit(1)
    print(f"Found {len(jsonl_files)} JSONL file(s) in {PARSED_DIR}")
    for path in jsonl_files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if len(text) < min_chars:
                    continue
                if obj.get("has_table") and len(text) < 400:
                    continue
                if obj.get("has_formula") and len(text) < 400:
                    continue
                obj["_source_file"] = path.stem
                chunks.append(obj)
                if len(chunks) >= max_chunks:
                    break
        if len(chunks) >= max_chunks:
            break
    print(f"Loaded {len(chunks):,} qualifying chunks (min_chars={min_chars})")
    return chunks


def group_chunks_by_paper(chunks: list[dict], min_per_row: int = 2, max_per_row: int = 4) -> list[list[dict]]:
    paper_buckets: dict[str, list[dict]] = {}
    for chunk in chunks:
        key = chunk.get("_source_file", "unknown")
        paper_buckets.setdefault(key, []).append(chunk)

    groups = []
    for paper_chunks in paper_buckets.values():
        n = len(paper_chunks)
        if n < min_per_row:
            if n > 0:
                groups.append(paper_chunks[:])
            continue
        i = 0
        while i < n:
            remaining = n - i
            if remaining < min_per_row:
                break
            window = random.randint(min_per_row, min(max_per_row, remaining))
            groups.append(paper_chunks[i : i + window])
            stride = random.randint(1, max(1, window // 2))
            i += stride

    random.shuffle(groups)
    print(f"Created {len(groups):,} chunk groups "
          f"(2-{max_per_row} chunks each, from {len(paper_buckets)} paper(s))")
    return groups


# ── parallel worker engine ─────────────────────────────────────────────────────

def _worker(
    worker_id: int,
    backend,                      # GroqBackend / CerebrasBackend / OllamaBackend instance
    fallback_ollama: OllamaBackend,
    work_queue: queue.Queue,
    result_list: list,
    lock: threading.Lock,
    pbar: tqdm,
    max_rows: Optional[int],
    skip_event: threading.Event,
):
    """Thread worker: pulls chunk groups from a shared queue and generates rows."""
    while True:
        if skip_event.is_set():
            break
        try:
            group = work_queue.get(timeout=1)
        except queue.Empty:
            break

        # Check if we've hit the row cap
        with lock:
            if max_rows and len(result_list) >= max_rows:
                work_queue.task_done()
                skip_event.set()
                break

        row = backend.generate(group)
        if row is None and isinstance(fallback_ollama, OllamaBackend) and fallback_ollama.is_available():
            row = fallback_ollama.generate(group)
        if row is None:
            row = make_template_row(group)

        with lock:
            if row:
                if max_rows is None or len(result_list) < max_rows:
                    result_list.append(row)
            pbar.update(1)

        work_queue.task_done()


def run_parallel(
    backends: list,
    groups: list[list[dict]],
    fallback_ollama: OllamaBackend,
    max_rows: Optional[int],
) -> list[dict]:
    """
    Distribute chunk groups across multiple backends (each in its own thread).
    Each GroqBackend uses a different API key → true parallel throughput.
    """
    n_workers  = len(backends)
    work_queue = queue.Queue()
    for g in groups:
        work_queue.put(g)

    results    = []
    lock       = threading.Lock()
    skip_event = threading.Event()

    total = min(len(groups), max_rows) if max_rows else len(groups)
    pbar  = tqdm(total=total, desc=f"Generating rows ({n_workers} workers)", unit="row")

    threads = []
    for i, backend in enumerate(backends):
        t = threading.Thread(
            target=_worker,
            args=(i, backend, fallback_ollama, work_queue, results, lock, pbar, max_rows, skip_event),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    pbar.close()

    # Drain remaining queue items silently (if row cap hit)
    while not work_queue.empty():
        try:
            work_queue.get_nowait()
            work_queue.task_done()
        except queue.Empty:
            break

    return results


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate RAG-aware chat-format dataset for Unsloth fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Speed tips:
  --model llama-3.1-8b-instant    Use fast 8B model on Groq (~2s/call vs ~10s for 70B)
  --workers 4                     Use 4 parallel threads (needs 4 Groq API keys in .env)
  --max-rows 1000                 Stop after 1000 high-quality rows
  --backend cerebras              Use Cerebras (2100 tok/s, free at cloud.cerebras.ai)

Keys for parallel mode (add to .env):
  GROQ_API_KEY=gsk_key1
  GROQ_API_KEY_2=gsk_key2
  GROQ_API_KEY_3=gsk_key3
  GROQ_API_KEY_4=gsk_key4
        """
    )
    parser.add_argument("--backend", choices=["groq", "cerebras", "ollama", "template"], default=None)
    parser.add_argument("--model",   type=str, default=None,
                        help="LLM model to use (e.g. llama-3.1-8b-instant for fast Groq)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker threads (default: 1 per API key found)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max",     type=int, default=None,
                        help="Max number of raw chunks to load (default: config FT_GEN_MAX_CHUNKS)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Stop after generating this many rows (e.g. 1000)")
    parser.add_argument("--out",     type=str, default=None)
    parser.add_argument("--chunks-per-row", type=int, default=None,
                        help="Fixed chunks per context row (2-4). Default: random 2-4.")
    args = parser.parse_args()

    max_chunks = args.max or FT_GEN_MAX_CHUNKS
    out_path   = Path(args.out) if args.out else FT_DATASET_PATH
    max_rows   = args.max_rows

    fixed_cpr = args.chunks_per_row
    if fixed_cpr is not None and not (2 <= fixed_cpr <= 6):
        print("[error] --chunks-per-row must be between 2 and 6"); sys.exit(1)
    min_cpr = fixed_cpr or 2
    max_cpr = fixed_cpr or 4

    # ── backend selection — build a unified mixed pool ────────────────────────
    ollama = OllamaBackend()
    active_backends: list = []
    groq_model      = args.model or GROQ_MODEL
    cerebras_model  = args.model or CEREBRAS_MODEL

    # ── 1. Groq (multi-key) ────────────────────────────────────────────────────
    if args.backend in ("groq", None):
        groq_keys = _collect_groq_keys()
        if groq_keys:
            print(f"Found {len(groq_keys)} Groq key(s) — checking...")
            probe = GroqBackend(groq_keys[0], groq_model)
            if probe.is_available():
                for key in groq_keys:
                    active_backends.append(GroqBackend(key, groq_model))
                print(f"  OK  +{len(groq_keys)} Groq worker(s) / model: {groq_model}")
            else:
                print("  FAIL (Groq unreachable)")
                if args.backend == "groq":
                    sys.exit(1)
        else:
            if args.backend == "groq":
                print("[error] No GROQ_API_KEY found in .env"); sys.exit(1)

    # ── 2. Cerebras (multi-key) ────────────────────────────────────────────────
    if args.backend in ("cerebras", None):
        cerebras_keys = _collect_cerebras_keys()
        if cerebras_keys:
            print(f"Found {len(cerebras_keys)} Cerebras key(s) — checking...")
            # Probe with the first key
            probe_cb = CerebrasBackend(model=cerebras_model)
            probe_cb.api_key = cerebras_keys[0]
            probe_cb.headers["Authorization"] = f"Bearer {cerebras_keys[0]}"
            if probe_cb.is_available():
                for key in cerebras_keys:
                    cb = CerebrasBackend(model=cerebras_model)
                    cb.api_key = key
                    cb.headers["Authorization"] = f"Bearer {key}"
                    active_backends.append(cb)
                print(f"  OK  +{len(cerebras_keys)} Cerebras worker(s) / model: {cerebras_model}")
            else:
                print("  FAIL (Cerebras unreachable — check key)")
                if args.backend == "cerebras":
                    sys.exit(1)
        else:
            if args.backend == "cerebras":
                print("[error] No CEREBRAS_API_KEY found in .env  (free at cloud.cerebras.ai)")
                sys.exit(1)

    # ── 3. Ollama fallback ─────────────────────────────────────────────────────
    if not active_backends and args.backend in ("ollama", None):
        print("Checking Ollama...", end=" ", flush=True)
        if ollama.is_available() and FT_GEN_USE_OLLAMA:
            active_backends = [ollama]
            print(f"OK")
        else:
            print("FAIL (falling back to template mode)")

    # ── 4. Template (last resort) ──────────────────────────────────────────────
    if not active_backends:
        active_backends = [None]  # sentinel

    # Override worker count if --workers specified
    if args.workers and active_backends[0] is not None:
        model_used = groq_model if isinstance(active_backends[0], GroqBackend) else cerebras_model
        # Expand or shrink pool to requested count by round-robining existing backends
        active_backends = [
            active_backends[i % len(active_backends)]
            for i in range(args.workers)
        ]

    # Build label
    n_groq     = sum(1 for b in active_backends if isinstance(b, GroqBackend))
    n_cerebras = sum(1 for b in active_backends if isinstance(b, CerebrasBackend))
    n_ollama   = sum(1 for b in active_backends if isinstance(b, OllamaBackend))
    parts = []
    if n_groq:     parts.append(f"Groq x{n_groq}")
    if n_cerebras: parts.append(f"Cerebras x{n_cerebras}")
    if n_ollama:   parts.append(f"Ollama x{n_ollama}")
    if not parts:  parts = ["template (offline)"]
    active_label = " + ".join(parts) + f"  ({len(active_backends)} total workers)"
    print(f"\nCombined pool: {active_label}")

    print()
    print(f"Generation backend : {active_label}")
    row_cap_str = f"{max_rows:,}" if max_rows else "unlimited"
    print(f"Row cap            : {row_cap_str}")
    cpr_str = f"{min_cpr}" if min_cpr == max_cpr else f"{min_cpr}-{max_cpr} (random)"
    print(f"Chunks per row     : {cpr_str}")
    print(f"Max chunks loaded  : {max_chunks:,}")
    print(f"Output path        : {out_path}\n")

    # ── load & group chunks ────────────────────────────────────────────────────
    chunks = load_chunks(min_chars=FT_GEN_MIN_CHARS, max_chunks=max_chunks)
    groups = group_chunks_by_paper(chunks, min_per_row=min_cpr, max_per_row=max_cpr)

    # ── generate rows ──────────────────────────────────────────────────────────
    if active_backends[0] is None:
        # Template-only mode (single thread)
        rows = []
        for group in tqdm(groups, desc="Generating rows (template)", unit="group"):
            if max_rows and len(rows) >= max_rows:
                break
            row = make_template_row(group)
            if row:
                rows.append(row)
    else:
        rows = run_parallel(
            backends       = active_backends,
            groups         = groups,
            fallback_ollama= ollama,
            max_rows       = max_rows,
        )

    skipped = len(groups) - len(rows)

    # ── dry-run preview ────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{'-'*70}")
        print(f"DRY RUN - {len(rows)} rows generated, {skipped} skipped\n")
        for i, row in enumerate(rows[:3], 1):
            msgs = row["messages"]
            print(f"[Row {i}]")
            print(f"  USER : {msgs[1]['content'][:300]}...")
            print(f"  ASST : {msgs[2]['content'][:400]}...")
            print()
        print("(No file written in dry-run mode)")
        return

    # ── write output ───────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{'-'*70}")
    print(f"Dataset saved  : {out_path}")
    print(f"Total rows     : {len(rows):,}")
    print(f"Backend used   : {active_label}")
    print(f"\nNext step -> python src/pipeline_03_finetune.py")


if __name__ == "__main__":
    main()

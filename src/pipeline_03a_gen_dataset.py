"""
pipeline_03a_gen_dataset.py
───────────────────────────
Synthetic Q&A dataset generator for Unsloth fine-tuning.

Reads all parsed JSONL chunk files and generates (instruction, output) pairs
suitable for LoRA fine-tuning in the Alpaca prompt format.

Generation backend priority (automatic fallback chain):
  1. Groq API          — high-quality 70B model, free, ~1000 tok/s
  2. Ollama (local)    — your running local LLM fallback
  3. Template-based    — offline rule-based, no LLM needed

Usage:
  python src/pipeline_03a_gen_dataset.py
  python src/pipeline_03a_gen_dataset.py --dry-run        # preview without saving
  python src/pipeline_03a_gen_dataset.py --backend ollama # force Ollama
  python src/pipeline_03a_gen_dataset.py --backend template
  python src/pipeline_03a_gen_dataset.py --max 500        # limit chunks
  python src/pipeline_03a_gen_dataset.py --model deepseek-r1-distill-llama-70b
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

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
    FT_GEN_USE_OLLAMA, FT_GEN_QUESTIONS_PER_CHUNK,
    GROQ_API_KEY_ENV, GROQ_MODEL, GROQ_BASE_URL,
    GROQ_TIMEOUT, GROQ_MAX_TOKENS, GROQ_TEMPERATURE,
)

# ── prompt constants ───────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an expert AI/ML researcher who creates high-quality training datasets.
Given a passage from a research paper, generate ONE specific, insightful question \
that a researcher or student might ask about the key concept, method, finding, or \
contribution described in the passage.
Then write a thorough, accurate, well-structured answer strictly grounded in the passage.

Rules:
- The question must be specific and non-trivial (not "What is this about?")
- The answer must be factually grounded in the passage text
- Do NOT make up information not present in the passage
- Prefer technical depth over surface-level summaries

Respond ONLY with valid JSON in this exact format (no markdown fences):
{
  "question": "...",
  "answer": "..."
}"""

# ── template fallback ──────────────────────────────────────────────────────────
_TEMPLATES = [
    ("What key method or mechanism is described in this research excerpt?",
     "The passage describes the following method: {text}"),
    ("What problem does this passage address, and what solution is proposed?",
     "According to the passage: {text}"),
    ("Explain the main technical contribution described in this excerpt.",
     "The main technical contribution is: {text}"),
    ("What are the key findings or results discussed in this passage?",
     "The passage reports the following: {text}"),
    ("How does the approach described in this passage improve upon prior work?",
     "Based on the passage: {text}"),
]

_HEADING_NOISE = re.compile(
    r"^(abstract|introduction|related work|conclusion|references|"
    r"acknowledgements?|appendix|table of contents)$",
    re.IGNORECASE,
)

_FORMULA_HEAVY = re.compile(r"[=\u2211\u220f\u222b\u2202\u2207\u2248\u2264\u2265\u2208\u2286\u2287]")


def _first_sentence(text: str) -> str:
    for sep in (".", "!", "?"):
        idx = text.find(sep)
        if idx > 30:
            return text[: idx + 1].strip()
    return text[:200].strip()


def make_template_pair(chunk: dict) -> Optional[dict]:
    """Offline template-based Q&A generation — no LLM needed."""
    text: str = chunk.get("text", "")
    heading: str = chunk.get("heading", "")

    if _HEADING_NOISE.match(heading):
        return None
    if _FORMULA_HEAVY.search(text) and len(text) < 400:
        return None

    template_q, template_a = random.choice(_TEMPLATES)
    answer = template_a.format(text=text[:900])
    return {"instruction": template_q, "input": "", "output": answer.strip()}


# ── shared LLM call logic ──────────────────────────────────────────────────────

def _parse_llm_response(raw: str) -> Optional[dict]:
    """Parse the JSON Q&A response from any LLM, stripping markdown fences."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",           "", raw, flags=re.MULTILINE)
    # Find the first {...} block in case the model adds preamble text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        question = data.get("question", "").strip()
        answer   = data.get("answer",   "").strip()
        if question and answer and len(answer) > 30:
            return {"instruction": question, "input": "", "output": answer}
    except json.JSONDecodeError:
        pass
    return None


# ── Groq backend ───────────────────────────────────────────────────────────────

class GroqBackend:
    def __init__(self, model: str):
        self.model   = model
        self.api_key = os.environ.get(GROQ_API_KEY_ENV, "")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        self.url = f"{GROQ_BASE_URL}/chat/completions"

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            r = requests.get(
                f"{GROQ_BASE_URL}/models",
                headers=self.headers,
                timeout=8,
            )
            return r.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, passage: str, retries: int = 3) -> Optional[dict]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": f"PASSAGE:\n{passage[:1500]}"},
            ],
            "max_tokens":   GROQ_MAX_TOKENS,
            "temperature":  GROQ_TEMPERATURE,
        }
        for attempt in range(retries):
            try:
                resp = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=GROQ_TIMEOUT,
                )
                # Handle Groq rate limiting (429)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("retry-after", 10))
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                result = _parse_llm_response(raw)
                if result:
                    return result
            except (requests.RequestException, KeyError, IndexError) as exc:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"\n  [groq warn] {exc}", flush=True)
        return None


# ── Ollama backend ─────────────────────────────────────────────────────────────

class OllamaBackend:
    def is_available(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, passage: str, retries: int = 2) -> Optional[dict]:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": f"PASSAGE:\n{passage[:1200]}"},
            ],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 512},
        }
        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    json=payload,
                    timeout=OLLAMA_TIMEOUT,
                )
                resp.raise_for_status()
                raw = resp.json()["message"]["content"]
                result = _parse_llm_response(raw)
                if result:
                    return result
            except (requests.RequestException, KeyError) as exc:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                else:
                    print(f"\n  [ollama warn] {exc}", flush=True)
        return None


# ── chunk loading ──────────────────────────────────────────────────────────────

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
                chunks.append(obj)
                if len(chunks) >= max_chunks:
                    break
        if len(chunks) >= max_chunks:
            break

    random.shuffle(chunks)
    print(f"Loaded {len(chunks):,} qualifying chunks (min_chars={min_chars})")
    return chunks


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Q&A dataset for Unsloth fine-tuning"
    )
    parser.add_argument(
        "--backend", choices=["groq", "ollama", "template"], default=None,
        help="Force a specific backend (default: auto-detect Groq → Ollama → template)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override Groq model (e.g. deepseek-r1-distill-llama-70b)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview 3 sample pairs without writing output"
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Max number of chunks to process (overrides config)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Override output file path"
    )
    args = parser.parse_args()

    max_chunks = args.max or FT_GEN_MAX_CHUNKS
    out_path   = Path(args.out) if args.out else FT_DATASET_PATH
    groq_model = args.model or GROQ_MODEL

    # ── backend selection ──────────────────────────────────────────────────────
    groq   = GroqBackend(model=groq_model)
    ollama = OllamaBackend()

    if args.backend == "groq" or (args.backend is None):
        print("Checking Groq API...", end=" ", flush=True)
        if groq.is_available():
            active_backend = "groq"
            print(f"OK  (model: {groq_model})")
        else:
            api_key_set = bool(os.environ.get(GROQ_API_KEY_ENV))
            if not api_key_set:
                print("FAIL  (GROQ_API_KEY not set in .env)")
            else:
                print("FAIL  (API unreachable)")

            if args.backend == "groq":
                print("[error] Groq forced but unavailable. Check your API key.")
                sys.exit(1)
            # fall through to Ollama
            active_backend = None

        if active_backend != "groq" and args.backend != "template":
            print("Checking Ollama...", end=" ", flush=True)
            if ollama.is_available() and FT_GEN_USE_OLLAMA:
                active_backend = "ollama"
                print(f"OK  (model: {OLLAMA_MODEL})")
            else:
                active_backend = "template"
                print("FAIL  (falling back to template mode)")

    elif args.backend == "ollama":
        print("Checking Ollama...", end=" ", flush=True)
        if ollama.is_available():
            active_backend = "ollama"
            print(f"OK  (model: {OLLAMA_MODEL})")
        else:
            print("FAIL  (Ollama not running)")
            sys.exit(1)
    else:
        active_backend = "template"
        print("Backend: template (offline)")

    backend_label = {
        "groq":     f"Groq / {groq_model}",
        "ollama":   f"Ollama / {OLLAMA_MODEL}",
        "template": "template (offline)",
    }[active_backend]

    print(f"\nGeneration backend : {backend_label}")
    print(f"Max chunks         : {max_chunks:,}")
    print(f"Output path        : {out_path}\n")

    # ── load chunks ────────────────────────────────────────────────────────────
    chunks = load_chunks(min_chars=FT_GEN_MIN_CHARS, max_chunks=max_chunks)

    # ── generate pairs ─────────────────────────────────────────────────────────
    pairs   = []
    skipped = 0

    for chunk in tqdm(chunks, desc="Generating Q&A pairs", unit="chunk"):
        for _ in range(FT_GEN_QUESTIONS_PER_CHUNK):
            pair = None

            if active_backend == "groq":
                pair = groq.generate(chunk["text"])
                if pair is None:
                    # graceful fallback
                    pair = ollama.generate(chunk["text"]) if ollama.is_available() else None
                if pair is None:
                    pair = make_template_pair(chunk)

            elif active_backend == "ollama":
                pair = ollama.generate(chunk["text"])
                if pair is None:
                    pair = make_template_pair(chunk)

            else:
                pair = make_template_pair(chunk)

            if pair:
                pair["_source"] = chunk.get("chunk_id", "")
                pairs.append(pair)
            else:
                skipped += 1

    # ── dry-run preview ────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{'─'*62}")
        print(f"DRY RUN -- {len(pairs)} pairs generated, {skipped} skipped")
        print("\nSample pairs:\n")
        for i, p in enumerate(pairs[:3], 1):
            print(f"  [{i}] INSTRUCTION: {p['instruction']}")
            print(f"      OUTPUT     : {p['output'][:300]}...\n")
        print("(No file written in dry-run mode)")
        return

    # ── write output ───────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n{'─'*62}")
    print(f"Dataset saved  : {out_path}")
    print(f"Total pairs    : {len(pairs):,}")
    print(f"Skipped chunks : {skipped}")
    print(f"Backend used   : {backend_label}")
    print(f"\nNext step -> python src/pipeline_03_finetune.py")


if __name__ == "__main__":
    main()

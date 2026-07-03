"""
RAGAS Evaluation — Step 1: Testset Generator  (v2 — improved generation)
=========================================================================
Reads parsed paper chunks from PARSED_DIR, samples them stratified across
documents, and uses Ollama (DeepSeek-R1) to generate question + ground-truth
answer pairs.  No external API key required — everything runs locally.

Key improvements over v1
————————————————————————
• Two-pass generation  — "ask" and "answer" as separate LLM calls so neither
  gets truncated by the model's chain-of-thought thinking block.
• Higher token budget  — OLLAMA_MAX_PREDICT raised to 1500 (safe for R1's
  400-600 token think block + actual output).
• Rotating templates   — 3 question-type templates (definition / method /
  result) cycle across chunks to prevent generic "What is this about?" output.
• Retry logic          — up to N_RETRIES per chunk before giving up.
• Quality filter       — generated answer must share meaningful content with
  the source chunk (n-gram overlap check).
• Consistent key names — prompt and parser both use "ground_truth".

Output:  <project_root>/eval/testset.json

Usage:
    python src/eval/generate_testset.py
    python src/eval/generate_testset.py --one-pass   # disable two-pass mode
"""

import json
import re
import random
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── path setup ────────────────────────────────────────────────────────────────
SRC_DIR     = Path(__file__).resolve().parent.parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import requests
from config import PARSED_DIR, OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

# ── output paths ──────────────────────────────────────────────────────────────
EVAL_DIR     = PROJECT_DIR / "eval"
EVAL_DIR.mkdir(exist_ok=True)
TESTSET_PATH = EVAL_DIR / "testset.json"

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

N_SAMPLES        = 25     # target number of QA pairs
MIN_CHARS        = 200    # skip tiny / incomplete chunks
MAX_CHARS        = 1000   # skip very long chunks
PREFER_NO_TABLE  = True
RANDOM_SEED      = 42

# Two-pass mode: generate question first, then answer in a second call.
# Each call only needs to produce a short output, so truncation is unlikely.
# Set to False (or pass --one-pass CLI flag) to use single-call JSON mode.
TWO_PASS = True

# Token budget per Ollama call.
# DeepSeek-R1:8b uses ~400-600 tokens for <think> before producing output.
# 1500 gives comfortable headroom for thinking + the actual response.
OLLAMA_MAX_PREDICT = 1800

# Retries per chunk before moving on
N_RETRIES = 2

# Quality gate: at least this fraction of content words in the answer must
# also appear in the chunk (guards against hallucinated answers).
MIN_OVERLAP_RATIO = 0.10


# ── Question-type templates (rotate per chunk) ─────────────────────────────
QUESTION_TEMPLATES = [
    # Template 0 — Definition / concept
    (
        "You are building an evaluation dataset for a research-paper Q&A system.\n\n"
        "Read the excerpt below and write ONE specific question asking what a key "
        "term, concept, or methodology MEANS (a definition-style question).\n"
        "The question must be answerable from the excerpt alone.\n\n"
        "Excerpt:\n{chunk}\n\n"
        "Reply with ONLY the question text. No preamble, no numbering."
    ),
    # Template 1 — Method / mechanism
    (
        "You are building an evaluation dataset for a research-paper Q&A system.\n\n"
        "Read the excerpt below and write ONE specific question about HOW something "
        "works, is computed, or is implemented (a method/mechanism question).\n"
        "The question must be answerable from the excerpt alone.\n\n"
        "Excerpt:\n{chunk}\n\n"
        "Reply with ONLY the question text. No preamble, no numbering."
    ),
    # Template 2 — Result / finding
    (
        "You are building an evaluation dataset for a research-paper Q&A system.\n\n"
        "Read the excerpt below and write ONE specific question about a RESULT, "
        "finding, metric, or comparison reported (a result-focused question).\n"
        "The question must be answerable from the excerpt alone.\n\n"
        "Excerpt:\n{chunk}\n\n"
        "Reply with ONLY the question text. No preamble, no numbering."
    ),
]

ANSWER_PROMPT_TMPL = (
    "You are an expert research assistant building a reference evaluation dataset.\n\n"
    "Answer the question below using ONLY the information in the excerpt.\n"
    "Your answer should:\n"
    "  - Be 3-5 sentences long.\n"
    "  - Include all relevant facts, numbers, and technical details from the excerpt that pertain to the question.\n"
    "  - Be self-contained (a reader should understand the answer without seeing the excerpt).\n"
    "  - NOT invent or infer anything not explicitly stated in the excerpt.\n\n"
    "Excerpt:\n{chunk}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# One-pass prompt (used when TWO_PASS=False or --one-pass flag is set)
ONE_PASS_PROMPT_TMPL = (
    "You are building an evaluation dataset for a research-paper Q&A system.\n\n"
    "Read the excerpt and produce ONE evaluation item:\n"
    '  - A specific, answerable question (about a concept, method, or result).\n'
    '  - A concise ground-truth answer drawn ONLY from the excerpt (1-3 sentences).\n\n'
    'Output a JSON object with exactly two keys: "question" and "ground_truth".\n'
    "No extra text, no markdown fences.\n\n"
    "Excerpt:\n{chunk}\n\n"
    "JSON:"
)


# ══════════════════════════════════════════════════════════════════════════════
#  CHUNK LOADING + SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def load_all_chunks() -> list[dict]:
    """Load every chunk from every JSONL in PARSED_DIR."""
    chunks: list[dict] = []
    jsonl_files = sorted(PARSED_DIR.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {PARSED_DIR}")
    for path in jsonl_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def _is_garbled(text: str) -> bool:
    """
    Detect chunks with garbled OCR (e.g. 'E L E C T R A' style spaced letters).
    Returns True if more than 40% of whitespace-delimited tokens are single chars.
    """
    tokens = text.split()
    if not tokens:
        return True
    single = sum(1 for t in tokens if len(t) == 1)
    return (single / len(tokens)) > 0.40


def stratified_sample(chunks: list[dict], n: int) -> list[dict]:
    """
    Stratified sampling spread evenly across documents.
    Returns up to n+15 candidates so the generator loop has room to skip
    chunks that fail all retries.
    """
    eligible = [
        c for c in chunks
        if MIN_CHARS <= len(c.get("text", "")) <= MAX_CHARS
        and not c.get("has_formula", False)
        and (not PREFER_NO_TABLE or not c.get("has_table", False))
        and not _is_garbled(c.get("text", ""))
    ]

    if not eligible:
        raise RuntimeError(
            f"No eligible chunks found. Adjust MIN_CHARS ({MIN_CHARS}) / "
            f"MAX_CHARS ({MAX_CHARS}) in generate_testset.py."
        )

    by_doc: dict[str, list[dict]] = {}
    for c in eligible:
        by_doc.setdefault(c["doc_id"], []).append(c)

    per_doc = max(3, (n + 15) // len(by_doc) + 1)

    sampled: list[dict] = []
    for doc_chunks in by_doc.values():
        k = min(per_doc, len(doc_chunks))
        sampled.extend(random.sample(doc_chunks, k))

    random.shuffle(sampled)
    return sampled


# ══════════════════════════════════════════════════════════════════════════════
#  OLLAMA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _call_ollama(prompt: str, max_tokens: int = OLLAMA_MAX_PREDICT) -> str:
    """
    POST a prompt to Ollama and return the cleaned response text.
    Strips DeepSeek-R1 <think>...</think> chain-of-thought blocks.
    Returns "" if thinking was truncated (response would be incomplete).
    """
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()

    if "<think>" in raw and "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    elif "<think>" in raw:
        # Thinking block was cut off — response is incomplete, discard
        return ""

    return raw


def _fix_loose_json(text: str) -> str:
    """Light repair of common LLM JSON issues (trailing commas, single quotes)."""
    text = re.sub(r",\s*([}\]])", r"\1", text)   # trailing commas
    text = re.sub(r"(?<!\\)'", '"', text)         # single -> double quotes
    return text


def _extract_json(text: str) -> dict | None:
    """
    Try multiple strategies to extract a JSON object from LLM output.
    Strategy 1: Direct brace extraction { ... }
    Strategy 2: Markdown fence ```json ... ``` or ``` ... ```
    Strategy 3: Loose JSON repair then retry
    """
    if not text:
        return None

    def _try_parse(s: str) -> dict | None:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
        try:
            obj = json.loads(_fix_loose_json(s))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    # Strategy 1
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > 0:
        result = _try_parse(text[start:end])
        if result is not None:
            return result

    # Strategy 2 & 3
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        inner = fence_match.group(1).strip()
        result = _try_parse(inner)
        if result is not None:
            return result

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY FILTER
# ══════════════════════════════════════════════════════════════════════════════

_STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "of", "to", "and", "or", "for",
    "on", "at", "by", "as", "be", "are", "was", "were", "this", "that",
    "with", "from", "its", "also", "can", "may", "such", "than", "these",
    "their", "which", "have", "has", "had", "not", "but", "so", "if",
}


def _content_tokens(text: str) -> set[str]:
    """Return lowercase non-stop-word alphabetic tokens (>=3 chars) from text."""
    return {
        w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text)
        if w.lower() not in _STOP_WORDS
    }


def _answer_overlaps_chunk(answer: str, chunk_text: str) -> bool:
    """
    Quality gate: at least MIN_OVERLAP_RATIO of the content words in the
    answer must also appear in the chunk. Rejects hallucinated answers.
    """
    ans_tokens   = _content_tokens(answer)
    chunk_tokens = _content_tokens(chunk_text)
    if not ans_tokens:
        return False
    overlap = ans_tokens & chunk_tokens
    return (len(overlap) / len(ans_tokens)) >= MIN_OVERLAP_RATIO


# ══════════════════════════════════════════════════════════════════════════════
#  QA GENERATION  (two-pass and one-pass modes)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_two_pass(chunk_text: str, template_idx: int) -> dict | None:
    """
    Two-pass generation:
      Pass 1 -> generate question only (short output, reliable)
      Pass 2 -> generate answer given question + chunk (grounded)
    Returns {"question": ..., "ground_truth": ...} or None.
    """
    excerpt = chunk_text[:800]

    # Pass 1: question
    q_prompt = QUESTION_TEMPLATES[template_idx % len(QUESTION_TEMPLATES)].format(
        chunk=excerpt
    )
    question = _call_ollama(q_prompt)
    if not question or len(question.strip()) < 15:
        return None

    # Strip model artifacts like leading "Q:" or numbering
    question = re.sub(r"^\s*[Qq]\s*[:.)]\s*", "", question).strip()
    question = re.sub(r"^\d+[.)]\s*", "", question).strip()

    if len(question) < 15:
        return None

    # Pass 2: answer
    a_prompt = ANSWER_PROMPT_TMPL.format(chunk=excerpt, question=question)
    answer = _call_ollama(a_prompt)
    if not answer or len(answer.strip()) < 15:
        return None

    answer = re.sub(r"^\s*[Aa]\s*[:.)]\s*", "", answer).strip()

    return {"question": question, "ground_truth": answer}


def _generate_one_pass(chunk_text: str, template_idx: int) -> dict | None:
    """
    One-pass generation: single Ollama call returning a JSON object.
    """
    excerpt = chunk_text[:800]
    prompt  = ONE_PASS_PROMPT_TMPL.format(chunk=excerpt)
    raw     = _call_ollama(prompt)
    qa      = _extract_json(raw)

    if qa is None:
        return None

    q = str(qa.get("question",     qa.get("q", ""))).strip()
    a = str(qa.get("ground_truth", qa.get("answer", ""))).strip()

    q = re.sub(r"^\s*[Qq]\s*[:.)]\s*", "", q).strip()
    a = re.sub(r"^\s*[Aa]\s*[:.)]\s*", "", a).strip()

    if len(q) < 15 or len(a) < 15:
        return None

    return {"question": q, "ground_truth": a}


def generate_qa_pair(
    chunk_text: str,
    template_idx: int,
    two_pass: bool = True,
) -> dict | None:
    """
    Generate a (question, ground_truth) pair from chunk_text.
    Retries up to N_RETRIES times; applies a quality overlap filter.
    Returns {"question": ..., "ground_truth": ...} or None.
    """
    _generator = _generate_two_pass if two_pass else _generate_one_pass

    for attempt in range(1, N_RETRIES + 1):
        try:
            qa = _generator(chunk_text, template_idx)
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(f"[attempt {attempt}] error: {exc}", end="  ")
            qa = None

        if qa is None:
            if attempt < N_RETRIES:
                time.sleep(0.5)
            continue

        # Quality gate: answer must share content words with the chunk
        if not _answer_overlaps_chunk(qa["ground_truth"], chunk_text):
            if attempt < N_RETRIES:
                time.sleep(0.5)
            continue

        return qa   # good pair

    return None   # exhausted retries


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    random.seed(RANDOM_SEED)

    two_pass   = TWO_PASS and "--one-pass" not in sys.argv
    mode_label = "two-pass" if two_pass else "one-pass"

    print("=" * 60)
    print("RAGAS Testset Generator  (v2)")
    print(f"Mode: {mode_label}  |  Model: {OLLAMA_MODEL}")
    print("=" * 60)

    # 1. Verify Ollama is reachable
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        print(f"\nOllama OK  ({OLLAMA_MODEL})")
    except requests.RequestException:
        print(f"\n[ERROR] Cannot reach Ollama at {OLLAMA_URL}")
        print("Make sure Ollama is running:  ollama serve")
        sys.exit(1)

    # 2. Load chunks
    print(f"\n[1/3] Loading chunks from {PARSED_DIR.name} ...")
    chunks = load_all_chunks()
    n_docs = len({c["doc_id"] for c in chunks})
    print(f"      {len(chunks):,} chunks across {n_docs} document(s)")

    # 3. Sample candidates
    print(f"\n[2/3] Stratified sampling (target: {N_SAMPLES} QA pairs) ...")
    candidates = stratified_sample(chunks, N_SAMPLES)
    print(f"      {len(candidates)} candidates selected")

    # 4. Generate QA pairs
    mode_note = (
        "2 Ollama calls/pair (question then answer)"
        if two_pass
        else "1 Ollama call/pair (question + answer JSON)"
    )
    print(f"\n[3/3] Generating QA pairs via {OLLAMA_MODEL}  [{mode_note}] ...")
    testset: list[dict] = []
    skipped = 0
    tmpl_names = ["definition", "method", "result"]

    for i, chunk in enumerate(candidates):
        if len(testset) >= N_SAMPLES:
            break

        template_idx = i % len(QUESTION_TEMPLATES)
        label = f"{chunk['source_file']}  p.{chunk['page_start']}"

        print(
            f"  [{i+1:2d}/{len(candidates)}]  "
            f"[{tmpl_names[template_idx]}]  "
            f"{label:<45}",
            end="  ",
            flush=True,
        )

        qa = generate_qa_pair(chunk["text"], template_idx, two_pass=two_pass)

        if qa:
            testset.append({
                **qa,
                "chunk_id":       chunk["chunk_id"],
                "source_file":    chunk["source_file"],
                "page_start":     chunk["page_start"],
                "page_end":       chunk["page_end"],
                "section":        chunk.get("heading") or "",
                "template":       tmpl_names[template_idx],
                "reference_text": chunk["text"],
            })
            print(f"OK  ({len(testset)}/{N_SAMPLES})")
        else:
            skipped += 1
            print("SKIP")

        time.sleep(0.1)

    # 5. Save
    with open(TESTSET_PATH, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Saved {len(testset)} QA pairs  ->  {TESTSET_PATH}")
    print(f"Skipped: {skipped} chunks (failed retries or quality gate)")
    print(f"{'='*60}")

    if len(testset) < N_SAMPLES:
        print(
            f"\n[WARNING] Only {len(testset)}/{N_SAMPLES} pairs generated.\n"
            "  Options:\n"
            "  - Widen MIN_CHARS / MAX_CHARS filters\n"
            "  - Add more papers to PARSED_DIR\n"
            f"  - Lower MIN_OVERLAP_RATIO (currently {MIN_OVERLAP_RATIO:.0%}) "
            "to relax the quality gate"
        )

    print("\nNext step:  python src/eval/run_eval.py")


if __name__ == "__main__":
    main()

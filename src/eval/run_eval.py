"""
RAGAS Evaluation — Step 2: Run Evaluation
==========================================
Loads the testset from eval/testset.json, runs every question through the
RAGEngine, then scores the results with four RAGAS metrics:

  Metric               What it measures
  ─────────────────    ──────────────────────────────────────────────────
  Faithfulness         Is the answer grounded in the retrieved context?
  Answer Relevancy     Is the answer actually relevant to the question?
  Context Recall       Did retrieval surface the info needed to answer?
  Context Precision    Were retrieved chunks precise (low noise)?

Scores are 0–1 (higher is better).
Results are saved to eval/results/results_YYYYMMDD_HHMMSS.json.

Usage:
    python src/eval/run_eval.py            # full run
    python src/eval/run_eval.py --no-rag   # skip RAGEngine; reuse cached results
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
SRC_DIR     = Path(__file__).resolve().parent.parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from config import OLLAMA_URL, OLLAMA_MODEL, EMBED_MODEL

EVAL_DIR     = PROJECT_DIR / "eval"
TESTSET_PATH = EVAL_DIR / "testset.json"
CACHE_PATH   = EVAL_DIR / "rag_results_cache.json"
RESULTS_DIR  = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ── RAGAS metric names (key = column name in RAGAS output dataframe) ──────────
METRIC_LABELS: dict[str, str] = {
    "faithfulness":      "Faithfulness       (answer grounded in context?)",
    "answer_relevancy":  "Answer Relevancy   (answer relevant to question?)",
    "context_recall":    "Context Recall     (retrieved needed information?)",
    "context_precision": "Context Precision  (retrieved chunks were precise?)",
}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP A — Collect RAG results
# ══════════════════════════════════════════════════════════════════════════════

def collect_rag_results(testset: list[dict]) -> list[dict]:
    """
    Run every question in the testset through the RAGEngine.
    Returns a list of dicts: {question, ground_truth, answer, contexts, ...}
    """
    from index import RAGEngine   # import here so --no-rag skips the heavy load

    engine = RAGEngine()

    results: list[dict] = []
    print(f"\n{'='*60}")
    print(f"Running {len(testset)} queries through RAGEngine ...")
    print(f"{'='*60}")

    for i, item in enumerate(testset):
        print(f"\n[{i+1}/{len(testset)}] {item['question'][:75]}...")
        try:
            t0     = time.time()
            result = engine.query(item["question"], rewrite_query=False)
            elapsed = time.time() - t0

            results.append({
                "question":     item["question"],
                "ground_truth": item["ground_truth"],
                "answer":       result.answer,
                "contexts":     [c.text for c in result.chunks],
                "source_file":  item.get("source_file", ""),
                "latency_s":    round(elapsed, 2),
            })
            print(f"  ✓  {len(result.chunks)} context(s) retrieved  |  {elapsed:.1f}s")

        except Exception as exc:
            print(f"  ✗  ERROR: {exc}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STEP B — RAGAS Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_ragas_evaluation(rag_results: list[dict]) -> "EvaluationResult":
    """
    Score the collected RAG results using RAGAS.

    Judge LLM  : Ollama (DeepSeek) via LangChain
    Embeddings : HuggingFace BGE-large (same model the project uses)
    """
    # ── lazy imports so the script still starts even if ragas isn't installed ─
    try:
        from ragas import evaluate, EvaluationDataset, SingleTurnSample
        from ragas.metrics import (
            Faithfulness, AnswerRelevancy,
            ContextRecall, ContextPrecision,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_ollama import ChatOllama
        from langchain_huggingface import HuggingFaceEmbeddings

    except ImportError as exc:
        print(f"\n[ERROR] Missing dependency: {exc}")
        print(
            "Install eval extras:\n"
            "  pip install ragas langchain-ollama langchain-huggingface"
        )
        sys.exit(1)

    # ── build RAGAS dataset ───────────────────────────────────────────────────
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        for r in rag_results
    ]
    dataset = EvaluationDataset(samples=samples)

    # ── configure judge LLM (Ollama / DeepSeek) ──────────────────────────────
    print(f"\nConfiguring RAGAS judge LLM : {OLLAMA_MODEL} @ {OLLAMA_URL}")
    llm = LangchainLLMWrapper(
        ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0)
    )

    # ── configure embeddings (BGE-large on CPU — used once, not per-query) ───
    print(f"Configuring RAGAS embeddings: {EMBED_MODEL}")
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    # ── define metrics ────────────────────────────────────────────────────────
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        ContextRecall(llm=llm),
        ContextPrecision(llm=llm),
    ]

    print(
        f"\nRunning RAGAS on {len(rag_results)} samples "
        "(this calls the LLM judge many times — may take 5-15 min) ..."
    )
    scores = evaluate(dataset=dataset, metrics=metrics)
    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  STEP C — Report + Save
# ══════════════════════════════════════════════════════════════════════════════

def _bar(score: float, width: int = 20) -> str:
    """Simple ASCII progress bar  e.g. '████████████░░░░░░░░  0.600'"""
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def print_report(rag_results: list[dict], scores) -> None:
    df = scores.to_pandas()
    overall = df.mean(numeric_only=True).to_dict()

    print("\n" + "=" * 70)
    print("RAGAS EVALUATION REPORT")
    print(
        f"Model: {OLLAMA_MODEL}   "
        f"Embed: {EMBED_MODEL.split('/')[-1]}   "
        f"Questions: {len(rag_results)}   "
        f"{datetime.now():%Y-%m-%d %H:%M}"
    )
    print("=" * 70)

    print("\n  OVERALL SCORES\n  " + "─" * 60)
    for key, label in METRIC_LABELS.items():
        val = overall.get(key)
        if val is None:
            print(f"  {label:<45}  N/A")
        else:
            print(f"  {label:<45}  {val:.3f}  {_bar(val)}")

    print("  " + "─" * 60)

    # ── latency summary ───────────────────────────────────────────────────────
    lats = [r["latency_s"] for r in rag_results if "latency_s" in r]
    if lats:
        avg_l = sum(lats) / len(lats)
        print(
            f"\n  LATENCY  avg={avg_l:.1f}s  "
            f"min={min(lats):.1f}s  max={max(lats):.1f}s"
        )

    # ── per-question table ────────────────────────────────────────────────────
    print("\n  PER-QUESTION BREAKDOWN\n  " + "─" * 60)
    score_cols = [k for k in METRIC_LABELS if k in df.columns]
    header = f"  {'#':>2}  {'Question':<50}  " + "  ".join(
        f"{k[:6]:>6}" for k in score_cols
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for i, row in df.iterrows():
        q = rag_results[i]["question"][:48] + ".."
        vals = "  ".join(
            f"{row[k]:.3f}" if k in row and row[k] == row[k] else " N/A "
            for k in score_cols
        )
        print(f"  {i+1:>2}  {q:<50}  {vals}")

    print("\n" + "=" * 70)


def save_results(rag_results: list[dict], scores) -> Path:
    """Save full results (metadata + per-question scores) to JSON."""
    df = scores.to_pandas()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"results_{ts}.json"

    score_cols = [k for k in METRIC_LABELS if k in df.columns]

    output = {
        "metadata": {
            "timestamp":   ts,
            "model":       OLLAMA_MODEL,
            "embed_model": EMBED_MODEL,
            "n_questions": len(rag_results),
            "metrics":     score_cols,
        },
        "overall_scores": {
            k: round(float(df[k].mean()), 4)
            for k in score_cols
        },
        "per_question": [
            {
                "question":     rag_results[i]["question"],
                "ground_truth": rag_results[i]["ground_truth"],
                "answer":       rag_results[i]["answer"],
                "source_file":  rag_results[i].get("source_file", ""),
                "latency_s":    rag_results[i].get("latency_s"),
                "n_contexts":   len(rag_results[i].get("contexts", [])),
                "scores": {
                    k: round(float(df.iloc[i][k]), 4)
                    for k in score_cols
                    if df.iloc[i][k] == df.iloc[i][k]   # skip NaN
                },
            }
            for i in range(min(len(rag_results), len(df)))
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    skip_rag = "--no-rag" in sys.argv

    print("=" * 60)
    print("RAGAS Evaluation Runner")
    print("=" * 60)

    # ── 1. Load testset ───────────────────────────────────────────────────────
    if not TESTSET_PATH.exists():
        print(f"\n[ERROR] Testset not found: {TESTSET_PATH}")
        print("Generate it first:  python src/eval/generate_testset.py")
        sys.exit(1)

    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset: list[dict] = json.load(f)

    print(f"\nLoaded {len(testset)} QA pairs from {TESTSET_PATH.name}")

    # ── 2. Collect RAG results (or reuse cache) ───────────────────────────────
    if skip_rag and CACHE_PATH.exists():
        print("[--no-rag] Loading cached RAG results ...")
        with open(CACHE_PATH, encoding="utf-8") as f:
            rag_results: list[dict] = json.load(f)
        print(f"  {len(rag_results)} cached results loaded")
    else:
        rag_results = collect_rag_results(testset)
        # cache so --no-rag works on subsequent runs
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(rag_results, f, indent=2, ensure_ascii=False)
        print(f"\nRAG results cached → {CACHE_PATH.name}")

    if not rag_results:
        print("[ERROR] No RAG results to evaluate. Exiting.")
        sys.exit(1)

    # ── 3. RAGAS evaluation ───────────────────────────────────────────────────
    scores = run_ragas_evaluation(rag_results)

    # ── 4. Print report ───────────────────────────────────────────────────────
    print_report(rag_results, scores)

    # ── 5. Save results ───────────────────────────────────────────────────────
    out_path = save_results(rag_results, scores)
    print(f"\nFull results saved → {out_path}")
    print(
        "\nTip: Re-run with --no-rag to re-score the same answers "
        "without re-running the RAG pipeline."
    )


if __name__ == "__main__":
    main()

"""
RAG Evaluation with RAGAS
  - TestsetGenerator  : builds QA dataset from your chunks
  - ragas.evaluate()  : Faithfulness, AnswerRelevancy, ContextPrecision,
                        ContextRecall, AnswerCorrectness
  - Custom            : Hit Rate @ K, MRR @ K  (chunk-ID retrieval)

Run: python src/evaluate.py
"""
import contextlib, io, json, os, pickle, sys, time, types, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # safe Unicode on Windows
from pathlib import Path

# ── Windows asyncio fix (RAGAS uses async; ProactorEventLoop causes crashes) ──
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── Load .env for local API keys (safe: ignored if file missing) ───────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — set env vars manually

# ── Compatibility shim ─────────────────────────────────────────────────────────
# ragas 0.4.3 imports ChatVertexAI from langchain_community which removed it in 0.4.x
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
class _ChatVertexAI: pass
_stub.ChatVertexAI = _ChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _stub)
# ──────────────────────────────────────────────────────────────────────────────
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from config import INDEX_DIR, OLLAMA_URL, OLLAMA_MODEL, EMBED_MODEL
from rag_engine import RAGEngine

BASE  = Path(__file__).parent.parent
DATA  = BASE / "evaluation_dataset.json"
OUT   = BASE / "evaluation_results.json"
STALE = 7  # days before auto-regenerating dataset

# ── Helpers ────────────────────────────────────────────────────────────────────

def _llm():
    """Judge LLM: Groq llama-3.3-70b-versatile (free tier)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set.\n"
            "  • Get a free key at: https://console.groq.com/keys\n"
            "  • Add it to your .env file: GROQ_API_KEY=gsk_..."
        )
    from langchain_groq import ChatGroq
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0,
                    api_key=api_key), "groq-llama-3.3-70b"

def _emb():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})

def _stale():
    """Only regenerate if dataset is missing or too small."""
    if not DATA.exists(): return True
    try: return len(json.loads(DATA.read_text())) < 5
    except: return True

# ── Step 1: Dataset ────────────────────────────────────────────────────────────

def generate_dataset(n=20):
    """
    Generate a QA evaluation dataset by asking the judge LLM to produce
    a question + reference answer from each sampled chunk.

    Uses direct LLM calls instead of RAGAS TestsetGenerator to avoid
    JSON-parsing failures with open-source models (Llama, Mixtral, etc.)
    that wrap their JSON output in conversational text.
    """
    import random, re
    from langchain_core.messages import SystemMessage, HumanMessage

    meta = pickle.loads((INDEX_DIR / "metadata.pkl").read_bytes())
    pool = [m for m in meta if len(m["text"]) > 300 and "<unk>" not in m["text"]]
    chunks = random.sample(pool, min(len(pool), n))
    print(f"  {len(chunks)} chunks sampled -> generating {n} QA pairs ...")

    llm, _ = _llm()
    system = SystemMessage(content=(
        "You are a research QA generator. Given a passage from a scientific paper, "
        "produce one specific, answerable question and a concise reference answer. "
        "Respond with ONLY a JSON object — no extra text, no markdown fences:\n"
        '{"question": "...", "reference_answer": "..."}'
    ))

    dataset = []
    for i, m in enumerate(chunks):
        try:
            resp = llm.invoke([system, HumanMessage(content=m["text"][:2000])])
            raw = resp.content if hasattr(resp, "content") else str(resp)
            # Robustly extract the first {...} block regardless of surrounding text
            match = re.search(r"\{[^{}]*\"question\"[^{}]*\"reference_answer\"[^{}]*\}", raw, re.S)
            if not match:
                match = re.search(r"\{.*?\}", raw, re.S)
            qa = json.loads(match.group()) if match else {}
            q  = qa.get("question", "").strip()
            ra = qa.get("reference_answer", "").strip()
            if not q:
                print(f"  [{i+1}/{n}] skipped — no question parsed")
                continue
            dataset.append({
                "question":             q,
                "reference_answer":     ra,
                "ground_truth_text":    m["text"],
                "ground_truth_chunk_id": m.get("chunk_id", ""),
            })
            print(f"  [{i+1}/{n}] Q: {q[:90]}")
        except Exception as e:
            print(f"  [{i+1}/{n}] skipped — {type(e).__name__}: {str(e)[:80]}")

    if not dataset:
        raise RuntimeError("Dataset generation produced 0 samples. Check your judge LLM.")

    DATA.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    print(f"  [OK] Saved {len(dataset)} QA pairs -> {DATA.name}")
    return dataset

# ── Step 2: Retrieval ──────────────────────────────────────────────────────────

def eval_retrieval(dataset, engine, k=5):
    hits, mrr = 0, 0.0
    for item in dataset:
        if not item.get("ground_truth_chunk_id"): continue
        ids = [c.chunk_id for c in engine.query(item["question"], top_k=k, top_n=k, verbose=False, rewrite_query=False).chunks]
        if item["ground_truth_chunk_id"] in ids:
            rank = ids.index(item["ground_truth_chunk_id"]) + 1
            hits += 1; mrr += 1 / rank
            item["retrieval_rank"] = rank
        else: item["retrieval_rank"] = None
    n = len(dataset) or 1
    return {"hit_rate": round(hits/n, 4), "mrr": round(mrr/n, 4)}

# ── Step 3: RAGAS ──────────────────────────────────────────────────────────────

def eval_ragas(dataset, engine):
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    llm_raw, _ = _llm()
    llm, emb = LangchainLLMWrapper(llm_raw), LangchainEmbeddingsWrapper(_emb())

    samples = []
    for item in dataset:
        engine.clear_history()
        r = engine.query(item["question"], verbose=False, rewrite_query=False)
        item["generated_answer"] = r.answer
        samples.append(SingleTurnSample(
            user_input=item["question"], response=r.answer,
            retrieved_contexts=[c.text for c in r.chunks] or [""],
            reference=item.get("reference_answer") or item.get("ground_truth_text", ""),
        ))

    results = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=emb),
                 ContextPrecision(llm=llm), ContextRecall(llm=llm), AnswerCorrectness(llm=llm, embeddings=emb)],
    )
    return {k: round(float(v), 4) for k, v in results.to_pandas().mean(numeric_only=True).items()}

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="RAG Eval via RAGAS — run with no args")
    p.add_argument("--force-regen", action="store_true")
    p.add_argument("--samples",     type=int, default=20)
    p.add_argument("--top-k",       type=int, default=5)
    args = p.parse_args()

    _, judge = _llm()
    print(f"\nRAG EVALUATION  {datetime.now():%Y-%m-%d %H:%M}  judge={judge}")

    dataset = generate_dataset(args.samples) if args.force_regen or _stale() else json.loads(DATA.read_text())
    print(f"Dataset: {len(dataset)} samples")

    with contextlib.redirect_stdout(io.StringIO()): engine = RAGEngine()

    r_scores = eval_retrieval(dataset, engine, k=args.top_k)
    g_scores = eval_ragas(dataset, engine)

    print("\n-- Retrieval ---------------------------")
    for k, v in r_scores.items(): print(f"  {k:<22} {v:.4f}  {'#'*int(v*20)}")
    print("\n-- Generation (RAGAS) ------------------")
    for k, v in g_scores.items(): print(f"  {k:<22} {v:.4f}  {'#'*int(v*20)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {"meta": {"timestamp": ts, "judge": judge, "samples": len(dataset), "top_k": args.top_k},
              "summary": {"retrieval": r_scores, "generation": g_scores}, "per_sample": dataset}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (BASE / f"evaluation_results_{ts}.json").write_text(json.dumps(result, indent=2))
    print(f"\n[OK] Saved -> {OUT.name}")

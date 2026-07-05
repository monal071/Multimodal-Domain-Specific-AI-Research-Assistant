"""
RAG Evaluation Script
Evaluates Retrieval (Hit Rate, MRR) and Generation (Faithfulness, Relevance)
using an LLM-as-a-judge approach.
"""

import json
import os
import pickle
import random
import textwrap
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from config import INDEX_DIR, OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
from rag_engine import RAGEngine

EVAL_DATASET_PATH = Path("evaluation_dataset.json")

def _query_ollama(prompt: str, model: str = OLLAMA_MODEL, max_tokens: int = 500, temperature: float = 0.0) -> str:
    """Helper to query Ollama synchronously."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()
        
        # strip DeepSeek <think> tags if present
        if "<think>" in result and "</think>" in result:
            result = result.split("</think>")[-1].strip()
        return result
    except Exception as e:
        print(f"[Ollama Error] {e}")
        return ""


def generate_synthetic_dataset(num_samples: int = 10, save_path: Path = EVAL_DATASET_PATH):
    """
    Randomly selects chunks and uses Ollama to generate a question 
    that the chunk perfectly answers.
    """
    print(f"Generating {num_samples} synthetic evaluation questions...")
    
    with open(INDEX_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
        
    if not metadata:
        print("No metadata found. Run indexing first.")
        return

    # Filter out very short chunks
    valid_chunks = [m for m in metadata if len(m["text"]) > 200]
    
    if len(valid_chunks) < num_samples:
        print(f"Warning: Only {len(valid_chunks)} valid chunks found.")
        num_samples = len(valid_chunks)

    selected_chunks = random.sample(valid_chunks, num_samples)
    dataset = []

    for m in tqdm(selected_chunks, desc="Generating QA pairs"):
        text = m["text"]
        prompt = textwrap.dedent(f"""\
            Given the following text excerpt, generate exactly ONE specific question that can be answered solely based on this text.
            Do not ask generic questions like "What is this text about?". Ask a specific technical or factual question.
            Output ONLY the question, without any other text.
            
            TEXT:
            {text}
            
            QUESTION:
        """)
        
        question = _query_ollama(prompt, model=OLLAMA_MODEL, max_tokens=1000)
        
        if question:
            dataset.append({
                "question": question,
                "ground_truth_chunk_id": m["chunk_id"],
                "ground_truth_text": text,
                "source_file": m["source_file"]
            })
            
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Saved {len(dataset)} evaluation pairs to {save_path}")


def evaluate_retrieval(dataset: list[dict], engine: RAGEngine, top_k: int = 5):
    """
    Evaluates Hit Rate and Mean Reciprocal Rank (MRR) for the retrieval step.
    """
    print(f"\n--- Evaluating Retrieval (Top-{top_k}) ---")
    
    hits = 0
    mrr_sum = 0.0
    
    for item in tqdm(dataset, desc="Evaluating Retrieval"):
        question = item["question"]
        gt_chunk_id = item["ground_truth_chunk_id"]
        
        # Run retrieval (using hybrid search)
        result = engine.query(question, top_k=top_k, top_n=top_k, verbose=False)
        retrieved_ids = [c.chunk_id for c in result.chunks]
        
        # Calculate Hit Rate and MRR
        if gt_chunk_id in retrieved_ids:
            hits += 1
            rank = retrieved_ids.index(gt_chunk_id) + 1
            mrr_sum += 1.0 / rank
            item["retrieval_rank"] = rank
        else:
            item["retrieval_rank"] = None
            
    if not dataset:
        print("Dataset is empty. Skipping retrieval evaluation.")
        return 0.0, 0.0
        
    hit_rate = hits / len(dataset)
    mrr = mrr_sum / len(dataset)
    
    print(f"Retrieval Hit Rate @ {top_k}: {hit_rate:.2%}")
    print(f"Retrieval MRR @ {top_k}:      {mrr:.4f}")
    return hit_rate, mrr


def evaluate_generation(dataset: list[dict], engine: RAGEngine, judge_model: str = OLLAMA_MODEL):
    """
    Evaluates Faithfulness and Relevance using the LLM as a judge.
    """
    print(f"\n--- Evaluating Generation (Judge Model: {judge_model}) ---")
    
    faithfulness_scores = []
    relevance_scores = []
    
    for item in tqdm(dataset, desc="Evaluating Generation"):
        question = item["question"]
        gt_text = item["ground_truth_text"]
        
        # 1. Generate Answer using RAG
        # Force a fresh history for each eval
        engine.clear_history()
        result = engine.query(question, verbose=False)
        answer = result.answer
        context = "\n".join([c.text for c in result.chunks])
        
        item["generated_answer"] = answer
        
        # 2. Score Faithfulness (1-5): Is the answer hallucinated or derived from context?
        faith_prompt = textwrap.dedent(f"""\
            You are an impartial judge. Your task is to evaluate the faithfulness of an AI-generated answer.
            Compare the generated answer against the provided context.
            Score from 1 to 5, where:
            1: The answer contradicts the context or includes major hallucinations.
            3: The answer is partially faithful but includes some unverified information.
            5: The answer is completely faithful to the provided context and contains no outside information.
            
            Output ONLY a single integer between 1 and 5.
            
            CONTEXT:
            {context}
            
            ANSWER:
            {answer}
            
            FAITHFULNESS SCORE:
        """)
        
        f_score_str = _query_ollama(faith_prompt, model=judge_model, max_tokens=500)
        try:
            # Extract just the first digit
            f_score = int(''.join(filter(str.isdigit, f_score_str))[0])
            faithfulness_scores.append(f_score)
            item["faithfulness_score"] = f_score
        except:
            item["faithfulness_score"] = None
            
        # 3. Score Relevance (1-5): Does the answer address the question?
        rel_prompt = textwrap.dedent(f"""\
            You are an impartial judge. Your task is to evaluate the relevance of an AI-generated answer to a user's question.
            Score from 1 to 5, where:
            1: The answer is completely irrelevant to the question.
            3: The answer addresses the topic but misses the specific question asked.
            5: The answer directly and clearly addresses the user's question without unnecessary tangents.
            
            Output ONLY a single integer between 1 and 5.
            
            QUESTION:
            {question}
            
            ANSWER:
            {answer}
            
            RELEVANCE SCORE:
        """)
        
        r_score_str = _query_ollama(rel_prompt, model=judge_model, max_tokens=500)
        try:
            r_score = int(''.join(filter(str.isdigit, r_score_str))[0])
            relevance_scores.append(r_score)
            item["relevance_score"] = r_score
        except:
            item["relevance_score"] = None
            
    if not dataset:
        print("Dataset is empty. Skipping generation evaluation.")
        return 0.0, 0.0

    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
    avg_rel = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    
    print(f"Average Faithfulness Score (1-5): {avg_faith:.2f}")
    print(f"Average Relevance Score (1-5):    {avg_rel:.2f}")
    
    return avg_faith, avg_rel


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate new synthetic dataset")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--judge-model", type=str, default=OLLAMA_MODEL, help="Ollama model to use as the judge")
    args = parser.parse_args()
    
    if args.generate or not EVAL_DATASET_PATH.exists():
        generate_synthetic_dataset(num_samples=args.samples)
        
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print("Loading RAG Engine...")
    # Suppress normal output for eval
    import sys
    original_stdout = sys.stdout
    with open(os.devnull, 'w') as f:
        sys.stdout = f
        engine = RAGEngine()
        # Mock print for generation to prevent streaming output during eval
        engine._generate = lambda prompt: _query_ollama(prompt, model=OLLAMA_MODEL, max_tokens=1000)
    sys.stdout = original_stdout
    
    evaluate_retrieval(dataset, engine, top_k=5)
    evaluate_generation(dataset, engine, judge_model=args.judge_model)
    
    # Save results
    results_path = Path("evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    print(f"\nDetailed evaluation results saved to {results_path}")

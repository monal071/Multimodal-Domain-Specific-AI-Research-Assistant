"""
Research Assistant — Gradio UI
================================
Simple chat interface wrapping the existing RAGEngine.

Usage:
    pip install gradio
    python src/gradio_app.py
    # Opens at http://localhost:7860
"""

import sys
import json
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import gradio as gr
import requests as _req
from config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, MAX_NEW_TOKENS
from rag_engine import RAGEngine

# ── Load engine once at startup ───────────────────────────────────────────────
print("Loading RAG engine...")
engine = RAGEngine()
print("Ready.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Core chat function  (streaming generator)
# ══════════════════════════════════════════════════════════════════════════════

def chat(message: str, history: list, rewrite: bool):
    """
    Gradio streaming chat handler.
    Yields (history, sources_markdown) pairs as tokens arrive.
    """
    from config import RETRIEVAL_TOP_K, RERANK_TOP_N
    import time

    if not message.strip():
        yield history, ""
        return

    # ── 1. Rewrite query ──────────────────────────────────────────────────────
    search_query = engine._generate_hyde_document(message) if rewrite else message
    rewrite_note = (
        f"*↩ Rewritten as: `{search_query}`*\n\n"
        if rewrite and search_query != message else ""
    )

    # ── 2. Retrieve + rerank ──────────────────────────────────────────────────
    q_vec      = engine._embed_query(search_query)
    candidates = engine._hybrid_search(search_query, q_vec, RETRIEVAL_TOP_K)
    ranked     = engine._rerank(message, candidates, RERANK_TOP_N)

    # ── 3. Build sources markdown ─────────────────────────────────────────────
    sources_md = "### 📚 Retrieved Sources\n\n"
    for i, c in enumerate(ranked, 1):
        section   = " › ".join(c.section_path) if c.section_path else (c.heading or "—")
        rrf       = f"RRF `{c.score:.4f}`"
        rrank     = f" · Rerank `{c.rerank_score:.3f}`" if c.rerank_score is not None else ""
        flags     = (" · `table`" if c.has_table else "") + (" · `formula`" if c.has_formula else "")
        sources_md += (
            f"**[{i}] {c.source_file}**\n"
            f"*{section}* · {rrf}{rrank}{flags}\n\n"
            f"> {c.text[:300].replace(chr(10), ' ')}{'…' if len(c.text) > 300 else ''}\n\n"
            "---\n\n"
        )

    # ── 4. Build prompt + stream answer ──────────────────────────────────────
    context = engine._build_context(ranked)
    prompt  = engine._build_prompt(message, context)

    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    answer_parts = []
    in_think = False
    buffer   = ""

    try:
        with _req.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":   OLLAMA_MODEL,
                "prompt":  prompt,
                "stream":  True,
                "options": {
                    "num_predict":    MAX_NEW_TOKENS,
                    "temperature":    0.0,
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
                data  = json.loads(raw_line)
                token = data.get("response", "")

                if token:
                    buffer += token
                    answer_parts.append(token)

                    # suppress <think>...</think> blocks from the UI
                    if "<think>" in buffer:
                        in_think = True
                    if in_think:
                        if "</think>" in buffer:
                            in_think = False
                            buffer   = buffer.split("</think>")[-1]
                        else:
                            if data.get("done"):
                                break
                            continue  # still in think block, don't yield

                    history[-1]["content"] = rewrite_note + buffer
                    yield history, sources_md

                if data.get("done"):
                    break

    except _req.RequestException as e:
        history[-1]["content"] = f"❌ Ollama error: {e}"
        yield history, sources_md
        return

    # ── 5. Final — save to engine memory ─────────────────────────────────────
    full_answer = "".join(answer_parts)
    if "<think>" in full_answer and "</think>" in full_answer:
        full_answer = full_answer.split("</think>")[-1].strip()
    engine._history.append({"question": message, "answer": full_answer})

    yield history, sources_md


def clear_history():
    """Clear the RAGEngine conversation memory."""
    engine.clear_history()
    return [], ""


# ══════════════════════════════════════════════════════════════════════════════
#  Gradio UI
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="Research Assistant") as demo:

    gr.Markdown(
        "# 🔬 Research Assistant\n"
        "**Multimodal Domain-Specific AI** · RAG + LoRA · "
        f"Model: `{OLLAMA_MODEL}` · Hybrid BM25 + FAISS retrieval"
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=520,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a question about the research papers…",
                    label="",
                    scale=5,
                    lines=1,
                    max_lines=4,
                    autofocus=True,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=80)

            with gr.Row():
                clear_btn = gr.Button("🗑 Clear history", size="sm")
                rewrite_toggle = gr.Checkbox(
                    value=True,
                    label="Query rewriting",
                    info="Rewrites your question into precise academic search terms",
                    scale=2,
                )

        with gr.Column(scale=2):
            sources_box = gr.Markdown(
                value="*Sources will appear here after each answer.*",
                label="Retrieved Sources",
                height=600,
            )

    # ── Example questions ─────────────────────────────────────────────────────
    gr.Examples(
        examples=[
            ["What is Flash Attention and why is it more memory-efficient?"],
            ["How does XLNet differ from BERT in pretraining?"],
            ["Explain the Switch Transformer Mixture-of-Experts architecture."],
            ["What is LoRA fine-tuning and how does it reduce trainable parameters?"],
            ["How does chain-of-thought prompting improve reasoning in LLMs?"],
        ],
        inputs=msg_box,
        label="Example questions",
    )

    # ── Event wiring ──────────────────────────────────────────────────────────
    submit_args = dict(
        fn=chat,
        inputs=[msg_box, chatbot, rewrite_toggle],
        outputs=[chatbot, sources_box],
    )

    msg_box.submit(**submit_args).then(lambda: "", outputs=msg_box)
    send_btn.click(**submit_args).then(lambda: "", outputs=msg_box)
    clear_btn.click(fn=clear_history, outputs=[chatbot, sources_box])


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )

"""
pipeline_03_finetune.py
───────────────────────
Unsloth LoRA fine-tuning pipeline for the Research Assistant project.

Loads the RAG-aware chat-format dataset produced by pipeline_03a_gen_dataset.py
(OpenAI messages schema), fine-tunes the chosen base model with 4-bit QLoRA
via Unsloth's FastLanguageModel + SFTTrainer using the model's native chat
template, and saves the LoRA adapter (+ optionally exports to GGUF for Ollama).

Dataset format expected (one JSON object per line):
    {
      "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "<context>...\\n\\nQuery: ..."},
        {"role": "assistant", "content": "### Core Contribution\\n..."}
      ]
    }

Prerequisites:
  1. Generate dataset first:
       python src/pipeline_03a_gen_dataset.py
  2. Install Unsloth (Linux / WSL2 / Colab recommended):
       pip install "unsloth[colab-new]>=2024.9" trl>=0.9.0 peft>=0.11.0 datasets>=2.19.0

Usage:
  python src/pipeline_03_finetune.py
  python src/pipeline_03_finetune.py --test          # 10-step smoke test on 5 samples
  python src/pipeline_03_finetune.py --epochs 1      # quick single-epoch run
  python src/pipeline_03_finetune.py --export-gguf   # also export GGUF after training
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Unsloth must be imported FIRST, before transformers/trl/peft ───────────────
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
except ImportError as e:
    print(
        "\n[ERROR] Unsloth is not installed or failed to import.\n"
        "  Install on Linux / WSL2 / Colab:\n"
        "    pip install 'unsloth[colab-new]>=2024.9' trl>=0.9.0 peft>=0.11.0 datasets>=2.19.0\n"
    )
    raise

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── path bootstrap ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FT_MODEL_ID, FT_DATASET_PATH, FT_OUTPUT_DIR, FT_GGUF_DIR,
    FT_LORA_R, FT_LORA_ALPHA, FT_LORA_DROPOUT, FT_TARGET_MODULES,
    FT_MAX_SEQ_LEN, FT_LOAD_IN_4BIT,
    FT_BATCH_SIZE, FT_GRAD_ACCUM, FT_EPOCHS, FT_LR,
    FT_WARMUP_STEPS, FT_WEIGHT_DECAY, FT_LR_SCHEDULER,
    FT_EXPORT_GGUF, FT_GGUF_QUANT,
)

# Required headings used to validate each training row's assistant content.
_REQUIRED_HEADINGS = [
    "### Core Contribution",
    "### Architectural & Mathematical Mechanics",
    "### Empirical Results",
    "### Citations",
]


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE  — load the saved LoRA adapter and run generation
#  (imported by rag_engine.py; no training dependencies needed at import time)
# ══════════════════════════════════════════════════════════════════════════════

class FinetunedGenerator:
    """
    Loads the fine-tuned Qwen3-8B LoRA adapter via Unsloth and runs inference.
    Instantiated once at RAGEngine startup and kept in GPU memory.
    """

    def __init__(self, adapter_path: Path, load_in_4bit: bool = True):
        from config import FT_INFERENCE_MAX_NEW_TOKENS

        print(f"[FT] Loading fine-tuned Qwen3-8B from: {adapter_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = str(adapter_path),
            max_seq_length = FT_INFERENCE_MAX_NEW_TOKENS + 512,
            dtype          = None,
            load_in_4bit   = load_in_4bit,
        )
        FastLanguageModel.for_inference(model)   # 2× speed
        self._model = model
        self._tok   = tokenizer
        print(f"[FT] ✓ Unsloth FastLanguageModel ready  (4bit={load_in_4bit})")

    # ──────────────────────────────────────────────────────────────────────────

    def generate(self, prompt: str, max_new_tokens: int = 2048) -> str:
        """Generate text from a plain-string prompt. Strips Qwen3 <think> tags."""
        import torch

        inputs = self._tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens     = max_new_tokens,
                temperature        = 0.7,
                do_sample          = False,
                repetition_penalty = 1.05,
                pad_token_id       = self._tok.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        answer = self._tok.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
        print(answer)

        if "<think>" in answer:
            answer = answer.split("</think>")[-1].strip() if "</think>" in answer \
                     else "[Thinking truncated — increase FT_INFERENCE_MAX_NEW_TOKENS.]"

        return answer

    def generate_short(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Lightweight call for HyDE query rewriting."""
        return self.generate(prompt, max_new_tokens=max_new_tokens)


# ── dataset helpers ─────────────────────────────────────────────────────────────

def load_dataset_from_jsonl(path: Path, test_mode: bool = False) -> list:
    """
    Load the RAG-aware chat-format JSONL dataset.

    Each line must be a JSON object with a 'messages' key containing a list
    of dicts with roles: system, user, assistant.  Rows that fail validation
    are skipped with a warning.
    """
    if not path.exists():
        print(f"[error] Dataset not found: {path}")
        print("  Run:  python src/pipeline_03a_gen_dataset.py")
        sys.exit(1)

    records = []
    skipped = 0

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            messages = obj.get("messages", [])

            # Validate structure: need exactly 3 messages with the right roles
            if len(messages) < 3:
                skipped += 1
                continue
            roles = [m.get("role") for m in messages[:3]]
            if roles != ["system", "user", "assistant"]:
                skipped += 1
                continue

            # Validate assistant content contains all 4 section headings
            assistant_content = messages[2].get("content", "")
            if not all(h in assistant_content for h in _REQUIRED_HEADINGS):
                skipped += 1
                continue

            records.append({"messages": messages})

            if test_mode and len(records) >= 5:
                break

    print(f"Loaded {len(records):,} valid training rows from {path.name} "
          f"({skipped} rows skipped)")
    return records


# ── GPU info ───────────────────────────────────────────────────────────────────

def _print_gpu_info():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU             : {name}  ({vram:.1f} GB VRAM)")
    else:
        print("GPU             : Not available — training will be very slow on CPU")
    bf16_ok = is_bfloat16_supported()
    print(f"bf16 support    : {'Yes' if bf16_ok else 'No (will use fp16)'}")
    return bf16_ok


# ── main training function ─────────────────────────────────────────────────────

def run_training(
    test_mode: bool = False,
    extra_epochs: int = None,
    force_export_gguf: bool = False,
):
    import torch
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    epochs = extra_epochs if extra_epochs is not None else FT_EPOCHS
    export_gguf = force_export_gguf or FT_EXPORT_GGUF

    print("=" * 62)
    print("  LoRA Fine-Tuning — Research Assistant")
    print("=" * 62)
    print(f"Base model      : {FT_MODEL_ID}")
    print(f"Engine          : Unsloth (FastLanguageModel)")
    print(f"LoRA rank       : {FT_LORA_R}  (alpha={FT_LORA_ALPHA})")
    print(f"Max seq length  : {FT_MAX_SEQ_LEN}")
    print(f"4-bit QLoRA     : {FT_LOAD_IN_4BIT}")
    print(f"Dataset format  : chat (messages)")

    supports_bf16 = _print_gpu_info()

    if test_mode:
        print("\n[TEST MODE] 10 steps, 5 samples only\n")

    print()

    # ── load base model & tokenizer via Unsloth ──────────────────────────────
    # Free any cached VRAM before loading the base model
    torch.cuda.empty_cache()
    print("Loading base model with Unsloth 4-bit quantisation...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = FT_MODEL_ID,
        max_seq_length = FT_MAX_SEQ_LEN,
        # Force all layers to GPU 0 — prevents accelerate from offloading
        # layers to CPU/disk when VRAM is tight (bitsandbytes 4-bit does
        # not support CPU offloading).
        device_map     = {"" : 0},
        dtype          = torch.bfloat16 if supports_bf16 else torch.float16,
        load_in_4bit   = FT_LOAD_IN_4BIT,
    )

    print("Attaching LoRA adapters via Unsloth...")
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = FT_LORA_R,
        lora_alpha                 = FT_LORA_ALPHA,
        lora_dropout               = FT_LORA_DROPOUT,
        target_modules             = FT_TARGET_MODULES,
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = 42,
        use_rslora                 = False,
    )

    # ── load & format dataset ────────────────────────────────────────────────
    raw_records = load_dataset_from_jsonl(FT_DATASET_PATH, test_mode=test_mode)
    dataset     = Dataset.from_list(raw_records)

    print("Tokenizing dataset with model chat template...")
    dataset = dataset.map(
        lambda examples: {
            "text": [
                tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for msgs in examples["messages"]
            ]
        },
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting dataset",
    )

    # ── SFT config (Unsloth-native) ──────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir                   = str(FT_OUTPUT_DIR / "checkpoints"),
        num_train_epochs             = epochs,
        per_device_train_batch_size  = FT_BATCH_SIZE,
        gradient_accumulation_steps  = FT_GRAD_ACCUM,
        warmup_steps                 = FT_WARMUP_STEPS if not test_mode else 2,
        max_steps                    = 10 if test_mode else -1,
        learning_rate                = FT_LR,
        fp16                         = not supports_bf16,
        bf16                         = supports_bf16,
        logging_steps                = 1 if test_mode else 10,
        optim                        = "adamw_8bit",
        weight_decay                 = FT_WEIGHT_DECAY,
        lr_scheduler_type            = FT_LR_SCHEDULER,
        seed                         = 42,
        report_to                    = "none",
        save_strategy                = "epoch" if not test_mode else "no",
        dataset_text_field           = "text",
        max_seq_length               = FT_MAX_SEQ_LEN,
        packing                      = False,
    )

    # ── build Unsloth SFTTrainer ─────────────────────────────────────────────
    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = dataset,
        args          = sft_config,
    )

    # ── train ────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer_stats = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Training runtime : {trainer_stats.metrics.get('train_runtime', 0):.1f}s")

    # ── save LoRA adapter ────────────────────────────────────────────────────
    FT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving LoRA adapter -> {FT_OUTPUT_DIR}")
    model.save_pretrained(str(FT_OUTPUT_DIR))
    tokenizer.save_pretrained(str(FT_OUTPUT_DIR))
    print("  LoRA adapter saved.")

    # ── optional: export to GGUF ─────────────────────────────────────────────
    if export_gguf and not test_mode:
        FT_GGUF_DIR.mkdir(parents=True, exist_ok=True)
        gguf_path = FT_GGUF_DIR / f"model-{FT_GGUF_QUANT}.gguf"
        print(f"\nExporting GGUF ({FT_GGUF_QUANT}) -> {gguf_path}")
        model.save_pretrained_gguf(
            str(FT_GGUF_DIR),
            tokenizer,
            quantization_method = FT_GGUF_QUANT,
        )
        print("  GGUF export complete.")
        _print_ollama_import_instructions(gguf_path)

    print("\nAll done!")
    _print_inference_example(FT_OUTPUT_DIR)


# ── post-training instructions ─────────────────────────────────────────────────

def _print_ollama_import_instructions(gguf_path: Path):
    model_name = "qwen3-8b-finetuned"   # update config.py OLLAMA_MODEL to this
    modelfile_content = f"""FROM {gguf_path.as_posix()}
PARAMETER temperature 0.7
PARAMETER num_predict 1024
SYSTEM "You are a deep learning research assistant. Answer the user's query using only the provided context. Structure your response with the following Markdown headings: ### Core Contribution, ### Architectural & Mathematical Mechanics, ### Empirical Results, ### Citations."
"""
    modelfile_path = gguf_path.parent / "Modelfile"
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    print(f"\n  To import into Ollama:")
    print(f"    ollama create {model_name} -f {modelfile_path}")
    print(f"    ollama run {model_name}")
    print(f"\n  Then update config.py:  OLLAMA_MODEL = \"{model_name}\"")


def _print_inference_example(adapter_path: Path):
    print("\n" + "─" * 62)
    print("Quick inference test (run after training):\n")
    print("""from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = r\"""" + str(adapter_path) + """\",
    max_seq_length = 2048,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)

messages = [
    {
        "role": "system",
        "content": (
            "You are a deep learning research assistant. "
            "Answer using only the provided context. "
            "Structure your response with: ### Core Contribution, "
            "### Architectural & Mathematical Mechanics, "
            "### Empirical Results, ### Citations."
        ),
    },
    {
        "role": "user",
        "content": (
            "<context>\\n"
            "[Chunk 1: Adam is an algorithm for first-order gradient-based "
            "optimization based on adaptive estimates of lower-order moments.]\\n"
            "[Chunk 2: The method computes individual adaptive learning rates "
            "from estimates of first and second moments of the gradients.]\\n"
            "</context>\\n\\n"
            "Query: How does Adam calculate parameter updates?"
        ),
    },
]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
""")


# ── entrypoint ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unsloth LoRA fine-tuning for Research Assistant (chat format)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Smoke test: 10 steps, 5 samples — verifies the pipeline works"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override FT_EPOCHS from config"
    )
    parser.add_argument(
        "--export-gguf", action="store_true",
        help="Export GGUF after training (overrides FT_EXPORT_GGUF=False in config)"
    )
    args = parser.parse_args()

    run_training(
        test_mode         = args.test,
        extra_epochs      = args.epochs,
        force_export_gguf = args.export_gguf,
    )


if __name__ == "__main__":
    main()

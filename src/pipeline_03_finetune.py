"""
pipeline_03_finetune.py
───────────────────────
Unsloth LoRA fine-tuning pipeline for the Research Assistant project.

Loads the synthetic Q&A dataset produced by pipeline_03a_gen_dataset.py,
fine-tunes the chosen base model with 4-bit QLoRA via Unsloth's SFTTrainer,
and saves the LoRA adapter (+ optionally exports to GGUF for Ollama).

Prerequisites:
  1. Generate dataset first:
       python src/pipeline_03a_gen_dataset.py
  2. Install Unsloth (Linux / WSL2 / Colab recommended):
       pip install "unsloth[colab-new]>=2024.9" trl>=0.9.0 peft>=0.11.0 datasets>=2.19.0
     On native Windows (slower, no CUDA kernels):
       pip install "unsloth[windows]>=2024.9" trl peft datasets

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

# ── Alpaca prompt template ─────────────────────────────────────────────────────
ALPACA_PROMPT = """\
Below is an instruction that describes a task. \
Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN_PLACEHOLDER = "<|end_of_text|>"  # replaced at runtime with tokenizer.eos_token


# ── dataset helpers ─────────────────────────────────────────────────────────────

def load_dataset_from_jsonl(path: Path) -> list:
    """Load Alpaca-format JSONL pairs into a list of dicts."""
    if not path.exists():
        print(f"[error] Dataset not found: {path}")
        print("  Run:  python src/pipeline_03a_gen_dataset.py")
        sys.exit(1)

    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Strip internal metadata field added by the generator
                obj.pop("_source", None)
                records.append(obj)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(records):,} training examples from {path.name}")
    return records


def formatting_func(examples: dict, eos_token: str) -> list:
    """Format raw examples into Alpaca prompt strings for SFTTrainer."""
    texts = []
    instructions = examples["instruction"]
    outputs       = examples["output"]

    for instruction, output in zip(instructions, outputs):
        text = ALPACA_PROMPT.format(instruction, output) + eos_token
        texts.append(text)

    return texts


# ── environment check ──────────────────────────────────────────────────────────

def _check_unsloth_available() -> bool:
    try:
        import unsloth  # noqa: F401
        return True
    except ImportError:
        return False


def _print_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU             : {name}  ({vram:.1f} GB VRAM)")
            # Determine if bf16 is supported (Ampere and newer)
            cap = torch.cuda.get_device_capability()
            bf16_ok = cap[0] >= 8
            print(f"bf16 support    : {'Yes' if bf16_ok else 'No (will use fp16)'}")
            return bf16_ok
        else:
            print("GPU             : Not available — training will be very slow on CPU")
            return False
    except Exception:
        return False


# ── main training function ─────────────────────────────────────────────────────

def run_training(
    test_mode: bool = False,
    extra_epochs: int = None,
    force_export_gguf: bool = False,
):
    # ── imports (deferred so the script can be imported without heavy deps) ────
    if not _check_unsloth_available():
        print(
            "\n[error] Unsloth is not installed.\n"
            "  On Linux / WSL2 / Colab:\n"
            "    pip install 'unsloth[colab-new]>=2024.9' trl>=0.9.0 peft>=0.11.0 datasets>=2.19.0\n"
            "  On native Windows:\n"
            "    pip install 'unsloth[windows]>=2024.9' trl peft datasets\n"
        )
        sys.exit(1)

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    print("=" * 62)
    print("  Unsloth LoRA Fine-Tuning — Research Assistant")
    print("=" * 62)
    print(f"Base model      : {FT_MODEL_ID}")
    print(f"LoRA rank       : {FT_LORA_R}  (alpha={FT_LORA_ALPHA})")
    print(f"Max seq length  : {FT_MAX_SEQ_LEN}")
    print(f"4-bit QLoRA     : {FT_LOAD_IN_4BIT}")

    supports_bf16 = _print_gpu_info()
    epochs = extra_epochs if extra_epochs is not None else FT_EPOCHS
    export_gguf = force_export_gguf or FT_EXPORT_GGUF

    if test_mode:
        print("\n[TEST MODE] 10 steps, 5 samples only\n")

    print()

    # ── load base model ────────────────────────────────────────────────────────
    print("Loading base model with 4-bit quantisation...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = FT_MODEL_ID,
        max_seq_length = FT_MAX_SEQ_LEN,
        dtype          = None,           # auto-detect (bf16 on Ampere, fp16 elsewhere)
        load_in_4bit   = FT_LOAD_IN_4BIT,
    )

    # ── attach LoRA adapters ───────────────────────────────────────────────────
    print("Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r                    = FT_LORA_R,
        lora_alpha           = FT_LORA_ALPHA,
        lora_dropout         = FT_LORA_DROPOUT,
        target_modules       = FT_TARGET_MODULES,
        bias                 = "none",
        use_gradient_checkpointing = "unsloth",  # Unsloth's optimised checkpointing
        random_state         = 42,
        use_rslora           = False,   # set True to use Rank-Stabilized LoRA
    )

    # ── load & format dataset ──────────────────────────────────────────────────
    raw_records = load_dataset_from_jsonl(FT_DATASET_PATH)

    if test_mode:
        raw_records = raw_records[:5]

    eos_token = tokenizer.eos_token or EOS_TOKEN_PLACEHOLDER
    dataset   = Dataset.from_list(raw_records)

    # ── training arguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = str(FT_OUTPUT_DIR / "checkpoints"),
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
        report_to                    = "none",   # set "wandb" or "tensorboard" if desired
        save_strategy                = "epoch" if not test_mode else "no",
    )

    # ── build trainer ──────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = dataset,
        dataset_text_field = None,   # we use formatting_func instead
        formatting_func = lambda examples: formatting_func(examples, eos_token),
        max_seq_length  = FT_MAX_SEQ_LEN,
        dataset_num_proc = 2,
        packing          = False,    # True speeds up training on short sequences
        args             = training_args,
    )

    # ── train ──────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer_stats = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Peak VRAM used : {trainer_stats.metrics.get('train_runtime', 0):.1f}s")

    # ── save LoRA adapter ──────────────────────────────────────────────────────
    FT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving LoRA adapter -> {FT_OUTPUT_DIR}")
    model.save_pretrained(str(FT_OUTPUT_DIR))
    tokenizer.save_pretrained(str(FT_OUTPUT_DIR))
    print("  LoRA adapter saved.")

    # ── optional: export to GGUF ───────────────────────────────────────────────
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
    model_name = "deepseek-r1:8b"   # overwrite the base model — app.py needs no changes
    modelfile_content = f"""FROM {gguf_path.as_posix()}
PARAMETER temperature 0.7
PARAMETER num_predict 512
SYSTEM "You are an expert AI/ML research assistant. Answer questions accurately based on research papers."
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

prompt = \"\"\"Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
What is the purpose of LoRA in large language model fine-tuning?

### Response:
\"\"\"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
""")


# ── entrypoint ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unsloth LoRA fine-tuning for Research Assistant"
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

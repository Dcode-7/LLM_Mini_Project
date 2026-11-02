"""Fine-tune Qwen2.5-0.5B-Instruct on the college helpdesk data with LoRA or QLoRA (Lecture 4).

    # LoRA  (frozen base in bf16/fp16; fp32 on CPU)
    python finetune/train_lora.py

    # QLoRA (frozen base loaded in 4-bit NF4 + double quantization) - needs an NVIDIA GPU
    python finetune/train_lora.py --qlora

Run finetune/prepare_dataset.py first. Outputs:
    finetune/adapters/college-<lora|qlora>/      the adapter only (a few MB) - base model still required
    finetune/results/train_<lora|qlora>.json      config, parameter counts, loss curve, memory, time
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--qlora", action="store_true", help="load the frozen base in 4-bit (NF4)")
    p.add_argument("--r", type=int, default=8, help="LoRA rank (bottleneck width)")
    p.add_argument("--alpha", type=int, default=16, help="LoRA alpha; update is scaled by alpha/r")
    p.add_argument("--dropout", type=float, default=0.05, help="dropout on the LoRA path")
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-len", type=int, default=1024)
    return p.parse_args()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize(example, tok, max_len):
    """Train only on the assistant answer: prompt tokens get label -100."""
    msgs = example["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = prompt + msgs[-1]["content"] + tok.eos_token
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = ([-100] * len(prompt_ids) + ids[len(prompt_ids):])[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def main():
    args = parse_args()
    method = "qlora" if args.qlora else "lora"
    output_dir = HERE / "adapters" / f"college-{method}"
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 1-2. Load the base model (optionally in 4-bit for QLoRA)
    if args.qlora:
        if not cuda:
            raise SystemExit("QLoRA needs a CUDA GPU (bitsandbytes). Use plain LoRA on CPU, or run on Colab.")
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NormalFloat4 - suits normally-distributed weights
            bnb_4bit_use_double_quant=True,     # quantize the quantization constants too
            bnb_4bit_compute_dtype=dtype,       # storage precision != computation precision
        )
        model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb, device_map={"": 0})
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
        if cuda:
            model.to("cuda")

    # bitsandbytes packs two 4-bit weights per stored element, so count those twice
    base_params = sum(p.numel() * (2 if type(p).__name__ == "Params4bit" else 1) for p in model.parameters())
    base_mem_mb = model.get_memory_footprint() / 1e6

    # 3. Attach LoRA: base weights are frozen, only A and B are trainable
    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=args.target_modules.split(","),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Base weights in memory: {base_mem_mb:.0f} MB | trainable LoRA params: {trainable:,}")

    # 4. Data
    train_ds = [tokenize(r, tok, args.max_len) for r in load_jsonl(HERE / "data" / "train.jsonl")]
    val_ds = [tokenize(r, tok, args.max_len) for r in load_jsonl(HERE / "data" / "val.jsonl")]

    # 5. Train only the LoRA parameters
    targs = TrainingArguments(
        output_dir=str(HERE / "outputs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=bf16,
        fp16=cuda and not bf16,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",  # paged optimizer absorbs memory spikes
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    final_eval = trainer.evaluate()

    # 6. Save ONLY the adapter + tokenizer
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    adapter_mb = sum(f.stat().st_size for f in output_dir.glob("*.safetensors")) / 1e6

    history = trainer.state.log_history
    results = {
        "method": method,
        "base_model": args.base_model,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "config": {"r": args.r, "alpha": args.alpha, "dropout": args.dropout, "epochs": args.epochs,
                   "lr": args.lr, "batch_size": args.batch_size, "grad_accum": 2,
                   "target_modules": args.target_modules.split(","),
                   "quantization": "4-bit NF4 + double quant" if args.qlora else str(dtype).replace("torch.", "")},
        "base_params": base_params,
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / base_params, 4),
        "base_weight_memory_mb": round(base_mem_mb, 1),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if cuda else None,
        "adapter_size_mb": round(adapter_mb, 2),
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "train_seconds": round(train_secs, 1),
        "final_eval_loss": round(final_eval["eval_loss"], 4),
        "train_loss": [{"step": h["step"], "loss": h["loss"]} for h in history if "loss" in h],
        "eval_loss": [{"epoch": h["epoch"], "loss": h["eval_loss"]} for h in history if "eval_loss" in h],
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"train_{method}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k not in ("train_loss", "eval_loss")}, indent=2))
    print(f"Adapter saved to {output_dir}. Serve it with LLM_BACKEND=lora LORA_ADAPTER_PATH={output_dir}")


if __name__ == "__main__":
    main()

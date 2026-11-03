"""Evaluate base vs base + LoRA vs base + QLoRA on the validation set.

    python finetune/evaluate_lora.py
"""

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from support_rag import FALLBACK_ANSWER  # noqa: E402

SAMPLE_COUNT = 6


def f1(pred: str, ref: str) -> float:
    p, r = re.findall(r"\w+", pred.lower()), re.findall(r"\w+", ref.lower())
    common = sum(min(p.count(w), r.count(w)) for w in set(p))
    if not p or not r or not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def facts(pred: str, ref: str) -> float | None:
    nums = set(re.findall(r"\d[\d,.]*\d|\d", ref))
    if not nums:
        return None
    return sum(n in pred for n in nums) / len(nums)


def generate_all(model, tok, rows):
    preds, t0 = [], time.time()
    for row in rows:
        prompt = tok.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False, repetition_penalty=1.05)
        preds.append(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return preds, (time.time() - t0) * 1000 / len(rows)


def score(preds, rows):
    f1s, fact_scores, refusals, n_off = [], [], 0, 0
    for pred, row in zip(preds, rows):
        ref = row["messages"][-1]["content"]
        if ref == FALLBACK_ANSWER:
            n_off += 1
            refusals += "don't have enough information" in pred.lower()
        else:
            f1s.append(f1(pred, ref))
            if (fs := facts(pred, ref)) is not None:
                fact_scores.append(fs)
    return {
        "token_f1": round(sum(f1s) / len(f1s), 3),
        "exact_facts": round(sum(fact_scores) / len(fact_scores), 3),
        "refusal_acc": round(refusals / max(n_off, 1), 3),
    }


def load_base(name, qlora):
    if qlora:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        return AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map={"": 0}).eval()
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    return m.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(HERE / "data" / "val.jsonl", encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(args.base_model)

    runs = [("base", None, False)]
    for method in ("lora", "qlora"):
        if (HERE / "adapters" / f"college-{method}").exists():
            runs.append((method, HERE / "adapters" / f"college-{method}", method == "qlora"))

    metrics, predictions = {}, {}
    for name, adapter, qlora in runs:
        model = load_base(args.base_model, qlora)
        if adapter:
            model = PeftModel.from_pretrained(model, adapter).eval()
        preds, ms = generate_all(model, tok, rows)
        metrics[name] = {**score(preds, rows), "ms_per_answer": round(ms)}
        predictions[name] = preds
        print(name, metrics[name])
        del model
        gc.collect()
        torch.cuda.empty_cache()

    answerable = [i for i, r in enumerate(rows) if r["messages"][-1]["content"] != FALLBACK_ANSWER]
    off_topic = [i for i, r in enumerate(rows) if r["messages"][-1]["content"] == FALLBACK_ANSWER]
    picks = answerable[:: max(1, len(answerable) // (SAMPLE_COUNT - 1))][: SAMPLE_COUNT - 1] + off_topic[:1]
    samples = [{
        "question": rows[i]["messages"][1]["content"].rsplit("Student question: ", 1)[-1],
        "reference": rows[i]["messages"][-1]["content"],
        **{name: predictions[name][i] for name in predictions},
    } for i in picks]

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "eval.json").write_text(
        json.dumps({"val_examples": len(rows), "metrics": metrics, "samples": samples}, indent=2, ensure_ascii=False))
    print(f"\n{'model':8} {'token_f1':>9} {'facts':>7} {'refusal':>8} {'ms/ans':>7}")
    for name, m in metrics.items():
        print(f"{name:8} {m['token_f1']:9.3f} {m['exact_facts']:7.3f} {m['refusal_acc']:8.2f} {m['ms_per_answer']:7}")


if __name__ == "__main__":
    main()

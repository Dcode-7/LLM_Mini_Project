"""Streamlit "Fine-tuning" page: shows the LoRA / QLoRA training and evaluation results.

The results are produced offline by finetune/train_lora.py and finetune/evaluate_lora.py and
committed as small JSON files, so the hosted app needs no torch/GPU to display them.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from finetune.lora_math import lora_params

RESULTS = Path(__file__).resolve().parent / "finetune" / "results"
SERIES_COLORS = {"LoRA": "#2a78d6", "QLoRA": "#eb6834"}   # fixed categorical order: slot 1, slot 2
LABELS = {"base": "Base (no fine-tuning)", "lora": "Base + LoRA", "qlora": "Base + QLoRA"}


def _load(name):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _mb(v):
    return "—" if v is None else f"{v:,.0f} MB" if v >= 10 else f"{v:.1f} MB"


def render_finetuning():
    st.title("Fine-tuning the LLM (LoRA / QLoRA)")
    st.markdown(
        "The base weights stay **frozen**; only the small low-rank matrices **A** and **B** are trained:"
    )
    st.latex(r"W' = W_0 + \frac{\alpha}{r}\,BA")

    runs = {m: r for m in ("lora", "qlora") if (r := _load(f"train_{m}.json"))}
    evaluation = _load("eval.json")
    if not runs:
        st.info("No training results yet. Run `python finetune/prepare_dataset.py` and "
                "`python finetune/train_lora.py [--qlora]`, then `python finetune/evaluate_lora.py`.")
    else:
        _training_section(runs)
    if evaluation:
        _evaluation_section(evaluation)
    _calculator()


def _training_section(runs):
    first = next(iter(runs.values()))
    st.subheader("1 · What was trained")
    c1, c2, c3 = st.columns(3)
    c1.metric("Base model parameters", f"{first['base_params'] / 1e6:,.0f} M")
    c2.metric("Trainable (LoRA A, B)", f"{first['trainable_params'] / 1e6:.2f} M",
              help="Frozen parameters are still used in every forward pass - they are just not updated.")
    c3.metric("Share trained", f"{first['trainable_pct']:.2f} %")
    st.caption(f"Training data: {first['train_examples']} examples · validation: {first['val_examples']} held-out "
               f"paraphrases + off-topic questions · device: {first['device']}")

    rows = {}
    for m, r in runs.items():
        cfg = r["config"]
        rows["LoRA" if m == "lora" else "QLoRA"] = {
            "Frozen base stored as": cfg["quantization"],
            "Base-weight memory": _mb(r["base_weight_memory_mb"]),
            "Peak GPU memory (training)": _mb(r["peak_gpu_memory_mb"]),
            "Rank r / alpha / dropout": f"{cfg['r']} / {cfg['alpha']} / {cfg['dropout']}",
            "Target modules": ", ".join(cfg["target_modules"]),
            "Epochs": cfg["epochs"],
            "Training time": f"{r['train_seconds']:.0f} s",
            "Final validation loss": f"{r['final_eval_loss']:.3f}",
            "Saved adapter size": _mb(r["adapter_size_mb"]),
        }
    st.subheader("2 · LoRA vs QLoRA")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("QLoRA stores the frozen base in 4-bit NF4 (with double quantization) and dequantizes it for "
               "computation - storage precision ≠ computation precision. The saved adapter does not include the "
               "base model: base + adapter are both needed for inference.")
    st.info("**Model weight size ≠ training memory.** The bf16 weights are only ~1 GB, yet LoRA training peaked at "
            "~9 GB (activations, gradients of A/B, optimizer state, and Qwen's 151k-token output layer computed for "
            "every position). On the 6 GB RTX 3050 the LoRA run spilled into shared system memory, while the QLoRA "
            "run stayed inside the card. Embeddings and the output layer are not quantized, which is why the 4-bit "
            "base still occupies ~720 MB.")

    st.subheader("3 · Training loss")
    curves = {("LoRA" if m == "lora" else "QLoRA"): pd.Series({p["step"]: p["loss"] for p in r["train_loss"]})
              for m, r in runs.items()}
    df = pd.DataFrame(curves)
    df.index.name = "step"
    st.line_chart(df, x_label="training step", y_label="loss", color=[SERIES_COLORS[c] for c in df.columns])


def _evaluation_section(ev):
    st.subheader("4 · Does fine-tuning help? (held-out validation set)")
    table = pd.DataFrame({
        LABELS[name]: {
            "Answer word-overlap F1 ↑": f"{m['token_f1']:.3f}",
            "Facts reproduced (numbers) ↑": f"{100 * m['exact_facts']:.0f} %",
            "Off-topic correctly refused ↑": f"{100 * m['refusal_acc']:.0f} %",
            "Generation time / answer": f"{m['ms_per_answer'] / 1000:.1f} s",
        }
        for name, m in ev["metrics"].items()
    })
    st.dataframe(table, use_container_width=True)
    st.caption(f"{ev['val_examples']} held-out questions (paraphrases never seen in training + off-topic questions), "
               "same retrieved context for every model, greedy decoding.")

    st.markdown("**Sample answers**")
    for s in ev["samples"]:
        with st.expander(s["question"]):
            st.markdown(f"**Reference (knowledge base):** {s['reference']}")
            for name in ev["metrics"]:
                st.markdown(f"**{LABELS[name]}:** {s[name]}")


def _calculator():
    st.subheader("5 · LoRA parameter calculator")
    c1, c2, c3 = st.columns(3)
    d_out = c1.number_input("d_out", min_value=1, value=4096, step=128)
    d_in = c2.number_input("d_in", min_value=1, value=4096, step=128)
    r = c3.number_input("rank r", min_value=1, value=8, step=1)
    full, lora, ratio = lora_params(int(d_out), int(d_in), int(r))
    st.markdown(f"Full update: **{full:,}** params · LoRA (A: {r}×{d_in}, B: {d_out}×{r}): **{lora:,}** params "
                f"→ **{ratio:,.0f}× fewer** trainable parameters *for this one matrix* (not the whole model).")

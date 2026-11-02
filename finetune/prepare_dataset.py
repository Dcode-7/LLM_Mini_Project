"""Build LoRA/QLoRA training data from the college knowledge base.

Each example uses the SAME prompt the API sends at inference time (system prompt +
retrieved context + question), so the adapter learns the college's response behaviour
(RAG -> current knowledge, LoRA -> specialised behaviour). Off-topic questions are
paired with the fallback answer so the model learns to refuse instead of inventing.

Split: the last alt_question of every entry is held out for validation.

    python finetune/prepare_dataset.py
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support_rag import FALLBACK_ANSWER, MIN_SCORE, KnowledgeBase, build_messages  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "data"

OFF_TOPIC_TRAIN = [
    "What is the capital of France?", "Write a poem about the rain", "Who is the best football player?",
    "How do I cook biryani?", "What will the stock market do tomorrow?", "Tell me a joke",
    "What is the fee at another college nearby?", "Can you do my assignment for me?",
    "What is the weather in Mumbai?", "Who will win the next election?",
]
OFF_TOPIC_VAL = ["Recommend a good movie", "What is the price of bitcoin?", "Translate hello into Japanese"]


def example(kb: KnowledgeBase, question: str, answer: str | None) -> dict:
    hits = kb.search(question)
    relevant = [h for h in hits if h.score >= MIN_SCORE] or hits[:1]
    messages = build_messages(question, relevant)
    messages.append({"role": "assistant", "content": answer or FALLBACK_ANSWER})
    return {"messages": messages}


def main():
    kb = KnowledgeBase()
    train, val = [], []
    for e in kb.entries:
        alts = e.get("alt_questions", [])
        for q in [e["question"], *alts[:-1]]:
            train.append(example(kb, q, e["answer"]))
        if alts:
            val.append(example(kb, alts[-1], e["answer"]))
    train += [example(kb, q, None) for q in OFF_TOPIC_TRAIN]
    val += [example(kb, q, None) for q in OFF_TOPIC_VAL]

    random.Random(42).shuffle(train)
    OUT_DIR.mkdir(exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with open(OUT_DIR / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train: {len(train)} examples, val: {len(val)} examples -> {OUT_DIR}")


if __name__ == "__main__":
    main()

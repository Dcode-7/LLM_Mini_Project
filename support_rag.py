"""College Support Assistant - RAG pipeline and LLM backends.

The retriever (knowledge base + TF-IDF search + prompt construction) is kept
separate from the LLM backend so the backend can be swapped through the
LLM_BACKEND environment variable without touching the rest of the app:

    extractive  - no LLM; returns the best-matching knowledge-base answer (default, no key needed)
    cloud       - any OpenAI-compatible chat-completions API (Groq, Hugging Face router, OpenRouter, ...)
    ollama      - a local model served by Ollama (http://localhost:11434)
    lora        - Qwen2.5-0.5B-Instruct + the LoRA/QLoRA adapter trained in finetune/ (local only)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("support_rag")

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = Path(os.getenv("KB_PATH", BASE_DIR / "data" / "college_kb.json"))
TOP_K = int(os.getenv("TOP_K", "3"))
# Below this similarity the question is treated as "not covered by the knowledge base".
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.17"))

FALLBACK_ANSWER = (
    "I'm sorry, I don't have enough information in the college knowledge base to answer that. "
    "Please contact the college front office at info@northbridge.edu or +91-20-5550-1000."
)
FALLBACK_TICKET = {
    "department": "Front Office",
    "category": "Unclassified",
    "priority": "Low",
    "action": "Route to a human helpdesk agent",
}

SYSTEM_PROMPT = (
    "You are the College Support Assistant for Northbridge College of Engineering & Technology. "
    "Answer the student's question using ONLY the facts in the provided context. "
    "Be concise (at most 5 sentences) and keep all numbers, fees, dates and contact details exactly as given. "
    "If the context does not contain the answer, reply exactly with: "
    f'"{FALLBACK_ANSWER}" '
    "Never invent policies, amounts, dates or contacts."
)


class BackendError(Exception):
    """Raised when the LLM backend fails. The message is safe to log but never sent to clients."""


@dataclass
class Hit:
    entry: dict
    score: float


class KnowledgeBase:
    """Loads the Q&A knowledge base once and builds a hybrid word + character TF-IDF index."""

    def __init__(self, path: Path = KB_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.entries: list[dict] = data["entries"]
        self.version = data.get("version", "unknown")
        docs = [
            " ".join([e["question"], e["question"], *e.get("alt_questions", []), e["category"], e["answer"]])
            for e in self.entries
        ]
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.word_matrix = self.word_vec.fit_transform(docs)
        self.char_matrix = self.char_vec.fit_transform(docs)

    def search(self, query: str, k: int = TOP_K) -> list[Hit]:
        word_sim = cosine_similarity(self.word_vec.transform([query]), self.word_matrix)[0]
        char_sim = cosine_similarity(self.char_vec.transform([query]), self.char_matrix)[0]
        scores = 0.6 * word_sim + 0.4 * char_sim
        top = scores.argsort()[::-1][:k]
        return [Hit(self.entries[i], float(scores[i])) for i in top]


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{h.entry['id']}] ({h.entry['department']} / {h.entry['category']})\n"
        f"Q: {h.entry['question']}\nA: {h.entry['answer']}"
        for h in hits
    )


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{build_context(hits)}\n\nStudent question: {question}"},
    ]


# --------------------------------------------------------------------------- backends


class ExtractiveBackend:
    """No-LLM backend: returns the best-matching curated answer. Works on any machine, no key required."""

    name = "extractive"

    def generate(self, question: str, hits: list[Hit]) -> str:
        return hits[0].entry["answer"]


class CloudBackend:
    """OpenAI-compatible /chat/completions API (Groq, Hugging Face router, OpenRouter, Together, ...)."""

    name = "cloud"

    def __init__(self):
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "30"))
        self.client = httpx.Client(timeout=self.timeout)
        if not self.api_key:
            logger.warning("LLM_BACKEND=cloud but LLM_API_KEY is not set; requests will fail")

    def generate(self, question: str, hits: list[Hit]) -> str:
        if not self.api_key:
            raise BackendError("cloud backend has no API key configured")
        try:
            resp = self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": build_messages(question, hits),
                    "temperature": 0.1,
                    "max_tokens": 400,
                },
            )
        except httpx.TimeoutException as exc:
            raise BackendError("cloud backend timed out") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"cloud backend network error: {type(exc).__name__}") from exc
        if resp.status_code != 200:
            # Status code only - provider error bodies can echo request details.
            raise BackendError(f"cloud backend returned HTTP {resp.status_code}")
        try:
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as exc:
            raise BackendError("cloud backend returned an unexpected response format") from exc


class OllamaBackend:
    """Local model served by Ollama, e.g. `ollama pull qwen2.5:0.5b`."""

    name = "ollama"

    def __init__(self):
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
        self.client = httpx.Client(timeout=float(os.getenv("LLM_TIMEOUT", "60")))

    def generate(self, question: str, hits: list[Hit]) -> str:
        try:
            resp = self.client.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": build_messages(question, hits),
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
        except httpx.ConnectError as exc:
            raise BackendError("Ollama is not reachable - is `ollama serve` running?") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"Ollama request failed: {type(exc).__name__}") from exc
        if resp.status_code == 404:
            raise BackendError(f"Ollama model '{self.model}' not found - run `ollama pull {self.model}`")
        if resp.status_code != 200:
            raise BackendError(f"Ollama returned HTTP {resp.status_code}")
        return resp.json()["message"]["content"].strip()


class LoRABackend:
    """Base model + LoRA adapter from finetune/train_lora.py. Needs requirements-finetune.txt (local GPU/CPU)."""

    name = "lora"

    def __init__(self):
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("LLM_BACKEND=lora needs `pip install -r requirements-finetune.txt`") from exc
        base = os.getenv("LORA_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        adapter = os.getenv("LORA_ADAPTER_PATH", str(BASE_DIR / "finetune" / "adapters" / "college-lora"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(adapter)
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        ).to(device)
        self.model = PeftModel.from_pretrained(model, adapter).eval()

    def generate(self, question: str, hits: list[Hit]) -> str:
        try:
            prompt = self.tokenizer.apply_chat_template(
                build_messages(question, hits), tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=300, do_sample=False)
            return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as exc:
            raise BackendError(f"LoRA generation failed: {type(exc).__name__}") from exc


BACKENDS = {"extractive": ExtractiveBackend, "cloud": CloudBackend, "ollama": OllamaBackend, "lora": LoRABackend}


def make_backend(name: str | None = None):
    name = (name or os.getenv("LLM_BACKEND", "extractive")).strip().lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown LLM_BACKEND '{name}'. Choose one of: {', '.join(BACKENDS)}")
    return BACKENDS[name]()


# --------------------------------------------------------------------------- pipeline


class SupportRAG:
    """Retrieve -> (fallback if nothing relevant) -> generate -> attach helpdesk ticket metadata."""

    def __init__(self, backend=None, kb: KnowledgeBase | None = None):
        self.kb = kb or KnowledgeBase()
        self.backend = backend or make_backend()
        logger.info("RAG ready: %d KB entries, backend=%s", len(self.kb.entries), self.backend.name)

    def answer(self, question: str) -> dict:
        hits = self.kb.search(question)
        best = hits[0]
        if best.score < MIN_SCORE:
            return {
                "answer": FALLBACK_ANSWER,
                "grounded": False,
                "ticket": FALLBACK_TICKET,
                "sources": [],
                "top_score": round(best.score, 3),
            }

        relevant = [h for h in hits if h.score >= MIN_SCORE]
        text = self.backend.generate(question, relevant)
        grounded = "don't have enough information" not in text.lower()
        e = best.entry
        return {
            "answer": text,
            "grounded": grounded,
            "ticket": {
                "department": e["department"],
                "category": e["category"],
                "priority": e["priority"],
                "action": e["action"],
            } if grounded else FALLBACK_TICKET,
            "sources": [
                {"id": h.entry["id"], "question": h.entry["question"], "score": round(h.score, 3)} for h in relevant
            ],
            "top_score": round(best.score, 3),
        }

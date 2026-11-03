"""Cloud backend tests against a simulated OpenAI-compatible API (no network, no real key)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pytest

from support_rag import BackendError, CloudBackend, Hit

HIT = [Hit({"id": "FEE-003", "department": "Accounts", "category": "Payment Reconciliation",
            "question": "Fee deducted but unpaid?", "answer": "Wait 48 hours, then email accounts."}, 0.5)]


def backend(monkeypatch, handler, **env):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    b = CloudBackend()
    b.client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def test_gpt_oss_gets_low_reasoning_effort_and_room_for_tokens(monkeypatch):
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": " Wait 48 hours. "}}]})

    b = backend(monkeypatch, handler, LLM_MODEL="openai/gpt-oss-20b")
    assert b.generate("fee unpaid?", HIT) == "Wait 48 hours."
    assert sent["model"] == "openai/gpt-oss-20b"
    assert sent["reasoning_effort"] == "low"
    assert sent["max_tokens"] >= 1024


def test_non_reasoning_model_gets_no_reasoning_effort(monkeypatch):
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    backend(monkeypatch, handler, LLM_MODEL="some-chat-model").generate("q", HIT)
    assert "reasoning_effort" not in sent


def test_retired_model_reports_error_code_without_secrets(monkeypatch):
    def handler(request):
        return httpx.Response(404, json={"error": {"message": "The model `llama-3.3-70b-versatile` has been decommissioned",
                                                   "code": "model_not_found"}})

    b = backend(monkeypatch, handler, LLM_MODEL="llama-3.3-70b-versatile")
    with pytest.raises(BackendError) as exc:
        b.generate("q", HIT)
    msg = str(exc.value)
    assert "HTTP 404" in msg and "code=model_not_found" in msg and "model=llama-3.3-70b-versatile" in msg
    assert "test-key" not in msg


def test_empty_answer_is_an_error(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    with pytest.raises(BackendError, match="empty answer"):
        backend(monkeypatch, handler).generate("q", HIT)

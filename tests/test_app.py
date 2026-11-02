"""Endpoint tests matching the Experiment 7 testing checklist. Run: pytest -v"""

import os
import sys
from pathlib import Path

os.environ["LLM_BACKEND"] = "extractive"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

import app as app_module
from support_rag import BackendError, FALLBACK_ANSWER


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["kb_entries"] > 50


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "College Support Assistant" in r.text


@pytest.mark.parametrize("question,expected_id", [
    ("My semester fee was deducted but the portal still shows unpaid.", "FEE-003"),
    ("how do i get bonafide certificate", "CRT-001"),
    ("what is the minimum attendance needed", "ACD-001"),
    ("hall ticket not downloading", "EXM-002"),
    ("library timings on sunday?", "LIB-001"),
    ("I want to complain about ragging by seniors", "WEL-001"),
    ("fan not working in my hostel room", "HOS-004"),
    ("forgot ERP password", "IT-001"),
])
def test_supported_questions(client, question, expected_id):
    r = client.post("/ask", json={"question": question})
    assert r.status_code == 200
    d = r.json()
    assert d["grounded"] is True
    assert d["sources"][0]["id"] == expected_id
    assert d["latency_ms"] >= 0


def test_helpdesk_ticket_routing(client):
    d = client.post("/ask", json={"question": "My semester fee was deducted but the portal still shows unpaid."}).json()
    assert d["ticket"] == {
        "department": "Accounts",
        "category": "Payment Reconciliation",
        "priority": "High",
        "action": "Verify transaction reference",
    }


@pytest.mark.parametrize("question", [
    "What is the capital of Australia?",
    "Who won the cricket world cup?",
    "Write me a poem about the moon",
])
def test_unsupported_question_falls_back(client, question):
    d = client.post("/ask", json={"question": question}).json()
    assert d["grounded"] is False
    assert d["answer"] == FALLBACK_ANSWER
    assert d["sources"] == []


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "    "}, {}])
def test_empty_question_rejected(client, payload):
    r = client.post("/ask", json=payload)
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_input"


def test_long_question_rejected(client):
    r = client.post("/ask", json={"question": "a" * 1001})
    assert r.status_code == 422
    assert "too long" in r.json()["detail"][0]


def test_backend_failure_is_controlled(client, monkeypatch):
    def boom(question, hits):
        raise BackendError("provider said: invalid key sk-SECRET123")

    monkeypatch.setattr(app_module.rag.backend, "generate", boom)
    r = client.post("/ask", json={"question": "What are the library timings?"})
    assert r.status_code == 503
    body = r.text
    assert "backend_unavailable" in body
    assert "SECRET" not in body and "Traceback" not in body

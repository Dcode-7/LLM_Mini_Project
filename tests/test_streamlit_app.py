"""Streamlit UI tests using Streamlit's headless AppTest. Run: pytest -v"""

import os
from pathlib import Path

os.environ["LLM_BACKEND"] = "extractive"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000"

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def run_with(question):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value(question).run()
    return at


def test_app_loads():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "College Support Assistant"


def test_supported_question_shows_answer_and_ticket():
    at = run_with("My semester fee was deducted but the portal still shows unpaid.")
    assert not at.exception
    msg = at.session_state.messages[-1]
    assert msg["grounded"] is True
    assert msg["ticket"]["department"] == "Accounts"
    assert msg["ticket"]["priority"] == "High"
    assert msg["latency_ms"] >= 0


def test_unsupported_question_falls_back():
    at = run_with("What is the capital of Australia?")
    assert at.session_state.messages[-1]["grounded"] is False


def test_blank_question_rejected():
    at = run_with("   ")
    assert at.session_state.messages == []
    assert "empty" in at.warning[0].value


def test_finetuning_page_renders():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("🧪 Fine-tuning (LoRA / QLoRA)").run()
    assert not at.exception
    assert at.title[0].value == "Fine-tuning the LLM (LoRA / QLoRA)"


def test_knowledge_base_page_renders():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("📄 Knowledge base (PDF)").run()
    assert not at.exception
    assert at.title[0].value == "Knowledge base: from PDF to answers"

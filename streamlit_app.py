"""College Support Assistant - Streamlit interface (deployed on Streamlit Community Cloud).

Uses the same RAG pipeline as the FastAPI service (support_rag.py) with the same
production checks: input validation, controlled errors, logging and latency.

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import logging
import os
import time
import uuid

import streamlit as st

st.set_page_config(page_title="College Support Assistant", page_icon="🎓", layout="centered")

# Streamlit Cloud stores secrets in st.secrets; expose them as env vars before support_rag reads its settings.
try:
    for key, value in st.secrets.items():
        if isinstance(value, (str, int, float)):
            os.environ.setdefault(key, str(value))
except Exception:  # no secrets.toml locally - fall back to .env / environment
    pass

from dotenv import load_dotenv

load_dotenv()

from support_rag import BackendError, SupportRAG  # noqa: E402

MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "1000"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
SUGGESTIONS = [
    "My semester fee was deducted but the portal still shows unpaid.",
    "How do I get a bonafide certificate?",
    "What is the minimum attendance requirement?",
    "I can't download my hall ticket",
    "What are the library timings?",
    "How do I report ragging?",
    "What is the eligibility for campus placements?",
]
PRIORITY_COLOR = {"High": "red", "Medium": "orange", "Low": "green"}

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("college_support_streamlit")


@st.cache_resource(show_spinner="Loading knowledge base…")
def get_rag() -> SupportRAG:
    # Built once per server process and shared by all sessions - not per question.
    return SupportRAG()


def validate(question: str) -> tuple[str | None, str | None]:
    q = (question or "").strip()
    if not q:
        return None, "Question must not be empty."
    if len(q) > MAX_QUESTION_CHARS:
        return None, f"Question is too long (maximum {MAX_QUESTION_CHARS} characters)."
    return q, None


def rate_limited() -> bool:
    now = time.monotonic()
    recent = [t for t in st.session_state.get("ask_times", []) if now - t < 60]
    limited = len(recent) >= RATE_LIMIT_PER_MINUTE
    if not limited:
        recent.append(now)
    st.session_state.ask_times = recent
    return limited


def ask(rag: SupportRAG, question: str) -> dict:
    """Returns a chat message dict: either a result or a safe error."""
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    try:
        result = rag.answer(question)
    except BackendError as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.error("ask_failed id=%s backend=%s latency_ms=%.1f reason=%s", request_id, rag.backend.name, latency, exc)
        return {"role": "assistant", "error": "The answer service is temporarily unavailable. Please try again later.",
                "request_id": request_id}
    except Exception:
        logger.exception("unhandled_error id=%s", request_id)
        return {"role": "assistant", "error": "Something went wrong.", "request_id": request_id}
    latency = (time.perf_counter() - start) * 1000
    logger.info(
        "ask_ok id=%s backend=%s q_len=%d grounded=%s top=%s latency_ms=%.1f",
        request_id, rag.backend.name, len(question), result["grounded"],
        result["sources"][0]["id"] if result["sources"] else "-", latency,
    )
    return {"role": "assistant", **result, "latency_ms": round(latency, 1), "backend": rag.backend.name,
            "request_id": request_id}


def render(msg: dict):
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            return
        if "error" in msg:
            st.error(f"{msg['error']} (ref `{msg['request_id']}`)")
            return
        st.markdown(msg["answer"])
        t = msg["ticket"]
        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Department:** {t['department']}")
            c2.markdown(f"**Priority:** :{PRIORITY_COLOR.get(t['priority'], 'gray')}[{t['priority']}]")
            c1.markdown(f"**Category:** {t['category']}")
            c2.markdown(f"**Action:** {t['action']}")
        grounded = "✅ grounded in knowledge base" if msg["grounded"] else "⚠️ not in knowledge base"
        st.caption(f"⏱ {msg['latency_ms']} ms · backend: {msg['backend']} · {grounded}")
        if msg["sources"]:
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s['id']}` {s['question']} — score {s['score']}")


rag = get_rag()
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("🎓 Northbridge College")
    st.caption("Fictional sample institution")
    page = st.radio("Page", ["💬 Assistant", "🧪 Fine-tuning (LoRA / QLoRA)"], label_visibility="collapsed")
    st.success(f"Online · backend: **{rag.backend.name}**")
    st.metric("Knowledge-base entries", len(rag.kb.entries))
    st.caption(f"KB version {rag.kb.version}")
    if page.startswith("💬"):
        st.subheader("Try asking")
        for s in SUGGESTIONS:
            if st.button(s, use_container_width=True):
                st.session_state.pending = s
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def assistant_page():
    st.title("College Support Assistant")
    st.caption("Ask about admissions, fees, exams, scholarships, hostel, library, placements, certificates and more. "
               "Each answer shows the department your query is routed to.")

    for m in st.session_state.messages:
        render(m)

    typed = st.chat_input("Type your question…", max_chars=MAX_QUESTION_CHARS)
    raw = typed if typed is not None else st.session_state.pop("pending", None)
    if raw is None:
        return
    question, problem = validate(raw)
    if problem:
        logger.info("validation_error reason=%s", problem)
        st.warning(problem)
    elif rate_limited():
        logger.warning("rate_limited")
        st.warning("Too many questions in a minute. Please slow down.")
    else:
        user_msg = {"role": "user", "content": question}
        st.session_state.messages.append(user_msg)
        render(user_msg)
        with st.spinner("Thinking…"):
            reply = ask(rag, question)
        st.session_state.messages.append(reply)
        render(reply)


def finetuning_page():
    from finetuning_view import render_finetuning

    render_finetuning()


if page.startswith("💬"):
    assistant_page()
else:
    finetuning_page()

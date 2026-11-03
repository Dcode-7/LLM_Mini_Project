"""College Support Assistant - FastAPI service.

    uvicorn app:app
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from support_rag import BackendError, SupportRAG

MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "1000"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("college_support_api")

rag: SupportRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = SupportRAG()
    yield


app = FastAPI(
    title="College Support Assistant API",
    description="RAG-based helpdesk for Northbridge College (fictional). Experiment 7 + Lecture 4 (LoRA/QLoRA).",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_CHARS, examples=["How do I get a bonafide certificate?"])

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty")
        return v


class Ticket(BaseModel):
    department: str
    category: str
    priority: str
    action: str


class Source(BaseModel):
    id: str
    question: str
    score: float
    page: int | None = None


class AskResponse(BaseModel):
    request_id: str
    answer: str
    grounded: bool
    ticket: Ticket
    sources: list[Source]
    backend: str
    latency_ms: float


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    msgs = []
    for err in exc.errors():
        if err["type"] == "string_too_long":
            msgs.append(f"Question is too long (maximum {MAX_QUESTION_CHARS} characters)")
        elif err["type"] == "missing":
            msgs.append("Field 'question' is required")
        else:
            msgs.append(str(err.get("msg", "Invalid input")).removeprefix("Value error, "))
    logger.info("validation_error path=%s reasons=%s", request.url.path, msgs)
    return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": msgs})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("unhandled_error path=%s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "Something went wrong."})


_hits: dict[str, deque] = defaultdict(deque)


def rate_limited(ip: str) -> bool:
    now = time.monotonic()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MINUTE:
        return True
    q.append(now)
    return False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": rag.backend.name if rag else None,
        "kb_entries": len(rag.kb.entries) if rag else 0,
        "kb_version": rag.kb.version if rag else None,
    }


@app.post("/ask", response_model=AskResponse, responses={422: {}, 429: {}, 503: {}})
def ask(body: AskRequest, request: Request):
    request_id = uuid.uuid4().hex[:8]
    client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    if rate_limited(client_ip):
        logger.warning("rate_limited id=%s", request_id)
        return JSONResponse(status_code=429, content={"error": "rate_limited", "detail": "Too many requests, slow down."})

    start = time.perf_counter()
    try:
        result = rag.answer(body.question)
    except BackendError as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.error("ask_failed id=%s backend=%s latency_ms=%.1f reason=%s", request_id, rag.backend.name, latency, exc)
        return JSONResponse(
            status_code=503,
            content={"error": "backend_unavailable", "detail": "The answer service is temporarily unavailable. Please try again later.", "request_id": request_id},
        )
    latency = (time.perf_counter() - start) * 1000
    logger.info(
        "ask_ok id=%s backend=%s q_len=%d grounded=%s top=%s latency_ms=%.1f",
        request_id, rag.backend.name, len(body.question), result["grounded"],
        result["sources"][0]["id"] if result["sources"] else "-", latency,
    )
    return AskResponse(request_id=request_id, backend=rag.backend.name, latency_ms=round(latency, 1), **{
        k: result[k] for k in ("answer", "grounded", "ticket", "sources")
    })


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

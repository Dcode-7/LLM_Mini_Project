"""PDF ingestion: extract, clean, structure and chunk the handbook.

    python pdf_ingest.py
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

# header/footer lines repeated on every page
HEADER_FOOTER_PATTERNS = [
    re.compile(r"^Northbridge College of Engineering & Technology - Student Helpdesk Handbook.*$"),
    re.compile(r"^Page \d+$"),
    re.compile(r"^Fictional sample data for an academic project$"),
]
SECTION_RE = re.compile(r"^Section (\d+) - (.+)$", re.M)
TOPIC_RE = re.compile(r"^\[([A-Z]+-\d{3})\]\s*", re.M)
ROUTING_RE = re.compile(
    r"(?P<department>.+?)\s*\|\s*Category:\s*(?P<category>.+?)\s*\|\s*"
    r"Priority:\s*(?P<priority>\w+)\s*\|\s*Action:\s*(?P<action>.+)$"
)
MAX_CHUNK_CHARS = 900
CHUNK_OVERLAP_CHARS = 150


@dataclass
class PageInfo:
    number: int
    raw_text: str
    clean_text: str
    removed_lines: list[str]
    status: str                 # "content" | "discarded: cover" | "discarded: contents"


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    aliases: list[str]
    section: str
    pages: list[int]
    department: str
    category: str
    priority: str
    action: str

    def as_entry(self) -> dict:
        return {
            "id": self.id, "question": self.title, "alt_questions": self.aliases, "answer": self.text,
            "section": self.section, "pages": self.pages, "department": self.department,
            "category": self.category, "priority": self.priority, "action": self.action,
        }


@dataclass
class IngestReport:
    path: Path
    file_kb: float
    metadata: dict
    pages: list[PageInfo]
    chunks: list[Chunk]
    sections: list[str]
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)


def _norm(text: str) -> str:
    """Join PDF line-wraps into flowing text and collapse whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def clean_page(raw: str) -> tuple[str, list[str]]:
    kept, removed = [], []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if any(p.match(s) for p in HEADER_FOOTER_PATTERNS):
            removed.append(s)
        else:
            kept.append(s)
    return "\n".join(kept), removed


def classify_page(number: int, clean: str) -> str:
    if number == 1:
        return "discarded: cover"
    if clean.startswith("Contents"):
        return "discarded: contents"
    return "content"


def split_topic(body: str) -> tuple[str, list[str], str, str]:
    """Split one topic block into title / aliases / answer / routing using its printed labels."""
    m = re.search(r"Also asked as:(.*?)Answer:(.*?)Routed to:(.*)", body, re.S)
    if not m:
        raise ValueError("topic block is missing one of: 'Also asked as:', 'Answer:', 'Routed to:'")
    title = body[: m.start()]
    aliases = [a.strip() for a in _norm(m.group(1)).split(";") if a.strip()]
    return _norm(title), aliases, _norm(m.group(2)), _norm(m.group(3))


def window(text: str, size: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            end = text.rfind(" ", start, end) if text.rfind(" ", start, end) > start else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def ingest(path: Path) -> IngestReport:
    t0 = time.perf_counter()
    reader = PdfReader(str(path))
    meta = {k.lstrip("/"): str(v) for k, v in (reader.metadata or {}).items()}
    raw_pages = [p.extract_text() or "" for p in reader.pages]
    t_extract = time.perf_counter()

    pages = []
    for i, raw in enumerate(raw_pages, 1):
        clean, removed = clean_page(raw)
        pages.append(PageInfo(i, raw, clean, removed, classify_page(i, clean)))
    t_clean = time.perf_counter()

    full, page_starts = "", []
    for p in pages:
        if p.status == "content":
            page_starts.append((len(full), p.number))
            full += p.clean_text + "\n"

    def pages_for(start: int, end: int) -> list[int]:
        return [n for i, (off, n) in enumerate(page_starts)
                if off < end and (page_starts[i + 1][0] if i + 1 < len(page_starts) else len(full)) > start]

    sections = [(m.start(), m.group(2).strip()) for m in SECTION_RE.finditer(full)]

    def section_at(pos: int) -> str:
        name = ""
        for off, title in sections:
            if off <= pos:
                name = title
        return name

    topics = list(TOPIC_RE.finditer(full))
    t_struct = time.perf_counter()

    chunks, warnings = [], []
    for i, m in enumerate(topics):
        start = m.start()
        end = min([t.start() for t in topics[i + 1:i + 2]] + [o for o, _ in sections if o > start] + [len(full)])
        body = full[m.end():end]
        try:
            title, aliases, answer, routing = split_topic(body)
        except ValueError as exc:
            warnings.append(f"{m.group(1)}: {exc} - skipped")
            continue
        r = ROUTING_RE.search(routing)
        if not r:
            warnings.append(f"{m.group(1)}: routing line not parsed - using defaults")
        meta_fields = r.groupdict() if r else {"department": "Front Office", "category": "General Information",
                                               "priority": "Low", "action": "Route to helpdesk"}
        pieces = window(answer)
        for j, piece in enumerate(pieces):
            chunks.append(Chunk(
                id=m.group(1) if len(pieces) == 1 else f"{m.group(1)}#{j + 1}",
                title=title, text=piece, aliases=aliases, section=section_at(start),
                pages=pages_for(start, end), **{k: v.strip() for k, v in meta_fields.items()},
            ))
    t_chunk = time.perf_counter()

    return IngestReport(
        path=Path(path), file_kb=Path(path).stat().st_size / 1024, metadata=meta, pages=pages, chunks=chunks,
        sections=[s for _, s in sections], warnings=warnings,
        timings_ms={
            "1. extract text (pypdf)": round((t_extract - t0) * 1000, 1),
            "2. clean pages": round((t_clean - t_extract) * 1000, 1),
            "3. detect sections & topics": round((t_struct - t_clean) * 1000, 1),
            "4. chunk + parse metadata": round((t_chunk - t_struct) * 1000, 1),
        },
    )


def naive_chunks(report: IngestReport, size: int = 500, overlap: int = 100) -> list[str]:
    """Fixed-size chunking, used only for comparison."""
    text = _norm(" ".join(p.clean_text for p in report.pages if p.status == "content"))
    return window(text, size, overlap)


if __name__ == "__main__":
    rep = ingest(Path(__file__).resolve().parent / "data" / "college_handbook.pdf")
    print(f"{rep.path.name}: {len(rep.pages)} pages, {rep.file_kb:.0f} KB, title={rep.metadata.get('Title')}")
    for p in rep.pages:
        print(f"  page {p.number:2}: {len(p.raw_text):5} raw chars -> {len(p.clean_text):5} clean, "
              f"{len(p.removed_lines)} header/footer lines removed, {p.status}")
    print(f"sections: {len(rep.sections)} | chunks: {len(rep.chunks)} | warnings: {rep.warnings or 'none'}")
    print("timings:", rep.timings_ms)
    c = rep.chunks[8]
    print(f"\nexample chunk {c.id} (section {c.section}, pages {c.pages}):\n  title: {c.title}\n  aliases: {c.aliases}"
          f"\n  text: {c.text}\n  routing: {c.department} | {c.category} | {c.priority} | {c.action}")

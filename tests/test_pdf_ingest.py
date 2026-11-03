"""PDF ingestion tests."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from pdf_ingest import ingest, window


@pytest.fixture(scope="module")
def report():
    return ingest(ROOT / "data" / "college_handbook.pdf")


@pytest.fixture(scope="module")
def source():
    return {e["id"]: e for e in json.loads((ROOT / "data" / "college_kb.json").read_text(encoding="utf-8"))["entries"]}


def test_pages_classified(report):
    assert len(report.pages) >= 10
    assert report.pages[0].status == "discarded: cover"
    assert report.pages[1].status == "discarded: contents"
    assert all(p.status == "content" for p in report.pages[2:])


def test_header_footer_removed(report):
    for p in report.pages[1:]:
        assert len(p.removed_lines) == 3
        assert "Student Helpdesk Handbook 2026-27" not in p.clean_text


def test_every_topic_extracted_exactly(report, source):
    assert not report.warnings
    assert {c.id for c in report.chunks} == set(source)
    for c in report.chunks:
        e = source[c.id]
        assert c.title == e["question"]
        assert c.text == e["answer"]
        assert c.aliases == e["alt_questions"]
        assert (c.department, c.category, c.priority, c.action) == (e["department"], e["category"], e["priority"], e["action"])


def test_sections_and_pages(report):
    assert len(report.sections) == 15
    fee = next(c for c in report.chunks if c.id == "FEE-003")
    assert fee.section == "Fees and Payments"
    assert all(len(c.pages) == 1 and c.pages[0] >= 3 for c in report.chunks)


def test_window_splits_long_text_with_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    parts = window(text, size=300, overlap=60)
    assert len(parts) > 1
    assert all(len(p) <= 300 for p in parts)
    assert parts[0].split()[-1] in parts[1]

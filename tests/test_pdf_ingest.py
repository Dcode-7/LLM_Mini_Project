"""PDF ingestion tests."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from pdf_ingest import ingest, window


@pytest.fixture(scope="module")
def report():
    return ingest(ROOT / "data" / "college_handbook.pdf")




def test_pages_classified(report):
    assert len(report.pages) >= 10
    assert report.pages[0].status == "discarded: cover"
    assert report.pages[1].status == "discarded: contents"
    assert all(p.status == "content" for p in report.pages[2:])


def test_header_footer_removed(report):
    for p in report.pages[1:]:
        assert len(p.removed_lines) == 3
        assert "Student Helpdesk Handbook 2026-27" not in p.clean_text


def test_every_topic_extracted_completely(report):
    assert not report.warnings
    # every "[ID]" topic heading printed in the content pages becomes exactly one chunk
    content = "\n".join(p.clean_text for p in report.pages if p.status == "content")
    printed = re.findall(r"^\[([A-Z]+-\d{3})\]", content, re.M)
    assert len(printed) >= 60
    assert sorted(c.id for c in report.chunks) == sorted(printed)
    for c in report.chunks:
        assert c.title and c.text and c.aliases               # title, answer and alternate phrasings parsed
        assert c.department and c.category and c.action       # routing line parsed
        assert c.priority in {"High", "Medium", "Low"}
        assert "Routed to:" not in c.text and "Answer:" not in c.text   # labels stripped from the answer


def test_known_topic_exact(report):
    c = next(c for c in report.chunks if c.id == "FEE-003")
    assert c.title == "My semester fee was deducted but the portal still shows unpaid. What should I do?"
    assert c.text.startswith("Online payments can take up to 48 hours to reconcile.")
    assert c.text.endswith("the bank auto-reverses the amount within 5-7 working days.")
    assert (c.department, c.category, c.priority, c.action) == (
        "Accounts", "Payment Reconciliation", "High", "Verify transaction reference")


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

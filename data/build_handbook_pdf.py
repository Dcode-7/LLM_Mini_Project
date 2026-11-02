"""Generate the Student Helpdesk Handbook PDF - the knowledge base the RAG app reads.

The Q&A source data (college_kb.json) is authored once and typeset into a realistic
multi-page PDF: cover page, contents page, running header/footer, numbered sections.
The app never reads the JSON at runtime - it ingests this PDF (see pdf_ingest.py).

    python data/build_handbook_pdf.py        (needs reportlab: pip install -r requirements-dev.txt)
"""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "college_kb.json"
OUTPUT = HERE / "college_handbook.pdf"

HEADER = "Northbridge College of Engineering & Technology - Student Helpdesk Handbook 2026-27"
SECTIONS = [  # (id prefix, section title, intro sentence)
    ("ADM", "Admissions", "Eligibility, application process, documents and cancellation."),
    ("FEE", "Fees and Payments", "Fee structure, payment methods, instalments, receipts and refunds."),
    ("SCH", "Scholarships", "Government and institute scholarships, deadlines and pending credits."),
    ("EXM", "Examinations", "Evaluation pattern, hall tickets, revaluation, backlogs and results."),
    ("ACD", "Academics and Attendance", "Attendance rules, calendar, electives, branch change and internships."),
    ("LIB", "Library", "Timings, borrowing rules, fines and e-resources."),
    ("HOS", "Hostel", "Accommodation, fees, rules, maintenance and mess."),
    ("PLC", "Training and Placements", "Placement eligibility, registration, statistics and training."),
    ("IT", "IT Services", "ERP password, campus Wi-Fi, college email and software licences."),
    ("TRN", "Transport", "College buses and travel concessions."),
    ("CRT", "Certificates and Documents", "Bonafide, transcripts, leaving certificate, duplicates and degree."),
    ("WEL", "Student Welfare and Safety", "Anti-ragging, grievances, counselling and the ICC."),
    ("MED", "Medical Facilities", "Campus medical centre and insurance."),
    ("SPT", "Sports and Clubs", "Sports facilities and student clubs."),
    ("GEN", "General Information", "Office timings, contact details, dress code and exchange programmes."),
]

styles = getSampleStyleSheet()
S = {
    "cover_title": ParagraphStyle("ct", parent=styles["Title"], fontSize=26, leading=32, textColor=colors.HexColor("#1c3f94")),
    "cover_sub": ParagraphStyle("cs", parent=styles["Normal"], fontSize=13, leading=18, alignment=TA_CENTER),
    "cover_note": ParagraphStyle("cn", parent=styles["Normal"], fontSize=9.5, leading=13, alignment=TA_CENTER,
                                 textColor=colors.HexColor("#555555")),
    "h1": ParagraphStyle("h1", parent=styles["Heading1"], fontSize=17, textColor=colors.HexColor("#1c3f94"), spaceAfter=4),
    "intro": ParagraphStyle("in", parent=styles["Italic"], fontSize=10, textColor=colors.HexColor("#444444"), spaceAfter=8),
    "q": ParagraphStyle("q", parent=styles["Heading3"], fontSize=11.5, leading=14.5, spaceBefore=8, spaceAfter=2),
    "alias": ParagraphStyle("al", parent=styles["Normal"], fontSize=8.5, leading=11, textColor=colors.HexColor("#666666")),
    "body": ParagraphStyle("bd", parent=styles["Normal"], fontSize=10.5, leading=14.5, spaceBefore=3, spaceAfter=3),
    "route": ParagraphStyle("rt", parent=styles["Normal"], fontSize=8.5, leading=11, textColor=colors.HexColor("#1c3f94")),
    "toc": ParagraphStyle("toc", parent=styles["Normal"], fontSize=11, leading=17),
}


def header_footer(canvas, doc):
    if doc.page == 1:
        return  # no running header on the cover
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(18 * mm, A4[1] - 12 * mm, HEADER)
    canvas.line(18 * mm, A4[1] - 13.5 * mm, A4[0] - 18 * mm, A4[1] - 13.5 * mm)
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.drawString(18 * mm, 10 * mm, "Fictional sample data for an academic project")
    canvas.restoreState()


def topic_block(e):
    parts = [
        Paragraph(f"[{e['id']}] {escape(e['question'])}", S["q"]),
        Paragraph("Also asked as: " + escape("; ".join(e.get("alt_questions", []))), S["alias"]),
        Paragraph("Answer: " + escape(e["answer"]), S["body"]),
        Paragraph(
            f"Routed to: {escape(e['department'])} | Category: {escape(e['category'])} | "
            f"Priority: {e['priority']} | Action: {escape(e['action'])}",
            S["route"],
        ),
    ]
    return KeepTogether(parts)


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    by_prefix = {}
    for e in data["entries"]:
        by_prefix.setdefault(e["id"].split("-")[0], []).append(e)
    missing = set(by_prefix) - {p for p, _, _ in SECTIONS}
    assert not missing, f"entries without a section: {missing}"

    story = [
        Spacer(1, 55 * mm),
        Paragraph("Student Helpdesk Handbook", S["cover_title"]),
        Spacer(1, 6 * mm),
        Paragraph("Northbridge College of Engineering &amp; Technology, Pune", S["cover_sub"]),
        Paragraph(f"Academic Year {data['version']} | Updated {data['last_updated']}", S["cover_sub"]),
        Spacer(1, 70 * mm),
        Paragraph("Issued by the Office of the Registrar. This handbook answers the questions students most often ask "
                  "the helpdesk and shows which office handles each request.", S["cover_note"]),
        Spacer(1, 4 * mm),
        Paragraph("The institution, fees, dates and contacts in this document are fictional sample data.", S["cover_note"]),
        PageBreak(),
        Paragraph("Contents", S["h1"]),
        Spacer(1, 4 * mm),
    ]
    rows = [[f"Section {i}", title, f"{(n := len(by_prefix.get(p, [])))} topic{'' if n == 1 else 's'}"] for i, (p, title, _) in enumerate(SECTIONS, 1)]
    toc = Table(rows, colWidths=[28 * mm, 100 * mm, 30 * mm])
    toc.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 10.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                             ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc"))]))
    story += [toc, PageBreak()]

    for i, (prefix, title, intro) in enumerate(SECTIONS, 1):
        story.append(Paragraph(f"Section {i} - {title}", S["h1"]))
        story.append(Paragraph(intro, S["intro"]))
        story += [topic_block(e) for e in by_prefix.get(prefix, [])]
        story.append(Spacer(1, 6 * mm))

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm, bottomMargin=18 * mm,
        title="Student Helpdesk Handbook 2026-27", author="Northbridge College (fictional)",
        subject=f"Helpdesk knowledge base, edition {data['version']}",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB) with {len(data['entries'])} topics")


if __name__ == "__main__":
    main()

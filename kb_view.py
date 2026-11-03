"""Streamlit "Knowledge base (PDF)" page: shows how the handbook PDF is processed and searched."""

import re

import pandas as pd
import streamlit as st

from pdf_ingest import CHUNK_OVERLAP_CHARS, MAX_CHUNK_CHARS, naive_chunks
from support_rag import MIN_SCORE, SupportRAG, build_messages

TOPIC_MARK = re.compile(r"\[[A-Z]+-\d{3}\]")


def render_kb(rag: SupportRAG):
    kb, rep = rag.kb, rag.kb.report
    st.title("Knowledge base: from PDF to answers")
    st.markdown(
        "The assistant's knowledge comes from one document, the **Student Helpdesk Handbook** (PDF). "
        "At startup it is processed by `pdf_ingest.py` and indexed by `support_rag.py`. "
        "This page shows every step and lets you inspect the result."
    )

    # ------------------------------------------------------------------ source
    st.subheader("0 · Source document")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pages", len(rep.pages))
    c2.metric("File size", f"{rep.file_kb:.0f} KB")
    c3.metric("Sections", len(rep.sections))
    c4.metric("Chunks indexed", len(rep.chunks))
    st.caption(f"**{rep.path.name}** · title: {rep.metadata.get('Title', '—')} · {rep.metadata.get('Subject', '')}")
    st.download_button("Download the handbook PDF", rep.path.read_bytes(), file_name=rep.path.name,
                       mime="application/pdf")

    st.markdown("**Processing pipeline**")
    st.code(
        "PDF ─► 1. extract text (pypdf, per page)\n"
        "    ─► 2. clean: drop running header/footer, skip cover & contents pages\n"
        "    ─► 3. detect structure: 'Section N - …' headings and '[ID] …' topics\n"
        f"    ─► 4. chunk: one chunk per topic (split if > {MAX_CHUNK_CHARS} chars, {CHUNK_OVERLAP_CHARS} overlap)\n"
        "         + parse 'Also asked as' and 'Routed to' lines into metadata\n"
        "    ─► 5. index: word TF-IDF (1-2 grams) + character TF-IDF (3-5 grams)\n"
        "    ─► 6. retrieve top-k ─► threshold ─► prompt ─► LLM ─► answer + ticket",
        language=None,
    )
    st.dataframe(pd.DataFrame({"step": rep.timings_ms.keys(), "time (ms)": rep.timings_ms.values()}),
                 hide_index=True, use_container_width=True)
    if rep.warnings:
        st.warning("Ingestion warnings: " + "; ".join(rep.warnings))

    # ------------------------------------------------------------------ extraction & cleaning
    st.subheader("1–2 · Text extraction and cleaning")
    pages_df = pd.DataFrame([{
        "page": p.number, "raw chars": len(p.raw_text), "clean chars": len(p.clean_text),
        "header/footer lines removed": len(p.removed_lines), "status": p.status,
    } for p in rep.pages])
    st.dataframe(pages_df, hide_index=True, use_container_width=True)
    n = st.selectbox("Inspect page", [p.number for p in rep.pages], index=2)
    page = rep.pages[n - 1]
    left, right = st.columns(2)
    left.markdown("**Raw text from pypdf**")
    left.code(page.raw_text[:2500], language=None)
    right.markdown(f"**After cleaning** · {page.status}")
    right.code(page.clean_text[:2500], language=None)
    if page.removed_lines:
        st.caption("Removed: " + " · ".join(f"`{l}`" for l in page.removed_lines))

    # ------------------------------------------------------------------ chunking
    st.subheader("3–4 · Structure-aware chunking")
    lengths = [len(c.text) for c in rep.chunks]
    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks", len(rep.chunks))
    c2.metric("Avg answer length", f"{sum(lengths) / len(lengths):.0f} chars")
    c3.metric("Shortest / longest", f"{min(lengths)} / {max(lengths)}")
    st.markdown(
        "Each `[ID]` topic becomes one chunk, so a chunk never mixes two policies. The printed labels "
        "*Also asked as*, *Answer* and *Routed to* are parsed into fields: the alternate phrasings improve "
        "retrieval, and the routing line becomes the helpdesk ticket."
    )
    section = st.selectbox("Filter by section", ["All"] + rep.sections)
    rows = [c for c in rep.chunks if section == "All" or c.section == section]
    st.dataframe(pd.DataFrame([{
        "id": c.id, "section": c.section, "page": ", ".join(map(str, c.pages)), "title": c.title,
        "chars": len(c.text), "department": c.department, "priority": c.priority,
    } for c in rows]), hide_index=True, use_container_width=True)
    chunk_id = st.selectbox("Inspect chunk", [c.id for c in rows])
    c = next(c for c in rep.chunks if c.id == chunk_id)
    with st.container(border=True):
        st.markdown(f"**[{c.id}] {c.title}**  \n*{c.section} · page {', '.join(map(str, c.pages))}*")
        st.markdown(f"**Text:** {c.text}")
        st.markdown(f"**Also asked as:** {'; '.join(c.aliases)}")
        st.markdown(f"**Metadata:** department `{c.department}` · category `{c.category}` · "
                    f"priority `{c.priority}` · action `{c.action}`")

    with st.expander("Why not fixed-size chunks? Compare with naive 500-character chunking"):
        naive = naive_chunks(rep)
        # a chunk mixes topics when a new "[ID]" topic starts anywhere after its first character
        mixed = sum(1 for t in naive if any(m.start() > 0 for m in TOPIC_MARK.finditer(t)))
        st.markdown(
            f"Splitting the same cleaned text every 500 characters (100 overlap) gives **{len(naive)} chunks**, and "
            f"**{mixed}** of them contain parts of two or more different topics. A retrieved chunk could then mix, "
            "for example, the hostel fee with the hostel application deadline, and topics cut mid-sentence lose "
            "their routing line. Two examples:"
        )
        for t in naive[3:5]:
            st.code(t, language=None)

    # ------------------------------------------------------------------ index & retrieval
    st.subheader("5–6 · Index and retrieval explorer")
    st.caption(f"Word vocabulary: {len(kb.word_vec.vocabulary_):,} terms · character n-gram vocabulary: "
               f"{len(kb.char_vec.vocabulary_):,} · score = 0.6 × word + 0.4 × character cosine similarity · "
               f"answer threshold = {MIN_SCORE}")
    query = st.text_input("Try a question", "my fees got deducted but portal says unpaid")
    if query.strip():
        hits = kb.search(query.strip(), k=5)
        st.dataframe(pd.DataFrame([{
            "rank": i + 1, "chunk": h.entry["id"], "page": h.entry["pages"][0], "title": h.entry["question"],
            "word score": round(h.word_score, 3), "char score": round(h.char_score, 3),
            "combined": round(h.score, 3), "used?": "yes" if h.score >= MIN_SCORE else "no",
        } for i, h in enumerate(hits)]), hide_index=True, use_container_width=True)
        relevant = [h for h in hits[:3] if h.score >= MIN_SCORE]
        if not relevant:
            st.warning(f"Best score {hits[0].score:.3f} is below {MIN_SCORE}: the app answers with the fallback "
                       "message and does not call the LLM.")
        else:
            with st.expander("Prompt sent to the LLM for this question"):
                for m in build_messages(query.strip(), relevant):
                    st.markdown(f"**{m['role']}**")
                    st.code(m["content"], language=None)

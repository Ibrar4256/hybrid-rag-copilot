# -*- coding: utf-8 -*-
"""Generates two PDFs from content.py: a Questions-only prep doc and a full
Answers doc (question + detailed answer). Run: python build_pdfs.py"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, ListFlowable, ListItem,
)

from content import TOPICS, SECTIONS

OUT_DIR = Path(__file__).parent

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleBig", parent=styles["Title"], fontSize=22, spaceAfter=6)
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=12,
                                 textColor=colors.HexColor("#555555"), spaceAfter=20)
h1_style = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16,
                            spaceBefore=24, spaceAfter=10, textColor=colors.HexColor("#1a3d5c"))
h2_topic_style = ParagraphStyle("H2Topic", parent=styles["Heading2"], fontSize=12,
                                  spaceBefore=10, spaceAfter=2, textColor=colors.HexColor("#1a3d5c"))
q_style = ParagraphStyle("Question", parent=styles["Normal"], fontSize=11, leading=15,
                           spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#111111"))
q_id_style = ParagraphStyle("QID", parent=styles["Normal"], fontSize=11, leading=15,
                              spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#8a1c1c"))
a_style = ParagraphStyle("Answer", parent=styles["Normal"], fontSize=10.5, leading=15,
                           spaceBefore=2, spaceAfter=14, textColor=colors.HexColor("#222222"))
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15,
                              spaceAfter=6)
toc_entry_style = ParagraphStyle("TOCEntry", parent=styles["Normal"], fontSize=10.5, leading=14)


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<b>", "\x00B\x00").replace("</b>", "\x00b\x00")
                .replace("<", "&lt;").replace(">", "&gt;")
                .replace("\x00B\x00", "<b>").replace("\x00b\x00", "</b>"))


def build_topics_flowables():
    flow = [Paragraph("Topics to Learn", h1_style),
            Paragraph(
                "Study these before attempting the questions. Each maps directly to a "
                "section below and to real decisions/bugs from this project.", body_style),
            Spacer(1, 6)]
    rows = [[Paragraph(f"<b>{_esc(name)}</b>", body_style), Paragraph(_esc(desc), body_style)]
            for name, desc in TOPICS]
    t = Table(rows, colWidths=[1.9 * inch, 4.6 * inch])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f7f7")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    flow.append(t)
    flow.append(PageBreak())
    return flow


def build_toc_flowables():
    flow = [Paragraph("Contents", h1_style)]
    for section_title, items in SECTIONS:
        flow.append(Paragraph(f"<b>{_esc(section_title)}</b>", toc_entry_style))
        ids = ", ".join(qid for qid, _, _ in items)
        flow.append(Paragraph(ids, toc_entry_style))
        flow.append(Spacer(1, 6))
    flow.append(PageBreak())
    return flow


def build_questions_pdf():
    doc = SimpleDocTemplate(
        str(OUT_DIR / "Research_Copilot_Interview_Questions.pdf"),
        pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
    )
    flow = []
    flow.append(Paragraph("Research Copilot — Interview Prep", title_style))
    flow.append(Paragraph(
        "Questions Only &mdash; RAG, Agentic Systems &amp; Production LLM Engineering", subtitle_style))
    flow.append(Paragraph(
        "50 scenario-based questions across 12 topics, grounded in real architecture "
        "decisions, real bugs, and real eval results from building the Research Copilot "
        "project (Weeks 1). Answers are in the companion PDF &mdash; attempt these first.",
        body_style))
    flow.append(Spacer(1, 16))
    flow += build_topics_flowables()
    flow += build_toc_flowables()

    for section_title, items in SECTIONS:
        flow.append(Paragraph(section_title, h1_style))
        for qid, question, _answer in items:
            flow.append(Paragraph(f"<b>{qid}.</b> {_esc(question)}", q_style))
        flow.append(Spacer(1, 10))

    doc.build(flow)
    print(f"Wrote {OUT_DIR / 'Research_Copilot_Interview_Questions.pdf'}")


def build_answers_pdf():
    doc = SimpleDocTemplate(
        str(OUT_DIR / "Research_Copilot_Interview_Answers.pdf"),
        pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
    )
    flow = []
    flow.append(Paragraph("Research Copilot — Interview Prep", title_style))
    flow.append(Paragraph(
        "Detailed Answers &mdash; RAG, Agentic Systems &amp; Production LLM Engineering", subtitle_style))
    flow.append(Paragraph(
        "Full explanations for all 50 questions, referencing this project's actual ADRs, "
        "real ablation numbers, and real bugs found and fixed during the eval-harness build. "
        "Use these to check your own answers, not as a substitute for attempting the "
        "questions cold first.", body_style))
    flow.append(Spacer(1, 16))
    flow += build_topics_flowables()
    flow += build_toc_flowables()

    for section_title, items in SECTIONS:
        flow.append(Paragraph(section_title, h1_style))
        for qid, question, answer in items:
            flow.append(Paragraph(f"<b>{qid}.</b> {_esc(question)}", q_id_style))
            flow.append(Paragraph(_esc(answer), a_style))
        flow.append(Spacer(1, 10))

    doc.build(flow)
    print(f"Wrote {OUT_DIR / 'Research_Copilot_Interview_Answers.pdf'}")


if __name__ == "__main__":
    build_questions_pdf()
    build_answers_pdf()

"""Build docs/prototype-overview.pdf: diagram + worked example + results for the Track 2 prototype.

Needs reportlab (pip install reportlab) and Liberation Sans. Run from the project root:
    python docs/build_overview.py docs/prototype-overview.pdf
"""

import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

FONT_DIR = "/usr/share/fonts/liberation-sans-fonts/"
pdfmetrics.registerFont(TTFont("Sans", FONT_DIR + "LiberationSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", FONT_DIR + "LiberationSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Italic", FONT_DIR + "LiberationSans-Italic.ttf"))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic")

INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5B6770")
RULES = colors.HexColor("#DCE8FB")      # rule-based code
RULES_EDGE = colors.HexColor("#3B6FD8")
LLM = colors.HexColor("#FDECC8")        # language model
LLM_EDGE = colors.HexColor("#C98A0B")
DATA = colors.HexColor("#D8F0DF")       # Hagstofa PX-API
DATA_EDGE = colors.HexColor("#2E8B57")
IO = colors.HexColor("#ECEFF2")         # user in/out
IO_EDGE = colors.HexColor("#6B7785")

title = ParagraphStyle("title", fontName="Sans-Bold", fontSize=22, leading=26, textColor=INK)
subtitle = ParagraphStyle("subtitle", fontName="Sans", fontSize=11.5, leading=15, textColor=MUTED)
h2 = ParagraphStyle("h2", fontName="Sans-Bold", fontSize=15, leading=19, textColor=INK, spaceAfter=6)
body = ParagraphStyle("body", fontName="Sans", fontSize=10, leading=13.5, textColor=INK)
small = ParagraphStyle("small", fontName="Sans", fontSize=8.8, leading=11.5, textColor=INK)
cell = ParagraphStyle("cell", fontName="Sans", fontSize=8.6, leading=10.6, textColor=INK, alignment=TA_LEFT)
cell_bold = ParagraphStyle("cellb", parent=cell, fontName="Sans-Bold")


# ------------------------------------------------------------------ diagram

STEPS = [
    # (kind, title, file, lines)
    ("io", "1  Question", "frontend → app.py",
     ["POST /spyrja_px_api", "Several questions in one", "prompt are split and", "answered one by one"]),
    ("llm", "2  Understand", "answer.py + normalize.py",
     ["Icelandic: fast LLM", "(qwen3:30b-a3b) translates", "to English, fixes typos,", "gives Icelandic keywords",
      "Then rules: years, places,", "breakdowns, intent"]),
    ("rules", "3  Search", "table_search.py",
     ["2,102 tables", "BM25 on 4-letter pieces", "of title, folders,", "variables, values",
      "+ vectors (qwen3-", "embedding) = hybrid"]),
    ("rules", "4  Candidates", "table_choice.py",
     ["Main tables for common", "topics always included", "Must cover the years", "asked (both ends)",
      "Marks places found", "(\"inniheldur: Flateyri\")"]),
    ("llm", "5  Choose table", "table_choice.py",
     ["qwen3:32b reads the", "top 10 and ranks", "the best 3", "Forced JSON answer",
      "Falls back to the", "search order"]),
    ("rules", "6  Selections", "answer.py",
     ["One value per dimension:", "time = years asked", "or newest; place =", "matching value;",
      "\"eftir X\" = all values;", "else total (Alls)"]),
    ("data", "7  Fetch numbers", "px_query.py",
     ["Validates every", "dimension against", "metadata, max 500", "cells, then POST", "to Hagstofa PX-API"]),
    ("llm", "8  Write answer", "answer.py",
     ["qwen3:32b writes a", "short answer from the", "numbers only, with", "period + source",
      "Notes if year or place", "is missing"]),
]

STYLE = {
    "io": (IO, IO_EDGE),
    "rules": (RULES, RULES_EDGE),
    "llm": (LLM, LLM_EDGE),
    "data": (DATA, DATA_EDGE),
}


class PipelineDiagram(Flowable):
    """Two rows of four boxes, snaking left-to-right then right-to-left."""

    def __init__(self, width, height):
        super().__init__()
        self.width, self.height = width, height

    def wrap(self, *args):
        return self.width, self.height

    def draw(self):
        c = self.canv
        cols, gap_x, gap_y = 4, 16 * mm, 14 * mm
        box_w = (self.width - (cols - 1) * gap_x) / cols
        box_h = (self.height - gap_y - 14 * mm) / 2
        legend_h = 14 * mm

        positions = []
        for i in range(len(STEPS)):
            row, col = divmod(i, cols)
            if row == 1:
                col = cols - 1 - col  # snake back
            x = col * (box_w + gap_x)
            y = legend_h + (1 - row) * (box_h + gap_y)
            positions.append((x, y))

        # arrows first, under the boxes
        c.setStrokeColor(MUTED)
        c.setFillColor(MUTED)
        c.setLineWidth(1.6)
        for i in range(len(STEPS) - 1):
            (x1, y1), (x2, y2) = positions[i], positions[i + 1]
            if y1 == y2:
                if x2 > x1:
                    self.arrow(x1 + box_w, y1 + box_h / 2, x2, y2 + box_h / 2)
                else:
                    self.arrow(x1, y1 + box_h / 2, x2 + box_w, y2 + box_h / 2)
            else:
                self.arrow(x1 + box_w / 2, y1, x2 + box_w / 2, y2 + box_h)

        for (x, y), (kind, name, file, lines) in zip(positions, STEPS):
            fill, edge = STYLE[kind]
            c.setFillColor(fill)
            c.setStrokeColor(edge)
            c.setLineWidth(1.3)
            c.roundRect(x, y, box_w, box_h, 3.5 * mm, stroke=1, fill=1)
            c.setFillColor(INK)
            c.setFont("Sans-Bold", 11.5)
            c.drawString(x + 4 * mm, y + box_h - 7.5 * mm, name)
            c.setFillColor(edge)
            c.setFont("Sans-Italic", 8.5)
            c.drawString(x + 4 * mm, y + box_h - 12 * mm, file)
            c.setFillColor(INK)
            c.setFont("Sans", 8.8)
            c.setFont("Sans", 9.4)
            for n, line in enumerate(lines):
                c.drawString(x + 4 * mm, y + box_h - 19 * mm - n * 4.8 * mm, line)

        # legend
        items = [("rules", "Rule-based code (fast, same result every time)"),
                 ("llm", "Language model (qwen3 on the GB10 GPU machine)"),
                 ("data", "Hagstofa PX-API (official numbers)"),
                 ("io", "User / web")]
        x = 0
        for kind, label in items:
            fill, edge = STYLE[kind]
            c.setFillColor(fill)
            c.setStrokeColor(edge)
            c.roundRect(x, 3 * mm, 6 * mm, 6 * mm, 1.5 * mm, stroke=1, fill=1)
            c.setFillColor(INK)
            c.setFont("Sans", 9)
            c.drawString(x + 8 * mm, 4.6 * mm, label)
            x += 8 * mm + c.stringWidth(label, "Sans", 9) + 10 * mm

    def arrow(self, x1, y1, x2, y2):
        c = self.canv
        pad = 1.5 * mm
        if y1 == y2:
            direction = 1 if x2 > x1 else -1
            x1, x2 = x1 + direction * pad, x2 - direction * pad
            c.line(x1, y1, x2 - direction * 3 * mm, y2)
            path = c.beginPath()
            path.moveTo(x2, y2)
            path.lineTo(x2 - direction * 3.2 * mm, y2 + 1.8 * mm)
            path.lineTo(x2 - direction * 3.2 * mm, y2 - 1.8 * mm)
            path.close()
        else:
            y1, y2 = y1 - pad, y2 + pad
            c.line(x1, y1, x2, y2 + 3 * mm)
            path = c.beginPath()
            path.moveTo(x2, y2)
            path.lineTo(x2 - 1.8 * mm, y2 + 3.2 * mm)
            path.lineTo(x2 + 1.8 * mm, y2 + 3.2 * mm)
            path.close()
        c.drawPath(path, stroke=0, fill=1)


# ------------------------------------------------------------------ tables

def table(rows, widths, header_fill=colors.HexColor("#E6EAEE"), kinds=None):
    data = [[Paragraph(str(v), cell_bold if r == 0 else cell) for v in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_fill),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C7CED6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for r, kind in enumerate(kinds or [], start=1):
        if kind:
            style.append(("BACKGROUND", (0, r), (0, r), STYLE[kind][0]))
    t.setStyle(TableStyle(style))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(15 * mm, 8 * mm, "HAM · Hagstofa hackathon 2026 · Track 2 (PX-API) · branch: prototype")
    canvas.drawRightString(landscape(A4)[0] - 15 * mm, 8 * mm, f"{doc.page}")
    canvas.restoreState()


def build(path):
    page_w, page_h = landscape(A4)
    doc = SimpleDocTemplate(path, pagesize=landscape(A4), leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=13 * mm, bottomMargin=14 * mm,
                            title="Track 2 prototype: from question to official numbers",
                            author="HAM team")
    width = page_w - 30 * mm
    story = []

    # ---- page 1: diagram
    story.append(Paragraph("From question to official numbers", title))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "Our Track 2 prototype answers questions about Icelandic statistics in Icelandic or English. "
        "The numbers always come straight from Hagstofa's PX-API; the language model only "
        "<b>reads the question</b>, <b>chooses the table</b> and <b>phrases the answer</b>.", subtitle))
    story.append(Spacer(1, 8 * mm))
    story.append(PipelineDiagram(width, 140 * mm))

    # ---- page 2: worked example
    story.append(PageBreak())
    story.append(Paragraph("Example: “Hvað áttu margir heima á Flateyri 1990?”", h2))
    story.append(Paragraph("What each step produces for one real question (about 10 seconds in total).", subtitle))
    story.append(Spacer(1, 4 * mm))
    rows = [
        ["Step", "What happens", "Result"],
        ["2  Understand", "The fast model translates the question (fixing typos and inflected place names) and suggests Icelandic title keywords; rules then pull out the year and place.",
         "“How many people lived in Flateyri in 1990?” · years: [1990] · keywords: flateyri, mannfjöldi"],
        ["3  Search", "Hybrid search over 2,102 tables. Flateyri is found among the value labels of locality tables.",
         "Population tables by locality (byggðakjarnar), old and new"],
        ["4  Candidates", "Tables that start after 1990 (most locality tables start in 1998) are dropped.",
         "MAN00104 … 1889–1990, marked “inniheldur: Flateyri, Flateyrarhr.”"],
        ["5  Choose table", "qwen3:32b reads the 10 candidates and ranks the best three.",
         "#1 MAN00104 – Mannfjöldi í einstökum byggðakjörnum … 1889–1990"],
        ["6  Selections", "Whole-word match on the place (not Flatey á Breiðafirði); year from the question.",
         "Byggðarkjarni: Flateyri, Flateyrarhr. · Ár: 1990"],
        ["7  Fetch numbers", "Query validated against the table's metadata, then sent to the PX-API.",
         "Flateyri, 1990 → 396"],
        ["8  Write answer", "The model writes one sentence from the number, with the period and the source.",
         "One sentence with 396, the year 1990 and Flateyri, then “Heimild:” with the table name and link"],
    ]
    story.append(table(rows, [32 * mm, 120 * mm, width - 152 * mm],
                       kinds=["llm", "rules", "rules", "llm", "rules", "data", "llm"]))
    story.append(Spacer(1, 7 * mm))
    story.append(Paragraph("Why it is built this way", h2))
    for point in [
        "<b>Numbers never come from the language model.</b> They are fetched from the PX-API and passed to it; the answer must use only those.",
        "<b>Every answer shows its source</b>: table name, link, the chosen dimensions and the raw data rows (expandable in the web page).",
        "<b>Honest when data is missing</b>: if the year or the place is not in the table, the answer says so instead of guessing "
        "(e.g. a question about 2030 gets the newest year, 2026, with a note).",
        "<b>Rules plus LLM where it helps</b>: rules are fast and predictable; the LLM handles the ambiguous parts "
        "(understanding everyday or misspelled questions, choosing between similar tables, phrasing).",
        "<b>Keeps working if the LLM fails</b>: table choice falls back to the search order, and the answer falls back to showing the data.",
    ]:
        story.append(Paragraph("•  " + point, body))
        story.append(Spacer(1, 2.2 * mm))

    # ---- page 3: results and limitations
    story.append(PageBreak())
    story.append(Paragraph("Results", h2))
    story.append(Paragraph(
        "How often the right table is chosen first, on the 83 questions in AI_test/spurningar.txt "
        "(analytical questions written by a team member) and our 24 shorter questions.", subtitle))
    story.append(Spacer(1, 2 * mm))
    results = [
        ["Method", "83 questions", "24 questions", "Time per choice"],
        ["Search alone (hybrid BM25 + vectors)", "35", "12", "0.3 s"],
        ["+ qwen3:4b choosing (laptop CPU), 40-question sample", "24 of 40 (search alone: 23)", "", "≈ 25 s"],
        ["+ qwen3:30b-a3b choosing (GPU)", "39", "17", "0.7 s"],
        ["+ qwen3:32b choosing (GPU) – used", "44", "17", "2–3 s"],
        ["Right table anywhere in the top 10", "55", "21", "–"],
    ]
    story.append(table(results, [95 * mm, 45 * mm, 35 * mm, 35 * mm]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Tested end to end (through the web API)", h2))
    tested = [
        ["Question", "Table", "Answer"],
        ["Hvað búa margir í Kópavogi?", "MAN02005", "40,286 (1 Jan 2026)"],
        ["Hvað bjuggu margir á Íslandi árið 2024?", "MAN00000", "383,726 (1 Jan 2024)"],
        ["Hver var verðbólgan á Íslandi árið 2024?", "VIS01000", "4.8 % (12-month change, Dec 2024)"],
        ["Hversu margir fluttu til Íslands árið 2024?", "MAN01400", "19,789 immigrants, 15,745 emigrants"],
        ["Hvað fæddust mörg börn 2023?", "MAN05203", "4,315"],
        ["Hvað áttu margir heima á Flateyri 1990?", "MAN00104", "396"],
        ["How many people live in Reykjavík?", "MAN03270", "139,804 (1 Jan 2026)"],
        ["Hvað búa margir á Íslandi árið 2030?", "MAN00000", "2030 not available; newest is 2026"],
        ["asdkj qwpe", "–", "“Ég skildi ekki spurninguna …”"],
    ]
    story.append(table(tested, [95 * mm, 30 * mm, width - 125 * mm]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Known limitations and next step", h2))
    for point in [
        "<b>Understanding and selections are rule-based.</b> New phrasings need new rules, and if a question's words don't look like the "
        "table's value labels, the total is used. <b>Next step:</b> let the LLM pick a value for each dimension from the table's metadata, "
        "validated by px_query.",
        "Monthly tables asked about a whole year use the last month: right for stocks (population, inflation), not for flows "
        "(passengers in a year would need a sum).",
        "Answers take 5–25 seconds; English questions are slowest (extra translation step). The model's Icelandic grammar is not perfect.",
        "Hagur (hagstofa-ai:latest) was not available to us; the model is a setting (MODEL=…), so it can be swapped in.",
    ]:
        story.append(Paragraph("•  " + point, small))
        story.append(Spacer(1, 1 * mm))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build(sys.argv[1])

"""Generate tests/corpus/lecture.pdf — the text-layer test document.

Three A4 pages of lecture notes built to trip every known failure:
  8a  title is a single large line inside the top band
  8b  title repeats the running header's exact words
  8c  an ISO date byline sits at ~14% of the page
  8d  the running header has an underline rule
plus a superscript equation, a vector figure with node labels, a margin
note, faint furniture, a handle, and a CamScanner watermark.

Page 1 is expected to yield exactly nine regions.
Run:  python tests/corpus/make_test_pdf.py   (needs reportlab; dev only)
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

W, H = A4
LEFT, BODY_W = 72, 380
OUT = Path(__file__).with_name("lecture.pdf")


def header_footer(c, n):
    c.setFillGray(0.35)
    c.setFont("Helvetica", 9)
    c.drawString(LEFT, H - 40, "Balanced Binary Search Trees")
    c.setLineWidth(0.6)
    c.line(LEFT, H - 45, W - LEFT, H - 45)
    c.drawCentredString(W / 2, 30, f"Page {n}")
    c.setFillGray(0)


def para(c, y, text, size=11, lead=14.5, width=BODY_W, font="Helvetica"):
    """Greedy word wrap. Returns the y below the paragraph."""
    c.setFont(font, size)
    line = ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if c.stringWidth(trial, font, size) > width:
            c.drawString(LEFT, y, line)
            y -= lead
            line = word
        else:
            line = trial
    c.drawString(LEFT, y, line)
    return y - lead


def tree(c, x, y):
    """A height-2 balanced BST: circles, edges, node labels."""
    nodes = {8: (x + 110, y + 120), 4: (x + 55, y + 65), 12: (x + 165, y + 65),
             2: (x + 25, y + 10), 6: (x + 85, y + 10), 10: (x + 135, y + 10),
             14: (x + 195, y + 10)}
    c.setLineWidth(1.2)
    for a, b in [(8, 4), (8, 12), (4, 2), (4, 6), (12, 10), (12, 14)]:
        c.line(*nodes[a], *nodes[b])
    for k, (nx, ny) in nodes.items():
        c.setFillGray(1)
        c.circle(nx, ny, 13, stroke=1, fill=1)
        c.setFillGray(0)
        c.setFont("Helvetica", 10)
        c.drawCentredString(nx, ny - 3.5, str(k))


def bars(c, x, y):
    c.setLineWidth(0.8)
    c.line(x, y, x + 230, y)
    c.line(x, y, x, y + 140)
    for i, hgt in enumerate([30, 55, 70, 95, 118]):
        c.setFillGray(0.55)
        c.rect(x + 15 + i * 42, y, 28, hgt, stroke=0, fill=1)
    c.setFillGray(0)


def page1(c):
    header_footer(c, 1)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(LEFT, H - 82, "Balanced Binary Search Trees")
    c.setFont("Helvetica", 10)
    c.drawString(LEFT, H - 118, "2026-09-19")
    y = para(c, H - 150,
             "A binary search tree keeps its keys in sorted order, so lookup, insertion and "
             "deletion can skip half of the remaining tree at every step. That promise only "
             "holds while the tree stays balanced. A balanced binary search tree maintains "
             "logarithmic height by performing rotations after insertions and deletions. The "
             "most common variants are AVL trees and red-black trees, and both guarantee the "
             "bound below for every sequence of operations.")
    c.setFont("Helvetica", 9)
    c.drawString(475, H - 165, "ask about AVL")
    c.setFont("Helvetica", 12)
    c.drawString(LEFT + 120, y - 16, "h = O(log n)")
    tree(c, LEFT + 70, y - 215)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(LEFT + 40, y - 245, "Fig 3. A balanced binary search tree of height 2")
    c.showPage()


def page2(c):
    header_footer(c, 2)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(LEFT, H - 95, "Rotations")
    y = para(c, H - 125,
             "A rotation changes the shape of a subtree without changing the in-order sequence "
             "of its keys. A single left rotation lifts the right child into the parent's place "
             "and makes the old parent its left child.")
    c.setFont("Helvetica", 12)
    x = LEFT + 120
    for part, sup in [("x", "2"), (" + y", "2"), (" = r", "2")]:
        c.setFont("Helvetica", 12)
        c.drawString(x, y - 18, part)
        x += c.stringWidth(part, "Helvetica", 12)
        c.setFont("Helvetica", 8)
        c.drawString(x, y - 13, sup)
        x += c.stringWidth(sup, "Helvetica", 8)
    para(c, y - 50,
         "Rotations are constant-time operations, so the cost of rebalancing is dominated by "
         "walking back up the path from the modified leaf to the root.")
    c.setFillGray(0.72)
    c.setFont("Helvetica", 8)
    c.drawString(LEFT, 52, "Draft notes, not for distribution")
    c.drawRightString(W - LEFT, 52, "@cse_notes")
    c.setFillGray(0)
    c.showPage()


def page3(c):
    header_footer(c, 3)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(LEFT, H - 95, "Red-black trees")
    y = para(c, H - 125,
             "A red-black tree colours every node and enforces that no red node has a red "
             "child. The longest path is therefore at most twice the shortest, which is "
             "enough to keep the height logarithmic.")
    bars(c, LEFT + 60, y - 190)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(LEFT + 60, y - 210, "Figure 4. Rotation cost by tree size")
    para(c, y - 250,
         "In practice red-black trees perform fewer rotations than AVL trees on insertion, "
         "which is why most standard libraries use them.")
    c.setFont("Helvetica", 8)
    c.setFillGray(0.4)
    c.drawString(LEFT, 48, "Scanned by CamScanner")
    c.setFillGray(0)
    c.showPage()


if __name__ == "__main__":
    c = canvas.Canvas(str(OUT), pagesize=A4)
    c.setTitle("Balanced Binary Search Trees")
    for p in (page1, page2, page3):
        p(c)
    c.save()
    print(f"wrote {OUT}")

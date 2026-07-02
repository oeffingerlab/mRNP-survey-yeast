"""
Assemble Part 1 manuscript figures into Illustrator-ready PDFs.
Uses reportlab for layout canvas + pypdf for embedding source vector PDFs.
Each source PDF is embedded as a full page placed at scale onto the canvas.
fonttype=42 preserved (reportlab does not re-encode embedded content).
"""
import os, shutil
import glob, os, shutil

# ── Clean old assembled PDFs before regenerating ─────────────────────────────
OUT_CLEAN = "/sessions/lucid-modest-goldberg/mnt/MS results/Assembled_Figures/"
old_files = glob.glob(OUT_CLEAN + "*.pdf")
for f in old_files:
    try:
        os.remove(f)
        print(f"  deleted: {os.path.basename(f)}")
    except OSError:
        pass  # sandbox can't delete; overwrite instead
print(f"Cleaned {len(old_files)} old file(s) from Assembled_Figures/")

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject
from reportlab.lib.pagesizes import A3, landscape
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.units import cm, inch
import io

OUT = "/sessions/lucid-modest-goldberg/mnt/MS results/Assembled_Figures/"
SRC = "/sessions/lucid-modest-goldberg/mnt/MS results/"

def pts(cm_val):
    return cm_val * 28.3465

# ─── Helper: scale a PDF page onto a target rect using pypdf ─────────────────
def embed_pdf_on_page(writer, src_path, x_pt, y_pt, target_w_pt, target_h_pt,
                       page_writer_idx=0, src_page=0):
    """
    Embed src_path[src_page] as an XObject onto writer.pages[page_writer_idx]
    at position (x_pt, y_pt) scaled to fit target_w_pt × target_h_pt.
    Returns the writer.
    """
    reader = PdfReader(src_path)
    src_pg = reader.pages[src_page]
    sw = float(src_pg.mediabox.width)
    sh = float(src_pg.mediabox.height)
    sx = target_w_pt / sw
    sy = target_h_pt / sh
    scale = min(sx, sy)
    # centre within target box
    dx = (target_w_pt - sw * scale) / 2
    dy = (target_h_pt - sh * scale) / 2
    t = Transformation().scale(scale, scale).translate(x_pt + dx, y_pt + dy)
    dest_page = writer.pages[page_writer_idx]
    dest_page.merge_transformed_page(src_pg, t, over=True)
    return writer

def blank_pdf_page(w_pt, h_pt):
    """Create a single-page blank PDF in a BytesIO buffer using reportlab."""
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=(w_pt, h_pt))
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, w_pt, h_pt, fill=1, stroke=0)
    c.save()
    buf.seek(0)
    return buf

def placeholder_box(c, x, y, w, h, label, fontsize=18):
    """Draw a grey placeholder box with label text on a reportlab canvas."""
    c.setStrokeColorRGB(0.6, 0.6, 0.6)
    c.setFillColorRGB(0.94, 0.94, 0.94)
    c.setLineWidth(1)
    c.rect(x, y, w, h, fill=1, stroke=1)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.setFont("Helvetica", fontsize)
    c.drawCentredString(x + w/2, y + h/2 - fontsize/3, label)

def panel_label(c, x, y, letter, fontsize=24):
    """Draw a bold panel label (A, B, C …) at position."""
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", fontsize)
    c.drawString(x, y, letter)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 1  (A–H)
# Layout:
#   Row 0: A (2/3 width) | B (1/3 width)   ← placeholders in CDEF composite
#   Row 1: C | D | E                        ← from CDEF composite
#   Row 2: F  (full width = same as G+H)    ← from CDEF composite
#   Row 3: G | H  (side by side)            ← from GH composite
# Strategy: embed CDEF in top ~75% of page, GH in bottom ~25%.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Assembling Figure 1...")

W1 = 11 * 72    # 792 pt  US Letter landscape
H1 = 8.5 * 72  # 612 pt
m1 = 6          # thin margin (pt) between CDEF and GH rows

# Heights match source figsize (inches × 72pt) so both embed at scale=1 (full width)
# 1/4 page per row group: each row = (H1-m1)/4 = 151.5pt
cdef_h = 6.25 * 72  # 3×(1/4 page) = 3/4 page → 450pt; CDEF source (11×6.25") ≤ this ✓
gh_h   = H1 - cdef_h - m1   # ≈155pt ≥ GH source (11×2.05"=147.6pt) ✓
y_gh   = 0
y_cdef = gh_h + m1

buf1 = io.BytesIO()
c1 = rl_canvas.Canvas(buf1, pagesize=(W1, H1))
c1.setFillColorRGB(1, 1, 1)
c1.rect(0, 0, W1, H1, fill=1, stroke=0)
c1.save()
buf1.seek(0)

writer1 = PdfWriter()
writer1.add_page(PdfReader(buf1).pages[0])

# CDEF composite (contains panels A, B, C, D, E, F) — top portion
embed_pdf_on_page(writer1, SRC + "figure1_panels_CDEF.pdf",
                  0, y_cdef, W1, cdef_h)

# GH composite (panels G, H) — bottom portion, same full width as F
embed_pdf_on_page(writer1, SRC + "figure1_panels_GH.pdf",
                  0, y_gh, W1, gh_h)

out1 = OUT + "Figure_1_ASSEMBLED.pdf"
with open(out1, "wb") as f:
    writer1.write(f)
print(f"  → Figure_1_ASSEMBLED.pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 2  (A–F)
# Layout: top (~35%) = Panel A (introns HCL, from figure2A_introns.pdf)
#         bottom (~65%) = Panels B–F (from figure2_composite.pdf)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Assembling Figure 2...")

margin = pts(0.8)   # shared margin value used by Figure 2+ sections
W2 = pts(22)
row_a_h    = pts(14)   # intron HCL + bar chart
row_bcde_h = pts(24)   # figure2_composite (B, C, D, E together)
H2 = margin + row_bcde_h + margin + row_a_h + margin

buf2 = io.BytesIO()
c2 = rl_canvas.Canvas(buf2, pagesize=(W2, H2))
c2.setFillColorRGB(1, 1, 1)
c2.rect(0, 0, W2, H2, fill=1, stroke=0)

y_bcde = margin
y_a    = y_bcde + row_bcde_h + margin

panel_label(c2, margin, y_a + row_a_h - pts(0.8), "A")

c2.save()
buf2.seek(0)

writer2 = PdfWriter()
cr2 = PdfReader(buf2)
writer2.add_page(cr2.pages[0])
embed_pdf_on_page(writer2, SRC+"figure2A_introns.pdf",
                  margin, y_a, W2 - 2*margin, row_a_h)
embed_pdf_on_page(writer2, SRC+"figure2_composite.pdf",
                  margin, y_bcde, W2 - 2*margin, row_bcde_h)

out2 = OUT + "Figure_2_ASSEMBLED.pdf"
with open(out2, "wb") as f:
    writer2.write(f)
print(f"  → {out2}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 3 — already a complete composite, just copy with final name
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Copying Figure 3...")
shutil.copy(SRC+"figure3_composite.pdf", OUT+"Figure_3_ASSEMBLED.pdf")
print(f"  → {OUT}Figure_3_ASSEMBLED.pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUPPLEMENTARY FIGURES S1–S7
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
supp_map = {
    "S1_AP-MS_Dataset_Characterization":    "figureS_APMS_analysis.pdf",
    "S2_AP-MS_Reproducibility_and_Data_Quality": "figureS1_composite.pdf",
    "S3_AP-RNAseq_Deep_Characterization":   "figureS_RNAseq_analysis.pdf",
    "S4_AP-RNAseq_Extended_Analysis":       "figureS2_composite.pdf",
    "S5_Cross-Dataset_Concordance":         "figureS_CrossDataset.pdf",
    "S6_Multi-omics_Clustering":            "figureS3_multiomics_clustering.pdf",
    "S7_Genomic_Chromatin_Context":         "figureS_GenomeContext.pdf",
}

print("Copying Supplementary Figures S1–S7...")
for label, src_name in supp_map.items():
    src = SRC + src_name
    dst = OUT + f"Supplementary_Figure_{label}.pdf"
    shutil.copy(src, dst)
    print(f"  → Supplementary_Figure_{label}.pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUPPLEMENTARY FIGURE S4 — append length scatter as extra page
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Appending length scatter to S4...")
s4_path = OUT + "Supplementary_Figure_S4_AP-RNAseq_Extended_Analysis.pdf"
scatter_path = SRC + "figure2C_length_scatter.pdf"
writer_s4 = PdfWriter()
for pg in PdfReader(s4_path).pages:
    writer_s4.add_page(pg)
for pg in PdfReader(scatter_path).pages:
    writer_s4.add_page(pg)
with open(s4_path, "wb") as f:
    writer_s4.write(f)
print(f"  → S4 now has {len(writer_s4.pages)} pages (extended analysis + length scatter as panel E)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUPPLEMENTARY FIGURE S8 — Spotlight figures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
spotlight_files = [f for f in os.listdir(SRC) if f.startswith("spotlight_") and f.endswith(".pdf")]
if spotlight_files:
    print(f"Assembling Supplementary Figure S8 (spotlights: {spotlight_files})...")
    writer_s8 = PdfWriter()
    for sf in sorted(spotlight_files):
        for pg in PdfReader(SRC + sf).pages:
            writer_s8.add_page(pg)
    s8_path = OUT + "Supplementary_Figure_S8_Spotlight_Analyses.pdf"
    with open(s8_path, "wb") as f:
        writer_s8.write(f)
    print(f"  → S8: {len(writer_s8.pages)} panels from {len(spotlight_files)} spotlight files")

print("\nAll done. Files in Assembled_Figures/:")
for f in sorted(os.listdir(OUT)):
    size_kb = os.path.getsize(OUT+f)//1024
    print(f"  {f}  ({size_kb} KB)")

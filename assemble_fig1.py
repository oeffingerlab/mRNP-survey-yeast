"""Assemble Figure 1 — landscape and portrait — from CDEF + GH source PDFs."""
import subprocess, sys, os, shutil

sys.path.insert(0, "/sessions/lucid-modest-goldberg/mnt/outputs/")
from figure_style import apply_style; apply_style()

import fitz   # PyMuPDF

MSDIR   = "/sessions/lucid-modest-goldberg/mnt/MS results/"
OUTDIR  = "/sessions/lucid-modest-goldberg/mnt/MS results/Assembled_Figures/current_2026-07-02/"
os.makedirs(OUTDIR, exist_ok=True)

def embed_pdf_on_page(page, src_path, x, y, target_w, target_h):
    """Embed src PDF (page 0) at (x,y) scaled to fit target_w×target_h (pts)."""
    src = fitz.open(src_path)
    src_rect = src[0].rect          # source dimensions in pts
    scale = min(target_w / src_rect.width, target_h / src_rect.height)
    placed_w = src_rect.width * scale
    placed_h = src_rect.height * scale
    # centre horizontally within target box
    ox = x + (target_w - placed_w) / 2
    oy = y + (target_h - placed_h) / 2
    dest_rect = fitz.Rect(ox, oy, ox + placed_w, oy + placed_h)
    page.show_pdf_page(dest_rect, src, 0)
    src.close()

# ─────────────────────────────────────────────────────────────────────────────
# LANDSCAPE  (792 × 612 pt = 11" × 8.5")
# ─────────────────────────────────────────────────────────────────────────────
W, H = 792, 612
m  = 0    # no outer margin — source already has its own margins
cdef_h = int(6.25 * 72)     # 450 pt  (source figsize height × 72)
gh_h   = H - cdef_h - m     # ≈ 162 pt  (rest of page)

doc = fitz.open()
page = doc.new_page(width=W, height=H)
embed_pdf_on_page(page, MSDIR + "figure1_panels_CDEF.pdf", 0, 0,      W, cdef_h)
embed_pdf_on_page(page, MSDIR + "figure1_panels_GH.pdf",   0, cdef_h, W, gh_h)
out_ls = OUTDIR + "Figure_1_ASSEMBLED.pdf"
import tempfile
with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
    tmp_ls = tmp.name
doc.save(tmp_ls); doc.close()
shutil.copy2(tmp_ls, out_ls); os.unlink(tmp_ls)
print(f"Landscape saved: {out_ls}")

# ─────────────────────────────────────────────────────────────────────────────
# Generate portrait GH source PDF via exec()
# Note: CDEF for portrait is taken directly from the landscape source so that
#       CDE panels and their scale bars are identical to the landscape figure.
# ─────────────────────────────────────────────────────────────────────────────

_OUTDIR_SAVED = OUTDIR

# ── Portrait GH (8.5" × 2.3") ────────────────────────────────────────────────
with open(MSDIR + "make_figure1_cliques.py") as f:
    src2 = f.read()

src2 = src2.replace("figsize=(11, 2.05)", "figsize=(8.5, 2.3)")
src2 = src2.replace("figure1_panels_GH.pdf", "figure1_panels_GH_portrait.pdf")
exec(src2)

OUTDIR = _OUTDIR_SAVED

# ─────────────────────────────────────────────────────────────────────────────
# PORTRAIT  (612 × 792 pt = 8.5" × 11")
# CDE panels: copied from landscape source (figure1_panels_CDEF.pdf — 792×450 pt)
#   → scaled to 612 pt wide → height = 450 × (612/792) = 347.7 pt ≈ 4.83"
# GH: portrait source (figure1_panels_GH_portrait.pdf — 612×166 pt)
# Both centred at full page width; content placed at top, GH follows immediately.
# ─────────────────────────────────────────────────────────────────────────────
Wp, Hp = 612, 792

# Compute natural scaled heights (both sources fill full width exactly)
_cdef_src = fitz.open(MSDIR + "figure1_panels_CDEF.pdf")
_gh_src   = fitz.open(MSDIR + "figure1_panels_GH_portrait.pdf")
cdef_nat_h = round(_cdef_src[0].rect.height * Wp / _cdef_src[0].rect.width)
gh_nat_h   = round(_gh_src[0].rect.height   * Wp / _gh_src[0].rect.width)
_cdef_src.close(); _gh_src.close()

gap       = 8   # small gap between CDEF and GH blocks (pts)
total_h   = cdef_nat_h + gap + gh_nat_h
y_top     = (Hp - total_h) // 2   # centre vertically on page

doc2 = fitz.open()
page2 = doc2.new_page(width=Wp, height=Hp)
embed_pdf_on_page(page2, MSDIR + "figure1_panels_CDEF.pdf",
                  0, y_top, Wp, cdef_nat_h)
embed_pdf_on_page(page2, MSDIR + "figure1_panels_GH_portrait.pdf",
                  0, y_top + cdef_nat_h + gap, Wp, gh_nat_h)
out_pt = OUTDIR + "Figure_1_ASSEMBLED_portrait.pdf"
with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
    tmp_pt = tmp.name
doc2.save(tmp_pt); doc2.close()
shutil.copy2(tmp_pt, out_pt); os.unlink(tmp_pt)
print(f"Portrait saved: {out_pt}")

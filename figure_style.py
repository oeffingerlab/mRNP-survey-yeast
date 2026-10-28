"""
Shared style constants for all manuscript figures.
Exact colors from original Illustrator files, Helvetica font, PDF/Illustrator-compatible output.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
# Helvetica TTFs installed in ~/.fonts (Helvetica-Regular/Bold/Italic/BoldItalic.ttf)
# Built from Liberation Sans via fonttools — metric-compatible Helvetica substitute.

# ── Global rcParams (call apply_style() at top of each script) ────────────
def apply_style():
    """Apply global figure style: Helvetica, readable labels, Illustrator-ready PDF."""
    plt.rcParams.update({
        # Font — Helvetica TTF (registered above); falls back to Liberation Sans
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Helvetica', 'Liberation Sans', 'Arial', 'DejaVu Sans'],
        'font.weight':       'normal',
        'axes.titleweight':  'normal',
        'axes.labelweight':  'normal',
        'font.size':         11,
        'axes.titlesize':    12,
        'axes.labelsize':    12,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'legend.fontsize':   10,
        # PDF / Illustrator compatibility
        'pdf.fonttype':      42,   # TrueType — editable text in Illustrator
        'ps.fonttype':       42,
        'svg.fonttype':      'none',
        # Clean look
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.linewidth':    0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'figure.dpi':        150,
    })

# ── Pathway group colors — matched to Figures 4-6 color scheme ───────────
# Colors sampled from New Figure 5_v2.ai and New Figure 6_v2.ai (pixel extraction)
GROUP_COLORS = {
    "CBC":               "#C93A3B",   # crimson red
    "Co-Txp/THO/TREX":  "#338FBF",   # steel blue / cyan-teal
    "Splicing":          "#3D477E",   # dark navy blue
    "3'end processing":  "#56C4CB",   # cyan-teal (unchanged)
    "Surveillance":      "#A87247",   # warm ochre-brown
    "TREX-2/NPC":        "#78C5A3",   # teal-green
    "Export/pore":       "#67BA94",   # medium teal-green
    "Cytoplasmic":       "#DC9C3C",   # golden orange
}

CLUSTER_COLORS = {
    1: '#A7948D',   2: '#78B8EF',   3: '#8EC691',   4: '#66CDDA',
    5: '#FEE886',   6: '#FDBA66',   7: '#BB7CCC',   8: '#EF8886',
}

BAIT_GROUP_APMS = {
    'CBP20':"CBC", 'CBP80':"CBC",
    'THP2':"Co-Txp/THO/TREX", 'YRA1':"Co-Txp/THO/TREX",
    'YHS7':"Co-Txp/THO/TREX", 'YRA2':"Co-Txp/THO/TREX", 'SUB2':"Co-Txp/THO/TREX",
    'SNU71':"Splicing", 'SNU114':"Splicing", 'PRP42':"Splicing", 'PRP6':"Splicing",
    'RNA14':"3'end processing", 'CLP1':"3'end processing",
    'YSH1':"3'end processing", 'SYC1':"3'end processing",
    'CFT1':"3'end processing", 'REF2':"3'end processing", 'HRP1':"3'end processing",
    'PAB1':"3'end processing", 'PAP1':"3'end processing",
    'NAB2':"3'end processing", 'NPL3':"3'end processing",
    'GBP2':"3'end processing", 'HRB1':"3'end processing", 'GFD1':"3'end processing",
    'RRP6':"Surveillance",
    'PML39':"TREX-2/NPC", 'SAC3':"TREX-2/NPC", 'THP1':"TREX-2/NPC",
    'MEX67':"Export/pore", 'DBP5':"Export/pore",
    'MLP1':"Export/pore", 'MLP2':"Export/pore",
    'SHE2':"Cytoplasmic",
}
BAIT_GROUP_RNA = {
    'CBP80':"CBC",
    'THP2':"Co-Txp/THO/TREX", 'YRA1':"Co-Txp/THO/TREX",
    'YHS7':"Co-Txp/THO/TREX", 'SUB2':"Co-Txp/THO/TREX",
    'SNU71':"Splicing",
    'CLP1':"3'end processing", 'SYC1':"3'end processing",
    'REF2':"3'end processing", 'HRP1':"3'end processing",
    'PAB1':"3'end processing", 'NAB2':"3'end processing",
    'NPL3':"3'end processing", 'GBP2':"3'end processing", 'GFD1':"3'end processing",
    'RRP6':"Surveillance",
    'SAC3':"TREX-2/NPC",
    'MEX67':"Export/pore", 'DBP5':"Export/pore",
}

# ── Colormaps (soft versions) ─────────────────────────────────────────────
# AP-MS heatmap: white → blush → deep rose
CMAP_APMS = LinearSegmentedColormap.from_list(
    'red_apms_soft', ['#FFFFFF', '#FDDDE6', '#F48CAF', '#C2185B'], N=256)

# Pearson / correlation: cream → warm sand → rust
CMAP_CORR = LinearSegmentedColormap.from_list(
    'warm_corr_soft', ['#FFFDE7', '#FFE082', '#E65100', '#3E2723'], N=256)

# Shortest path: near-white → periwinkle → indigo (dark=close)
CMAP_SP = LinearSegmentedColormap.from_list(
    'sp_dist_soft', ['#3949AB', '#7E9BD6', '#C5D3EE', '#F5F5F5'], N=256)

# RNA-seq: white → powder blue → steel blue
CMAP_RNA = LinearSegmentedColormap.from_list(
    'blue_rna_soft', ['#FFFFFF', '#DBEAFE', '#3B82F6', '#1E3A5F'], N=256)

# ── Pathway orders ────────────────────────────────────────────────────────
PATHWAY_ORDER_APMS = [
    'CBP20','CBP80',
    'THP2','YRA1','YHS7','YRA2','SUB2',
    'SNU71','SNU114','PRP42','PRP6',
    'RNA14','CLP1','YSH1','SYC1','CFT1','REF2','HRP1',
    'PAB1','PAP1','NAB2','NPL3','GBP2','HRB1','GFD1',
    'RRP6',
    'PML39','SAC3','THP1',
    'MEX67','DBP5','MLP1','MLP2','SHE2',
]
PATHWAY_ORDER_RNA = [
    'CBP80','THP2','YRA1','SNU71',
    'CLP1','SYC1','REF2','HRP1',
    'PAB1','NAB2','NPL3','GBP2','GFD1',
    'RRP6','SUB2','SAC3','MEX67','DBP5',
]
CLUSTER_ORDER = [1, 2, 3, 4, 5, 6, 7, 8]
CLUSTER_LABELS = {1: "C1\nCBC/splicing/Npl3p", 2: "C2\nhnRNP/Poly(A)", 3: "C3\nTHO/TREX", 4: "C4\nHrp1p", 5: "C5\n3'end proc.", 6: "C6\nTREX-2", 7: "C7\nGfd1p/CBC", 8: "C8\nExport"}

# ── Helper: colored bait label bars ──────────────────────────────────────
def bait_color_bar_h(ax, baits, bait_group_map):
    colors = [GROUP_COLORS.get(bait_group_map.get(b,''), '#CCCCCC') for b in baits]
    n = len(baits)
    for i, c in enumerate(colors):
        ax.add_patch(mpatches.Rectangle((i/n, 0), 1/n, 1,
            color=c, transform=ax.transAxes, clip_on=False))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

def bait_color_bar_v(ax, baits, bait_group_map):
    colors = [GROUP_COLORS.get(bait_group_map.get(b,''), '#CCCCCC') for b in baits]
    n = len(baits)
    for i, c in enumerate(colors):
        ax.add_patch(mpatches.Rectangle((0, i/n), 1, 1/n,
            color=c, transform=ax.transAxes, clip_on=False))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

def fmt_bait(b):
    """Convert all-caps gene name to yeast protein nomenclature: CBP20 → Cbp20p."""
    return b[0].upper() + b[1:].lower() + 'p'

def fmt_baits(blist):
    """Apply fmt_bait to a list of bait names."""
    return [fmt_bait(b) for b in blist]

def group_legend_patches(bait_group_map):
    seen = []
    patches = []
    for grp, col in GROUP_COLORS.items():
        if grp in bait_group_map.values() and grp not in seen:
            patches.append(mpatches.Patch(color=col, label=grp))
            seen.append(grp)
    return patches

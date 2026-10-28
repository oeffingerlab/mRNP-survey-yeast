"""
Figure 2A — AP-RNA-seq HCL heatmap with intron-containing genes highlighted.

Strip uses THREE colors:
  - Light gray      : no intron
  - Dusty blue      : non-RP intron-containing genes
  - Soft terracotta : RP (ribosomal protein) genes

Layout (right of heatmap, left → right):
  heatmap | pathway color path bar | gap | enrichment scale bar
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_RNA,
                           fmt_bait, fmt_baits,
                           CLUSTER_COLORS, CMAP_RNA,
                           CLUSTER_LABELS, CLUSTER_ORDER)
apply_style()

OUTDIR  = "output/"
DATADIR = "data/"
GFF     = "data/saccharomyces_cerevisiae.20260618.gff"

COLOR_NONE  = '#E8E8E8'
COLOR_NONRP = '#8BB4D0'
COLOR_RP    = '#E8A87C'

PATHWAY_ORDER_RNA = [
    'CBP80','THP2','YRA1','SNU71',
    'CLP1','SYC1','REF2','HRP1',
    'PAB1','NAB2','NPL3','GBP2','GFD1',
    'RRP6','SUB2','SAC3','MEX67','DBP5',
]

# ── Extract intron/RP genes from GFF ────────────────────────────────────────
INTRON_RE = re.compile(r'Name=([^;]+)_intron')
ORF_RE    = re.compile(r'^Y[A-P][LR]\d{3}[CW](-[A-Z])?$')

intron_genes = set()
with open(GFF) as fh:
    for line in fh:
        if '\tintron\t' not in line: continue
        m = INTRON_RE.search(line)
        if m and ORF_RE.match(m.group(1)):
            intron_genes.add(m.group(1))

rp_genes = set()
with open(GFF) as fh:
    for line in fh:
        if line.startswith('#'): continue
        parts = line.strip().split('\t')
        if len(parts) < 9 or parts[2] != 'gene': continue
        id_m   = re.search(r'ID=([^;]+)', parts[8])
        gene_m = re.search(r'gene=([^;]+)', parts[8])
        if id_m and gene_m:
            orf = id_m.group(1)
            if re.match(r'^Y[A-P][LR]\d{3}[CW](-[A-Z])?$', orf) and re.match(r'^RP[LS]\d', gene_m.group(1), re.IGNORECASE):
                rp_genes.add(orf)

print(f"Intron ORFs: {len(intron_genes)},  RP genes: {len(rp_genes)}")

# ── Load data ────────────────────────────────────────────────────────────────
df_rna   = pd.read_csv(DATADIR + "rnaseq_bait_means.tsv",  sep='\t', index_col=0)
df_clust = pd.read_csv(DATADIR + "rna_gene_clusters.tsv",  sep='\t', index_col=0)

rna_baits   = [b for b in PATHWAY_ORDER_RNA if b in df_rna.columns]
df_rna      = df_rna[rna_baits]
df_rna_filt = df_rna[df_rna.max(axis=1) > 0.3]

dist_g = pdist(df_rna_filt.values, metric='correlation')
dist_g = np.nan_to_num(dist_g, nan=1.0)
link_g = linkage(dist_g, method='average', optimal_ordering=True)
gene_ord = leaves_list(link_g)
df_hcl   = df_rna_filt.iloc[gene_ord]

def classify(g):
    if g in rp_genes and g in intron_genes: return 2   # intron-containing RP
    if g in intron_genes: return 1
    return 0

strip_vals       = np.array([classify(g) for g in df_hcl.index])
n_rp_in_data    = (strip_vals == 2).sum()
n_nonrp_in_data = (strip_vals == 1).sum()
print(f"RP={n_rp_in_data}, non-RP intron={n_nonrp_in_data}")

# ── Color bar helper (vertical, top=first bait) ──────────────────────────────
def bait_color_bar_v(ax, baits):
    colors = [GROUP_COLORS.get(BAIT_GROUP_RNA.get(b, ''), '#CCCCCC') for b in baits]
    n = len(baits)
    for i, c in enumerate(colors):
        ax.add_patch(mpatches.Rectangle((0, (n-1-i)/n), 1, 1/n,
            color=c, transform=ax.transAxes, clip_on=False))

# ════════════════════════════════════════════════════════════════════════════
# FIGURE
# Layout: heatmap axes spans left=0.12 → right=0.78
#   right of axes: pathway color path bar, then gap, then scale bar
# ════════════════════════════════════════════════════════════════════════════
STRIP_HEIGHT = 0.04

fig = plt.figure(figsize=(22, 6))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    1, 1, figure=fig,
    left=0.12, right=0.78, top=0.88, bottom=0.16,
)

ax_a = fig.add_subplot(gs[0, 0])

im_a = ax_a.imshow(df_hcl.values.T, aspect='auto', cmap=CMAP_RNA, vmin=0, vmax=1.0)

ax_a.set_title(
    'AP-RNA-seq Hierarchical Clustering  (genes × baits, max-FPKM normalized)',
    fontsize=16, fontweight='bold')
ax_a.set_ylabel('Baits (pathway order, early → late)', fontsize=15)
ax_a.set_xlabel(
    f'{len(df_hcl)} co-enriched genes  |  Pearson / average linkage  '
    f'|  {n_rp_in_data} intron-containing RP genes  +  {n_nonrp_in_data} non-RP intron genes highlighted',
    fontsize=11)
ax_a.set_yticks(range(len(rna_baits)))
ax_a.set_yticklabels(fmt_baits(rna_baits), fontsize=16, color='black')
ax_a.set_xticks([])

# ── Pathway color path bar: immediately right of heatmap ─────────────────────
# inset_axes coords are relative to the parent axes (0,0)=bottom-left, (1,1)=top-right
# x > 1 = right of axes
ax_strip_right = ax_a.inset_axes([1.008, 0, 0.015, 1])
ax_strip_right.axis('off')
bait_color_bar_v(ax_strip_right, rna_baits)

# ── Enrichment scale bar: right of pathway strip, with gap ──────────────────
# strip occupies 1.018 → 1.046; gap; colorbar at 1.058 →
cax = ax_a.inset_axes([1.030, 0.30, 0.022, 0.50])
cbar = plt.colorbar(im_a, cax=cax)
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Enrichment\n(max=1)', fontsize=9, labelpad=4)

# ── Intron gene strip (below heatmap) ────────────────────────────────────────
ax_strip_bot = ax_a.inset_axes([0, -STRIP_HEIGHT - 0.01, 1, STRIP_HEIGHT])
ax_strip_bot.axis('off')
strip_cmap = ListedColormap([COLOR_NONE, COLOR_NONRP, COLOR_RP])
ax_strip_bot.imshow(strip_vals.reshape(1, -1).astype(float),
                    aspect='auto', cmap=strip_cmap, vmin=0, vmax=2,
                    interpolation='nearest')
ax_strip_bot.text(-0.005, 0.5, 'Intron\ngenes', ha='right', va='center',
                  fontsize=7, transform=ax_strip_bot.transAxes)

# ── Intron strip legend (below strip) ────────────────────────────────────────
p_rp    = mpatches.Patch(color=COLOR_RP,    label=f'Intron-containing RP genes (n={n_rp_in_data})')
p_nonrp = mpatches.Patch(color=COLOR_NONRP, label=f'Non-RP intron genes (n={n_nonrp_in_data})')
p_none  = mpatches.Patch(color=COLOR_NONE,  label='No intron / not detected')
ax_a.legend(handles=[p_rp, p_nonrp, p_none],
            loc='upper left',
            bbox_to_anchor=(0.0, -0.09 - STRIP_HEIGHT),
            fontsize=12, frameon=True, ncol=3)

# Panel label
ax_a.text(-0.07, 1.03, 'A', transform=ax_a.transAxes, fontsize=32, fontweight='bold')

out = OUTDIR + "figure2A_introns.pdf"
fig.savefig(out, facecolor='white')   # no tight crop — preserves full 22" width
plt.close()
print(f"Saved: {out}")

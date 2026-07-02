"""
Figure 2A duplicate — with intron-containing genes highlighted.

A thin strip below the heatmap marks (in orange) every gene that contains
at least one spliceosomal intron, using systematic IDs extracted from
saccharomyces_cerevisiae.20260618.gff (uploaded by user).

Cluster distribution of intron genes is printed to stdout and annotated
as a small in-figure table.
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
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, "/sessions/lucid-modest-goldberg/mnt/outputs/")
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_RNA,
                           fmt_bait, fmt_baits,
                           CLUSTER_COLORS, CMAP_RNA,
                           CLUSTER_LABELS, CLUSTER_ORDER)
apply_style()

OUTDIR  = "/sessions/lucid-modest-goldberg/mnt/MS results/"
DATADIR = "/sessions/lucid-modest-goldberg/mnt/outputs/"
GFF     = "/sessions/lucid-modest-goldberg/mnt/uploads/saccharomyces_cerevisiae.20260618.gff"

# ── Pathway / bait order ────────────────────────────────────────────────────
PATHWAY_ORDER_RNA = [
    'CBP80','THP2','YRA1','SNU71',
    'CLP1','SYC1','REF2','HRP1',
    'PAB1','NAB2','NPL3','GBP2','GFD1',
    'RRP6','SUB2','SAC3','MEX67','DBP5',
]

# ── Extract intron-containing ORFs from GFF ─────────────────────────────────
INTRON_RE = re.compile(r'Name=([^;]+)_intron')
ORF_RE    = re.compile(r'^Y[A-P][LR]\d{3}[CW]$')

intron_genes = set()
with open(GFF) as fh:
    for line in fh:
        if '\tintron\t' not in line:
            continue
        m = INTRON_RE.search(line)
        if m and ORF_RE.match(m.group(1)):
            intron_genes.add(m.group(1))

print(f"Intron-containing ORFs in GFF: {len(intron_genes)}")

# ── Load RNA-seq data ───────────────────────────────────────────────────────
df_rna   = pd.read_csv(DATADIR + "rnaseq_bait_means.tsv",  sep='\t', index_col=0)
df_clust = pd.read_csv(DATADIR + "rna_gene_clusters.tsv",  sep='\t', index_col=0)

rna_baits   = [b for b in PATHWAY_ORDER_RNA if b in df_rna.columns]
df_rna      = df_rna[rna_baits]
df_rna_filt = df_rna[df_rna.max(axis=1) > 0.3]

# HCL gene ordering (same as Figure 2A)
dist_g = pdist(df_rna_filt.values, metric='correlation')
dist_g = np.nan_to_num(dist_g, nan=1.0)
link_g = linkage(dist_g, method='average')
gene_ord = leaves_list(link_g)
df_hcl   = df_rna_filt.iloc[gene_ord]

# ── Intron overlay: binary array in HCL order ───────────────────────────────
is_intron = np.array([1 if g in intron_genes else 0 for g in df_hcl.index])
n_intron_in_data = is_intron.sum()
print(f"Intron genes in filtered RNA-seq data: {n_intron_in_data} / {len(df_hcl)}")

# ── Cluster distribution of intron genes ───────────────────────────────────
df_ic = df_clust[['cluster']].copy()
df_ic['is_intron'] = df_ic.index.isin(intron_genes).astype(int)
clust_counts = df_ic[df_ic['is_intron'] == 1].groupby('cluster').size()
# as fraction of each cluster
clust_total  = df_ic.groupby('cluster').size()
clust_frac   = (clust_counts / clust_total * 100).reindex(CLUSTER_ORDER).fillna(0)
print("\nIntron gene count and % per cluster (HCL k=8 order):")
for k in CLUSTER_ORDER:
    cnt  = int(clust_counts.get(k, 0))
    tot  = int(clust_total.get(k, 0))
    frac = clust_frac[k]
    print(f"  {CLUSTER_LABELS[k].split(chr(10))[0]:20s}  {cnt:3d} / {tot:4d}  ({frac:.1f}%)")

# ── Color bar helper ────────────────────────────────────────────────────────
def bait_color_bar_v(ax, baits):
    colors = [GROUP_COLORS.get(BAIT_GROUP_RNA.get(b,''), '#CCCCCC') for b in baits]
    n = len(baits)
    for i, c in enumerate(colors):
        ax.add_patch(mpatches.Rectangle((0,(n-1-i)/n), 1, 1/n,
            color=c, transform=ax.transAxes, clip_on=False))

# ════════════════════════════════════════════════════════════════════════════
# FIGURE  (Panel A + intron strip + cluster bar)
# ════════════════════════════════════════════════════════════════════════════
INTRON_COLOR = '#E07B39'   # warm orange — distinct from heatmap blues
STRIP_HEIGHT = 0.04        # fraction of heatmap axes height

fig = plt.figure(figsize=(22, 12))
fig.patch.set_facecolor('white')

# Single-column GridSpec — colorbar handled via shrink, cluster legend via bbox_to_anchor
gs = gridspec.GridSpec(
    2, 1,
    figure=fig,
    height_ratios=[10, 2.2],   # heatmap row, cluster-bar row
    hspace=0.42,               # generous gap so bar-title lands in white space
    left=0.07, right=0.80, top=0.92, bottom=0.08,
)

# ── Heatmap ─────────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

im_a = ax_a.imshow(df_hcl.values.T, aspect='auto', cmap=CMAP_RNA,
                   vmin=0, vmax=1.0)

ax_a.set_title(
    'AP-RNA-seq Hierarchical Clustering  (genes × baits, max-FPKM normalized)\n'
    'Orange strip = intron-containing genes',
    fontsize=10, fontweight='bold')
ax_a.set_ylabel('Baits (pathway order → early to late)', fontsize=8)
ax_a.set_xlabel(
    f'{len(df_hcl)} co-enriched genes  |  clustered by Pearson / average linkage  '
    f'|  {n_intron_in_data} intron-containing genes highlighted',
    fontsize=8)
ax_a.set_yticks(range(len(rna_baits)))
ax_a.set_yticklabels(fmt_baits(rna_baits), fontsize=11, color='black')
ax_a.set_xticks([])

# Pathway-stage color bar on right of heatmap
ax_a_right = ax_a.inset_axes([1.005, 0, 0.012, 1])
ax_a_right.axis('off')
bait_color_bar_v(ax_a_right, rna_baits)

# Small colorbar as inset_axes — does NOT steal width from ax_a,
# so ax_a and ax_bar stay the same width
cax = ax_a.inset_axes([1.055, 0.35, 0.020, 0.28])   # [x, y, w, h] in axes fraction — right of stage bar, vertically centred
cbar = plt.colorbar(im_a, cax=cax, label='Enrichment (max=1)')
cbar.ax.tick_params(labelsize=7)
cbar.set_label('Enrichment\n(max=1)', fontsize=7)

# ── Intron strip (inset below the heatmap) ──────────────────────────────────
# Place a thin axes flush under the heatmap axes (in data-independent fraction coords)
ax_strip = ax_a.inset_axes([0, -STRIP_HEIGHT - 0.005, 1, STRIP_HEIGHT])
ax_strip.axis('off')

n_genes = len(df_hcl)
# Draw the strip as a 1-row image: intron = orange, other = light grey
strip_arr = np.array([[INTRON_COLOR if is_intron[i] else '#E8E8E8'
                        for i in range(n_genes)]])

# imshow won't accept a string color array — use a float array + colormap
strip_vals = is_intron.reshape(1, -1).astype(float)
from matplotlib.colors import ListedColormap
strip_cmap = ListedColormap(['#E8E8E8', INTRON_COLOR])
ax_strip.imshow(strip_vals, aspect='auto', cmap=strip_cmap,
                vmin=0, vmax=1, interpolation='nearest')

# Label
ax_strip.text(-0.005, 0.5, 'Intron\ngenes', ha='right', va='center',
              fontsize=7, transform=ax_strip.transAxes)

# ── Cluster distribution bar (bottom row) ───────────────────────────────────
ax_bar = fig.add_subplot(gs[1])

# Draw one stacked-proportion bar per cluster, colours from CLUSTER_COLORS
# Bar shows: intron fraction (coloured) vs rest (grey)
bar_x = []
bar_intron_frac = []
bar_labels = []
bar_colors_list = []
n_per_clust = []

for k in CLUSTER_ORDER:
    cnt  = int(clust_counts.get(k, 0))
    tot  = int(clust_total.get(k, 0))
    frac = clust_frac[k]
    bar_x.append(k)
    bar_intron_frac.append(frac)
    bar_labels.append(f"{CLUSTER_LABELS[k].split(chr(10))[0]}\n({cnt}/{tot})")
    bar_colors_list.append(CLUSTER_COLORS[k])
    n_per_clust.append(cnt)

x_pos = np.arange(len(CLUSTER_ORDER))
bars = ax_bar.bar(x_pos, bar_intron_frac,
                  color=INTRON_COLOR, edgecolor='grey', linewidth=0.5,
                  width=0.7)

# Cluster identity tick labels coloured by cluster
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(bar_labels, fontsize=7.5, ha='center')
for tick, k in zip(ax_bar.get_xticklabels(), CLUSTER_ORDER):
    tick.set_color(CLUSTER_COLORS[k])
    tick.set_fontweight('bold')

# Annotate % above each bar
for xi, frac, cnt in zip(x_pos, bar_intron_frac, n_per_clust):
    if cnt > 0:
        ax_bar.text(xi, frac + 0.4, f'{frac:.1f}%',
                    ha='center', va='bottom', fontsize=7.5, color='#444444')

ax_bar.set_ylabel('Intron genes (%)', fontsize=8)
ax_bar.set_ylim(0, max(bar_intron_frac) * 1.35 + 1)
ax_bar.set_xlim(-0.5, len(CLUSTER_ORDER) - 0.5)
# Title placed via set_title with extra pad — hspace=0.42 puts it in the white gap
ax_bar.set_title('Intron-containing gene distribution across RNA-seq clusters (k=8)',
                 fontsize=9, pad=10)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

# ── Cluster legend to the RIGHT of the bar chart (mirrors Figure 2B style) ──
leg_handles = [
    mpatches.Patch(color=CLUSTER_COLORS[k],
                   label=CLUSTER_LABELS[k].replace('\n', ' '))
    for k in CLUSTER_ORDER
]
ax_bar.legend(handles=leg_handles,
              fontsize=7, loc='center left', bbox_to_anchor=(1.02, 0.5),
              framealpha=0.9, title='RNA cluster', title_fontsize=7.5,
              handlelength=1.2, handleheight=1.0)

# ── Intron strip legend (below heatmap xlabel) ────────────────────────────────
intron_patch = mpatches.Patch(color=INTRON_COLOR, label=f'Intron-containing gene  (n={n_intron_in_data})')
other_patch  = mpatches.Patch(color='#E8E8E8',    label='No intron / not detected')
ax_a.legend(handles=[intron_patch, other_patch],
            loc='upper left', bbox_to_anchor=(0.0, -0.07 - STRIP_HEIGHT),
            fontsize=8, frameon=True, ncol=2)

# Panel label
ax_a.text(-0.025, 1.03, 'A', transform=ax_a.transAxes, fontsize=14, fontweight='bold')

fig.suptitle(
    'Figure 2A  |  AP-RNA-seq Heatmap with Intron-Containing Gene Highlight',
    fontsize=11, fontweight='bold', y=0.97)

out = OUTDIR + "figure2A_introns.pdf"
fig.savefig(out, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {out}")

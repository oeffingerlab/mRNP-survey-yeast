"""
Figure 1 panel remake – AP-MS focused
Panels: A/B (placeholders), C (AP-MS heatmap), D (Pearson), E (Shortest path), F (RNA-seq HCL)
Layout: Row 0: A (2/3 width) | B (1/3 width)
        Row 1: C | D | E  (equally sized)
        Row 2: F (full width)
Fixes: bait names in protein convention (Cbp80p etc.), group color strips on RIGHT,
       correct top-to-bottom strip direction, vertical legends next to E and F.
"""
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
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS as BAIT_GROUP,
                           fmt_bait, fmt_baits,
                           CMAP_APMS, CMAP_CORR, CMAP_RNA)
apply_style()

OUTDIR  = "/sessions/lucid-modest-goldberg/mnt/MS results/"
DATADIR = "/sessions/lucid-modest-goldberg/mnt/outputs/"

PATHWAY_ORDER_APMS = [
    'CBP20','CBP80',
    'THP2','YRA1','YHS7','YRA2',
    'SNU71','SNU114','PRP42','PRP6',
    'RNA14','CLP1','YSH1','SYC1','CFT1','REF2','HRP1',
    'PAB1','PAP1','NAB2','NPL3','GBP2','HRB1','GFD1',
    'RRP6','SUB2','PML39',
    'SAC3','THP1',
    'MEX67','DBP5','MLP1','MLP2','SHE2',
]
PATHWAY_ORDER_RNA = [
    'CBP80','THP2','YRA1','SNU71',
    'CLP1','SYC1','REF2','HRP1',
    'PAB1','NAB2','NPL3','GBP2','GFD1',
    'RRP6','SUB2','SAC3','MEX67','DBP5',
]

# ── Load AP-MS matrix ──────────────────────────────────────────────────────
df_apms = pd.read_csv(DATADIR + "apms_nuclear_matrix.tsv", sep='\t', index_col=0)
df_apms = df_apms.drop(columns=['MTR2'], errors='ignore')
apms_baits = [b for b in PATHWAY_ORDER_APMS if b in df_apms.columns]
df_apms = df_apms[apms_baits]
prey_order = (df_apms > 0).sum(axis=1).sort_values(ascending=False).index
df_apms = df_apms.loc[prey_order]
print(f"AP-MS matrix: {df_apms.shape}")

# ── Compute Pearson correlation from AP-MS matrix ─────────────────────────
# Recomputed directly from apms_nuclear_matrix.tsv (all baits, excl. MTR2)
# Includes YHS7 and all baits previously missing from Alex's Feb-2015 file
df_apms_full = pd.read_csv(DATADIR + "apms_nuclear_matrix.tsv", sep='\t', index_col=0)
df_apms_full = df_apms_full.drop(columns=['MTR2'], errors='ignore')
df_p = df_apms_full.corr(method='pearson')
pearson_ordered = [b for b in PATHWAY_ORDER_APMS if b in df_p.index and b in df_p.columns]
df_pearson = df_p.loc[pearson_ordered, pearson_ordered]
print(f"Pearson matrix: {df_pearson.shape}")

# ── Shared-prey count matrix ───────────────────────────────────────────────
# For each bait pair (i,j): number of nuclear preys detected by both
binary = (df_apms.values > 0).astype(np.int16)  # int cast required — bool @ bool returns bool in numpy
shared_prey_matrix = (binary.T @ binary).astype(float)  # shape: (n_baits, n_baits)
shared_prey_plot = shared_prey_matrix.copy()
np.fill_diagonal(shared_prey_plot, np.nan)              # mask self-comparisons
print(f"Shared prey matrix: max off-diag={np.nanmax(shared_prey_plot):.0f}")

# ── RNA-seq HCL ───────────────────────────────────────────────────────────
df_rna = pd.read_csv(DATADIR + "rnaseq_bait_means.tsv", sep='\t', index_col=0)
rna_baits = [b for b in PATHWAY_ORDER_RNA if b in df_rna.columns]
df_rna = df_rna[rna_baits]
df_rna = df_rna[df_rna.max(axis=1) > 0.3]
dist = np.nan_to_num(pdist(df_rna.values, metric='correlation'), nan=1.0)
gene_order = leaves_list(linkage(dist, method='average'))
df_rna_sorted = df_rna.iloc[gene_order]
print(f"RNA-seq HCL: {df_rna_sorted.shape}")

# ── Color strip helpers ────────────────────────────────────────────────────
STRIP_W = 0.032   # consistent width for all group color strips

def strip_right(ax, baits, w=STRIP_W, gap=0.01):
    """Vertical group-color strip on the RIGHT of ax, top→bottom = baits[0]→baits[-1]."""
    colors = [GROUP_COLORS.get(BAIT_GROUP.get(b, ''), '#CCCCCC') for b in baits]
    n = len(colors)
    s = ax.inset_axes([1.0 + gap, 0, w, 1])
    s.set_xlim(0, 1); s.set_ylim(n, 0)   # ylim inverted: top=0, bottom=n
    s.axis('off')
    for i, c in enumerate(colors):
        s.add_patch(mpatches.Rectangle((0, i), 1, 1, color=c))
    return s

def strip_top(ax, baits, h=0.025, gap=0.01):
    """Horizontal group-color strip on TOP of ax, left→right = baits[0]→baits[-1]."""
    colors = [GROUP_COLORS.get(BAIT_GROUP.get(b, ''), '#CCCCCC') for b in baits]
    n = len(colors)
    s = ax.inset_axes([0, 1.0 + gap, 1, h])
    s.set_xlim(0, n); s.set_ylim(0, 1)
    s.axis('off')
    for i, c in enumerate(colors):
        s.add_patch(mpatches.Rectangle((i, 0), 1, 1, color=c))
    return s

def legend_patches(baits):
    """Functional-group legend patches for the groups represented in baits."""
    seen = set(BAIT_GROUP.get(b,'') for b in baits)
    return [mpatches.Patch(color=c, label=g)
            for g, c in GROUP_COLORS.items() if g in seen]

# ════════════════════════════════════════════════════════════════════════════
# BUILD FIGURE
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(11, 6.25))
fig.patch.set_facecolor('white')

# right=0.86 reserves 14% right margin for legends + colorbars outside panels
gs = gridspec.GridSpec(3, 3, figure=fig,
                        height_ratios=[0.7, 2.76, 1.5],
                        hspace=0.45, wspace=0.30,
                        left=0.08, right=0.86, top=0.94, bottom=0.05)

# ── Panel A ────────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0:2])
ax_a.set_facecolor('#F8F8F8')
for sp in ax_a.spines.values(): sp.set_edgecolor('#CCCCCC'); sp.set_linewidth(1)
ax_a.text(0.5, 0.55, 'Pathway model', ha='center', va='center',
          fontsize=11, color='#AAAAAA', style='italic')
ax_a.text(0.5, 0.38, '(to be placed in Illustrator)', ha='center', va='center',
          fontsize=8, color='#BBBBBB')
ax_a.set_xticks([]); ax_a.set_yticks([])
ax_a.text(-0.06, 1.10, 'A', transform=ax_a.transAxes, fontsize=16, fontweight='bold')

# ── Panel B ────────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 2])
ax_b.set_facecolor('#F8F8F8')
for sp in ax_b.spines.values(): sp.set_edgecolor('#CCCCCC'); sp.set_linewidth(1)
ax_b.text(0.5, 0.55, 'Panel B', ha='center', va='center',
          fontsize=11, color='#AAAAAA', style='italic')
ax_b.text(0.5, 0.38, '(to be placed in Illustrator)', ha='center', va='center',
          fontsize=8, color='#BBBBBB')
ax_b.set_xticks([]); ax_b.set_yticks([])
ax_b.text(-0.06, 1.10, 'B', transform=ax_b.transAxes, fontsize=16, fontweight='bold')

# ── Panel C: AP-MS heatmap ─────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
log_apms = np.log10(df_apms.values + 1)
im_c = ax_c.imshow(log_apms.T, aspect='auto', cmap=CMAP_APMS,
                    vmin=0, vmax=np.percentile(log_apms[log_apms>0], 95))
ax_c.set_title('AP-MS Spectral Counts\n(bait × prey)', fontsize=9, fontweight='bold', pad=5)
ax_c.set_xlabel('Prey proteins', fontsize=4)
ax_c.set_ylabel('Baits', fontsize=4)
ax_c.set_yticks(range(len(apms_baits)))
ax_c.set_yticklabels(fmt_baits(apms_baits), fontsize=4)
ax_c.set_xticks([])
ax_c.set_box_aspect(1)
strip_right(ax_c, apms_baits)
cb_c = plt.colorbar(im_c, ax=ax_c, fraction=0.072, shrink=0.70, pad=0.10)
cb_c.set_label('log₁₀(SpC+1)', fontsize=5.5)
cb_c.ax.tick_params(labelsize=5)
ax_c.text(-0.22, 1.05, 'C', transform=ax_c.transAxes, fontsize=16, fontweight='bold')

# ── Panel D: Pearson bait-bait correlation ──────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
pv = df_pearson.values.copy()
pv[np.eye(len(df_pearson), dtype=bool)] = np.nan
im_d = ax_d.imshow(pv, aspect='equal', cmap=CMAP_CORR, vmin=-0.2, vmax=1.0)
ax_d.set_title('Pearson Correlation\n(bait × bait)', fontsize=9, fontweight='bold', pad=5)
ax_d.set_xticks(range(len(pearson_ordered)))
ax_d.set_xticklabels(fmt_baits(pearson_ordered), fontsize=4, rotation=90, ha='right')
ax_d.set_yticks(range(len(pearson_ordered)))
ax_d.set_yticklabels(fmt_baits(pearson_ordered), fontsize=4)
strip_top(ax_d, pearson_ordered, h=0.032)
strip_right(ax_d, pearson_ordered)
cb_d = plt.colorbar(im_d, ax=ax_d, fraction=0.072, shrink=0.70, pad=0.10)
cb_d.set_label('Pearson r', fontsize=5.5)
cb_d.ax.tick_params(labelsize=5)
ax_d.text(-0.22, 1.10, 'D', transform=ax_d.transAxes, fontsize=16, fontweight='bold')

# ── Panel E: Shared nuclear prey count (bait × bait) ──────────────────────
ax_e = fig.add_subplot(gs[1, 2])
vmax_e = int(np.nanmax(shared_prey_plot))
im_e = ax_e.imshow(shared_prey_plot, aspect='equal',
                    cmap='Blues', vmin=0, vmax=vmax_e)
ax_e.set_title('Shared Nuclear Preys\n(bait × bait)', fontsize=9, fontweight='bold', pad=5)
ax_e.set_xticks(range(len(apms_baits)))
ax_e.set_xticklabels(fmt_baits(apms_baits), fontsize=4, rotation=90, ha='right')
ax_e.set_yticks(range(len(apms_baits)))
ax_e.set_yticklabels(fmt_baits(apms_baits), fontsize=4)
strip_top(ax_e, apms_baits, h=0.032)
strip_right(ax_e, apms_baits)
cb_e = plt.colorbar(im_e, ax=ax_e, fraction=0.072, shrink=0.70, pad=0.06)
cb_e.set_label('# shared nuclear preys', fontsize=5.5)
cb_e.ax.tick_params(labelsize=5)
ax_e.text(-0.22, 1.10, 'E', transform=ax_e.transAxes, fontsize=16, fontweight='bold')

# Legend next to E (vertical, just right of panel E / right margin)
ax_e.legend(handles=legend_patches(apms_baits),
            loc='upper left', bbox_to_anchor=(1.42, 1.02),
            ncol=1, fontsize=5.0, frameon=True,
            title='Functional group', title_fontsize=5.5,
            borderaxespad=0, labelspacing=0.25,
            markerscale=0.7, handlelength=0.8, handletextpad=0.3)

# ── Panel F: RNA-seq HCL ───────────────────────────────────────────────────
ax_f = fig.add_subplot(gs[2, :])
im_f = ax_f.imshow(df_rna_sorted.values.T, aspect='auto',
                    cmap=CMAP_RNA, vmin=0, vmax=1.0)
ax_f.set_title('AP-RNA-seq Hierarchical Clustering  (genes × baits, max-FPKM normalized)',
               fontsize=9, fontweight='bold', pad=5)
ax_f.set_xlabel(f'Genes ({len(df_rna_sorted)} co-enriched, clustered Pearson/avg-linkage)',
                fontsize=5)
ax_f.set_ylabel('Baits (pathway order)', fontsize=5)
ax_f.set_yticks(range(len(rna_baits)))
ax_f.set_yticklabels(fmt_baits(rna_baits), fontsize=4)
ax_f.set_xticks([])
strip_right(ax_f, rna_baits, w=STRIP_W/2)
cb_f = plt.colorbar(im_f, ax=ax_f, fraction=0.020, shrink=0.7, pad=0.08, orientation='vertical')
cb_f.set_label('Enrichment (max=1)', fontsize=5.5)
cb_f.ax.tick_params(labelsize=5)
ax_f.text(-0.07, 1.05, 'F', transform=ax_f.transAxes, fontsize=16, fontweight='bold')

# Legend next to F (vertical, just right of panel F / right margin)
ax_f.legend(handles=legend_patches(rna_baits),
            loc='upper left', bbox_to_anchor=(1.16, 1.02),
            ncol=1, fontsize=5.0, frameon=True,
            title='Functional group', title_fontsize=5.5,
            borderaxespad=0, labelspacing=0.25,
            markerscale=0.7, handlelength=0.8, handletextpad=0.3)

outpath = OUTDIR + "figure1_panels_CDEF.pdf"
fig.savefig(outpath, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {outpath}")

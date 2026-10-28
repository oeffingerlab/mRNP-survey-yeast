"""
Supplemental Figure 5, panels A, B and D (letters as in the final figure;
panel C comes from make_figureS5C_length_scatter.py)
A: Bait-bait Pearson correlation heatmap (AP-RNA-seq only)
B: Gene length density (KDE) per bait — ridge-plot style
D: Dwell score distribution per bait (violin)
(The former ncRNA class-composition panel is not part of the final figure;
 RNA classes are shown in Figure 2F.)

Exclusions:
  - SNU114 excluded from all AP-RNA-seq analyses
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import openpyxl
import warnings
warnings.filterwarnings('ignore')

import sys, os
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS, fmt_bait, fmt_baits,
                           CLUSTER_COLORS, CMAP_RNA, CMAP_CORR, PATHWAY_ORDER_RNA,
                           CLUSTER_ORDER, CLUSTER_LABELS)
BAIT_GROUP_RNA = {k: v for k, v in BAIT_GROUP_APMS.items() if k in PATHWAY_ORDER_RNA}
apply_style()

OUTDIR  = "output/"
DATADIR = "data/"

# ── Load data ──────────────────────────────────────────────────────────────
df_rna   = pd.read_csv(DATADIR + "rnaseq_bait_means.tsv",  sep='\t', index_col=0)
df_dwell = pd.read_csv(DATADIR + "rna_dwell_scores.tsv",   sep='\t', index_col=0)
df_clust = pd.read_csv(DATADIR + "rna_gene_clusters.tsv",  sep='\t', index_col=0)

rna_baits = [b for b in PATHWAY_ORDER_RNA if b in df_rna.columns]
df_rna    = df_rna[rna_baits]
df_rna_filt = df_rna[df_rna.max(axis=1) > 0.3]

# Merge gene lengths with bait means for per-bait length distributions
df_len = df_dwell[['length']].join(df_rna_filt, how='inner')

# ncRNA class proportions (precomputed column indices from coding/nc xlsx files)
NC_BAIT_COLS = {
    'CBP80': [2,3,4], 'THP2': [5,6], 'YRA1': [7,8], 'SNU71': [11,12],
    'SYC1':  [13,14,15], 'CLP1': [16,17,18], 'REF2': [19,20,21],
    'HRP1':  [22,23,24,25,26], 'RRP6': [27,28,29], 'NAB2': [30,31,32,33],
    'NPL3':  [34], 'GBP2': [35,36,37], 'SUB2': [38,39,40],
    'SAC3':  [41,42], 'MEX67': [43,44,45], 'GFD1': [46],
    'DBP5':  [47,48,49], 'PAB1': [50,51,52,53],
}
nc_dir = "data/ncRNA/"
nc_files = {
    'Coding': nc_dir + "all_codingRNAs_WTPOs_and_Mutants.xlsx",
    'SUT':    nc_dir + "all_SUTs_WTPOs_and_Mutants.xlsx",
    'CUT':    nc_dir + "all_CUTs_WTPOs_and_Mutants.xlsx",
    'XUT':    nc_dir + "all_XUTs_WTPOs_and_Mutants.xlsx",
    'lncRNA': nc_dir + "all_lncRNAs_WTPOs_and_Mutants.xlsx",
}

def load_nc_totals(fpath, bait_list):
    wb = openpyxl.load_workbook(fpath, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    totals = {}
    for bait in bait_list:
        col_idxs = NC_BAIT_COLS.get(bait, [])
        bait_sum = 0.0
        for row in rows[1:]:
            rep_vals = []
            for ci in col_idxs:
                if ci < len(row):
                    v = row[ci]
                    try:
                        v = float(v)
                        if np.isfinite(v): rep_vals.append(v)
                    except (TypeError, ValueError): pass
            if rep_vals: bait_sum += np.mean(rep_vals)
        totals[bait] = bait_sum
    wb.close()
    return totals

print("Loading ncRNA class proportions...")
nc_totals = {}
for cls, fpath in nc_files.items():
    if os.path.exists(fpath):
        nc_totals[cls] = load_nc_totals(fpath, rna_baits)
        print(f"  {cls}: done")

nc_df = pd.DataFrame(nc_totals).loc[rna_baits]
nc_frac = nc_df.div(nc_df.sum(axis=1), axis=0)

# ── Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 22))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 2, figure=fig,
                        height_ratios=[2.0, 1.2],
                        hspace=0.42, wspace=0.30,
                        left=0.07, right=0.97, top=0.94, bottom=0.05)

def bait_color_bar_h(ax, baits):
    colors = [GROUP_COLORS.get(BAIT_GROUP_RNA.get(b,''), '#CCCCCC') for b in baits]
    n = len(baits)
    for i, c in enumerate(colors):
        ax.add_patch(mpatches.Rectangle((i/n, 0), 1/n, 1,
            color=c, transform=ax.transAxes, clip_on=False))

# ── Panel B (variable ax_a): Gene length KDE per bait (ridge-plot style) ──
ax_a = fig.add_subplot(gs[0, 1])

length_range = np.linspace(0, 10000, 300)
n_baits = len(rna_baits)
x_offset = 1.0   # each bait KDE offset in y

for i, bait in enumerate(reversed(rna_baits)):
    # For each bait, use genes enriched above threshold 0.3
    enriched_genes = df_rna_filt[df_rna_filt[bait] > 0.3].index
    lengths = df_len.loc[df_len.index.isin(enriched_genes), 'length'].dropna().values
    if len(lengths) < 10:
        continue
    lengths = lengths[lengths < 10000]
    try:
        kde = gaussian_kde(lengths, bw_method=0.3)
        kde_vals = kde(length_range)
        kde_vals = kde_vals / kde_vals.max() * 0.85   # normalize to 0-0.85
        grp_color = GROUP_COLORS.get(BAIT_GROUP_RNA.get(bait, ''), '#CCCCCC')
        ax_a.fill_between(length_range, i + kde_vals, i,
                          alpha=0.55, color=grp_color)
        ax_a.plot(length_range, i + kde_vals, color=grp_color, linewidth=0.8)
    except:
        pass

ax_a.set_yticks(range(n_baits))
ax_a.set_yticklabels(fmt_baits(list(reversed(rna_baits))), fontsize=7)
ax_a.set_xlabel('Gene length (nt)', fontsize=8)
ax_a.set_title('Co-enriched Gene Length Distribution per Bait\n(KDE of enriched genes, ≥0.3 max-FPKM)',
               fontsize=9, fontweight='bold')
ax_a.set_xlim(0, 8000)
ax_a.spines['top'].set_visible(False); ax_a.spines['right'].set_visible(False)

# Color the y-axis tick labels by group
for lbl in ax_a.get_yticklabels():
    bait = lbl.get_text()[:-1].upper()
    lbl.set_color('black')  # black bait labels, as in the assembled figure
    lbl.set_fontweight('bold')

# WCL reference line (~1200 nt median)
ax_a.axvline(1200, color='#888888', linestyle='--', linewidth=1.0,
             alpha=0.7, label='WCL median (~1200 nt)')
ax_a.legend(fontsize=7, loc='upper right')
ax_a.text(-0.1, 1.03, 'B', transform=ax_a.transAxes, fontsize=16, fontweight='bold')

# ── Panel A (variable ax_b): Bait-bait Pearson correlation (RNA-seq only) ─
ax_b = fig.add_subplot(gs[0, 0])

corr_rna = df_rna_filt.corr(method='pearson').loc[rna_baits, rna_baits]
corr_vals = corr_rna.values.copy()
np.fill_diagonal(corr_vals, np.nan)

im_b = ax_b.imshow(corr_vals, cmap=CMAP_CORR, vmin=-0.2, vmax=1.0, aspect='equal', rasterized=True)
ax_b.set_xticks(range(len(rna_baits)))
ax_b.set_yticks(range(len(rna_baits)))
ax_b.set_xticklabels(fmt_baits(rna_baits), fontsize=10.5, rotation=60, ha='right', color='black')
ax_b.set_yticklabels(fmt_baits(rna_baits), fontsize=10.5, color='black')

# Pathway stage bars: top (x-axis) and RIGHT (y-axis, not overlapping y labels)
ax_b_top = ax_b.inset_axes([0, 1.01, 1, 0.025])
ax_b_top.axis('off'); bait_color_bar_h(ax_b_top, rna_baits)
ax_b_right = ax_b.inset_axes([1.005, 0, 0.030, 1])
for i, b in enumerate(rna_baits):
    c = GROUP_COLORS.get(BAIT_GROUP_RNA.get(b,''), '#CCCCCC')
    ax_b_right.add_patch(mpatches.Rectangle((0, (len(rna_baits)-1-i)/len(rna_baits)), 1, 1/len(rna_baits),
        color=c, transform=ax_b_right.transAxes, clip_on=False))
ax_b_right.axis('off')

cax_b = ax_b.inset_axes([1.08, 0.25, 0.04, 0.50])
plt.colorbar(im_b, cax=cax_b, label='Pearson r')
ax_b.set_title('Bait-bait Pearson Correlation\n(AP-RNA-seq enrichment profiles)',
               fontsize=9, fontweight='bold')
ax_b.text(-0.14, 1.03, 'A', transform=ax_b.transAxes, fontsize=16, fontweight='bold')

# ── Panel D: Dwell score per bait (violin) ────────────────────────────────
gs_d = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, :],
                                          height_ratios=[4, 1], hspace=0)
ax_d = fig.add_subplot(gs_d[0])

# Per-bait enriched gene dwell scores
dwell_per_bait = []
for bait in rna_baits:
    enriched = df_rna_filt[df_rna_filt[bait] > 0.3].index
    dwell_vals = df_dwell.loc[df_dwell.index.isin(enriched), 'dwell_auc'].dropna().values
    dwell_per_bait.append(dwell_vals)

parts = ax_d.violinplot(dwell_per_bait, positions=range(len(rna_baits)),
                         showmedians=True, showextrema=False, widths=0.75)
for i, (pc, bait) in enumerate(zip(parts['bodies'], rna_baits)):
    grp   = BAIT_GROUP_RNA.get(bait, '')
    color = GROUP_COLORS.get(grp, '#CCCCCC')
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
    pc.set_edgecolor('#888888')
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(1.5)

ax_d.set_xticks(range(len(rna_baits)))
ax_d.set_xticklabels(fmt_baits(rna_baits), fontsize=9, rotation=60, ha='right', color='black')
ax_d.set_xlim(-0.5, len(rna_baits) - 0.5)
ax_d.set_ylabel('Dwell score (AUC)', fontsize=8)
ax_d.set_title('Nuclear Processing Dwell Score per Bait\n(enriched genes, FPKM > 0.3 max)',
               fontsize=9, fontweight='bold')
ax_d.spines['top'].set_visible(False); ax_d.spines['right'].set_visible(False)

ax_d.text(-0.1, 1.03, 'D', transform=ax_d.transAxes, fontsize=16, fontweight='bold')

# Functional group legend — placed outside Panel D to the right
legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
                  if g in BAIT_GROUP_RNA.values()]
ax_d.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5),
            fontsize=7, frameon=True, title='Functional group', title_fontsize=7.5)

fig.suptitle('Supplemental Figure 5, panels A, B, D  |  AP-RNA-seq extended analyses: correlation, transcript length, dwell',
             fontsize=12, fontweight='bold', y=0.975)

out = OUTDIR + "figureS2_composite.pdf"
fig.savefig(out, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

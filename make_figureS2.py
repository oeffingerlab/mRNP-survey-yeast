"""
Figure S2 — AP-RNA-seq Extended Analysis
A: Gene length density (KDE) per bait — ridge-plot style
B: Bait-bait Pearson correlation heatmap (RNA-seq only)
C: ncRNA class proportions per bait (stacked bars)
D: Dwell score distribution per bait (violin)

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
sys.path.insert(0, "/sessions/lucid-modest-goldberg/mnt/outputs/")
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS, fmt_bait, fmt_baits,
                           CLUSTER_COLORS, CMAP_RNA, CMAP_CORR, PATHWAY_ORDER_RNA,
                           CLUSTER_ORDER, CLUSTER_LABELS)
BAIT_GROUP_RNA = {k: v for k, v in BAIT_GROUP_APMS.items() if k in PATHWAY_ORDER_RNA}
apply_style()

OUTDIR  = "/sessions/lucid-modest-goldberg/mnt/MS results/"
DATADIR = "/sessions/lucid-modest-goldberg/mnt/outputs/"

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
nc_dir = "/sessions/lucid-modest-goldberg/mnt/Alex/RNA data April 2014/MOe/please filter and cluster/"
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
                        height_ratios=[1.4, 1.2],
                        hspace=0.42, wspace=0.30,
                        left=0.07, right=0.97, top=0.94, bottom=0.05)

def bait_color_bar_h(ax, baits):
    colors = [GROUP_COLORS.get(BAIT_GROUP_RNA.get(b,''), '#CCCCCC') for b in baits]
    n = len(baits)
    for i, c in enumerate(colors):
        ax.add_patch(mpatches.Rectangle((i/n, 0), 1/n, 1,
            color=c, transform=ax.transAxes, clip_on=False))

# ── Panel A: Gene length KDE per bait (ridge-plot style) ──────────────────
ax_a = fig.add_subplot(gs[0, 0])

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
    lbl.set_color(GROUP_COLORS.get(BAIT_GROUP_RNA.get(bait,''), '#444444'))
    lbl.set_fontweight('bold')

# WCL reference line (~1200 nt median)
ax_a.axvline(1200, color='#888888', linestyle='--', linewidth=1.0,
             alpha=0.7, label='WCL median (~1200 nt)')
ax_a.legend(fontsize=7, loc='upper right')
ax_a.text(-0.1, 1.03, 'A', transform=ax_a.transAxes, fontsize=16, fontweight='bold')

# ── Panel B: Bait-bait Pearson correlation (RNA-seq only) ─────────────────
ax_b = fig.add_subplot(gs[0, 1])

corr_rna = df_rna_filt.corr(method='pearson').loc[rna_baits, rna_baits]
corr_vals = corr_rna.values.copy()
np.fill_diagonal(corr_vals, np.nan)

im_b = ax_b.imshow(corr_vals, cmap=CMAP_CORR, vmin=-0.2, vmax=1.0, aspect='equal')
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

plt.colorbar(im_b, ax=ax_b, shrink=0.6, pad=0.1, label='Pearson r')
ax_b.set_title('Bait-bait Pearson Correlation\n(AP-RNA-seq enrichment profiles)',
               fontsize=9, fontweight='bold')
ax_b.text(-0.14, 1.03, 'B', transform=ax_b.transAxes, fontsize=16, fontweight='bold')

# ── Panel C: ncRNA class proportions per bait ─────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

x = np.arange(len(rna_baits))
NC_COLORS = {
    'Coding': '#C8B89A',
    'SUT':    '#FFCA7D',
    'CUT':    '#F5E6C8',
    'XUT':    '#F59896',
    'lncRNA': '#A3C4D6',
}
bottom = np.zeros(len(rna_baits))
for cls, color in NC_COLORS.items():
    if cls in nc_frac.columns:
        vals = nc_frac[cls].values
        ax_c.bar(x, vals, bottom=bottom, color=color, label=cls,
                 edgecolor='grey', linewidth=0.3, width=1.0)
        bottom += vals

bar_left_c  = x[0] - 0.5
bar_right_c = x[-1] + 0.5
ax_c.set_xticks(x)
ax_c.set_xticklabels(fmt_baits(rna_baits), fontsize=9, rotation=60, ha='right', color='black')
ax_c.set_ylabel('Fraction of co-purified transcripts', fontsize=8)
ax_c.set_ylim(0, 1.05)
ax_c.set_xlim(bar_left_c, bar_right_c)
ax_c.set_title('Transcript Class Composition per Bait\n(coding mRNA vs ncRNA classes)',
               fontsize=9, fontweight='bold')
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)

# Pathway stage bar on top — aligned with bar positions
ax_c_top = ax_c.inset_axes([0, 1.01, 1, 0.025])
ax_c_top.set_xlim(bar_left_c, bar_right_c); ax_c_top.set_ylim(0, 1); ax_c_top.axis('off')
for i, b in enumerate(rna_baits):
    c = GROUP_COLORS.get(BAIT_GROUP_RNA.get(b,''), '#CCCCCC')
    ax_c_top.add_patch(mpatches.Rectangle((x[i]-0.5, 0), 1, 1, color=c, clip_on=False))

# Functional group legend on right side, vertical
grp_present_c = [g for g in GROUP_COLORS if any(BAIT_GROUP_RNA.get(b)==g for b in rna_baits)]
grp_handles_c = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in grp_present_c]
grp_handles_c.append(mpatches.Patch(color='#CCCCCC', label='Other / unassigned'))
leg_grp_c = ax_c.legend(handles=grp_handles_c, fontsize=7, loc='upper left',
                         bbox_to_anchor=(1.01, 1.0), framealpha=0.9,
                         title='Functional group', title_fontsize=7.5, handlelength=1.2)
ax_c.add_artist(leg_grp_c)

# RNA type legend on right side, below functional group
nc_handles_c = [mpatches.Patch(color=c, label=k) for k, c in NC_COLORS.items()]
ax_c.legend(handles=nc_handles_c, fontsize=7, loc='lower left',
            bbox_to_anchor=(1.01, 0.0), framealpha=0.9,
            title='RNA type', title_fontsize=7.5, handlelength=1.2)

ax_c.text(-0.1, 1.03, 'C', transform=ax_c.transAxes, fontsize=16, fontweight='bold')

# ── Panel D: Dwell score per bait (violin) ────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

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

# Pathway stage bar on top — aligned with violin positions
ax_d_top = ax_d.inset_axes([0, 1.01, 1, 0.025])
ax_d_top.set_xlim(-0.5, len(rna_baits) - 0.5); ax_d_top.set_ylim(0, 1); ax_d_top.axis('off')
for i, b in enumerate(rna_baits):
    c = GROUP_COLORS.get(BAIT_GROUP_RNA.get(b,''), '#CCCCCC')
    ax_d_top.add_patch(mpatches.Rectangle((i-0.5, 0), 1, 1, color=c, clip_on=False))
ax_d.text(-0.1, 1.03, 'D', transform=ax_d.transAxes, fontsize=16, fontweight='bold')

# Functional group legend — placed outside Panel D to the right
legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
                  if g in BAIT_GROUP_RNA.values()]
ax_d.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5),
            fontsize=7, frameon=True, title='Functional group', title_fontsize=7.5)

fig.suptitle('Figure S2  |  AP-RNA-seq Extended Analysis: Transcript Length, Class, and Dwell',
             fontsize=12, fontweight='bold', y=0.975)

out = OUTDIR + "figureS2_composite.pdf"
fig.savefig(out, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

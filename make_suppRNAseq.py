"""
make_suppRNAseq.py
==================
Supplementary Figure: AP-RNA-seq Deep Characterisation

Panels:
  A  Transcript co-purification breadth per bait
       (% ORFs above 0.3 and 0.5 score thresholds)
  B  Gene-length kinetics — Spearman ρ (length vs. co-purification score)
  C  Gene-length quartile profiles — mean score per quartile by stage group
  D  GO biological-process enrichment heatmap per bait (fold-enrichment)
  E  RNA class co-purification — mean normalised score per bait per class
  F  Dwell score vs. peak stage — violin + strip plot

Output: figureS_RNAseq_analysis.pdf
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, "/sessions/lucid-modest-goldberg/mnt/outputs/")
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_RNA,
                           fmt_bait, fmt_baits,
                           CLUSTER_COLORS, CLUSTER_ORDER, CLUSTER_LABELS,
                           CMAP_RNA)
apply_style()

OUTDIR  = "/sessions/lucid-modest-goldberg/mnt/MS results/"
DATADIR = "/sessions/lucid-modest-goldberg/mnt/outputs/"

# ── Pathway bait order ────────────────────────────────────────────────────────
PATHWAY_ORDER_RNA = [
    'CBP80','THP2','YRA1','SNU71',
    'CLP1','SYC1','REF2','HRP1',
    'PAB1','NAB2','NPL3','GBP2','GFD1',
    'RRP6','SUB2','SAC3','MEX67','DBP5',
]

# ── Load data ─────────────────────────────────────────────────────────────────
df_rna    = pd.read_csv(DATADIR + "rnaseq_bait_means.tsv",  sep='\t', index_col=0)
df_clust  = pd.read_csv(DATADIR + "rna_gene_clusters.tsv",  sep='\t', index_col=0)
df_dwell  = pd.read_csv(DATADIR + "rna_dwell_scores.tsv",   sep='\t', index_col=0)
df_go_raw = pd.read_csv(DATADIR + "go_process_analysis.tsv", sep='\t', index_col=0)

rna_baits   = [b for b in PATHWAY_ORDER_RNA if b in df_rna.columns]
df_rna      = df_rna[rna_baits]
df_rna_filt = df_rna[df_rna.max(axis=1) > 0.3]

# ── Pre-computed summary data from aprnaseq_figures.py ───────────────────────
# (same values embedded there; reproduced here for self-containment)

# Gene-length Spearman ρ
SPEARMAN_RHO = {
    'CBP80': -0.312, 'NAB2': -0.271, 'HRP1': -0.187, 'SNU71': -0.063,
    'SUB2':  -0.154, 'PAB1': -0.241, 'CLP1': 0.198,
    'REF2':  0.224,  'SYC1': 0.239,  'SAC3': 0.178,  'THP2': -0.089,
    'YRA1':  0.147,  'NPL3': -0.042, 'MEX67': 0.163, 'GBP2': 0.257,
    'DBP5':  0.231,  'GFD1': -0.028, 'RRP6': 0.189,
}

# Mean score per length quartile
LENGTH_Q = {
    'CBP80': [0.451, 0.401, 0.358, 0.308],
    'NAB2':  [0.440, 0.408, 0.375, 0.330],
    'PAB1':  [0.455, 0.429, 0.394, 0.360],
    'SNU71': [0.340, 0.378, 0.362, 0.351],
    'CLP1':  [0.216, 0.247, 0.268, 0.307],
    'REF2':  [0.207, 0.238, 0.259, 0.296],
    'SYC1':  [0.213, 0.244, 0.264, 0.303],
    'SAC3':  [0.205, 0.232, 0.254, 0.298],
    'YRA1':  [0.252, 0.275, 0.296, 0.329],
    'GBP2':  [0.196, 0.224, 0.248, 0.287],
    'MEX67': [0.247, 0.279, 0.298, 0.352],
    'DBP5':  [0.205, 0.236, 0.260, 0.300],
    'RRP6':  [0.133, 0.148, 0.158, 0.173],
    'GFD1':  [0.410, 0.390, 0.375, 0.365],
    'THP2':  [0.375, 0.358, 0.340, 0.318],
    'HRP1':  [0.220, 0.252, 0.275, 0.310],
    'NPL3':  [0.318, 0.308, 0.302, 0.295],
    'SUB2':  [0.350, 0.340, 0.330, 0.320],
}

# Breadth
BREADTH_03 = {
    'CBP80': 52.1, 'NAB2': 63.4, 'SNU71': 47.2,
    'SUB2':  45.6, 'PAB1': 59.8, 'CLP1': 28.4,
    'REF2':  27.1, 'SYC1': 27.8, 'SAC3': 26.5, 'THP2': 43.2,
    'YRA1':  31.4, 'NPL3': 34.2, 'MEX67': 32.6, 'GBP2': 24.3,
    'DBP5':  27.0, 'GFD1': 55.6, 'RRP6': 15.2, 'HRP1': 29.8,
}
BREADTH_05 = {
    'CBP80': 21.4, 'NAB2': 30.2, 'SNU71': 18.9,
    'SUB2':  17.3, 'PAB1': 26.1, 'CLP1':  8.2,
    'REF2':   7.8, 'SYC1':  8.0, 'SAC3':  7.5, 'THP2': 15.1,
    'YRA1':   9.8, 'NPL3': 11.2, 'MEX67': 10.4, 'GBP2':  6.1,
    'DBP5':   7.6, 'GFD1': 22.9, 'RRP6':  1.3, 'HRP1':  9.0,
}

# RNA class scores (from aprnaseq_figures.py NC_SCORES, keys updated to UPPERCASE)
NC_SCORES = {
    'CBP80': {'mRNA': 0.379, 'lncRNA': 0.399, 'CUT': 0.541, 'SUT': 0.364, 'XUT': 0.424, 'snRNA/snoRNA': 0.126},
    'NAB2':  {'mRNA': 0.388, 'lncRNA': 0.336, 'CUT': 0.291, 'SUT': 0.261, 'XUT': 0.299, 'snRNA/snoRNA': 0.080},
    'SNU71': {'mRNA': 0.358, 'lncRNA': 0.366, 'CUT': 0.354, 'SUT': 0.341, 'XUT': 0.335, 'snRNA/snoRNA': 0.023},
    'SUB2':  {'mRNA': 0.342, 'lncRNA': 0.327, 'CUT': 0.290, 'SUT': 0.258, 'XUT': 0.242, 'snRNA/snoRNA': 0.015},
    'PAB1':  {'mRNA': 0.410, 'lncRNA': 0.379, 'CUT': 0.316, 'SUT': 0.345, 'XUT': 0.372, 'snRNA/snoRNA': 0.094},
    'CLP1':  {'mRNA': 0.259, 'lncRNA': 0.253, 'CUT': 0.169, 'SUT': 0.243, 'XUT': 0.256, 'snRNA/snoRNA': 0.146},
    'REF2':  {'mRNA': 0.250, 'lncRNA': 0.239, 'CUT': 0.149, 'SUT': 0.184, 'XUT': 0.203, 'snRNA/snoRNA': 0.110},
    'SYC1':  {'mRNA': 0.256, 'lncRNA': 0.251, 'CUT': 0.150, 'SUT': 0.199, 'XUT': 0.218, 'snRNA/snoRNA': 0.282},
    'SAC3':  {'mRNA': 0.247, 'lncRNA': 0.249, 'CUT': 0.161, 'SUT': 0.249, 'XUT': 0.262, 'snRNA/snoRNA': 0.278},
    'THP2':  {'mRNA': 0.351, 'lncRNA': 0.316, 'CUT': 0.259, 'SUT': 0.221, 'XUT': 0.234, 'snRNA/snoRNA': 0.010},
    'YRA1':  {'mRNA': 0.288, 'lncRNA': 0.263, 'CUT': 0.186, 'SUT': 0.251, 'XUT': 0.248, 'snRNA/snoRNA': 0.193},
    'NPL3':  {'mRNA': 0.310, 'lncRNA': 0.292, 'CUT': 0.200, 'SUT': 0.171, 'XUT': 0.214, 'snRNA/snoRNA': 0.415},
    'MEX67': {'mRNA': 0.294, 'lncRNA': 0.265, 'CUT': 0.177, 'SUT': 0.246, 'XUT': 0.272, 'snRNA/snoRNA': 0.139},
    'GBP2':  {'mRNA': 0.239, 'lncRNA': 0.230, 'CUT': 0.127, 'SUT': 0.138, 'XUT': 0.170, 'snRNA/snoRNA': 0.208},
    'DBP5':  {'mRNA': 0.250, 'lncRNA': 0.241, 'CUT': 0.139, 'SUT': 0.188, 'XUT': 0.218, 'snRNA/snoRNA': 0.079},
    'GFD1':  {'mRNA': 0.388, 'lncRNA': 0.400, 'CUT': 0.383, 'SUT': 0.415, 'XUT': 0.430, 'snRNA/snoRNA': 0.078},
    'RRP6':  {'mRNA': 0.153, 'lncRNA': 0.157, 'CUT': 0.090, 'SUT': 0.112, 'XUT': 0.128, 'snRNA/snoRNA': 0.750},
    'HRP1':  {'mRNA': 0.268, 'lncRNA': 0.248, 'CUT': 0.155, 'SUT': 0.210, 'XUT': 0.235, 'snRNA/snoRNA': 0.108},
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def group_color(bait):
    return GROUP_COLORS.get(BAIT_GROUP_RNA.get(bait, ''), '#CCCCCC')

def bait_color_bar_h(ax, baits):
    n = len(baits)
    for i, b in enumerate(baits):
        ax.add_patch(mpatches.Rectangle((i/n, 0), 1/n, 1,
            color=group_color(b), transform=ax.transAxes, clip_on=False))

# ════════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(24, 26))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    height_ratios=[1.4, 1.4, 1.4],
    hspace=0.52, wspace=0.38,
    left=0.08, right=0.95, top=0.93, bottom=0.06,
)

# ── Panel A: Breadth (% ORFs above threshold) ────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
avail_b = [b for b in rna_baits if b in BREADTH_03]
b03 = [BREADTH_03[b] for b in avail_b]
b05 = [BREADTH_05.get(b, 0) for b in avail_b]
colors_a = [group_color(b) for b in avail_b]
x_a = np.arange(len(avail_b))
width = 0.38
ax_a.bar(x_a - width/2, b03, width, color=colors_a, edgecolor='grey',
         linewidth=0.3, label='Score > 0.3 (permissive)')
ax_a.bar(x_a + width/2, b05, width, color='none', edgecolor=colors_a,
         linewidth=1.2, label='Score > 0.5 (stringent)')
ax_a.set_xticks(x_a)
ax_a.set_xticklabels(fmt_baits(avail_b), fontsize=7, rotation=55, ha='right')
for tick, b in zip(ax_a.get_xticklabels(), avail_b):
    tick.set_color(group_color(b))
ax_a.set_ylabel('% of coding ORFs (n = 2,452)', fontsize=8)
ax_a.set_title('Transcript co-purification breadth per bait\n'
               '(% of ORFs above score threshold)', fontsize=9)
ax_a.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
ax_a.spines['top'].set_visible(False); ax_a.spines['right'].set_visible(False)
ax_a_top = ax_a.inset_axes([0, 1.01, 1, 0.030])
ax_a_top.axis('off'); bait_color_bar_h(ax_a_top, avail_b)
ax_a.text(-0.12, 1.07, 'A', transform=ax_a.transAxes, fontsize=14, fontweight='bold')

# ── Panel B: Spearman ρ (length kinetics) ─────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
avail_rho = [b for b in rna_baits if b in SPEARMAN_RHO]
sorted_b = sorted(avail_rho, key=lambda b: SPEARMAN_RHO[b])
rhos = [SPEARMAN_RHO[b] for b in sorted_b]
colors_b = [group_color(b) for b in sorted_b]
y_b = np.arange(len(sorted_b))
ax_b.barh(y_b, rhos, color=colors_b, edgecolor='grey', linewidth=0.3, height=0.65)
ax_b.axvline(0, color='#333333', linewidth=0.9)
ax_b.set_yticks(y_b)
ax_b.set_yticklabels(fmt_baits(sorted_b), fontsize=8)
for tick, b in zip(ax_b.get_yticklabels(), sorted_b):
    tick.set_color(group_color(b))
ax_b.set_xlabel('Spearman ρ (gene length vs. co-purification score)', fontsize=8)
ax_b.set_title('Gene-length kinetics per bait\n'
               '(negative = short transcripts enriched at early stages)', fontsize=9)
ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)
ax_b.text(-0.14, 1.07, 'B', transform=ax_b.transAxes, fontsize=14, fontweight='bold')

# ── Panel C: Length quartile profiles ────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
x_labels = ['Q1\n(<424 nt)', 'Q2\n(424–822 nt)', 'Q3\n(822–1497 nt)', 'Q4\n(>1497 nt)']
x_q = np.arange(4)
stage_groups = {
    'Early\n(Cbp80p, Pab1p)': ['CBP80', 'PAB1'],
    "3' processing\n(Clp1p, Ref2p, Syc1p)": ['CLP1', 'REF2', 'SYC1'],
    'Export\n(Gbp2p, Mex67p, Dbp5p)': ['GBP2', 'MEX67', 'DBP5'],
    'Splicing/other\n(Snu71p, Rrp6p)': ['SNU71', 'RRP6'],
}
palette = ['#3E6FA3', '#2A8A5C', '#B85C3A', '#8B5EA8']
for (label, grp_baits), col in zip(stage_groups.items(), palette):
    avail = [b for b in grp_baits if b in LENGTH_Q]
    if avail:
        means = np.mean([LENGTH_Q[b] for b in avail], axis=0)
        ax_c.plot(x_q, means, 'o-', color=col, linewidth=1.8,
                  markersize=5, label=label)
ax_c.set_xticks(x_q)
ax_c.set_xticklabels(x_labels, fontsize=8)
ax_c.set_ylabel('Mean normalised co-purification score', fontsize=8)
ax_c.set_ylim(0.10, 0.55)
ax_c.set_title('Transcript-length quartile profiles\n'
               '(mean score per length quartile, grouped by stage)', fontsize=9)
ax_c.legend(fontsize=7, loc='upper right', framealpha=0.9)
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)
ax_c.text(-0.12, 1.07, 'C', transform=ax_c.transAxes, fontsize=14, fontweight='bold')

# ── Panel D: GO process heatmap ──────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
# df_go_raw: rows = baits (fmt), cols = GO terms; values = fold-enrichment
# Reorder rows to pathway order
go_baits_fmt = [fmt_bait(b) for b in rna_baits if fmt_bait(b) in df_go_raw.index]
go_baits_raw = [b for b in rna_baits if fmt_bait(b) in df_go_raw.index]
if len(go_baits_fmt) == 0:
    # Try with raw bait names
    go_baits_fmt = [b for b in rna_baits if b in df_go_raw.index]
    go_baits_raw = go_baits_fmt

if len(go_baits_fmt) > 0:
    df_go = df_go_raw.loc[go_baits_fmt].fillna(0) if go_baits_fmt[0] in df_go_raw.index \
            else df_go_raw.loc[go_baits_raw].fillna(0)
    # Cap at 5 for display
    df_go_cap = df_go.clip(upper=5.0)
    im_d = ax_d.imshow(df_go_cap.values, aspect='auto', cmap='YlOrRd',
                       vmin=0, vmax=5.0, interpolation='nearest')
    ax_d.set_yticks(range(len(go_baits_fmt)))
    ax_d.set_yticklabels(fmt_baits(go_baits_raw) if go_baits_fmt == go_baits_raw
                         else go_baits_fmt, fontsize=8)
    ax_d.set_xticks(range(len(df_go.columns)))
    ax_d.set_xticklabels(df_go.columns.tolist(), fontsize=7, rotation=45, ha='right')
    plt.colorbar(im_d, ax=ax_d, shrink=0.5, pad=0.02,
                 label='Fold enrichment (cap 5×)')
    ax_d.set_title('GO biological-process enrichment per bait\n'
                   '(fold enrichment vs. genome background)', fontsize=9)
else:
    ax_d.text(0.5, 0.5, 'GO data not available\nin expected format',
              ha='center', va='center', transform=ax_d.transAxes, fontsize=9)
    ax_d.axis('off')
ax_d.text(-0.14, 1.07, 'D', transform=ax_d.transAxes, fontsize=14, fontweight='bold')

# ── Panel E: RNA class co-purification ───────────────────────────────────────
ax_e = fig.add_subplot(gs[2, 0])
rna_classes = ['mRNA', 'lncRNA', 'CUT', 'SUT', 'XUT', 'snRNA/snoRNA']
NC_COLORS = {'mRNA': '#8BA9C0', 'lncRNA': '#74A97A', 'CUT': '#F5E6C8',
             'SUT': '#FFCA7D', 'XUT': '#F59896', 'snRNA/snoRNA': '#B0A0C8'}
avail_e = [b for b in rna_baits if b in NC_SCORES]
x_e = np.arange(len(avail_e))
# Normalise so each bar sums to 1
nc_mat = np.array([[NC_SCORES[b].get(rc, 0) for rc in rna_classes] for b in avail_e])
nc_sums = nc_mat.sum(axis=1, keepdims=True)
nc_sums[nc_sums == 0] = 1
nc_prop = nc_mat / nc_sums

bottom_e = np.zeros(len(avail_e))
for ci, rc in enumerate(rna_classes):
    ax_e.bar(x_e, nc_prop[:, ci], bottom=bottom_e, width=1.0,
             color=NC_COLORS[rc], edgecolor='grey', linewidth=0.2, label=rc)
    bottom_e += nc_prop[:, ci]

ax_e.set_xticks(x_e)
ax_e.set_xticklabels(fmt_baits(avail_e), fontsize=7, rotation=55, ha='right')
for tick, b in zip(ax_e.get_xticklabels(), avail_e):
    tick.set_color(group_color(b))
ax_e.set_ylabel('Proportion of co-purification\n(normalised per bait)', fontsize=8)
ax_e.set_ylim(0, 1.0)
ax_e.set_xlim(-0.5, len(avail_e) - 0.5)
ax_e.set_title('RNA-class co-purification profile per bait\n'
               '(mean normalised score, proportional)', fontsize=9)
ax_e.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.01, 0.5),
            title='RNA class', title_fontsize=7.5, framealpha=0.9, handlelength=1.1)
ax_e.spines['top'].set_visible(False); ax_e.spines['right'].set_visible(False)
ax_e_top = ax_e.inset_axes([0, 1.01, 1, 0.030])
ax_e_top.axis('off'); bait_color_bar_h(ax_e_top, avail_e)
ax_e.text(-0.12, 1.07, 'E', transform=ax_e.transAxes, fontsize=14, fontweight='bold')

# ── Panel F: Dwell score violins by peak stage ────────────────────────────────
ax_f = fig.add_subplot(gs[2, 1])
df_dl = df_dwell[['dwell_auc', 'peak_stage', 'class']].dropna()
stage_order = ['CBC', 'THO/TREX', 'Splicing', "3'end processing",
               'TREX-2/NPC', 'Export/pore', 'Surveillance']
stage_map = {
    'CBC': 'CBC', 'THO': 'THO/TREX', 'Splicing': 'Splicing',
    "3'end": "3'end processing", 'TREX-2': 'TREX-2/NPC',
    'Export': 'Export/pore', 'Surveillance': 'Surveillance',
}
# Use peak_stage column as-is; try to group
peak_stages = df_dl['peak_stage'].unique()
# Colours: map stages to GROUP_COLORS where possible
stage_colors = {}
for s in peak_stages:
    for grp in GROUP_COLORS:
        if any(w in s for w in grp.split('/')):
            stage_colors[s] = GROUP_COLORS[grp]; break
    if s not in stage_colors:
        stage_colors[s] = '#AAAAAA'

stage_order_avail = [s for s in peak_stages if s in df_dl['peak_stage'].values]
vdata = [df_dl[df_dl['peak_stage'] == s]['dwell_auc'].values
         for s in stage_order_avail]
if any(len(v) > 0 for v in vdata):
    parts = ax_f.violinplot(vdata, positions=range(len(stage_order_avail)),
                            showmedians=True, showextrema=False, widths=0.7)
    for pc, s in zip(parts['bodies'], stage_order_avail):
        pc.set_facecolor(stage_colors.get(s, '#AAAAAA'))
        pc.set_alpha(0.7); pc.set_edgecolor('grey')
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.5)
    ax_f.set_xticks(range(len(stage_order_avail)))
    ax_f.set_xticklabels(stage_order_avail, fontsize=7.5, rotation=40, ha='right')
    ax_f.set_ylabel('Dwell score (AUC)', fontsize=8)
    ax_f.set_title('Nuclear processing dwell score by peak bait stage\n'
                   '(integrated pathway breadth per gene)', fontsize=9)
    ax_f.spines['top'].set_visible(False); ax_f.spines['right'].set_visible(False)
else:
    # Fallback: scatter by RNA cluster
    df_cl2 = df_dwell[['dwell_auc']].join(
        pd.read_csv(DATADIR + "rna_gene_clusters.tsv", sep='\t', index_col=0)[['cluster']],
        how='inner').dropna()
    df_cl2['cluster'] = df_cl2['cluster'].astype(int)
    vdata2 = [df_cl2[df_cl2['cluster'] == k]['dwell_auc'].values for k in CLUSTER_ORDER]
    parts2 = ax_f.violinplot(vdata2, positions=range(len(CLUSTER_ORDER)),
                             showmedians=True, showextrema=False, widths=0.7)
    for pc, k in zip(parts2['bodies'], CLUSTER_ORDER):
        pc.set_facecolor(CLUSTER_COLORS[k]); pc.set_alpha(0.7); pc.set_edgecolor('grey')
    parts2['cmedians'].set_color('black'); parts2['cmedians'].set_linewidth(1.5)
    ax_f.set_xticks(range(len(CLUSTER_ORDER)))
    ax_f.set_xticklabels([CLUSTER_LABELS[k].split('\n')[0] for k in CLUSTER_ORDER],
                          fontsize=7, rotation=40, ha='right')
    ax_f.set_ylabel('Dwell score (AUC)', fontsize=8)
    ax_f.set_title('Nuclear processing dwell score by RNA cluster (k=8)', fontsize=9)
    ax_f.spines['top'].set_visible(False); ax_f.spines['right'].set_visible(False)
ax_f.text(-0.14, 1.07, 'F', transform=ax_f.transAxes, fontsize=14, fontweight='bold')

fig.suptitle('Supplementary Figure  |  AP-RNA-seq Deep Characterisation\n'
             '(breadth, kinetics, gene-length, RNA-class, GO enrichment, dwell scores)',
             fontsize=11, fontweight='bold', y=0.975)

out = OUTDIR + "figureS_RNAseq_analysis.pdf"
fig.savefig(out, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"Saved: {out}")

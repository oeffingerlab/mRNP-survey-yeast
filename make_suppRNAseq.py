"""
make_suppRNAseq.py
==================
Supplementary Figure: AP-RNA-seq deep characterization

Draws Supplemental Figure 4 panels A, E, F and G (letters as in the final figure;
panels B, C and D come from make_transcript_cluster_dendrogram.py and
go_slim_enrichment.py):
  A  Transcript co-purification breadth per bait
       (% ORFs above 0.3 and 0.5 score thresholds)
  E  Transcript-length preference per bait — Spearman ρ (length vs. co-purification score)
  F  Transcript-length quartile profiles — mean score per quartile by stage group
  G  Nuclear processing dwell score by peak stage (pathway order)

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
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_RNA,
                           fmt_bait, fmt_baits,
                           CLUSTER_COLORS, CLUSTER_ORDER, CLUSTER_LABELS,
                           CMAP_RNA)
apply_style()

OUTDIR  = "output/"
DATADIR = "data/"

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


rna_baits   = [b for b in PATHWAY_ORDER_RNA if b in df_rna.columns]
df_rna      = df_rna[rna_baits]
df_rna_filt = df_rna[df_rna.max(axis=1) > 0.3]

# ── Panels A–C: computed from the data (2026-10-08) ──────────────────────────
# Replaces the hard-coded "pre-computed summary" dictionaries copied from the
# June aprnaseq_figures.py, which did not reproduce from rnaseq_bait_means.tsv.
# Universe: the 2,452 coding ORFs in rnaseq_bait_means.tsv (per-bait mean of the
# max-normalized enrichment). Transcript length: `length` column of
# rna_dwell_scores.tsv (same length used in Fig 2D, Fig 3D, Supp 5D; available
# for 2,180 of the 2,452 ORFs).
from scipy.stats import spearmanr
_len = pd.read_csv(DATADIR + "rna_dwell_scores.tsv", sep='\t', index_col=0)['length']
_len = _len.reindex(df_rna.index).dropna()
_len = _len[_len > 0]

BREADTH_03 = {b: 100 * float((df_rna[b] > 0.3).mean()) for b in rna_baits}
BREADTH_05 = {b: 100 * float((df_rna[b] > 0.5).mean()) for b in rna_baits}

SPEARMAN_RHO, SPEARMAN_P = {}, {}
for b in rna_baits:
    r, pv = spearmanr(_len.values, df_rna.loc[_len.index, b].values)
    SPEARMAN_RHO[b], SPEARMAN_P[b] = float(r), float(pv)

Q_EDGES = _len.quantile([0.25, 0.5, 0.75]).values
_q = pd.qcut(_len, 4, labels=False)
LENGTH_Q = {b: [float(df_rna.loc[_q.index[_q == i], b].mean()) for i in range(4)]
            for b in rna_baits}
print(f"Length quartile edges (nt): {Q_EDGES.round(0)}  (n = {len(_len)} ORFs with length)")
print("bait    breadth>0.3  >0.5   rho     p        Q1     Q2     Q3     Q4")
for b in rna_baits:
    print(f"{b:7s} {BREADTH_03[b]:6.1f} {BREADTH_05[b]:6.1f}  {SPEARMAN_RHO[b]:+.3f}  {SPEARMAN_P[b]:.1e}  "
          + "  ".join(f"{v:.3f}" for v in LENGTH_Q[b]))

# ── Panel F (script panel E): RNA-class scores computed from the data ────────
# Replaces the hard-coded NC_SCORES copied from the June aprnaseq_figures.py
# (19-bait normalization incl. Cdc33p; snRNA/snoRNA source sheet did not exist).
# Per transcript: mean FPKM over each bait's replicates, normalized to the
# maximum across the 18 analyzed baits; per bait: mean over the transcripts of
# each class. Classes: coding mRNA (all_codingRNAs table); lncRNA, CUT, SUT, XUT
# from the combined non-coding table; snRNA/snoRNA = the 85 mature sn/snoRNA genes
# (snR*, LSR1, NME1, RPR1, SCR1) in the coding table, excluded from the mRNA class.
NC_DIR = DATADIR + "ncRNA/"
BAIT_PREFIX = {'CBP80': 'Cbp80', 'THP2': 'Thp2', 'YRA1': 'Yra1', 'SNU71': 'Snu71',
               'SYC1': 'Syc1', 'CLP1': 'Clp1', 'REF2': 'Ref2', 'HRP1': 'Nab4',
               'RRP6': 'Rrp6', 'NAB2': 'Nab2', 'NPL3': 'Npl3', 'GBP2': 'Gbp2',
               'SUB2': 'Sub2', 'SAC3': 'Sac3', 'MEX67': 'Mex67', 'GFD1': 'Gfd1',
               'DBP5': 'Dbp5', 'PAB1': 'Pab1'}   # Nab4 = Hrp1p


def _class_scores(df):
    m = pd.DataFrame({b: df[[c for c in df.columns if str(c).startswith(pre)]].mean(axis=1)
                      for b, pre in BAIT_PREFIX.items()})
    mx = m.max(axis=1).replace(0, np.nan)
    return m.div(mx, axis=0).dropna(how='all').mean()


_cod = pd.read_excel(NC_DIR + "all_codingRNAs_WTPOs_and_Mutants.xlsx", sheet_name=0)
_nc = pd.read_excel(NC_DIR + "all_noncodingRNAs_WTPOs_and_Mutants.xlsx", sheet_name=0)
_g = _nc['gene'].astype(str)
_sno = _cod['gene'].astype(str).str.match(r'^(snR\d+|SNR\d+|LSR1|NME1|RPR1|SCR1)', case=False).values
_cls = {'mRNA': _class_scores(_cod[~_sno]),
        'lncRNA': _class_scores(_nc[_g.str.startswith('lncRNA').values]),
        'CUT': _class_scores(_nc[_g.str.match(r'^CUT').values]),
        'SUT': _class_scores(_nc[_g.str.match(r'^SUT').values]),
        'XUT': _class_scores(_nc[_g.str.match(r'^XUT').values]),
        'snRNA/snoRNA': _class_scores(_cod[_sno])}
NC_SCORES = {b: {k: float(v[b]) for k, v in _cls.items()} for b in BAIT_PREFIX}
print("RNA-class mean normalized scores:")
print(pd.DataFrame(NC_SCORES).T.round(3).to_string())

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
fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    height_ratios=[1.4, 1.4],
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
ax_a.bar(x_a - width/2, b03, width, color=colors_a, edgecolor='gray',
         linewidth=0.3, label='Score > 0.3 (permissive)')
ax_a.bar(x_a + width/2, b05, width, color='none', edgecolor=colors_a,
         linewidth=1.2, label='Score > 0.5 (stringent)')
ax_a.set_xticks(x_a)
ax_a.set_xticklabels(fmt_baits(avail_b), fontsize=7, rotation=55, ha='right')
for tick, b in zip(ax_a.get_xticklabels(), avail_b):
    tick.set_color('black')  # black bait labels, as in the assembled figure
ax_a.set_ylabel('% of coding ORFs (n = 2,452)', fontsize=8)
ax_a.set_title('Transcript co-purification breadth per bait\n'
               '(% of ORFs above score threshold)', fontsize=9)
ax_a.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
ax_a.spines['top'].set_visible(False); ax_a.spines['right'].set_visible(False)
ax_a.text(-0.12, 1.07, 'A', transform=ax_a.transAxes, fontsize=14, fontweight='bold')

# ── Panel E (variable ax_b): Spearman ρ (length preference) ───────────────────
ax_b = fig.add_subplot(gs[0, 1])
avail_rho = [b for b in rna_baits if b in SPEARMAN_RHO]
sorted_b = sorted(avail_rho, key=lambda b: SPEARMAN_RHO[b])
rhos = [SPEARMAN_RHO[b] for b in sorted_b]
colors_b = [group_color(b) for b in sorted_b]
y_b = np.arange(len(sorted_b))
ax_b.barh(y_b, rhos, color=colors_b, edgecolor='gray', linewidth=0.3, height=0.65)
ax_b.axvline(0, color='#333333', linewidth=0.9)
ax_b.set_yticks(y_b)
ax_b.set_yticklabels(fmt_baits(sorted_b), fontsize=8)
for tick, b in zip(ax_b.get_yticklabels(), sorted_b):
    tick.set_color('black')  # black bait labels, as in the assembled figure
ax_b.set_xlabel('Spearman ρ (gene length vs. co-purification score)', fontsize=8)
ax_b.set_title('Transcript-length preference per bait\n'
               '(negative = enrichment on shorter transcripts)', fontsize=9)
ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)
ax_b.text(-0.14, 1.07, 'E', transform=ax_b.transAxes, fontsize=14, fontweight='bold')

# ── Panel F (variable ax_c): Length quartile profiles ─────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
_e = [int(round(v)) for v in Q_EDGES]
x_labels = [f'Q1\n(<{_e[0]} nt)', f'Q2\n({_e[0]}–{_e[1]} nt)', f'Q3\n({_e[1]}–{_e[2]} nt)', f'Q4\n(>{_e[2]} nt)']
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
ax_c.set_ylabel('Mean normalized co-purification score', fontsize=8)
ax_c.set_ylim(0.0, 0.55)
ax_c.set_title('Transcript-length quartile profiles\n'
               '(mean score per length quartile, grouped by stage)', fontsize=9)
ax_c.legend(fontsize=7, loc='upper right', framealpha=0.9)
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)
ax_c.text(-0.12, 1.07, 'F', transform=ax_c.transAxes, fontsize=14, fontweight='bold')

# (GO-slim panels C, D: go_slim_enrichment.py. The former RNA-class panel is not part of the final figure;
#  RNA classes are shown in Figure 2F, make_figure2F_class_enrichment.py.)

# ── Panel G (variable ax_f): Dwell score by peak stage ───────────────────────
# Stages and their baits (as used for rna_dwell_scores.tsv; reproduces dwell_auc
# and peak_stage exactly): CBC = Cbp80p; THO/TREX = Sub2p, Thp2p; Splicing = Snu71p;
# hnRNP = Nab2p, Hrp1p, Npl3p, Pab1p, Gbp2p; 3' proc. = Clp1p, Syc1p, Ref2p;
# TREX-2/NPC = Sac3p, Gfd1p; Export = Mex67p, Dbp5p, Yra1p; Surveillance = Rrp6p.
ax_f = fig.add_subplot(gs[1, 1])
df_dl = df_dwell[['dwell_auc', 'peak_stage']].dropna()
STAGE_ORDER = ['CBC', 'THO/TREX', 'Splicing', 'hnRNP', "3' proc.",
               'TREX-2/NPC', 'Export', 'Surveillance']
STAGE_COLORS = {
    'CBC': GROUP_COLORS['CBC'],
    'THO/TREX': GROUP_COLORS['Co-Txp/THO/TREX'],
    'Splicing': GROUP_COLORS['Splicing'],
    'hnRNP': '#78B8EF',   # hnRNP/poly(A) (same blue as transcript cluster C2)
    "3' proc.": GROUP_COLORS["3'end processing"],
    'TREX-2/NPC': GROUP_COLORS['TREX-2/NPC'],
    'Export': GROUP_COLORS['Export/pore'],
    'Surveillance': GROUP_COLORS['Surveillance'],
}
assert set(df_dl['peak_stage']) <= set(STAGE_ORDER), set(df_dl['peak_stage'])
stages = [s for s in STAGE_ORDER if (df_dl['peak_stage'] == s).any()]
vdata = [df_dl.loc[df_dl['peak_stage'] == s, 'dwell_auc'].values for s in stages]
parts = ax_f.violinplot(vdata, positions=range(len(stages)),
                        showmedians=True, showextrema=False, widths=0.7)
for pc, s in zip(parts['bodies'], stages):
    pc.set_facecolor(STAGE_COLORS[s]); pc.set_alpha(0.7); pc.set_edgecolor('gray')
parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.5)
ax_f.set_xticks(range(len(stages)))
ax_f.set_xticklabels([f"{s.replace(chr(39), chr(8242))}\n(n = {len(v)})" for s, v in zip(stages, vdata)],
                     fontsize=7.5, rotation=40, ha='right')
ax_f.set_ylabel('Dwell score (AUC)', fontsize=8)
ax_f.set_title('Nuclear processing dwell score by peak stage\n'
               '(stages in pathway order)', fontsize=9)
ax_f.spines['top'].set_visible(False); ax_f.spines['right'].set_visible(False)
ax_f.text(-0.14, 1.07, 'G', transform=ax_f.transAxes, fontsize=14, fontweight='bold')

fig.suptitle('Supplemental Figure 4, panels A, E, F, G  |  AP-RNA-seq deep characterization\n'
             '(breadth, length preference, length quartiles, dwell score by peak stage)',
             fontsize=11, fontweight='bold', y=0.975)

out = OUTDIR + "figureS_RNAseq_analysis.pdf"
fig.savefig(out, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"Saved: {out}")

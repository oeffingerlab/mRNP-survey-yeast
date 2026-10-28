"""
Figure 1E (formerly Figure 3A) — within- vs between-module co-purification
correlation per bait, combined AP-MS + AP-RNA-seq, 18 shared baits.

Reproduces `Fig3A_within_between_18baits.pdf` (built inline 2026-10-07 from
Option 3 of make_fig3a_options.py) as a saved, reproducible script, and
reports the statistic quoted in Results para 47 / Methods 159 / Fig 1E legend.

Method
  AP-MS    data/apms_nuclear_matrix.tsv (380 nuclear preys; MTR2 dropped)
  RNA-seq  data/rnaseq_bait_means.tsv, transcripts with max enrichment > 0.3
           across the 18 baits (n = 2,434)
  Combined bait–bait correlation = mean of the AP-MS Pearson r and the
           AP-RNA-seq Pearson r (18 shared baits), as in make_figureS3.py.
  Modules  four functional modules assigned by pathway role (CLUSTER_4; same
           assignment as make_figureS3.py / Supplemental Figure 6).
  Per bait mean combined r with the other members of its module (within) and
           with all baits of other modules (between).
  Test     one-sided Mann–Whitney U, within- > between-module correlation over
           all bait PAIRS (34 within-module pairs vs 119 between-module pairs).
           Expected: p = 0.016 (within mean 0.260, between mean 0.156).

Output   output/figure1E_within_between.pdf (+ .png preview), panel letter "E"
Usage    python3 make_figure1E_within_between.py [--letter E]
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS, fmt_bait,
                          PATHWAY_ORDER_RNA, PATHWAY_ORDER_APMS)
apply_style()

DATADIR = os.path.join(HERE, "data")
OUTDIR = os.path.join(HERE, "output")
os.makedirs(OUTDIR, exist_ok=True)
LETTER = sys.argv[sys.argv.index("--letter") + 1] if "--letter" in sys.argv else "E"

# ── Data: combined bait–bait correlation (identical to make_figureS3.py) ───
df_apms = pd.read_csv(os.path.join(DATADIR, "apms_nuclear_matrix.tsv"), sep='\t', index_col=0)
df_apms = df_apms.drop(columns=['MTR2'], errors='ignore')
df_rna = pd.read_csv(os.path.join(DATADIR, "rnaseq_bait_means.tsv"), sep='\t', index_col=0)
rna_baits = [b for b in PATHWAY_ORDER_RNA if b in df_rna.columns]
apms_baits = [b for b in PATHWAY_ORDER_APMS if b in df_apms.columns]
shared = [b for b in rna_baits if b in apms_baits]
df_rna_filt = df_rna[rna_baits]
df_rna_filt = df_rna_filt[df_rna_filt.max(axis=1) > 0.3]
corr_apms = df_apms[shared].corr(method='pearson').loc[shared, shared]
corr_rna = df_rna_filt[shared].corr(method='pearson').loc[shared, shared]
corr_comb = (corr_apms + corr_rna) / 2.0

# ── Functional modules (pathway-assigned; same as make_figureS3.py) ────────
CLUSTER_4 = {
    'CBP80': 1, 'THP2': 1, 'YRA1': 1,
    'SNU71': 2, 'SYC1': 2, 'REF2': 2, 'CLP1': 2, 'HRP1': 2,
    'NAB2': 3, 'NPL3': 3, 'GBP2': 3, 'PAB1': 3, 'GFD1': 3, 'RRP6': 3,
    'SAC3': 4, 'MEX67': 4, 'DBP5': 4, 'SUB2': 4,
}
MODULE_NAMES = {1: 'CBC/THO/TREX', 2: "Splicing/3'end",
                3: 'hnRNP/Export adaptor', 4: 'Terminal export'}
MODULE_COLS = {1: '#DB2827', 2: '#465EAB', 3: '#56C4CB', 4: '#0A7F40'}

order = [b for k in (1, 2, 3, 4) for b in shared if CLUSTER_4[b] == k]
n_rna = len(df_rna_filt)

# ── Per-bait within / between ─────────────────────────────────────────────
within, between = [], []
for b in order:
    k = CLUSTER_4[b]
    same = [o for o in order if o != b and CLUSTER_4[o] == k]
    other = [o for o in order if CLUSTER_4[o] != k]
    within.append(corr_comb.loc[b, same].mean())
    between.append(corr_comb.loc[b, other].mean())

# ── Statistic over bait pairs ─────────────────────────────────────────────
pw, pb = [], []
for i, x in enumerate(order):
    for y in order[i + 1:]:
        (pw if CLUSTER_4[x] == CLUSTER_4[y] else pb).append(corr_comb.loc[x, y])
mw = mannwhitneyu(pw, pb, alternative='greater')

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12.9, 4.03))
fig.patch.set_facecolor('white')
x = np.arange(len(order))
bounds = np.cumsum([0] + [sum(CLUSTER_4[b] == k for b in order) for k in (1, 2, 3, 4)])
for i, k in enumerate((1, 2, 3, 4)):
    ax.axvspan(bounds[i] - 0.5, bounds[i + 1] - 0.5, color=MODULE_COLS[k], alpha=0.12, zorder=0)
    ax.text((bounds[i] + bounds[i + 1] - 1) / 2, 1.04, MODULE_NAMES[k],
            transform=ax.get_xaxis_transform(), ha='center', va='bottom',
            fontsize=7.5, fontweight='bold', color=MODULE_COLS[k])
ax.axhline(0, color='#DDDDDD', linewidth=0.8, zorder=1)
ax.plot(x, between, 's--', color='#AAAAAA', linewidth=1.2, markersize=5, zorder=2)
ax.plot(x, within, '-', color='#2E4057', linewidth=1.5, zorder=3)
for i, b in enumerate(order):
    ax.scatter(i, within[i], s=80, color=GROUP_COLORS.get(BAIT_GROUP_APMS.get(b, ''), '#CCCCCC'),
               edgecolors='white', linewidths=1.0, zorder=4)
ax.set_xticks(x)
ax.set_xticklabels([fmt_bait(b) for b in order], rotation=45, ha='right', fontsize=7)
ax.set_xlim(-0.8, len(order) - 0.5)
ax.set_ylim(-0.15, 1.05)
ax.set_ylabel('Mean Pearson r\n(combined AP-MS + AP-RNA-seq)', fontsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
handles = [Line2D([0], [0], color='#2E4057', marker='o', markersize=7, markerfacecolor='#2E4057',
                  label='Mean within-module r'),
           Line2D([0], [0], color='#AAAAAA', linestyle='--', marker='s', markersize=5,
                  label='Mean between-module r')]
ax.legend(handles=handles, loc='upper left', fontsize=7.5, frameon=True)
ax.text(-0.075, 1.06, LETTER, transform=ax.transAxes, fontsize=16, fontweight='bold')

fig.savefig(os.path.join(OUTDIR, "figure1E_within_between.pdf"), bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(OUTDIR, "figure1E_within_between.png"), dpi=150, bbox_inches='tight', facecolor='white')

# ── Report ────────────────────────────────────────────────────────────────
print(f"18 shared baits; {len(df_apms)} AP-MS nuclear preys; {n_rna} transcripts (max > 0.3)")
print(f"pairs: within n={len(pw)} mean r={np.mean(pw):.3f}; between n={len(pb)} mean r={np.mean(pb):.3f}")
print(f"one-sided Mann–Whitney U = {mw.statistic:.0f}, p = {mw.pvalue:.4f}")
for k in (1, 2, 3, 4):
    m = [i for i, b in enumerate(order) if CLUSTER_4[b] == k]
    print(f"  {MODULE_NAMES[k]:22s} within {np.mean([within[i] for i in m]):+.3f}  "
          f"between {np.mean([between[i] for i in m]):+.3f}")
for b, w, bt in zip(order, within, between):
    print(f"  {fmt_bait(b):7s} within {w:+.3f}  between {bt:+.3f}")

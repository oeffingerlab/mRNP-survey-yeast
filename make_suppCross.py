"""
make_suppCross.py
==================
Supplementary Figure: Cross-Dataset Concordance (AP-MS vs. AP-RNA-seq)

Panels:
  A  Bait-pair concordance scatter — AP-MS Pearson r vs. AP-RNA-seq Spearman ρ
  B  Cross-dataset bait-bait correlation heatmaps (side by side, same ordering)
  C  Concordance quadrant map — per-bait-pair functional interpretation
  D  Discordance map — pairs with opposite co-enrichment patterns

Output: figureS_CrossDataset.pdf
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS, BAIT_GROUP_RNA,
                           fmt_bait, fmt_baits, PATHWAY_ORDER_APMS, CMAP_APMS)
apply_style()

OUTDIR  = "output/"
DATADIR = "data/"

# ── Load data ────────────────────────────────────────────────────────────────
df_cc    = pd.read_csv(DATADIR + "cross_bait_corr.tsv",      sep='\t')
df_quad  = pd.read_csv(DATADIR + "concordance_quadrant.tsv", sep='\t')
df_apms  = pd.read_csv(DATADIR + "apms_nuclear_matrix.tsv",  sep='\t', index_col=0)
df_rna   = pd.read_csv(DATADIR + "rnaseq_bait_means.tsv",    sep='\t', index_col=0)

PATHWAY_ORDER_RNA = [
    'CBP80','THP2','YRA1','SNU71',
    'CLP1','SYC1','REF2','HRP1',
    'PAB1','NAB2','NPL3','GBP2','GFD1',
    'RRP6','SUB2','SAC3','MEX67','DBP5',
]

# Shared baits (in both datasets)
apms_baits = [b for b in PATHWAY_ORDER_APMS if b in df_apms.columns]
rna_baits  = [b for b in PATHWAY_ORDER_RNA  if b in df_rna.columns]
shared     = [b for b in rna_baits if b in apms_baits]

# ── Helpers ──────────────────────────────────────────────────────────────────
def grp_color_apms(b):
    return GROUP_COLORS.get(BAIT_GROUP_APMS.get(b, ''), '#CCCCCC')
def grp_color_rna(b):
    return GROUP_COLORS.get(BAIT_GROUP_RNA.get(b, ''), '#CCCCCC')

def bait_color_bar_h(ax, baits, group_fn):
    n = len(baits)
    for i, b in enumerate(baits):
        ax.add_patch(mpatches.Rectangle((i/n, 0), 1/n, 1,
            color=group_fn(b), transform=ax.transAxes, clip_on=False))

def bait_color_bar_v(ax, baits, group_fn):
    n = len(baits)
    for i, b in enumerate(baits):
        ax.add_patch(mpatches.Rectangle((0, (n-1-i)/n), 1, 1/n,
            color=group_fn(b), transform=ax.transAxes, clip_on=False))

def add_equal_strips(fig, ax, baits, fn_top, fn_right, strip_in=0.09, gap_in=0.02):
    """Add color strips above and right of ax with the SAME physical width (strip_in inches)."""
    pos  = ax.get_position()           # figure-fraction coords
    fw, fh = fig.get_size_inches()
    ax_w_in = pos.width  * fw          # actual axis width in inches
    ax_h_in = pos.height * fh          # actual axis height in inches
    top_frac   = strip_in / ax_h_in   # height fraction for top strip
    right_frac = strip_in / ax_w_in   # width  fraction for right strip
    gap_top    = gap_in   / ax_h_in
    gap_right  = gap_in   / ax_w_in
    ax_top = ax.inset_axes([0, 1 + gap_top, 1, top_frac])
    ax_top.axis('off')
    bait_color_bar_h(ax_top, baits, fn_top)
    ax_right = ax.inset_axes([1 + gap_right, 0, right_frac, 1])
    ax_right.axis('off')
    bait_color_bar_v(ax_right, baits, fn_right)
    return top_frac + gap_top, right_frac + gap_right   # space used (for colorbar placement)

# ════════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    height_ratios=[1, 1],
    hspace=0.04, wspace=0.12,
    left=0.08, right=0.96, top=0.93, bottom=0.06,
)

# ── Panel A: Bait-pair scatter (MS r vs RNA ρ) — 75% height+width of cell ────
gs_a = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 0],
                                         height_ratios=[3, 1], width_ratios=[3, 1],
                                         hspace=0, wspace=0)
ax_a = fig.add_subplot(gs_a[0, 0])

if 'ms_corr' in df_cc.columns and 'rna_spear' in df_cc.columns:
    ms_r   = df_cc['ms_corr'].values
    rna_rho = df_cc['rna_spear'].values
    # Color by functional group agreement
    pair_col = []
    for _, row in df_cc.iterrows():
        pair_str = row.get('pair', '')
        if '-' in str(pair_str):
            b1, b2 = pair_str.split('-', 1)
            g1 = BAIT_GROUP_RNA.get(b1, BAIT_GROUP_APMS.get(b1, ''))
            g2 = BAIT_GROUP_RNA.get(b2, BAIT_GROUP_APMS.get(b2, ''))
            pair_col.append('#3A6FA3' if g1 == g2 else '#AAAAAA')
        else:
            pair_col.append('#AAAAAA')
    ax_a.scatter(ms_r, rna_rho, c=pair_col, s=22, alpha=0.75,
                 linewidths=0.3, edgecolors='gray', zorder=3)
elif 'ms_r' in df_cc.columns and 'rna_r' in df_cc.columns:
    ms_r   = df_cc['ms_r'].fillna(0).values
    rna_rho = df_cc['rna_r'].fillna(0).values
    ax_a.scatter(ms_r, rna_rho, c='#5B8DB8', s=22, alpha=0.7,
                 linewidths=0.3, edgecolors='gray', zorder=3)
else:
    # Use concordance_quadrant for a scatter of MS vs RNA correlations
    if 'ms_r' in df_quad.columns and 'rna_r' in df_quad.columns:
        ms_r   = df_quad['ms_r'].fillna(0).values
        rna_rho = df_quad['rna_r'].fillna(0).values
        pair_col = ['#3A6FA3' if r1*r2 > 0 else '#C0392B'
                    for r1, r2 in zip(ms_r, rna_rho)]
        ax_a.scatter(ms_r, rna_rho, c=pair_col, s=22, alpha=0.7,
                     linewidths=0.3, edgecolors='gray', zorder=3)

ax_a.axhline(0, color='#555555', lw=0.8, ls='--')
ax_a.axvline(0, color='#555555', lw=0.8, ls='--')
ax_a.set_xlabel('AP-MS bait-pair Pearson r', fontsize=8)
ax_a.set_ylabel('AP-RNA-seq bait-pair correlation', fontsize=8)
ax_a.set_title('Cross-dataset bait-pair concordance\n'
               '(each point = one bait pair; blue = same complex)', fontsize=9)
ax_a.spines['top'].set_visible(False); ax_a.spines['right'].set_visible(False)
same_patch  = mpatches.Patch(color='#3A6FA3', label='Same functional group')
diff_patch  = mpatches.Patch(color='#AAAAAA', label='Different groups')
ax_a.legend(handles=[same_patch, diff_patch], fontsize=7.5, loc='lower right', framealpha=0.9)
ax_a.text(-0.12, 1.06, 'A', transform=ax_a.transAxes, fontsize=16, fontweight='bold')

# ── Panel B: Side-by-side bait correlation heatmaps ─────────────────────────
# Use shared baits, identical hierarchical ordering
df_apms_sh = df_apms[shared].loc[df_apms.index]
df_rna_sh  = df_rna[shared].dropna()

corr_ms  = df_apms[shared].corr(method='pearson').loc[shared, shared]
corr_rna = df_rna[shared].corr(method='spearman').loc[shared, shared]

# Cluster by average of both
avg_corr = (corr_ms.values + corr_rna.values) / 2
dist_sh  = pdist(avg_corr, metric='euclidean')
link_sh  = linkage(dist_sh, method='average')
sh_ord   = leaves_list(link_sh)
shared_c = [shared[i] for i in sh_ord]
corr_ms_c  = corr_ms.iloc[sh_ord, sh_ord]
corr_rna_c = corr_rna.iloc[sh_ord, sh_ord]

cmap_corr = plt.cm.RdBu_r
norm_corr = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=1.0)

gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1],
                                         wspace=0.12, hspace=0)
ax_b1 = fig.add_subplot(gs_b[0])
ax_b2 = fig.add_subplot(gs_b[1])

im_b1 = ax_b1.imshow(corr_ms_c.values, aspect='equal', cmap=cmap_corr, norm=norm_corr, rasterized=True)
n_sh = len(shared_c)
ax_b1.set_xticks(range(n_sh)); ax_b1.set_yticks(range(n_sh))
ax_b1.set_xticklabels(fmt_baits(shared_c), fontsize=6, rotation=60, ha='right')
ax_b1.set_yticklabels(fmt_baits(shared_c), fontsize=6)
# All tick labels black — color conveyed by the stage-color strip insets
for lbl in ax_b1.get_xticklabels(): lbl.set_color('black')
for lbl in ax_b1.get_yticklabels(): lbl.set_color('black')

im_b2 = ax_b2.imshow(corr_rna_c.values, aspect='equal', cmap=cmap_corr, norm=norm_corr, rasterized=True)
ax_b2.set_xticks(range(n_sh)); ax_b2.set_yticks(range(n_sh))
ax_b2.set_xticklabels(fmt_baits(shared_c), fontsize=6, rotation=60, ha='right')
ax_b2.set_yticklabels([])
for lbl in ax_b2.get_xticklabels(): lbl.set_color('black')

# ── Pathway-stage color bar strips — equal physical width on both axes ───────
_, b1_right_space = add_equal_strips(fig, ax_b1, shared_c, grp_color_apms, grp_color_apms)
_, b2_right_space = add_equal_strips(fig, ax_b2, shared_c, grp_color_rna,  grp_color_rna)

# Colorbar placed as cax inset further right of ax_b2 — does not resize any axis
# Position it beyond the right strip that was just added
cax_b = ax_b2.inset_axes([1 + b2_right_space + 0.06, 0.10, 0.05, 0.80])
cb_b = plt.colorbar(im_b2, cax=cax_b)
cb_b.ax.tick_params(labelsize=7)
cb_b.set_label('Correlation\ncoefficient', fontsize=7)

ax_b1.text(-0.18, 1.10, 'B', transform=ax_b1.transAxes, fontsize=16, fontweight='bold')
ax_b1.set_title('AP-MS Pearson r\n(shared baits, joint ordering)', fontsize=8.5)
ax_b2.set_title('AP-RNA-seq Spearman ρ\n(same ordering)', fontsize=8.5)

# ── Panel C: Concordance quadrant scatter plot — 75% height+width of cell ────
gs_c = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 0],
                                         height_ratios=[3, 1], width_ratios=[3, 1],
                                         hspace=0, wspace=0)
ax_c = fig.add_subplot(gs_c[0, 0])

# Color scheme: each quadrant a clearly distinct color
QUAD_COLORS = {
    'Both high':         '#1A5276',   # dark blue  — true functional complex
    'MS high, RNA low':  '#1E8449',   # dark green — stable complex, divergent RNA
    'RNA high, MS low':  '#922B21',   # dark red   — shared RNA, no stable complex
    'Both low':          '#AAAAAA',   # gray       — unrelated
}
QUAD_LABELS = {
    'Both high':         'Both high (n={n})\ntrue functional complex',
    'MS high, RNA low':  'MS high / RNA low (n={n})\nstable complex, divergent RNA',
    'RNA high, MS low':  'RNA high / MS low (n={n})\nshared RNA, no stable complex',
    'Both low':          'Both low (n={n})\nunrelated',
}

def _quad_key(q_str):
    """Return the matching QUAD_COLORS key for a quad string (handles embedded newlines)."""
    for key in QUAD_COLORS:
        if key.lower() in str(q_str).lower():
            return key
    return 'Both low'

if 'ms_r' in df_quad.columns and 'rna_r' in df_quad.columns:
    df_q = df_quad.dropna(subset=['ms_r', 'rna_r']).copy()

    # Quadrant thresholds: the same cut-offs that define the quad column in
    # concordance_quadrant.tsv (Methods): AP-MS r > 0.35, AP-RNA-seq rho > 0.50
    ms_thr, rna_thr = 0.35, 0.50

    # Plot each quadrant group separately for legend
    leg_handles = []
    if 'quad' in df_q.columns:
        for key, col in QUAD_COLORS.items():
            mask = df_q['quad'].apply(_quad_key) == key
            grp  = df_q[mask]
            n    = len(grp)
            if n == 0:
                continue
            ax_c.scatter(grp['ms_r'], grp['rna_r'],
                         c=col, s=28, alpha=0.82,
                         linewidths=0.4, edgecolors='gray', zorder=4)
            lbl = QUAD_LABELS[key].format(n=n)
            leg_handles.append(mpatches.Patch(color=col, label=lbl))
    else:
        ax_c.scatter(df_q['ms_r'], df_q['rna_r'],
                     c='#5B8DB8', s=28, alpha=0.8,
                     linewidths=0.4, edgecolors='gray', zorder=4)

    # Quadrant dividing lines
    ax_c.axvline(ms_thr,  color='#888888', lw=0.9, ls='--', zorder=2)
    ax_c.axhline(rna_thr, color='#888888', lw=0.9, ls='--', zorder=2)
    ax_c.axhline(0,       color='#CCCCCC', lw=0.5, ls='-',  zorder=1)

    # Quadrant corner text annotations
    x_lo = df_q['ms_r'].min()
    x_hi = df_q['ms_r'].max()
    y_lo = df_q['rna_r'].min()
    y_hi = df_q['rna_r'].max()
    # Quadrant names placed on either side of the dashed lines (above the top points
    # and just below the horizontal line) so they do not overlap the pair labels
    y_top = y_hi + 0.07
    ax_c.text(ms_thr + 0.01, y_top, 'Both high',         color='#1A5276', fontsize=6.5, va='top', ha='left')
    ax_c.text(ms_thr - 0.01, y_top, 'RNA high / MS low', color='#922B21', fontsize=6.5, va='top', ha='right')
    ax_c.text(ms_thr + 0.01, rna_thr - 0.01,
              'MS high / RNA low', color='#1E8449', fontsize=6.5, va='top', ha='left')
    ax_c.text(ms_thr - 0.01, rna_thr - 0.01,
              'Both low',         color='#666666', fontsize=6.5, va='top', ha='right')

    # Label notable bait pairs (all "Both high"; top 3 "RNA high/MS low" by rna_r)
    def _label_pair(row, color, kw=None):
        kw = kw or {}
        a  = fmt_bait(str(row['bait_A']))
        b  = fmt_bait(str(row['bait_B']))
        ax_c.annotate(f"{a}–{b}",
                      (row['ms_r'], row['rna_r']),
                      textcoords='offset points', xytext=(4, 2),
                      fontsize=5.5, color=color, zorder=6, **kw)

    if 'quad' in df_q.columns and 'bait_A' in df_q.columns:
        # All "Both high" pairs
        for _, row in df_q[df_q['quad'].apply(_quad_key) == 'Both high'].iterrows():
            _label_pair(row, '#1A5276')
        # Top 3 "RNA high / MS low" by rna_r
        rna_high = df_q[df_q['quad'].apply(_quad_key) == 'RNA high, MS low']
        for _, row in rna_high.nlargest(3, 'rna_r').iterrows():
            _label_pair(row, '#922B21')

    # Legend (outside right)
    if leg_handles:
        ax_c.legend(handles=leg_handles, fontsize=7, loc='lower right',
                    framealpha=0.9, title='Concordance quadrant', title_fontsize=7)

    ax_c.set_xlabel('AP-MS bait-pair Pearson r', fontsize=8)
    ax_c.set_ylabel('AP-RNA-seq bait-pair Spearman ρ', fontsize=8)
    ax_c.set_xlim(x_lo - 0.02, x_hi + 0.04)
    ax_c.set_ylim(y_lo - 0.05, y_hi + 0.08)
else:
    ax_c.text(0.5, 0.5, 'Concordance quadrant data\nnot in expected format',
              ha='center', va='center', transform=ax_c.transAxes)
    ax_c.axis('off')

ax_c.set_title('Concordance quadrant scatter\n'
               '(each point = one bait pair; dashed lines: AP-MS r = 0.35, AP-RNA-seq ρ = 0.50)',
               fontsize=9)
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)
ax_c.text(-0.15, 1.06, 'C', transform=ax_c.transAxes, fontsize=16, fontweight='bold')

# ── Panel D: Discordance map ─────────────────────────────────────────────────
# Use nested GridSpec so D occupies only the left half of gs[1,1],
# matching the width of B1 or B2 (each half of gs[0,1]).
# The right half is spacer/colorbar overflow area.
gs_d = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.12)
ax_d = fig.add_subplot(gs_d[0])
# For shared baits: show difference (MS corr - RNA corr) as heatmap
diff_mat = corr_ms_c.values - corr_rna_c.values
np.fill_diagonal(diff_mat, 0)
norm_d = TwoSlopeNorm(vmin=-0.8, vcenter=0, vmax=0.8)
im_d = ax_d.imshow(diff_mat, aspect='equal', cmap='RdBu_r', norm=norm_d, rasterized=True)
ax_d.set_xticks(range(n_sh)); ax_d.set_yticks(range(n_sh))
ax_d.set_xticklabels(fmt_baits(shared_c), fontsize=6, rotation=60, ha='right')
ax_d.set_yticklabels(fmt_baits(shared_c), fontsize=6)
# Tick labels all black — color conveyed by the stage-color strip insets
for lbl in ax_d.get_xticklabels(): lbl.set_color('black')
for lbl in ax_d.get_yticklabels(): lbl.set_color('black')

# Pathway-stage color bar strips — equal physical width
_, d_right_space = add_equal_strips(fig, ax_d, shared_c, grp_color_rna, grp_color_rna)

# Colorbar as cax inset — does not resize ax_d
cax_d = ax_d.inset_axes([1 + d_right_space + 0.06, 0.10, 0.05, 0.80])
cb_d = plt.colorbar(im_d, cax=cax_d)
cb_d.ax.tick_params(labelsize=7)
cb_d.set_label('Δ correlation\n(AP-MS − RNA-seq)\nred = MS-high\nblue = RNA-high', fontsize=7)

ax_d.set_title('Cross-dataset discordance map\n'
               '(Δ correlation: positive = higher in AP-MS, negative = higher in AP-RNA-seq)',
               fontsize=9)
ax_d.text(-0.16, 1.06, 'D', transform=ax_d.transAxes, fontsize=16, fontweight='bold')

fig.suptitle('Supplementary Figure  |  Cross-Dataset Concordance: AP-MS vs. AP-RNA-seq\n'
             '(Pearson r in AP-MS vs. Spearman ρ in AP-RNA-seq, shared baits)',
             fontsize=11, fontweight='bold', y=0.975)

out = OUTDIR + "figureS_CrossDataset.pdf"
fig.savefig(out, bbox_inches='tight', facecolor='white', dpi=150)
# Panel C alone, for placement in the assembled figure
_bb = ax_c.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
fig.savefig(OUTDIR + "figureS_CrossDataset_panelC.pdf", bbox_inches=_bb.expanded(1.03, 1.03),
            facecolor='white')
plt.close()
print(f"Saved: {out}")

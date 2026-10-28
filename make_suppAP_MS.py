"""
make_suppAP_MS.py
=================
Supplementary Figure: AP-MS Dataset Characterization

Panels:
  A  Nuclear-filtered prey enrichment heatmap (baits × preys, clustered)
  B  Bait–bait Pearson correlation heatmap
  C  Prey universality — histogram of #baits each prey appears in
  D  Specificity scatter — total spectral counts vs. number of baits
  E  Functional-group signal composition per bait (stacked bar)
  F  Non-bait mRNP proteins recovered as AP-MS preys

Output: figureS_APMS_analysis.pdf
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS,
                           fmt_bait, fmt_baits,
                           CMAP_APMS, PATHWAY_ORDER_APMS)
apply_style()

OUTDIR  = "output/"
DATADIR = "data/"

# ── Load data ────────────────────────────────────────────────────────────────
df_raw   = pd.read_csv(DATADIR + "apms_nuclear_matrix.tsv",   sep='\t', index_col=0)
df_nbp   = pd.read_csv(DATADIR + "non_bait_prey_profiles.tsv", sep='\t', index_col=0)
df_full  = pd.read_csv(DATADIR + "apms_full_matrix.tsv",      sep='\t', index_col=0)

# Bait order from figure_style (pathway stage order)
apms_baits = [b for b in PATHWAY_ORDER_APMS if b in df_raw.columns]
df_nuc = df_raw[apms_baits].copy()

# Binary presence (SC ≥ 2 in ≥1 bait)
df_bin = (df_nuc >= 2).astype(int)
# Prey universality: number of baits each prey appears in
prey_nbait = df_bin.sum(axis=1)

# ── Color helpers ────────────────────────────────────────────────────────────
def group_color(bait):
    return GROUP_COLORS.get(BAIT_GROUP_APMS.get(bait, ''), '#CCCCCC')

def bait_color_bar_h(ax, baits):
    n = len(baits)
    for i, b in enumerate(baits):
        ax.add_patch(mpatches.Rectangle((i/n, 0), 1/n, 1,
            color=group_color(b), transform=ax.transAxes, clip_on=False))

def bait_color_bar_v(ax, baits):
    n = len(baits)
    for i, b in enumerate(baits):
        ax.add_patch(mpatches.Rectangle((0, (n-1-i)/n), 1, 1/n,
            color=group_color(b), transform=ax.transAxes, clip_on=False))

# ── Hierarchical clustering of preys (for heatmap) ──────────────────────────
# Log-scale enrichment for heatmap: log2(SC+1), normalized per prey to max=1
df_log = np.log2(df_nuc + 1)
prey_max = df_log.max(axis=1)
prey_max[prey_max == 0] = 1
df_norm = df_log.div(prey_max, axis=0)

# Filter to preys detected in ≥2 baits
df_plot = df_norm[prey_nbait >= 2].copy()
# Cluster preys
dist_p = pdist(df_plot.values, metric='correlation')
dist_p = np.nan_to_num(dist_p, nan=1.0)
link_p = linkage(dist_p, method='average')
prey_ord = leaves_list(link_p)
df_plot = df_plot.iloc[prey_ord]

# ════════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(24, 26))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    height_ratios=[2.2, 1.4, 1.4],
    width_ratios=[2.0, 1.1, 1.1],
    hspace=0.42, wspace=0.38,
    left=0.07, right=0.96, top=0.93, bottom=0.06,
)

# ── Panel A: Prey heatmap ────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, :])
im_a = ax_a.imshow(df_plot.values.T, aspect='auto', cmap=CMAP_APMS,
                   vmin=0, vmax=1.0, interpolation='nearest')
ax_a.set_title('AP-MS nuclear-filtered prey enrichment  '
               f'(n = {len(df_plot)} preys detected in ≥2 baits)',
               fontsize=10, fontweight='bold')
ax_a.set_yticks(range(len(apms_baits)))
ax_a.set_yticklabels(fmt_baits(apms_baits), fontsize=8)
ax_a.set_xticks([])
ax_a.set_ylabel('Bait (pathway order)', fontsize=8)
ax_a.set_xlabel(f'Preys  (hierarchical clustering, Pearson / average linkage)',
                fontsize=8)
# Color bars
ax_a_right = ax_a.inset_axes([1.002, 0, 0.010, 1])
ax_a_right.axis('off'); bait_color_bar_v(ax_a_right, apms_baits)
cax_a = ax_a.inset_axes([1.025, 0.40, 0.012, 0.22])
cb = plt.colorbar(im_a, cax=cax_a, label='Enrichment\n(log2 SC+1, max=1)')
cb.ax.tick_params(labelsize=7); cb.set_label('Enrichment\n(log2 SC+1, max=1)', fontsize=7)
ax_a.text(-0.02, 1.03, 'A', transform=ax_a.transAxes, fontsize=16, fontweight='bold')

# ── Panel B: Bait–bait correlation ──────────────────────────────────────────
ax_b = fig.add_subplot(gs[1, 0])
corr = df_nuc.corr(method='pearson')
corr = corr.loc[apms_baits, apms_baits]
# Cluster baits
dist_b = pdist(corr.values, metric='euclidean')
link_b = linkage(dist_b, method='average')
b_ord  = leaves_list(link_b)
corr_c = corr.iloc[b_ord, b_ord]
ordered_baits_b = [apms_baits[i] for i in b_ord]

from matplotlib.colors import TwoSlopeNorm
cmap_corr = plt.cm.RdBu_r
norm_corr = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=1.0)
im_b = ax_b.imshow(corr_c.values, aspect='equal', cmap=cmap_corr, norm=norm_corr,
                   interpolation='nearest')
n_b = len(ordered_baits_b)
ax_b.set_xticks(range(n_b))
ax_b.set_yticks(range(n_b))
ax_b.set_xticklabels(fmt_baits(ordered_baits_b), fontsize=8, rotation=60, ha='right')
ax_b.set_yticklabels(fmt_baits(ordered_baits_b), fontsize=8)
for i, b in enumerate(ordered_baits_b):
    ax_b.get_xticklabels()[i].set_color('black')
    ax_b.get_yticklabels()[i].set_color('black')
plt.colorbar(im_b, ax=ax_b, shrink=0.5, pad=0.10, label='Pearson r')
ax_b.set_title('Bait–bait Pearson correlation\n(nuclear-filtered prey profiles)',
               fontsize=9)
# Stage color bars
ax_b_top = ax_b.inset_axes([0, 1.005, 1, 0.035])
ax_b_top.axis('off'); bait_color_bar_h(ax_b_top, ordered_baits_b)
ax_b_lft = ax_b.inset_axes([1.0 + 0.01, 0, 0.035, 1])
ax_b_lft.axis('off'); bait_color_bar_v(ax_b_lft, ordered_baits_b)
ax_b.text(-0.10, 1.10, 'B', transform=ax_b.transAxes, fontsize=16, fontweight='bold')
# Functional-group legend — right of Panel B
_leg_group_order = list(GROUP_COLORS.keys())
_leg_grp_handles = ([mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in _leg_group_order]
                    + [mpatches.Patch(color='#CCCCCC', label='Other/unknown')])
ax_b.legend(handles=_leg_grp_handles, bbox_to_anchor=(1.4, 1.0), loc='upper left',
            ncol=1, fontsize=9, title='Functional group', title_fontsize=9.5,
            frameon=True, handlelength=1.2, labelspacing=0.4, borderpad=0.5)

# ── Panel C: Prey universality histogram ─────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 1])
counts = prey_nbait.value_counts().sort_index()
ax_c.bar(counts.index, counts.values, color='#5B8DB8', edgecolor='gray',
         linewidth=0.4, width=0.8)
ax_c.set_xlabel('Number of baits prey detected in', fontsize=9)
ax_c.set_ylabel('Number of preys', fontsize=9)
ax_c.set_title('Prey universality\n(nuclear-filtered, SC ≥ 2)', fontsize=9)
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)
# Annotate cumulative fractions
total_preys = len(prey_nbait)
for nbait, cnt in counts.items():
    cum = (prey_nbait >= nbait).sum()
    if cnt > 3:
        ax_c.text(nbait, cnt + 1, str(cnt), ha='center', fontsize=7, color='#333333')
ax_c.text(0.97, 0.95,
          f'Total unique preys: {total_preys}\n'
          f'In ≥2 baits: {(prey_nbait >= 2).sum()}\n'
          f'In ≥5 baits: {(prey_nbait >= 5).sum()}',
          ha='right', va='top', fontsize=7.5,
          transform=ax_c.transAxes, bbox=dict(boxstyle='round,pad=0.3',
          facecolor='#F5F5F5', edgecolor='#CCCCCC', alpha=0.9))
ax_c.text(-0.15, 1.06, 'C', transform=ax_c.transAxes, fontsize=16, fontweight='bold')

# ── Panel D: Specificity scatter (total SC vs #baits) ───────────────────────
ax_d = fig.add_subplot(gs[1, 2])
total_sc = df_nuc.sum(axis=1)
# Color by group membership (use most-enriched bait's group)
top_bait = df_nuc.idxmax(axis=1)
dot_colors = [GROUP_COLORS.get(BAIT_GROUP_APMS.get(b, ''), '#BBBBBB') for b in top_bait]
np.random.seed(0)  # fixed seed: reproducible x-jitter in Supp Fig 2D
sc = ax_d.scatter(prey_nbait + np.random.uniform(-0.2, 0.2, len(prey_nbait)),
                  np.log10(total_sc + 1),
                  c=dot_colors, s=22, alpha=0.75, linewidths=0.3,
                  edgecolors='gray', rasterized=True)
ax_d.set_xlabel('Number of baits prey detected in', fontsize=9)
ax_d.set_ylabel('Total spectral counts  [log₁₀(SC+1)]', fontsize=9)
ax_d.set_title('Prey specificity vs. abundance\n(color = group of top-enriching bait)',
               fontsize=9)
ax_d.spines['top'].set_visible(False); ax_d.spines['right'].set_visible(False)
# Legend patches
seen = set()
leg_handles = []
for b, c in zip(top_bait, dot_colors):
    grp = BAIT_GROUP_APMS.get(b, '')
    col = GROUP_COLORS.get(grp, '#BBBBBB')
    if grp and grp not in seen:
        leg_handles.append(mpatches.Patch(color=col, label=grp))
        seen.add(grp)
ax_d.legend(handles=leg_handles, fontsize=8, loc='upper left',
            bbox_to_anchor=(1.01, 1.0),
            framealpha=0.9, handlelength=1.0, title='Functional group',
            title_fontsize=8.5)
ax_d.text(-0.15, 1.06, 'D', transform=ax_d.transAxes, fontsize=16, fontweight='bold')

# ── Panel E: Signal composition (stacked bar) ────────────────────────────────
ax_e = fig.add_subplot(gs[2, :2])
# For each bait: fraction of total SC from each functional group
group_order = list(GROUP_COLORS.keys())
comp_data = {}
for b in apms_baits:
    col_data = {}
    for grp in group_order:
        preys_in_grp = [p for p in df_nuc.index
                        if BAIT_GROUP_APMS.get(p.upper(), '') == grp]
        col_data[grp] = df_nuc.loc[df_nuc.index.isin(preys_in_grp), b].sum()
    col_data['Other'] = df_nuc.loc[
        ~df_nuc.index.isin([p for grp in group_order
                             for p in df_nuc.index
                             if BAIT_GROUP_APMS.get(p.upper(), '') == grp]), b].sum()
    total = sum(col_data.values())
    if total > 0:
        comp_data[b] = {k: v/total for k, v in col_data.items()}
    else:
        comp_data[b] = {k: 0 for k in list(col_data.keys())}

x_e = np.arange(len(apms_baits))
bottom_e = np.zeros(len(apms_baits))
for grp in group_order:
    vals = np.array([comp_data[b].get(grp, 0) for b in apms_baits])
    ax_e.bar(x_e, vals, bottom=bottom_e, width=1.0,
             color=GROUP_COLORS[grp], edgecolor='gray', linewidth=0.2,
             label=grp)
    bottom_e += vals
# 'Other' in gray
other_vals = np.array([comp_data[b].get('Other', 0) for b in apms_baits])
ax_e.bar(x_e, other_vals, bottom=bottom_e, width=1.0,
         color='#CCCCCC', edgecolor='gray', linewidth=0.2, label='Other/unknown')

ax_e.set_xticks(x_e)
ax_e.set_xticklabels(fmt_baits(apms_baits), fontsize=8.5, rotation=60, ha='right')
ax_e.set_ylabel('Fraction of total spectral counts', fontsize=9)
ax_e.set_ylim(0, 1.0)
ax_e.set_xlim(-0.5, len(apms_baits) - 0.5)
ax_e.set_title('Functional-group signal composition per bait\n'
               '(fraction of total spectral counts from each mRNP complex class)',
               fontsize=9)
ax_e.spines['top'].set_visible(False); ax_e.spines['right'].set_visible(False)
ax_e_top = ax_e.inset_axes([0, 1.01, 1, 0.030])
ax_e_top.axis('off'); bait_color_bar_h(ax_e_top, apms_baits)
# Legend to right
grp_handles = ([mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in group_order] +
               [mpatches.Patch(color='#CCCCCC', label='Other/unknown')])
# functional-group legend placed between A and B via fig.legend below
ax_e.text(-0.04, 1.10, 'E', transform=ax_e.transAxes, fontsize=16, fontweight='bold')

# ── Functional-group legend — between Panel A (row 0) and Panel B (row 1) ──────
# GridSpec: height_ratios=[2.2,1.4,1.4], hspace=0.42, top=0.93, bottom=0.06
# Gap between rows 0 and 1 is centered at ~y=0.59 in figure fraction
# (functional-group legend now on ax_b — see above)

# ── Panel F: Non-bait prey recovery ─────────────────────────────────────────
ax_f = fig.add_subplot(gs[2, 2])
# df_nbp has bait-proteins-as-preys: rows = preys that are also baits, cols = bait pulldowns
nbp_baits = [b for b in apms_baits if b in df_nbp.index]
nbp_cols  = [b for b in apms_baits if b in df_nbp.columns]

if len(nbp_baits) > 0 and len(nbp_cols) > 0:
    df_nbp_sub = df_nbp.loc[nbp_baits, nbp_cols]
    # Convert to binary (detected = SC ≥ 2)
    df_nbp_bin = (df_nbp_sub >= 2).astype(int)
    # For each bait-as-prey, count how many pulldowns it appears in
    prey_counts = df_nbp_bin.sum(axis=1).sort_values(ascending=True)
    y_f = np.arange(len(prey_counts))
    ax_f.barh(y_f, prey_counts.values,
              color=[group_color(b) for b in prey_counts.index],
              edgecolor='gray', linewidth=0.4, height=0.7)
    ax_f.set_yticks(y_f)
    ax_f.set_yticklabels(fmt_baits(list(prey_counts.index)), fontsize=7.5)
    for tick, b in zip(ax_f.get_yticklabels(), prey_counts.index):
        tick.set_color('black')
    ax_f.set_xlabel('Number of bait pulldowns detected in', fontsize=8)
    ax_f.set_title('mRNP bait proteins recovered\nas preys (cross-pulldown)',
                   fontsize=9)
    ax_f.spines['top'].set_visible(False); ax_f.spines['right'].set_visible(False)
    ax_f.set_xlim(0, len(nbp_cols) + 1)
else:
    # Fallback: show prey count per bait as simple bar
    n_preys = (df_nuc >= 2).sum(axis=0)
    n_preys_ord = pd.Series([n_preys.get(b, 0) for b in apms_baits], index=apms_baits)
    n_preys_ord = n_preys_ord.sort_values(ascending=True)
    y_f = np.arange(len(n_preys_ord))
    ax_f.barh(y_f, n_preys_ord.values,
              color=[group_color(b) for b in n_preys_ord.index],
              edgecolor='gray', linewidth=0.4, height=0.7)
    ax_f.set_yticks(y_f)
    ax_f.set_yticklabels(fmt_baits(list(n_preys_ord.index)), fontsize=7.5)
    for tick, b in zip(ax_f.get_yticklabels(), n_preys_ord.index):
        tick.set_color('black')
    ax_f.set_xlabel('Number of unique preys (SC ≥ 2)', fontsize=8)
    ax_f.set_title('Prey count per bait\n(nuclear-filtered, SC ≥ 2)',
                   fontsize=9)
    ax_f.spines['top'].set_visible(False); ax_f.spines['right'].set_visible(False)

ax_f.text(-0.18, 1.06, 'F', transform=ax_f.transAxes, fontsize=16, fontweight='bold')

fig.suptitle('Figure S1  |  AP-MS Dataset Characterization\n'
             '(nuclear-localized preys, all baits, spectral-count enrichment)',
             fontsize=11, fontweight='bold', y=0.975)

out = OUTDIR + "figureS_APMS_analysis.pdf"
fig.savefig(out, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"Saved: {out}")

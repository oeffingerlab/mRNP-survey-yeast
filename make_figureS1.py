"""
Figure S1 — AP-MS Quality Control and Reproducibility
Letters as in the final Supplemental Figure 1:
A: Placeholder — Coomassie gels (placed in Illustrator)
B: Total spectral counts per bait (bar chart)
C: Nuclear prey detection breadth
D: Prey sharing — fraction of nuclear preys per breadth category per bait
(The bait-bait Pearson panel formerly drawn here as "B" duplicated Figure 1D and was
 removed 2026-10-28; the clustered view is Supplemental Figure 2B.)

Exclusions:
  - MTR2 excluded (data quality)
  - Poison/mutant AP-MS conditions excluded
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
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS, fmt_baits,
                           CMAP_APMS, CMAP_CORR, PATHWAY_ORDER_APMS)
BAIT_GROUP = BAIT_GROUP_APMS
apply_style()

OUTDIR  = "output/"
DATADIR = "data/"

# ── Load data ──────────────────────────────────────────────────────────────
df_apms = pd.read_csv(DATADIR + "apms_nuclear_matrix.tsv", sep='\t', index_col=0)
df_apms = df_apms.drop(columns=['MTR2'], errors='ignore')
apms_baits = [b for b in PATHWAY_ORDER_APMS if b in df_apms.columns]
df_apms = df_apms[apms_baits]

# Prey breadth (number of baits each prey is detected in)
prey_breadth = (df_apms > 0).sum(axis=1)
bait_breadth  = (df_apms > 0).sum(axis=0)   # preys per bait

# ── Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(3, 2, figure=fig,
                        height_ratios=[1.2, 1.2, 1.1],
                        hspace=0.45, wspace=0.30,
                        left=0.07, right=0.97, top=0.94, bottom=0.05)

def bait_color_bar_h(ax, baits):
    colors = [GROUP_COLORS.get(BAIT_GROUP.get(b,''), '#CCCCCC') for b in baits]
    n = len(baits)
    for i, c in enumerate(colors):
        ax.add_patch(mpatches.Rectangle((i/n, 0), 1/n, 1,
            color=c, transform=ax.transAxes, clip_on=False))

# ── Panel A: Placeholder for western blots ────────────────────────────────
ax_a = fig.add_subplot(gs[0, :])
ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)
ax_a.set_facecolor('#F8F8F8')
for spine in ax_a.spines.values():
    spine.set_edgecolor('#CCCCCC'); spine.set_linewidth(1)
ax_a.text(0.5, 0.55, 'Western blots', ha='center', va='center',
          fontsize=14, color='#AAAAAA', style='italic')
ax_a.text(0.5, 0.38, '(to be placed in Illustrator)', ha='center', va='center',
          fontsize=9, color='#BBBBBB')
ax_a.set_xticks([]); ax_a.set_yticks([])
ax_a.text(-0.03, 1.02, 'A', transform=ax_a.transAxes, fontsize=16, fontweight='bold')

# ── Panel C: Total spectral counts per bait ───────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

total_sc = df_apms.sum(axis=0).loc[apms_baits]
bar_colors = [GROUP_COLORS.get(BAIT_GROUP.get(b,''), '#CCCCCC') for b in apms_baits]
x = np.arange(len(apms_baits))
ax_c.bar(x, total_sc.values, color=bar_colors, edgecolor='white', linewidth=0.3, width=0.75)
ax_c.set_xticks(x)
ax_c.set_xticklabels(fmt_baits(apms_baits), fontsize=7, rotation=60, ha='right')
ax_c.set_ylabel('Total spectral counts\n(nuclear preys)', fontsize=10)
ax_c.set_title('AP-MS Signal Intensity per Bait\n(sum of nuclear prey spectral counts)',
               fontsize=9, fontweight='bold')
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)

ax_c.text(-0.1, 1.04, 'B', transform=ax_c.transAxes, fontsize=16, fontweight='bold')

# ── Panel D: Nuclear prey detection per bait (breadth) ────────────────────
ax_d = fig.add_subplot(gs[1, 1])

n_preys = bait_breadth.loc[apms_baits]
ax_d.bar(x, n_preys.values, color=bar_colors, edgecolor='white', linewidth=0.3, width=0.75)
ax_d.set_xticks(x)
ax_d.set_xticklabels(fmt_baits(apms_baits), fontsize=7, rotation=60, ha='right')
ax_d.set_ylabel('Number of nuclear preys detected', fontsize=10)
ax_d.set_title('Nuclear Prey Detection per Bait\n(unique proteins with SpC > 0)',
               fontsize=9, fontweight='bold')
ax_d.spines['top'].set_visible(False); ax_d.spines['right'].set_visible(False)

ax_d.text(-0.1, 1.04, 'C', transform=ax_d.transAxes, fontsize=16, fontweight='bold')

# ── Panel E: Prey universality stacked bars (breadth categories per bait) ─
ax_e = fig.add_subplot(gs[2, :])

# Categories: universal (≥20 baits), broad (10-19), moderate (4-9), specific (1-3)
cat_breaks = [(1, 3, 'Specific (1–3 baits)', '#F5ADAB'),
              (4, 9,  'Moderate (4–9)',        '#FFCA7D'),
              (10,19, 'Broad (10–19)',          '#A3D6A6'),
              (20, 999,'Universal (≥20 baits)', '#7DCAC2')]

bottom = np.zeros(len(apms_baits))
for lo, hi, label, color in cat_breaks:
    mask = (prey_breadth >= lo) & (prey_breadth <= hi)
    preys_in_cat = prey_breadth[mask].index
    counts = df_apms.loc[preys_in_cat, apms_baits].gt(0).sum(axis=0).values.astype(float)
    ax_e.bar(x, counts, bottom=bottom, color=color, label=label,
             edgecolor='white', linewidth=0.3, width=0.75)
    bottom += counts

ax_e.set_xticks(x)
ax_e.set_xticklabels(fmt_baits(apms_baits), fontsize=7.5, rotation=60, ha='right')
ax_e.set_ylabel('Number of nuclear preys', fontsize=10)
ax_e.set_title('Prey Sharing Specificity per Bait\n(prey breadth categories)',
               fontsize=9, fontweight='bold')
ax_e.spines['top'].set_visible(False); ax_e.spines['right'].set_visible(False)
ax_e.legend(fontsize=7, loc='upper right', framealpha=0.7)

ax_e_top = ax_e.inset_axes([0, 1.01, 1, 0.025])
ax_e_top.axis('off'); bait_color_bar_h(ax_e_top, apms_baits)
ax_e.text(-0.05, 1.04, 'D', transform=ax_e.transAxes, fontsize=16, fontweight='bold')

fig.suptitle('Supplemental Figure 1  |  AP-MS reproducibility and data quality',
             fontsize=12, fontweight='bold', y=0.975)

out = OUTDIR + "figureS1_composite.pdf"
fig.savefig(out, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

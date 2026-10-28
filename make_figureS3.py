"""
Supplemental Figure 6 — Multi-omics clustering robustness
A: Multi-omics bait dendrogram (AP-MS + AP-RNA-seq combined Pearson, average linkage)
B: PCA of combined bait-bait correlation matrix
C: Full combined correlation matrix ordered by functional module
D: Functional-module interaction network (combined r > 0.35; fixed circular within-module layout)
Colors match the exact palette from the original Illustrator files.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS,
                           fmt_bait, fmt_baits, CMAP_CORR,
                           PATHWAY_ORDER_RNA, PATHWAY_ORDER_APMS)
apply_style()

OUTDIR  = "output/"
DATADIR = "data/"

# ── Load data ──────────────────────────────────────────────────────────────
df_apms = pd.read_csv(DATADIR + "apms_nuclear_matrix.tsv", sep='\t', index_col=0)
df_apms = df_apms.drop(columns=['MTR2'], errors='ignore')
df_rna  = pd.read_csv(DATADIR + "rnaseq_bait_means.tsv",  sep='\t', index_col=0)

rna_baits  = [b for b in PATHWAY_ORDER_RNA  if b in df_rna.columns]
apms_baits = [b for b in PATHWAY_ORDER_APMS if b in df_apms.columns]
shared     = [b for b in rna_baits if b in apms_baits]

df_rna_filt = df_rna[rna_baits]
df_rna_filt = df_rna_filt[df_rna_filt.max(axis=1) > 0.3]

corr_apms = df_apms[shared].corr(method='pearson').loc[shared, shared]
corr_rna  = df_rna_filt[shared].corr(method='pearson').loc[shared, shared]
corr_comb = (corr_apms + corr_rna) / 2.0
corr_comb = corr_comb.loc[shared, shared]

# ── Hierarchical linkage for dendrogram (Panel A) ─────────────────────────
dist_multi = 1.0 - corr_comb.values
dist_multi = np.clip(dist_multi, 0, None)
np.fill_diagonal(dist_multi, 0)
link_multi = linkage(dist_multi[np.triu_indices(len(shared), k=1)], method='average')

# ── 4-cluster assignment ───────────────────────────────────────────────────
CLUSTER_4 = {
    'CBP80': 1, 'THP2': 1, 'YRA1': 1,
    'SNU71': 2, 'SYC1': 2, 'REF2': 2, 'CLP1': 2, 'HRP1': 2,
    'NAB2': 3, 'NPL3': 3, 'GBP2': 3, 'PAB1': 3, 'GFD1': 3, 'RRP6': 3,
    'SAC3': 4, 'MEX67': 4, 'DBP5': 4, 'SUB2': 4,
}
CLUSTER_NAMES = {
    1: 'CBC/THO/TREX', 2: "Splicing/3'end",
    3: 'hnRNP/Export adaptor', 4: 'Terminal export'
}
# Two-line versions for the in-circle labels (cluster center has no nodes, so safe)
CLUSTER_NAMES_2L = {
    1: 'CBC/\nTHO-TREX',
    2: "Splicing/\n3′end",
    3: "hnRNP/\nExport adaptor",
    4: "Terminal\nexport"
}
# Colors matched to pathway group colors from original Illustrator files
CLUSTER_COLS = {
    1: '#DB2827',   # cherry red  (CBC dominant)
    2: '#465EAB',   # dark blue   (Splicing)
    3: '#56C4CB',   # teal        (3'end processing)
    4: '#0A7F40',   # clover green (Export/pore)
}

bait_colors = [GROUP_COLORS.get(BAIT_GROUP_APMS.get(b,''), '#CCCCCC') for b in shared]

# ── Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor('white')
fig.suptitle('Supplemental Figure 6 | Multi-omics clustering robustness',
             fontsize=13, y=0.99)

gs = gridspec.GridSpec(2, 2, figure=fig,
                        height_ratios=[1, 1],
                        hspace=0.38, wspace=0.32,
                        left=0.06, right=0.97, top=0.96, bottom=0.04)

legend_patches = [mpatches.Patch(color=CLUSTER_COLS[k], label=CLUSTER_NAMES[k])
                  for k in [1,2,3,4]]
grp_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
               if any(BAIT_GROUP_APMS.get(b)==g for b in shared)]

# ════════════════════════════════════════════════════════════════════════════
# Panel A — Multi-omics bait dendrogram (80% of cell height — 20% smaller than B/C/D)
# ════════════════════════════════════════════════════════════════════════════
gs_a = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0],
                                         height_ratios=[4, 1], hspace=0)
ax_dend = fig.add_subplot(gs_a[0])

dend = dendrogram(link_multi, labels=shared, ax=ax_dend,
                  orientation='top', color_threshold=0,
                  above_threshold_color='black',
                  leaf_font_size=7)

# Leaf labels: BLACK, protein convention (Cbp80p) per standing style rules
ax_dend.set_xticklabels([fmt_bait(t.get_text()) for t in ax_dend.get_xticklabels()])
for lbl in ax_dend.get_xticklabels():
    lbl.set_color('black')
    lbl.set_fontweight('bold')

# Per-leaf MODULE markers: one square under each leaf, colored by that bait's
# module. Replaces spanning boxes (which implied a module is one contiguous
# block; members are actually split across the tree, e.g. Sub2p sits with Thp2p).
leaf_order = dend['leaves']
leaf_names = [shared[i] for i in leaf_order]

ax_dend_ylim = ax_dend.get_ylim()
span  = ax_dend_ylim[1] - ax_dend_ylim[0]
bar_y = ax_dend_ylim[0] - 0.05 * span
sq_h  = 0.035 * span
ax_dend.set_ylim(bar_y - 0.03 * span, ax_dend_ylim[1])

import matplotlib.patches as _mp2
for i, nm in enumerate(leaf_names):
    col = CLUSTER_COLS.get(CLUSTER_4.get(nm, 0), '#CCCCCC')
    xc  = i * 10 + 5                      # scipy leaf centers: 5, 15, 25, ...
    ax_dend.add_patch(_mp2.Rectangle((xc - 2.6, bar_y), 5.2, sq_h,
                    facecolor=col, edgecolor='white', linewidth=0.6,
                    transform=ax_dend.transData, clip_on=False))

ax_dend.set_title('Multi-omics Bait Dendrogram\n'
                  '(AP-MS + AP-RNA-seq combined Pearson, average linkage)',
                  fontsize=11, fontweight='bold')
ax_dend.set_ylabel('Distance (1 – Pearson)', fontsize=10)
ax_dend.spines['top'].set_visible(False); ax_dend.spines['right'].set_visible(False)
ax_dend.text(-0.08, 1.04, 'A', transform=ax_dend.transAxes, fontsize=16, fontweight='bold')

# ════════════════════════════════════════════════════════════════════════════
# Panel B — PCA scatter
# ════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, 1])
pca    = PCA(n_components=2)
coords = pca.fit_transform(corr_comb.values)
var_exp = pca.explained_variance_ratio_ * 100

# Convex-hull polygons removed (misleading for small n and to computational
# readers). Points colored by module (matches the module legend); labels black.
for i, b in enumerate(shared):
    color = CLUSTER_COLS.get(CLUSTER_4.get(b, 0), '#CCCCCC')
    ax_a.scatter(coords[i,0], coords[i,1], s=100, color=color,
                 edgecolors='white', linewidths=1.3, zorder=4)
    ax_a.text(coords[i,0]+0.012, coords[i,1]+0.008, fmt_bait(b),
              fontsize=8, color='black', zorder=5)

ax_a.set_xlabel(f'PC1 ({var_exp[0]:.1f}% variance)', fontsize=10)
ax_a.set_ylabel(f'PC2 ({var_exp[1]:.1f}% variance)', fontsize=10)
ax_a.set_title('PCA of the combined AP-MS + AP-RNA-seq\nbait correlation matrix (same matrix as panels A, C, D)', fontsize=11)
ax_a.spines['top'].set_visible(False); ax_a.spines['right'].set_visible(False)
ax_a.legend(handles=legend_patches, fontsize=8, loc='lower right',
            title='Functional module', title_fontsize=8.5)
ax_a.text(-0.08, 1.04, 'B', transform=ax_a.transAxes, fontsize=16, fontweight='bold')

# ════════════════════════════════════════════════════════════════════════════
# Panel C — Full heatmap ordered by cluster
# ════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[1, 0])

ordered = []
for k in [1,2,3,4]:
    ordered += [b for b in shared if CLUSTER_4.get(b,0)==k]
corr_ord = corr_comb.loc[ordered, ordered].values.copy()
np.fill_diagonal(corr_ord, np.nan)

im = ax_b.imshow(corr_ord, cmap=CMAP_CORR, vmin=-0.2, vmax=1.0, aspect='equal', rasterized=True)
ax_b.set_xticks(range(len(ordered)))
ax_b.set_yticks(range(len(ordered)))
ax_b.set_xticklabels(fmt_baits(ordered), fontsize=11, rotation=60, ha='right', color='black')
ax_b.set_yticklabels(fmt_baits(ordered), fontsize=11, color='black')

sizes = [len([b for b in shared if CLUSTER_4.get(b,0)==k]) for k in [1,2,3,4]]
cum   = np.cumsum([0]+sizes)
for c in cum[1:-1]:
    ax_b.axhline(c-0.5, color='white', linewidth=2)
    ax_b.axvline(c-0.5, color='white', linewidth=2)

# Pathway stage color bars: right of y-axis and above x-axis
def _pbar_v(ax, baits):
    colors = [GROUP_COLORS.get(BAIT_GROUP_APMS.get(b,''), '#CCCCCC') for b in baits]
    n = len(baits)
    for i, c in enumerate(colors):
        ax.add_patch(mpatches.Rectangle((0, (n-1-i)/n), 1, 1/n, color=c,
                     transform=ax.transAxes, clip_on=False))

def _pbar_h(ax, baits):
    colors = [GROUP_COLORS.get(BAIT_GROUP_APMS.get(b,''), '#CCCCCC') for b in baits]
    n = len(baits)
    for i, c in enumerate(colors):
        ax.add_patch(mpatches.Rectangle((i/n, 0), 1/n, 1, color=c,
                     transform=ax.transAxes, clip_on=False))

ax_b_right = ax_b.inset_axes([1.005, 0, 0.030, 1])
ax_b_right.axis('off'); _pbar_v(ax_b_right, ordered)
ax_b_top = ax_b.inset_axes([0, 1.01, 1, 0.025])
ax_b_top.axis('off'); _pbar_h(ax_b_top, ordered)

# Colorbar via inset_axes so it does NOT steal space from ax_b's cell
cax_b = ax_b.inset_axes([1.08, 0.25, 0.04, 0.50])
cb_b = plt.colorbar(im, cax=cax_b, label='Pearson r')
cb_b.ax.tick_params(labelsize=9)
cb_b.set_label('Pearson r', fontsize=10)
ax_b.set_title('Combined correlation matrix\n(ordered by functional module)', fontsize=11)
# Cluster legend
ax_b.legend(handles=legend_patches, fontsize=7.5, loc='upper left',
            bbox_to_anchor=(1.20, 1.0), title='Functional module', title_fontsize=8)
# Functional group legend
_grp_patches_b = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
                  if any(BAIT_GROUP_APMS.get(b,'') == g for b in shared)]
from matplotlib.legend import Legend as _Leg
_grp_legend_b = _Leg(ax_b, _grp_patches_b, [p.get_label() for p in _grp_patches_b],
                      loc='upper left', bbox_to_anchor=(1.20, 0.48),
                      fontsize=7.5, title='Functional group', title_fontsize=8,
                      frameon=True, ncol=1, handlelength=1.0)
ax_b.add_artist(_grp_legend_b)
ax_b.text(-0.12, 1.04, 'C', transform=ax_b.transAxes, fontsize=16, fontweight='bold')


# ════════════════════════════════════════════════════════════════════════════
# Panel D — Functional-module network with fixed circular within-module layout
# ════════════════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs[1, 1])

THRESHOLD = 0.35
G = nx.Graph()
G.add_nodes_from(shared)
for i, b1 in enumerate(shared):
    for j, b2 in enumerate(shared):
        if j <= i: continue
        r = corr_comb.loc[b1, b2]
        if r > THRESHOLD:
            G.add_edge(b1, b2, weight=float(r))

# Fixed cluster centers — guaranteed separation, no spring randomness
CLUSTER_CENTERS_XY = {1: (-1.8, 1.8), 2: (1.8, 1.8), 3: (1.8, -1.8), 4: (-1.8, -1.8)}
CLUSTER_RADIUS     = {1: 0.48, 2: 0.62, 3: 0.72, 4: 0.55}

# Arrange each cluster's nodes in a small circle around its center
pos = {}
for k in [1, 2, 3, 4]:
    nodes_k = [b for b in shared if CLUSTER_4.get(b, 0) == k]
    cx, cy  = CLUSTER_CENTERS_XY[k]
    r       = CLUSTER_RADIUS[k]
    n       = len(nodes_k)
    for i, b in enumerate(nodes_k):
        angle   = 2 * np.pi * i / n - np.pi / 2   # start from 12 o'clock
        pos[b]  = (cx + r * np.cos(angle), cy + r * np.sin(angle))

# Draw edges
edge_list = list(G.edges())
edge_weights = [G[u][v]['weight'] for u, v in edge_list] if edge_list else []
max_w = max(edge_weights) if edge_weights else 1.0
for (u, v), w in zip(edge_list, edge_weights):
    ku, kv = CLUSTER_4.get(u, 0), CLUSTER_4.get(v, 0)
    if ku == kv:
        color = CLUSTER_COLS.get(ku, '#CCCCCC')
        lw    = 0.5 + 2.2 * (w / max_w)
        alpha = 0.30 + 0.45 * (w / max_w)
    else:
        color = '#CCCCCC'
        lw    = 0.4
        alpha = 0.20
    ax_d.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
              color=color, linewidth=lw, alpha=alpha, zorder=1)

NODE_R_OUTER = 0.12
NODE_R_INNER = 0.085
LABEL_GAP    = 0.16   # distance beyond outer ring to label anchor

# Draw nodes; labels point outward from their cluster center
for b in shared:
    x, y  = pos[b]
    color = GROUP_COLORS.get(BAIT_GROUP_APMS.get(b, ''), '#CCCCCC')
    k     = CLUSTER_4.get(b, 0)
    ax_d.add_patch(plt.Circle((x, y), NODE_R_OUTER,
                               color=CLUSTER_COLS.get(k, '#CCCCCC'), alpha=0.55, zorder=2))
    ax_d.add_patch(plt.Circle((x, y), NODE_R_INNER,
                               color=color, zorder=3, ec='white', linewidth=1.4))

    # Outward unit vector from cluster center to node
    cx, cy_c = CLUSTER_CENTERS_XY[k]
    dx, dy   = x - cx, y - cy_c
    dist     = np.sqrt(dx**2 + dy**2) or 1.0
    ux, uy   = dx / dist, dy / dist

    lx = x + ux * (NODE_R_OUTER + LABEL_GAP)
    ly = y + uy * (NODE_R_OUTER + LABEL_GAP)

    ang = np.degrees(np.arctan2(uy, ux))
    if   -45  <= ang <=  45:  ha, va = 'left',   'center'
    elif  45  <  ang <= 135:  ha, va = 'center',  'bottom'
    elif ang  >  135 or ang < -135: ha, va = 'right',  'center'
    else:                     ha, va = 'center',  'top'

    ax_d.text(lx, ly, fmt_bait(b), ha=ha, va=va,
              fontsize=8.5, color='#1A1A1A', zorder=4)

# Cluster name labels placed at the CENTER of each node ring.
# Nodes always sit on the outer circle so the center is always empty;
# protein labels point OUTWARD, guaranteeing zero overlap with cluster names.
for k in [1, 2, 3, 4]:
    cx, cy = CLUSTER_CENTERS_XY[k]
    ax_d.text(cx, cy, CLUSTER_NAMES_2L[k],
              ha='center', va='center',
              fontsize=9, color=CLUSTER_COLS[k], fontweight='bold',
              linespacing=1.35, zorder=6,
              bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                        alpha=0.88, edgecolor=CLUSTER_COLS[k], linewidth=1.2))

ax_d.set_xlim(-3.1, 3.1); ax_d.set_ylim(-3.1, 3.1)
ax_d.set_aspect('equal'); ax_d.axis('off')
ax_d.set_title(f'Multi-omics Network (combined r > {THRESHOLD})\n'
               f'Inner node = functional group; outer ring = functional module',
               fontsize=11)
ax_d.legend(handles=grp_patches, loc='lower center', ncol=3, fontsize=7.5,
            bbox_to_anchor=(0.5, -0.05),
            title='Inner node color = functional group', title_fontsize=8)
ax_d.text(-0.04, 1.04, 'D', transform=ax_d.transAxes, fontsize=16, fontweight='bold')

# ── Save ───────────────────────────────────────────────────────────────────
outpath = OUTDIR + "figureS6_multiomics_clustering.pdf"
fig.savefig(outpath, format='pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {outpath}")

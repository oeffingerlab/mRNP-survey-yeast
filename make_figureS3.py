"""
Figure S7 — Multi-omics Clustering Robustness
A: Multi-omics bait dendrogram (AP-MS + AP-RNA-seq combined Pearson, average linkage)
B: PCA of combined bait-bait correlation matrix
C: Full combined correlation matrix ordered by cluster
D: Force-directed cluster network (combined r > 0.35)
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
sys.path.insert(0, "/sessions/lucid-modest-goldberg/mnt/outputs/")
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS,
                           fmt_bait, fmt_baits, CMAP_CORR,
                           PATHWAY_ORDER_RNA, PATHWAY_ORDER_APMS)
apply_style()

OUTDIR  = "/sessions/lucid-modest-goldberg/mnt/MS results/"
DATADIR = "/sessions/lucid-modest-goldberg/mnt/outputs/"

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
fig.suptitle('Figure S7 | Multi-omics Clustering Robustness',
             fontsize=13, y=0.99)

gs = gridspec.GridSpec(2, 2, figure=fig,
                        hspace=0.38, wspace=0.32,
                        left=0.06, right=0.97, top=0.96, bottom=0.04)

legend_patches = [mpatches.Patch(color=CLUSTER_COLS[k], label=CLUSTER_NAMES[k])
                  for k in [1,2,3,4]]
grp_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
               if any(BAIT_GROUP_APMS.get(b)==g for b in shared)]

# ════════════════════════════════════════════════════════════════════════════
# Panel A — Multi-omics bait dendrogram
# ════════════════════════════════════════════════════════════════════════════
ax_dend = fig.add_subplot(gs[0, 0])

dend = dendrogram(link_multi, labels=shared, ax=ax_dend,
                  orientation='top', color_threshold=0,
                  above_threshold_color='black',
                  leaf_font_size=7)

# Color leaf labels by functional group
for lbl in ax_dend.get_xticklabels():
    bait = lbl.get_text().upper()
    col  = GROUP_COLORS.get(BAIT_GROUP_APMS.get(bait, ''), '#555555')
    lbl.set_color(col)
    lbl.set_fontweight('bold')

# Annotate cluster blocks below the dendrogram
leaf_order = dend['leaves']
leaf_names  = [shared[i] for i in leaf_order]
clust_ids = {b: CLUSTER_4.get(b, 0) for b in shared}

ax_dend_ylim = ax_dend.get_ylim()
bar_y = ax_dend_ylim[0] - 0.03 * (ax_dend_ylim[1] - ax_dend_ylim[0])
ax_dend.set_ylim(bar_y - 0.06 * (ax_dend_ylim[1] - ax_dend_ylim[0]), ax_dend_ylim[1])

prev_cl = clust_ids[leaf_names[0]]
block_start = 0
block_info = []
for i, nm in enumerate(leaf_names):
    cl = clust_ids[nm]
    if cl != prev_cl or i == len(leaf_names)-1:
        block_info.append((block_start, i-1 if cl != prev_cl else i, prev_cl))
        block_start = i
        prev_cl = cl

import matplotlib.patches as _mp2
for (bstart, bend, cl_id) in block_info:
    x0 = bstart * 10 + 5
    x1 = (bend + 1) * 10 - 5
    col = CLUSTER_COLS.get(cl_id, '#CCCCCC')
    ax_dend.add_patch(_mp2.FancyBboxPatch((x0, bar_y), x1-x0,
                    0.03*(ax_dend_ylim[1]-ax_dend_ylim[0]),
                    boxstyle="round,pad=0.5", color=col, alpha=0.7,
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
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull

pca    = PCA(n_components=2)
coords = pca.fit_transform(corr_comb.values)
var_exp = pca.explained_variance_ratio_ * 100

for k in [1,2,3,4]:
    idx = [i for i,b in enumerate(shared) if CLUSTER_4.get(b,0)==k]
    if len(idx) >= 3:
        pts = coords[idx]
        try:
            hull = ConvexHull(pts)
            hp = pts[hull.vertices]
            hp = np.vstack([hp, hp[0]])
            ax_a.fill(hp[:,0], hp[:,1], color=CLUSTER_COLS[k], alpha=0.18, zorder=1)
            ax_a.plot(hp[:,0], hp[:,1], color=CLUSTER_COLS[k], linewidth=1.2, alpha=0.5, zorder=2)
        except Exception:
            pass

for i, b in enumerate(shared):
    color = GROUP_COLORS.get(BAIT_GROUP_APMS.get(b,''), '#CCCCCC')
    ax_a.scatter(coords[i,0], coords[i,1], s=100, color=color,
                 edgecolors='white', linewidths=1.3, zorder=4)
    ax_a.text(coords[i,0]+0.012, coords[i,1]+0.008, fmt_bait(b),
              fontsize=8, color='#222222', zorder=5)

ax_a.set_xlabel(f'PC1 ({var_exp[0]:.1f}% variance)', fontsize=10)
ax_a.set_ylabel(f'PC2 ({var_exp[1]:.1f}% variance)', fontsize=10)
ax_a.set_title('PCA of Combined AP-MS + AP-RNA-seq\nBait Correlation Matrix', fontsize=11)
ax_a.spines['top'].set_visible(False); ax_a.spines['right'].set_visible(False)
ax_a.legend(handles=legend_patches, fontsize=8, loc='lower right',
            title='Multi-omics cluster', title_fontsize=8.5)
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

im = ax_b.imshow(corr_ord, cmap=CMAP_CORR, vmin=-0.2, vmax=1.0, aspect='equal')
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

plt.colorbar(im, ax=ax_b, shrink=0.5, pad=0.08, label='Pearson r')
ax_b.set_title('Combined Correlation Matrix\n(ordered by multi-omics cluster)', fontsize=11)
# Cluster legend (keep small, top-right inside colorbar area)
ax_b.legend(handles=legend_patches, fontsize=7.5, loc='upper left',
            bbox_to_anchor=(1.55, 1.0), title='Cluster', title_fontsize=8)
# Functional group legend to right of Panel B
ax_b.figure.add_axes([0, 0, 0, 0])  # dummy; actual placement via ax_b legend below
_grp_leg = ax_b.get_figure().add_axes([0, 0, 0.001, 0.001])  # tiny invisible placeholder
_grp_leg.axis('off')
_grp_patches_b = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
                  if any(BAIT_GROUP_APMS.get(b,'') == g for b in shared)]
ax_b.figure.legends.clear()  # remove any auto-placed fig legends
# Use a second legend on ax_b via add_artist
from matplotlib.legend import Legend as _Leg
_grp_legend_b = _Leg(ax_b, _grp_patches_b, [p.get_label() for p in _grp_patches_b],
                      loc='upper left', bbox_to_anchor=(1.55, 0.48),
                      fontsize=7.5, title='Functional group', title_fontsize=8,
                      frameon=True, ncol=1, handlelength=1.0)
ax_b.add_artist(_grp_legend_b)
ax_b.text(-0.12, 1.04, 'C', transform=ax_b.transAxes, fontsize=16, fontweight='bold')


# ════════════════════════════════════════════════════════════════════════════
# Panel D — Cluster network with fixed circular within-cluster layout
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
               f'Inner node = functional group; outer ring = cluster',
               fontsize=11)
ax_d.legend(handles=grp_patches, loc='lower center', ncol=3, fontsize=7.5,
            bbox_to_anchor=(0.5, -0.05),
            title='Inner node color = functional group', title_fontsize=8)
ax_d.text(-0.04, 1.04, 'D', transform=ax_d.transAxes, fontsize=16, fontweight='bold')

# ── Save ───────────────────────────────────────────────────────────────────
outpath = OUTDIR + "figureS7_multiomics_clustering.pdf"
fig.savefig(outpath, format='pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {outpath}")

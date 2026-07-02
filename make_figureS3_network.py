"""
Supplementary Figure S3 — AP-MS Network Topology (2026-07-02 recomputation)
Recomputed entirely from apms_nuclear_matrix.tsv; replaces Alex's Cytoscape PDF.
Full 34-bait set (MTR2 excluded), includes YHS7.

Panels:
  A — Directed bait-to-bait network (edge A→B if A pulls down B as prey)
  B — Betweenness centrality per bait (in directed bait-to-bait graph)
  C — Shortest-path heatmap, pathway order (same metric as Figure 1E)
  D — Shortest-path heatmap, hierarchical clustering order
  E — Group-averaged shortest-path heatmap (functional group × functional group)
  F — Pairwise shortest-path distance distribution (histogram)
  G — Undirected bait co-purification network (edge if A pulls down B or vice versa)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import shutil, warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, "/sessions/lucid-modest-goldberg/mnt/outputs/")
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS,
                           fmt_bait, fmt_baits, CMAP_SP, PATHWAY_ORDER_APMS as _PO)
apply_style()
BAIT_GROUP = BAIT_GROUP_APMS

OUTDIR  = "/sessions/lucid-modest-goldberg/mnt/outputs/"
MSDIR   = "/sessions/lucid-modest-goldberg/mnt/MS results/"
DATADIR = "/sessions/lucid-modest-goldberg/mnt/outputs/"

PATHWAY_ORDER = [
    'CBP20','CBP80',
    'THP2','YRA1','YHS7','YRA2',
    'SNU71','SNU114','PRP42','PRP6',
    'RNA14','CLP1','YSH1','SYC1','CFT1','REF2','HRP1',
    'PAB1','PAP1','NAB2','NPL3','GBP2','HRB1','GFD1',
    'RRP6','SUB2','PML39',
    'SAC3','THP1',
    'MEX67','DBP5','MLP1','MLP2','SHE2',
]

# ── Load AP-MS matrix ──────────────────────────────────────────────────────
df = pd.read_csv(DATADIR + "apms_nuclear_matrix.tsv", sep='\t', index_col=0)
df = df.drop(columns=['MTR2'], errors='ignore')
baits = [b for b in PATHWAY_ORDER if b in df.columns]
df_baits = df[baits]
print(f"Matrix: {df_baits.shape} — baits: {len(baits)}")

# ── Build directed bait-to-bait graph ─────────────────────────────────────
# Edge A → B if bait A has bait B as a prey (adj_score > 0)
G_dir = nx.DiGraph()
G_dir.add_nodes_from(baits)
for src in baits:
    for tgt in baits:
        if src != tgt and df_baits.loc[tgt, src] > 0 if tgt in df_baits.index else False:
            G_dir.add_edge(src, tgt, weight=float(df_baits.loc[tgt, src]))

# ── Build undirected bait-to-bait graph ───────────────────────────────────
# Edge A — B if A pulls down B OR B pulls down A
G_undir = nx.Graph()
G_undir.add_nodes_from(baits)
for src in baits:
    for tgt in baits:
        if src < tgt:
            w_fwd = float(df_baits.loc[tgt, src]) if tgt in df_baits.index else 0
            w_rev = float(df_baits.loc[src, tgt]) if src in df_baits.index else 0
            if w_fwd > 0 or w_rev > 0:
                G_undir.add_edge(src, tgt, weight=max(w_fwd, w_rev))

print(f"Directed edges: {G_dir.number_of_edges()}, Undirected edges: {G_undir.number_of_edges()}")

# ── Shortest-path matrix (via bipartite bait-prey graph) ──────────────────
G_bp = nx.Graph()
for bait in baits:
    for prey, sc in df_baits[bait].items():
        if sc > 0:
            G_bp.add_edge(bait, prey)

sp_matrix = pd.DataFrame(np.nan, index=baits, columns=baits)
for src in baits:
    try:
        lengths = nx.single_source_shortest_path_length(G_bp, src, cutoff=8)
        for tgt in baits:
            if tgt in lengths:
                sp_matrix.loc[src, tgt] = lengths[tgt]
    except Exception:
        pass
sp_matrix = sp_matrix.fillna(9)
np.fill_diagonal(sp_matrix.values, 0)
print(f"Shortest-path: shape={sp_matrix.shape}, max={sp_matrix.values.max():.0f}")

# ── Betweenness centrality ─────────────────────────────────────────────────
bc = nx.betweenness_centrality(G_dir, normalized=True)
bc_series = pd.Series(bc).reindex(baits).fillna(0)

# ── Hierarchical clustering order for shortest-path ───────────────────────
sp_sym = (sp_matrix.values + sp_matrix.values.T) / 2
np.fill_diagonal(sp_sym, 0)
try:
    Z = linkage(squareform(sp_sym, checks=False), method='average')
    clust_order = leaves_list(Z)
    baits_clust = [baits[i] for i in clust_order]
except Exception:
    baits_clust = baits[:]

# ── Group-averaged shortest-path ──────────────────────────────────────────
group_order = [g for g in GROUP_COLORS if any(BAIT_GROUP.get(b)==g for b in baits)]
group_baits = {g: [b for b in baits if BAIT_GROUP.get(b)==g] for g in group_order}
# average SP between groups
n_grp = len(group_order)
grp_sp = np.zeros((n_grp, n_grp))
for i, gi in enumerate(group_order):
    for j, gj in enumerate(group_order):
        bi, bj = group_baits[gi], group_baits[gj]
        if bi and bj:
            vals = sp_matrix.loc[bi, bj].values.flatten()
            vals_nondiag = vals if i != j else vals[vals > 0]
            grp_sp[i, j] = np.nanmean(vals_nondiag) if len(vals_nondiag) > 0 else 0

# ── Network layout (shared, reproducible) ─────────────────────────────────
# Circular layout with baits in processing-stage order
n = len(baits)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
pos_circ = {b: (np.cos(a), np.sin(a)) for b, a in zip(baits, angles)}

# Spring layout for undirected (more visually spread)
np.random.seed(42)
pos_spring = nx.spring_layout(G_undir, k=2.5, seed=42, iterations=150,
                               weight='weight')

def node_colors(node_list):
    return [GROUP_COLORS.get(BAIT_GROUP.get(b,''), '#CCCCCC') for b in node_list]

def draw_colorstrip_x(ax, bait_list, h=0.025, gap=0.01):
    """Horizontal group-color strip above x-axis."""
    n = len(bait_list)
    s = ax.inset_axes([0, 1.0+gap, 1, h])
    s.set_xlim(0, n); s.set_ylim(0, 1); s.axis('off')
    for i, b in enumerate(bait_list):
        c = GROUP_COLORS.get(BAIT_GROUP.get(b,''), '#CCCCCC')
        s.add_patch(mpatches.Rectangle((i, 0), 1, 1, color=c))

def draw_colorstrip_y(ax, bait_list, w=0.025, gap=0.01):
    """Vertical group-color strip right of y-axis."""
    n = len(bait_list)
    s = ax.inset_axes([1.0+gap, 0, w, 1])
    s.set_xlim(0, 1); s.set_ylim(n, 0); s.axis('off')
    for i, b in enumerate(bait_list):
        c = GROUP_COLORS.get(BAIT_GROUP.get(b,''), '#CCCCCC')
        s.add_patch(mpatches.Rectangle((0, i), 1, 1, color=c))

# ════════════════════════════════════════════════════════════════════════════
# FIGURE LAYOUT — 4 rows × 2 cols
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(11, 22))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(4, 2, figure=fig,
                       height_ratios=[1.6, 0.9, 0.9, 1.4],
                       hspace=0.55, wspace=0.40,
                       left=0.10, right=0.92, top=0.96, bottom=0.04)

# ── Panel A: Directed bait-to-bait network ────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_facecolor('white')
ax_a.set_aspect('equal')
ax_a.axis('off')

n_colors = node_colors(baits)
nx.draw_networkx_nodes(G_dir, pos_circ, ax=ax_a,
                        node_color=n_colors, node_size=300, linewidths=0.5,
                        edgecolors='white')
# Edges — width proportional to weight, alpha based on weight
edges = list(G_dir.edges(data='weight', default=1))
if edges:
    weights = np.array([e[2] for e in edges], dtype=float)
    max_w = weights.max() if weights.max() > 0 else 1
    edge_widths = 0.5 + 2.0 * (weights / max_w)
    edge_alphas = 0.3 + 0.6 * (weights / max_w)
    for (u, v, w), lw, al in zip(edges, edge_widths, edge_alphas):
        c = GROUP_COLORS.get(BAIT_GROUP.get(u,''), '#AAAAAA')
        ax_a.annotate('', xy=pos_circ[v], xytext=pos_circ[u],
                      arrowprops=dict(arrowstyle='->', color=c,
                                      lw=lw, alpha=al,
                                      connectionstyle='arc3,rad=0.1'))

# Node labels — only show bait name, small font
for b, (x, y) in pos_circ.items():
    ha = 'left' if x > 0 else 'right'
    va = 'bottom' if y > 0 else 'top'
    offset_x = 0.08 * np.sign(x) if abs(x) > 0.1 else 0
    offset_y = 0.08 * np.sign(y) if abs(y) > 0.1 else 0
    ax_a.text(x + offset_x, y + offset_y, fmt_bait(b),
              fontsize=4.5, ha=ha, va=va, fontweight='bold',
              color='black')

ax_a.set_ylim(-1.35, 1.55)  # extra top margin so nodes don't crowd the title
ax_a.set_title('Directed bait-to-bait network\n(A→B: A pulls down B as prey)',
               fontsize=9, fontweight='bold', pad=20)
ax_a.text(-0.05, 1.03, 'A', transform=ax_a.transAxes, fontsize=16, fontweight='bold')

# Legend for functional groups
leg_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
               if any(BAIT_GROUP.get(b)==g for b in baits)]
ax_a.legend(handles=leg_patches, loc='upper left', bbox_to_anchor=(1.02, 1.0),
            fontsize=6, title='Functional group', title_fontsize=6.5,
            frameon=True, ncol=1, handlelength=1.2, labelspacing=0.4)

# ── Panel B: Betweenness centrality ───────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
bc_sorted = bc_series.reindex(baits)
x_b = np.arange(len(baits))
bar_cols = node_colors(baits)
ax_b.bar(x_b, bc_sorted.values, color=bar_cols, edgecolor='white', linewidth=0.3, width=0.75)
ax_b.set_xticks(x_b)
ax_b.set_xticklabels(fmt_baits(baits), fontsize=4.5, rotation=90)
ax_b.set_ylabel('Betweenness centrality\n(directed, normalized)', fontsize=8)
ax_b.set_title('Betweenness centrality per bait\n(directed bait-to-bait graph)',
               fontsize=9, fontweight='bold', pad=5)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.axhline(0, color='black', linewidth=0.5)
# Annotate top 5
top5_bc = bc_sorted.nlargest(5)
for prot, val in top5_bc.items():
    xi = baits.index(prot)
    ax_b.text(xi, val + 0.002, f'{val:.2f}', ha='center', va='bottom',
              fontsize=4.5, fontweight='bold')
ax_b.text(-0.10, 1.03, 'B', transform=ax_b.transAxes, fontsize=16, fontweight='bold')

# ── Panel C: Shortest-path heatmap — pathway order ────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
sp_vals = sp_matrix.reindex(index=baits, columns=baits).values.astype(float)
im_c = ax_c.imshow(sp_vals, aspect='equal', cmap='Blues', vmin=0, vmax=3)
ax_c.set_title('Shortest-path distance (pathway order)', fontsize=9, fontweight='bold', pad=5)
ax_c.set_xticks(range(len(baits)))
ax_c.set_xticklabels(fmt_baits(baits), fontsize=4.5, rotation=90)
ax_c.set_yticks(range(len(baits)))
ax_c.set_yticklabels(fmt_baits(baits), fontsize=4.5)
draw_colorstrip_x(ax_c, baits)
draw_colorstrip_y(ax_c, baits)
plt.colorbar(im_c, ax=ax_c, shrink=0.65, pad=0.18, label='Graph distance (steps)')
ax_c.text(-0.18, 1.09, 'C', transform=ax_c.transAxes, fontsize=16, fontweight='bold')

# ── Panel D: Shortest-path heatmap — hierarchical order ───────────────────
ax_d = fig.add_subplot(gs[1, 1])
sp_clust = sp_matrix.reindex(index=baits_clust, columns=baits_clust).values.astype(float)
im_d = ax_d.imshow(sp_clust, aspect='equal', cmap='Blues', vmin=0, vmax=3)
ax_d.set_title('Shortest-path distance (clustered order)', fontsize=9, fontweight='bold', pad=5)
ax_d.set_xticks(range(len(baits_clust)))
ax_d.set_xticklabels(fmt_baits(baits_clust), fontsize=4.5, rotation=90)
ax_d.set_yticks(range(len(baits_clust)))
ax_d.set_yticklabels(fmt_baits(baits_clust), fontsize=4.5)
draw_colorstrip_x(ax_d, baits_clust)
draw_colorstrip_y(ax_d, baits_clust)
plt.colorbar(im_d, ax=ax_d, shrink=0.65, pad=0.18, label='Graph distance (steps)')
ax_d.text(-0.18, 1.09, 'D', transform=ax_d.transAxes, fontsize=16, fontweight='bold')

# ── Panel E: Group-averaged shortest-path ─────────────────────────────────
ax_e = fig.add_subplot(gs[2, 0])
np.fill_diagonal(grp_sp, np.nan)  # mask diagonal for clarity
im_e = ax_e.imshow(grp_sp, aspect='auto', cmap='Blues', vmin=0, vmax=3)
ax_e.set_title('Mean shortest-path by functional group', fontsize=9, fontweight='bold', pad=5)
ax_e.set_xticks(range(n_grp))
ax_e.set_xticklabels(group_order, fontsize=6, rotation=45, ha='right')
ax_e.set_yticks(range(n_grp))
ax_e.set_yticklabels(group_order, fontsize=6)
# Color x tick labels by group color
for tick, g in zip(ax_e.get_xticklabels(), group_order):
    tick.set_color(GROUP_COLORS.get(g, 'black'))
    tick.set_fontweight('bold')
for tick, g in zip(ax_e.get_yticklabels(), group_order):
    tick.set_color(GROUP_COLORS.get(g, 'black'))
    tick.set_fontweight('bold')
# Annotate cells with value
for i in range(n_grp):
    for j in range(n_grp):
        if not np.isnan(grp_sp[i, j]):
            ax_e.text(j, i, f'{grp_sp[i,j]:.1f}', ha='center', va='center',
                      fontsize=5, color='white' if grp_sp[i,j] > 2.0 else 'black')
plt.colorbar(im_e, ax=ax_e, shrink=0.65, pad=0.05, label='Mean distance (steps)')
ax_e.text(-0.18, 1.09, 'E', transform=ax_e.transAxes, fontsize=16, fontweight='bold')
np.fill_diagonal(grp_sp, 0)  # restore

# ── Panel F: Path-length distribution histogram ────────────────────────────
ax_f = fig.add_subplot(gs[2, 1])
# All pairwise off-diagonal values
sp_vals_flat = sp_matrix.values.copy().astype(float)
np.fill_diagonal(sp_vals_flat, np.nan)
dists = sp_vals_flat[~np.isnan(sp_vals_flat)].flatten()
dists = dists[dists < 9]  # exclude unreachable (=9)
bins = np.arange(-0.5, 9.5, 1)
counts, edges = np.histogram(dists, bins=bins)
ax_f.bar(edges[:-1]+0.5, counts, width=0.8, color='#338FBF', edgecolor='white', linewidth=0.5)
ax_f.set_xlabel('Shortest-path distance (steps)', fontsize=8)
ax_f.set_ylabel('Bait-pair count', fontsize=8)
ax_f.set_title('Pairwise path-length distribution\n(bait-to-bait via shared preys)',
               fontsize=9, fontweight='bold', pad=5)
ax_f.set_xticks(range(1, 4))
ax_f.spines['top'].set_visible(False)
ax_f.spines['right'].set_visible(False)
# Annotate bars (only distances 1–3 are present in our dense network)
for i, cnt in enumerate(counts[1:4], 1):
    if cnt > 0:
        ax_f.text(i, cnt + 1, str(cnt), ha='center', va='bottom', fontsize=6, fontweight='bold')
ax_f.text(-0.12, 1.03, 'F', transform=ax_f.transAxes, fontsize=16, fontweight='bold')

# ── Panel G: Undirected bait co-purification network ─────────────────────
ax_g = fig.add_subplot(gs[3, :])
ax_g.set_facecolor('white')
ax_g.set_aspect('equal')
ax_g.axis('off')

# Draw on the spring layout
n_colors_u = node_colors(list(G_undir.nodes()))

# Edge widths and colors
edges_u = list(G_undir.edges(data='weight', default=1))
if edges_u:
    wts = np.array([e[2] for e in edges_u], dtype=float)
    max_wu = wts.max() if wts.max() > 0 else 1
    for (u, v, w) in edges_u:
        lw = 0.3 + 1.5 * (w / max_wu)
        al = 0.2 + 0.6 * (w / max_wu)
        ax_g.plot([pos_spring[u][0], pos_spring[v][0]],
                  [pos_spring[u][1], pos_spring[v][1]],
                  color='#888888', lw=lw, alpha=al, zorder=1)

nx.draw_networkx_nodes(G_undir, pos_spring, ax=ax_g,
                        node_color=node_colors(list(G_undir.nodes())),
                        node_size=250, linewidths=0.5, edgecolors='white')

# Node labels
for b, (x, y) in pos_spring.items():
    ax_g.text(x, y - 0.08, fmt_bait(b), fontsize=4.5,
              ha='center', va='top', fontweight='bold',
              color='black', zorder=3)

ax_g.set_title('Undirected bait co-purification network\n(edge: A pulls down B or B pulls down A)',
               fontsize=9, fontweight='bold', pad=5)
ax_g.text(-0.01, 1.03, 'G', transform=ax_g.transAxes, fontsize=16, fontweight='bold')

# Add degrees as a note
deg_u = dict(G_undir.degree())
top3 = sorted(deg_u.items(), key=lambda x: x[1], reverse=True)[:3]
note = 'Top degree: ' + ', '.join(f'{fmt_bait(b)} ({d})' for b, d in top3)
ax_g.text(0.5, -0.01, note, transform=ax_g.transAxes,
          fontsize=6, ha='center', va='top', color='#555555', style='italic')

# ────────────────────────────────────────────────────────────────────────────
tmp_path = OUTDIR + "figureS3_network_topology.pdf"
fig.savefig(tmp_path, format='pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {tmp_path}")

# Copy to MS results/ and assembled folder
import shutil
shutil.copy2(tmp_path, MSDIR + "figureS3_network_topology.pdf")
target = MSDIR + "Assembled_Figures/current_2026-07-02/Supplementary_Figure_S3_AP-MS_Network_Topology.pdf"
shutil.copy2(tmp_path, target)
print(f"Copied to: {target}")

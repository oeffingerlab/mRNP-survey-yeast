"""
Panel order (2026-10-27): A–D weighted shortest-path panels, E directed network, F betweenness, G undirected network.
Supplementary Figure S3 V2 — AP-MS Network Topology (WEIGHTED shortest paths)
Panels C and D replaced with WEIGHTED shortest-path heatmaps using Dijkstra
on the bait-prey bipartite graph with edge cost = 1/log10(SC+1).
All other panels (A, B, E, F, G) unchanged from V1.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import shutil, warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS,
                           fmt_bait, fmt_baits, CMAP_SP, PATHWAY_ORDER_APMS as _PO)
apply_style()
BAIT_GROUP = BAIT_GROUP_APMS

OUTDIR  = "data/"
MSDIR   = "output/"
DATADIR = "data/"

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

# ── Build directed bait-to-bait graph (for Panel A & B) ───────────────────
G_dir = nx.DiGraph()
G_dir.add_nodes_from(baits)
for src in baits:
    for tgt in baits:
        if src != tgt and df_baits.loc[tgt, src] > 0 if tgt in df_baits.index else False:
            G_dir.add_edge(src, tgt, weight=float(df_baits.loc[tgt, src]))

# ── Build undirected bait-to-bait graph (for Panel G) ────────────────────
G_undir = nx.Graph()
G_undir.add_nodes_from(baits)
for src in baits:
    for tgt in baits:
        if src < tgt:
            w_fwd = float(df_baits.loc[tgt, src]) if tgt in df_baits.index else 0
            w_rev = float(df_baits.loc[src, tgt]) if src in df_baits.index else 0
            if w_fwd > 0 or w_rev > 0:
                G_undir.add_edge(src, tgt, weight=max(w_fwd, w_rev))

# ── WEIGHTED bipartite bait-prey graph ────────────────────────────────────
# Edge cost = 1/log10(SC+1): higher spectral count → shorter distance
# Two baits sharing a high-SC prey are very close; sharing only a weak prey = far
G_bp_w = nx.Graph()
for bait in baits:
    for prey, sc in df_baits[bait].items():
        if sc > 0:
            cost = 1.0 / np.log10(float(sc) + 1)
            G_bp_w.add_edge(bait, prey, weight=cost)

wsp_matrix = pd.DataFrame(np.inf, index=baits, columns=baits)
for src in baits:
    try:
        lengths = nx.single_source_dijkstra_path_length(G_bp_w, src, weight='weight')
        for tgt in baits:
            if tgt in lengths:
                wsp_matrix.loc[src, tgt] = lengths[tgt]
    except Exception:
        pass
wsp_matrix = wsp_matrix.replace(np.inf, 9.0)
np.fill_diagonal(wsp_matrix.values, 0)

offdiag = wsp_matrix.values[~np.eye(len(baits), dtype=bool)]
wsp_max_real = float(np.max(offdiag[offdiag < 9]))
print(f"Weighted SP: min={offdiag.min():.2f}, median={np.median(offdiag):.2f}, "
      f"p90={np.percentile(offdiag,90):.2f}, max={wsp_max_real:.2f}")

# ── Betweenness centrality ─────────────────────────────────────────────────
bc = nx.betweenness_centrality(G_dir, normalized=True)
bc_series = pd.Series(bc).reindex(baits).fillna(0)

# ── Hierarchical clustering on weighted SP ────────────────────────────────
wsp_sym = (wsp_matrix.values + wsp_matrix.values.T) / 2
np.fill_diagonal(wsp_sym, 0)
try:
    Z = linkage(squareform(np.clip(wsp_sym, 0, 5), checks=False), method='average')
    clust_order = leaves_list(Z)
    baits_clust = [baits[i] for i in clust_order]
except Exception:
    baits_clust = baits[:]

# ── Group-averaged weighted SP ─────────────────────────────────────────────
group_order = [g for g in GROUP_COLORS if any(BAIT_GROUP.get(b)==g for b in baits)]
group_baits = {g: [b for b in baits if BAIT_GROUP.get(b)==g] for g in group_order}
n_grp = len(group_order)
grp_wsp = np.zeros((n_grp, n_grp))
for i, gi in enumerate(group_order):
    for j, gj in enumerate(group_order):
        bi, bj = group_baits[gi], group_baits[gj]
        if bi and bj:
            vals = wsp_matrix.loc[bi, bj].values.flatten()
            vals_nondiag = vals if i != j else vals[vals > 0]
            vals_finite = vals_nondiag[vals_nondiag < 9]
            grp_wsp[i, j] = np.nanmean(vals_finite) if len(vals_finite) > 0 else 0

# ── Network layout ─────────────────────────────────────────────────────────
n = len(baits)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
pos_circ = {b: (np.cos(a), np.sin(a)) for b, a in zip(baits, angles)}
np.random.seed(42)
pos_spring = nx.spring_layout(G_undir, k=2.5, seed=42, iterations=150, weight='weight')

def node_colors(node_list):
    return [GROUP_COLORS.get(BAIT_GROUP.get(b,''), '#CCCCCC') for b in node_list]

def draw_colorstrip_x(ax, bait_list, h=0.025, gap=0.01):
    n = len(bait_list)
    s = ax.inset_axes([0, 1.0+gap, 1, h])
    s.set_xlim(0, n); s.set_ylim(0, 1); s.axis('off')
    for i, b in enumerate(bait_list):
        c = GROUP_COLORS.get(BAIT_GROUP.get(b,''), '#CCCCCC')
        s.add_patch(mpatches.Rectangle((i, 0), 1, 1, color=c))

def draw_colorstrip_y(ax, bait_list, w=0.025, gap=0.01):
    n = len(bait_list)
    s = ax.inset_axes([1.0+gap, 0, w, 1])
    s.set_xlim(0, 1); s.set_ylim(n, 0); s.axis('off')
    for i, b in enumerate(bait_list):
        c = GROUP_COLORS.get(BAIT_GROUP.get(b,''), '#CCCCCC')
        s.add_patch(mpatches.Rectangle((0, i), 1, 1, color=c))

# Colorbar vmax: use 95th percentile of real distances for better contrast
VMAX = round(np.percentile(offdiag[offdiag < 9], 98), 1)
print(f"Colorbar vmax: {VMAX}")

# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(11, 22))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(4, 2, figure=fig,
                       height_ratios=[0.9, 0.9, 1.6, 1.4],
                       hspace=0.55, wspace=0.40,
                       left=0.10, right=0.92, top=0.96, bottom=0.04)

# ── Panel E (variable ax_a): Directed bait-to-bait network ────────────────────────────────
ax_a = fig.add_subplot(gs[2, 0])
ax_a.set_facecolor('white'); ax_a.set_aspect('equal'); ax_a.axis('off')
nx.draw_networkx_nodes(G_dir, pos_circ, ax=ax_a,
                        node_color=node_colors(baits), node_size=300,
                        linewidths=0.5, edgecolors='white')
edges = list(G_dir.edges(data='weight', default=1))
if edges:
    weights = np.array([e[2] for e in edges], dtype=float)
    max_w = weights.max() or 1
    for (u, v, w) in edges:
        lw = 0.5 + 2.0 * (w / max_w)
        al = 0.3 + 0.6 * (w / max_w)
        c = GROUP_COLORS.get(BAIT_GROUP.get(u,''), '#AAAAAA')
        ax_a.annotate('', xy=pos_circ[v], xytext=pos_circ[u],
                      arrowprops=dict(arrowstyle='->', color=c, lw=lw, alpha=al,
                                      connectionstyle='arc3,rad=0.1'))
for b, (x, y) in pos_circ.items():
    ha = 'left' if x > 0 else 'right'
    va = 'bottom' if y > 0 else 'top'
    ax_a.text(x + 0.08*np.sign(x) if abs(x)>0.1 else x,
              y + 0.08*np.sign(y) if abs(y)>0.1 else y,
              fmt_bait(b), fontsize=4.5, ha=ha, va=va, fontweight='bold', color='black')
ax_a.set_ylim(-1.35, 1.55)
ax_a.set_title('Directed bait-to-bait network\n(A→B: A pulls down B as prey)',
               fontsize=9, fontweight='bold', pad=20)
ax_a.text(-0.05, 1.03, 'E', transform=ax_a.transAxes, fontsize=16, fontweight='bold')
leg_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
               if any(BAIT_GROUP.get(b)==g for b in baits)]
ax_a.legend(handles=leg_patches, loc='upper left', bbox_to_anchor=(1.02, 1.0),
            fontsize=6, title='Functional group', title_fontsize=6.5,
            frameon=True, ncol=1, handlelength=1.2, labelspacing=0.4)

# ── Panel F (variable ax_b): Betweenness centrality ───────────────────────────────────────
ax_b = fig.add_subplot(gs[2, 1])
bc_sorted = bc_series.reindex(baits)
x_b = np.arange(len(baits))
ax_b.bar(x_b, bc_sorted.values, color=node_colors(baits),
         edgecolor='white', linewidth=0.3, width=0.75)
ax_b.set_xticks(x_b)
ax_b.set_xticklabels(fmt_baits(baits), fontsize=4.5, rotation=90)
ax_b.set_ylabel('Betweenness centrality\n(directed, normalized)', fontsize=8)
ax_b.set_title('Betweenness centrality per bait\n(directed bait-to-bait graph)',
               fontsize=9, fontweight='bold', pad=5)
ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)
ax_b.axhline(0, color='black', linewidth=0.5)
for prot, val in bc_sorted.nlargest(5).items():
    xi = baits.index(prot)
    ax_b.text(xi, val+0.002, f'{val:.2f}', ha='center', va='bottom', fontsize=4.5, fontweight='bold')
ax_b.text(-0.10, 1.03, 'F', transform=ax_b.transAxes, fontsize=16, fontweight='bold')

# ── Panel A (variable ax_c): WEIGHTED SP heatmap — pathway order ──────────────────────────
ax_c = fig.add_subplot(gs[0, 0])
wsp_vals = wsp_matrix.reindex(index=baits, columns=baits).values.astype(float)
wsp_vals_disp = np.where(wsp_vals >= 9, np.nan, wsp_vals)
im_c = ax_c.imshow(wsp_vals_disp, aspect='equal', cmap='Blues', vmin=0, vmax=VMAX, rasterized=True)
ax_c.set_title('Weighted shortest-path distance\n(pathway order; edge cost = 1/log₁₀(SC+1))',
               fontsize=8.5, fontweight='bold', pad=5)
ax_c.set_xticks(range(len(baits))); ax_c.set_yticks(range(len(baits)))
ax_c.set_xticklabels(fmt_baits(baits), fontsize=4.5, rotation=90)
ax_c.set_yticklabels(fmt_baits(baits), fontsize=4.5)
draw_colorstrip_x(ax_c, baits); draw_colorstrip_y(ax_c, baits)
cax_c = ax_c.inset_axes([1.08, 0.15, 0.05, 0.70])
cb_c = plt.colorbar(im_c, cax=cax_c)
cb_c.set_label('Weighted distance\n(lower = more shared\nhigh-SC preys)', fontsize=6)
cb_c.ax.tick_params(labelsize=5)
ax_c.text(-0.18, 1.09, 'A', transform=ax_c.transAxes, fontsize=16, fontweight='bold')

# ── Panel B (variable ax_d): WEIGHTED SP heatmap — hierarchical order ─────────────────────
ax_d = fig.add_subplot(gs[0, 1])
wsp_clust = wsp_matrix.reindex(index=baits_clust, columns=baits_clust).values.astype(float)
wsp_clust_disp = np.where(wsp_clust >= 9, np.nan, wsp_clust)
im_d = ax_d.imshow(wsp_clust_disp, aspect='equal', cmap='Blues', vmin=0, vmax=VMAX, rasterized=True)
ax_d.set_title('Weighted shortest-path distance\n(hierarchical cluster order)',
               fontsize=8.5, fontweight='bold', pad=5)
ax_d.set_xticks(range(len(baits_clust))); ax_d.set_yticks(range(len(baits_clust)))
ax_d.set_xticklabels(fmt_baits(baits_clust), fontsize=4.5, rotation=90)
ax_d.set_yticklabels(fmt_baits(baits_clust), fontsize=4.5)
draw_colorstrip_x(ax_d, baits_clust); draw_colorstrip_y(ax_d, baits_clust)
cax_d = ax_d.inset_axes([1.08, 0.15, 0.05, 0.70])
cb_d = plt.colorbar(im_d, cax=cax_d)
cb_d.set_label('Weighted distance\n(lower = more shared\nhigh-SC preys)', fontsize=6)
cb_d.ax.tick_params(labelsize=5)
ax_d.text(-0.18, 1.09, 'B', transform=ax_d.transAxes, fontsize=16, fontweight='bold')

# ── Panel C (variable ax_e): Group-averaged WEIGHTED SP ───────────────────────────────────
ax_e = fig.add_subplot(gs[1, 0])
grp_disp = grp_wsp.copy()
np.fill_diagonal(grp_disp, np.nan)
im_e = ax_e.imshow(grp_disp, aspect='auto', cmap='Blues', vmin=0, vmax=VMAX, rasterized=True)
ax_e.set_title('Mean weighted shortest-path by functional group', fontsize=9, fontweight='bold', pad=5)
ax_e.set_xticks(range(n_grp)); ax_e.set_yticks(range(n_grp))
ax_e.set_xticklabels(group_order, fontsize=6, rotation=45, ha='right')
ax_e.set_yticklabels(group_order, fontsize=6)
for tick, g in zip(ax_e.get_xticklabels(), group_order):
    tick.set_color(GROUP_COLORS.get(g,'black')); tick.set_fontweight('bold')
for tick, g in zip(ax_e.get_yticklabels(), group_order):
    tick.set_color(GROUP_COLORS.get(g,'black')); tick.set_fontweight('bold')
for i in range(n_grp):
    for j in range(n_grp):
        if not np.isnan(grp_disp[i, j]):
            ax_e.text(j, i, f'{grp_disp[i,j]:.2f}', ha='center', va='center',
                      fontsize=4.5, color='white' if grp_disp[i,j] > VMAX*0.65 else 'black')
plt.colorbar(im_e, ax=ax_e, shrink=0.65, pad=0.05, label='Mean weighted distance')
ax_e.text(-0.18, 1.09, 'C', transform=ax_e.transAxes, fontsize=16, fontweight='bold')

# ── Panel D (variable ax_f): Weighted distance distribution ────────────────────────────────
ax_f = fig.add_subplot(gs[1, 1])
finite_dists = offdiag[offdiag < 9]
bins = np.arange(0, VMAX + 0.15, 0.1)
counts, edges = np.histogram(finite_dists, bins=bins)
ax_f.bar(edges[:-1], counts, width=0.09, color='#338FBF', edgecolor='none', linewidth=0)
ax_f.set_xlabel('Weighted shortest-path distance', fontsize=8)
ax_f.set_ylabel('Bait-pair count', fontsize=8)
ax_f.set_title('Pairwise weighted-distance distribution\n(bait-to-bait via shared preys)',
               fontsize=9, fontweight='bold', pad=5)
ax_f.spines['top'].set_visible(False); ax_f.spines['right'].set_visible(False)
# Annotate median and quartiles
ax_f.axvline(np.median(finite_dists), color='#C93A3B', lw=1.2, ls='--',
             label=f'Median={np.median(finite_dists):.2f}')
ax_f.axvline(np.percentile(finite_dists,25), color='#888888', lw=0.8, ls=':',
             label=f'Q1={np.percentile(finite_dists,25):.2f}')
ax_f.axvline(np.percentile(finite_dists,75), color='#888888', lw=0.8, ls=':',
             label=f'Q3={np.percentile(finite_dists,75):.2f}')
ax_f.legend(fontsize=6, frameon=False)
ax_f.text(-0.12, 1.03, 'D', transform=ax_f.transAxes, fontsize=16, fontweight='bold')

# ── Panel G: Undirected bait co-purification network ─────────────────────
ax_g = fig.add_subplot(gs[3, :])
ax_g.set_facecolor('white'); ax_g.set_aspect('equal'); ax_g.axis('off')
edges_u = list(G_undir.edges(data='weight', default=1))
if edges_u:
    wts = np.array([e[2] for e in edges_u], dtype=float)
    max_wu = wts.max() or 1
    for (u, v, w) in edges_u:
        ax_g.plot([pos_spring[u][0], pos_spring[v][0]],
                  [pos_spring[u][1], pos_spring[v][1]],
                  color='#888888', lw=0.3+1.5*(w/max_wu), alpha=0.2+0.6*(w/max_wu), zorder=1)
nx.draw_networkx_nodes(G_undir, pos_spring, ax=ax_g,
                        node_color=node_colors(list(G_undir.nodes())),
                        node_size=250, linewidths=0.5, edgecolors='white')
for b, (x, y) in pos_spring.items():
    ax_g.text(x, y-0.08, fmt_bait(b), fontsize=4.5,
              ha='center', va='top', fontweight='bold', color='black', zorder=3)
ax_g.set_title('Undirected bait co-purification network\n(edge: A pulls down B or B pulls down A)',
               fontsize=9, fontweight='bold', pad=5)
ax_g.text(-0.01, 1.03, 'G', transform=ax_g.transAxes, fontsize=16, fontweight='bold')
deg_u = dict(G_undir.degree())
top3 = sorted(deg_u.items(), key=lambda x: x[1], reverse=True)[:3]
ax_g.text(0.5, -0.01, 'Top degree: '+', '.join(f'{fmt_bait(b)} ({d})' for b,d in top3),
          transform=ax_g.transAxes, fontsize=6, ha='center', va='top',
          color='#555555', style='italic')

# ── Save ──────────────────────────────────────────────────────────────────
os.makedirs(MSDIR, exist_ok=True)
tmp_path = MSDIR + "figureS3_network_topology_V2.pdf"
fig.savefig(tmp_path, format='pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {tmp_path}")

# Optional: Ghostscript-flattened copy for Illustrator (skipped if gs is not installed)
import subprocess
if shutil.which('gs'):
    gs_out = MSDIR + "figureS3_network_topology_V2_flattened.pdf"
    subprocess.run([
        'gs', '-dBATCH', '-dNOPAUSE', '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4', '-dPDFSETTINGS=/prepress',
        f'-sOutputFile={gs_out}', tmp_path
    ], check=True, capture_output=True)
    print(f"Ghostscript-flattened: {gs_out}")
else:
    print("Ghostscript (gs) not found; flattened copy skipped.")

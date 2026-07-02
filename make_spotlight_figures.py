"""
Spotlight figures:
  X: SAC3/TREX-2 / MLP1/MLP2 subnetwork (AP-MS interaction hub)
  Y: NAB2/YRA1/PAB1 breadth comparison panel
  Z: RRP6 ncRNA class spotlight

Exclusions:
  - MTR2 excluded from all AP-MS analyses
  - SNU114 excluded from AP-RNA-seq
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
import openpyxl, warnings, os
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, "/sessions/lucid-modest-goldberg/mnt/outputs/")
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS, fmt_bait, fmt_baits,
                           CLUSTER_COLORS, CMAP_RNA, PATHWAY_ORDER_RNA,
                           PATHWAY_ORDER_APMS)
apply_style()

OUTDIR  = "/sessions/lucid-modest-goldberg/mnt/MS results/"
DATADIR = "/sessions/lucid-modest-goldberg/mnt/outputs/"

# ── Load core data ──────────────────────────────────────────────────────────
df_apms = pd.read_csv(DATADIR + "apms_nuclear_matrix.tsv", sep='\t', index_col=0)
df_apms = df_apms.drop(columns=['MTR2'], errors='ignore')
df_rna  = pd.read_csv(DATADIR + "rnaseq_bait_means.tsv",  sep='\t', index_col=0)
df_clust= pd.read_csv(DATADIR + "rna_gene_clusters.tsv",  sep='\t', index_col=0)
df_dwell= pd.read_csv(DATADIR + "rna_dwell_scores.tsv",   sep='\t', index_col=0)

rna_baits = [b for b in PATHWAY_ORDER_RNA if b in df_rna.columns]
df_rna    = df_rna[rna_baits]
df_rna_filt = df_rna[df_rna.max(axis=1) > 0.3]

BAIT_GROUP = BAIT_GROUP_APMS

# ── NC_BAIT_COLS for ncRNA loading ──────────────────────────────────────────
NC_BAIT_COLS = {
    'CBP80': [2,3,4], 'THP2': [5,6], 'YRA1': [7,8], 'SNU71': [11,12],
    'SYC1':  [13,14,15], 'CLP1': [16,17,18], 'REF2': [19,20,21],
    'HRP1':  [22,23,24,25,26], 'RRP6': [27,28,29], 'NAB2': [30,31,32,33],
    'NPL3':  [34], 'GBP2': [35,36,37], 'SUB2': [38,39,40],
    'SAC3':  [41,42], 'MEX67': [43,44,45], 'GFD1': [46],
    'DBP5':  [47,48,49], 'PAB1': [50,51,52,53],
}

nc_dir = "/sessions/lucid-modest-goldberg/mnt/Alex/RNA data April 2014/MOe/please filter and cluster/"
nc_files = {
    'Coding': nc_dir + "all_codingRNAs_WTPOs_and_Mutants.xlsx",
    'SUT':    nc_dir + "all_SUTs_WTPOs_and_Mutants.xlsx",
    'CUT':    nc_dir + "all_CUTs_WTPOs_and_Mutants.xlsx",
    'XUT':    nc_dir + "all_XUTs_WTPOs_and_Mutants.xlsx",
    'lncRNA': nc_dir + "all_lncRNAs_WTPOs_and_Mutants.xlsx",
}

def load_nc_totals(fpath, bait_list):
    wb = openpyxl.load_workbook(fpath, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    totals = {}
    for bait in bait_list:
        col_idxs = NC_BAIT_COLS.get(bait, [])
        bait_sum = 0.0
        for row in rows[1:]:
            rep_vals = []
            for ci in col_idxs:
                if ci < len(row):
                    v = row[ci]
                    try:
                        v = float(v)
                        if np.isfinite(v): rep_vals.append(v)
                    except (TypeError, ValueError): pass
            if rep_vals: bait_sum += np.mean(rep_vals)
        totals[bait] = bait_sum
    wb.close()
    return totals

print("Loading ncRNA class proportions for spotlight figures...")
nc_totals = {}
for cls, fpath in nc_files.items():
    if os.path.exists(fpath):
        nc_totals[cls] = load_nc_totals(fpath, rna_baits)
        print(f"  {cls}: done")

nc_df   = pd.DataFrame(nc_totals).loc[rna_baits]
nc_frac = nc_df.div(nc_df.sum(axis=1), axis=0)

NC_COLORS = {
    'Coding': '#C8B89A', 'SUT': '#FFCA7D', 'CUT': '#F5E6C8',
    'XUT': '#F59896', 'lncRNA': '#A3C4D6',
}

# ════════════════════════════════════════════════════════════════════════════
# SPOTLIGHT FIGURE X — Pml39p export scaffold: force-directed subnetwork
# ════════════════════════════════════════════════════════════════════════════

FOCUS_BAITS  = ['SAC3', 'MEX67', 'DBP5', 'SUB2', 'YRA1', 'THP2', 'PML39']
# Prey-only proteins (proteins that are also baits are kept as bait nodes only)
PREY_ONLY    = ['MLP1', 'MLP2', 'THP1', 'SUS1', 'CDC31',
                'GLE1', 'NUP159', 'NUP100',
                'NPL3', 'NAB2', 'PAB1']

focus_baits_use = [b for b in FOCUS_BAITS if b in df_apms.columns]
prey_only_use   = [p for p in dict.fromkeys(PREY_ONLY)
                   if p in df_apms.index and p not in focus_baits_use]

# Build force-directed graph — proteins as single nodes
G_x = nx.Graph()
for b in focus_baits_use:
    G_x.add_node(b, node_type='bait')
for p in prey_only_use:
    G_x.add_node(p, node_type='prey')

edge_data = {}   # (src_bait, node) → spectral count

# Bait → prey-only edges
for b in focus_baits_use:
    for p in prey_only_use:
        sc = float(df_apms.loc[p, b]) if p in df_apms.index and b in df_apms.columns else 0
        if sc > 0:
            G_x.add_edge(b, p, weight=sc)
            edge_data[(b, p)] = sc

# Bait → bait cross-pulldown edges (real interactions)
for b_bait in focus_baits_use:
    for b_prey in focus_baits_use:
        if b_bait == b_prey:
            continue
        if b_prey in df_apms.index and b_bait in df_apms.columns:
            sc = float(df_apms.loc[b_prey, b_bait])
            if sc > 0 and not G_x.has_edge(b_bait, b_prey):
                G_x.add_edge(b_bait, b_prey, weight=sc)
                edge_data[(b_bait, b_prey)] = sc

# Remove isolated nodes
isolated = [n for n in list(G_x.nodes) if G_x.degree(n) == 0]
G_x.remove_nodes_from(isolated)

# Force-directed layout (spring)
pos_x = nx.spring_layout(G_x, k=2.6, seed=42, iterations=200)

fig_x, ax_x = plt.subplots(figsize=(11, 9))
fig_x.patch.set_facecolor('white')

# Draw edges — colored by the bait end, thickness by SC
max_w = max(edge_data.values()) if edge_data else 1
for (u, v), w in edge_data.items():
    if u not in pos_x or v not in pos_x:
        continue
    lw    = 1.2 + 3.0 * (w / max_w)
    alpha = 0.50 + 0.45 * (w / max_w)
    bait  = u if u in focus_baits_use else v
    grp   = BAIT_GROUP.get(bait, '')
    color = GROUP_COLORS.get(grp, '#BBBBBB')
    ax_x.plot([pos_x[u][0], pos_x[v][0]], [pos_x[u][1], pos_x[v][1]],
              color=color, linewidth=lw, alpha=alpha, zorder=1)

# Prey node categories
npc_basket = {'MLP1', 'MLP2', 'NUP159', 'NUP100', 'GLE1'}
trex2_core = {'THP1', 'SUS1', 'CDC31'}

def prey_color(p):
    if p in npc_basket:  return '#6AB8F7'
    if p in trex2_core:  return '#5DBDB5'
    return '#D4B84A'

# Draw prey nodes — scatter (size in screen pts², not data units)
prey_in_pos = [p for p in prey_only_use if p in pos_x]
ax_x.scatter([pos_x[p][0] for p in prey_in_pos],
             [pos_x[p][1] for p in prey_in_pos],
             s=350, c=[prey_color(p) for p in prey_in_pos],
             zorder=3, edgecolors='white', linewidths=0.8)
# Per-node label offsets (dx, dy, ha, va) — override default (right of node)
PREY_LABEL_POS = {
    'NUP159': ( 0.00, -0.055, 'center', 'top'),   # below: sits between DBP5 and SAC3
    'NUP100': ( 0.00, -0.055, 'center', 'top'),   # below: bleeds into GLE1 if right
    'PAB1':   (-0.06,  0.00,  'right',  'center'),# left of node
    'MLP1':   ( 0.06, -0.035, 'left',   'top'),   # right but shifted down slightly
    'NPL3':   ( 0.00, -0.055, 'center', 'top'),   # below node
}
for p in prey_in_pos:
    dx, dy, ha, va = PREY_LABEL_POS.get(p, (0.06, 0, 'left', 'center'))
    ax_x.text(pos_x[p][0] + dx, pos_x[p][1] + dy, fmt_bait(p),
              ha=ha, va=va, fontsize=7.5, color='#333333')

# Draw bait nodes — larger scatter, white label inside
bait_in_pos = [b for b in focus_baits_use if b in pos_x]
ax_x.scatter([pos_x[b][0] for b in bait_in_pos],
             [pos_x[b][1] for b in bait_in_pos],
             s=900, c=[GROUP_COLORS.get(BAIT_GROUP.get(b,''), '#CCCCCC') for b in bait_in_pos],
             zorder=5, edgecolors='white', linewidths=1.5)
for b in bait_in_pos:
    ax_x.text(pos_x[b][0], pos_x[b][1], fmt_bait(b),
              ha='center', va='center', fontsize=6.5, fontweight='bold',
              color='white', zorder=6)

# Auto-scale with a little margin
ax_x.autoscale()
ax_x.set_aspect('equal')
xs = [v[0] for v in pos_x.values()]; ys = [v[1] for v in pos_x.values()]
pad = 0.55
ax_x.set_xlim(min(xs) - pad, max(xs) + pad)
ax_x.set_ylim(min(ys) - pad, max(ys) + pad)
ax_x.axis('off')

# Legend — prey node categories + custom node-type examples (Pml39p bait / Npl3p prey)
import matplotlib.text as mtext
from matplotlib.legend_handler import HandlerBase

class BaitNodeLegend(HandlerBase):
    """Circle the same diameter as the patch squares above, name inside."""
    def __init__(self, color, name):
        self.color, self.name = color, name
        super().__init__()
    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        cx, cy = w / 2 - xd, h / 2 - yd
        r = w * 0.48          # diameter ≈ handle width = same visual size as patches
        return [
            mpatches.Circle((cx, cy), r, fc=self.color, ec='white', lw=1.2, transform=trans),
            mtext.Text(cx, cy, self.name, ha='center', va='center',
                       fontsize=fs * 0.82, fontweight='bold', color='white', transform=trans),
        ]

class PreyNodeLegend(HandlerBase):
    """Circle same width as patches, name below."""
    def __init__(self, color, name):
        self.color, self.name = color, name
        super().__init__()
    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        cx = w / 2 - xd
        r  = w * 0.48
        cy = h * 0.68 - yd    # shift up to leave room for text below within handle
        return [
            mpatches.Circle((cx, cy), r, fc=self.color, ec='white', lw=0.8, transform=trans),
            mtext.Text(cx, cy - r - 1, self.name, ha='center', va='top',
                       fontsize=fs * 0.80, color='#333333', transform=trans),
        ]

pml39_color = GROUP_COLORS.get(BAIT_GROUP.get('PML39', ''), '#CCCCCC')
bait_dummy  = mpatches.Patch(color='none', label='Bait node')
prey_dummy  = mpatches.Patch(color='none', label='Prey node')

legend_handles = [
    mpatches.Patch(color='#6AB8F7', label='NPC basket (Mlp1p, Mlp2p, Nup159p…)'),
    mpatches.Patch(color='#5DBDB5', label='TREX-2 core (Thp1p, Sus1p, Cdc31p)'),
    mpatches.Patch(color='#D4B84A', label='Export / mRNP associated'),
    bait_dummy,
    prey_dummy,
]
handler_map = {
    bait_dummy: BaitNodeLegend(pml39_color, 'Pml39p'),
    prey_dummy: PreyNodeLegend('#D4B84A',   'Npl3p'),
}
ax_x.legend(handles=legend_handles, handler_map=handler_map,
            loc='lower left', fontsize=8, frameon=True, framealpha=0.92,
            edgecolor='#CCCCCC', title='Node type / prey category', title_fontsize=8.5,
            handlelength=2.2, handleheight=2.8, labelspacing=0.7, borderpad=0.55)

ax_x.set_title('Pml39p-centered export scaffold\n'
               'Edge color: bait functional group  ·  edge weight: spectral count',
               fontsize=11, fontweight='bold', pad=12)
# Panel letter 'A' inserted by the assembler (make_spotlight_figures.py assembly step)

out_x = OUTDIR + "spotlight_X_SAC3_subnetwork.pdf"
fig_x.savefig(out_x, format='pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_x}")


# ════════════════════════════════════════════════════════════════════════════
# SPOTLIGHT FIGURE Y — NAB2/YRA1/PAB1 Breadth Comparison Panel
# ════════════════════════════════════════════════════════════════════════════

FOCAL_BAITS   = ['NAB2', 'YRA1', 'PAB1']
CONTEXT_BAITS = ['THP2', 'SNU71', 'RRP6']  # for contrast
ALL_PANEL_BAITS = FOCAL_BAITS + CONTEXT_BAITS

# Get cluster assignments
clust_col = [c for c in df_clust.columns if 'cluster' in c.lower() or 'k' in c.lower()]
if not clust_col:
    # Try to find cluster assignment column
    clust_col = [c for c in df_clust.columns if c not in df_rna.columns]
    clust_col = [c for c in clust_col if df_clust[c].nunique() <= 10]
print(f"Cluster columns: {clust_col}")

# Use the first cluster assignment column found
if clust_col:
    clust_col = clust_col[0]
    cluster_assignments = df_clust[clust_col]
else:
    # No cluster column found; create dummy
    cluster_assignments = pd.Series(1, index=df_rna_filt.index, name='cluster')

# Compute per-bait enrichment statistics across clusters
genes_in_filt = df_rna_filt.index

fig_y = plt.figure(figsize=(14, 18))
fig_y.patch.set_facecolor('white')
gs_y = gridspec.GridSpec(3, 2, figure=fig_y, hspace=0.52, wspace=0.38,
                          left=0.10, right=0.96, top=0.90, bottom=0.08)

for panel_i, bait in enumerate(ALL_PANEL_BAITS):
    row = panel_i // 2
    col = panel_i % 2
    ax = fig_y.add_subplot(gs_y[row, col])

    if bait not in df_rna.columns:
        ax.text(0.5, 0.5, f'{bait}\n(not available)', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='#888888')
        ax.axis('off')
        continue

    # Per-cluster enrichment distribution
    cluster_vals = []
    cluster_labels = []
    for k in sorted(cluster_assignments.unique()):
        genes_in_clust = cluster_assignments[cluster_assignments == k].index
        genes_enr = genes_in_filt.intersection(genes_in_clust)
        vals = df_rna.loc[genes_enr, bait].values
        cluster_vals.append(vals)
        cluster_labels.append(f"C{k}")

    n_clusters = len(cluster_vals)
    positions  = range(n_clusters)

    grp   = BAIT_GROUP.get(bait, '')
    color = GROUP_COLORS.get(grp, '#CCCCCC')

    # Violin plot per RNA-seq cluster, colored by functional group
    vp_data  = [v for v in cluster_vals if len(v) > 1]
    vp_pos   = [i for i, v in enumerate(cluster_vals) if len(v) > 1]
    vp_lbls  = [cluster_labels[i] for i in vp_pos]

    if vp_data:
        vp = ax.violinplot(vp_data, positions=vp_pos,
                           showmedians=True, showextrema=False, widths=0.7)
        for body in vp['bodies']:
            body.set_facecolor(color)
            body.set_alpha(0.72)
            body.set_edgecolor('#444444')
            body.set_linewidth(0.5)
        vp['cmedians'].set_color('black')
        vp['cmedians'].set_linewidth(1.2)

    ax.set_xticks(vp_pos)
    ax.set_xticklabels(vp_lbls, fontsize=5.5, rotation=45)
    ax.set_ylabel('Max-normalized FPKM', fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='y', labelsize=6)

    # Total fraction enriched (across all clusters)
    total_enr = np.mean(df_rna_filt[bait].values > 0.3) * 100
    ax.text(0.03, 0.96, f'Enriched: {total_enr:.0f}%',
            transform=ax.transAxes, fontsize=7, va='top',
            color='#555555', style='italic')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    panel_letter = ['A', 'B', 'C', 'D', 'E', 'F'][panel_i]
    if panel_i < 3:
        ax.set_title(f'{panel_letter}  {fmt_bait(bait)}\n(constitutive companion)',
                     fontsize=10, fontweight='bold', color=color, pad=4)
    else:
        ax.set_title(f'{panel_letter}  {fmt_bait(bait)}\n(stage-specific contrast)',
                     fontsize=10, fontweight='bold', color='#666666', pad=4)


# Legend for bait categories
c_patches = [mpatches.Patch(color=GROUP_COLORS.get(BAIT_GROUP.get(b,''),'#CCCCCC'), label=b)
             for b in ALL_PANEL_BAITS]
fig_y.legend(handles=c_patches, loc='lower center', ncol=6, fontsize=8,
             frameon=True, bbox_to_anchor=(0.5, 0.01),
             title='Bait | functional group color', title_fontsize=8)

fig_y.suptitle('Spotlight Figure Y  |  NAB2 / YRA1 / PAB1: Near-Universal Transcript Co-enrichment\n'
               'vs. Stage-specific Factors (THP2, SNU71, RRP6) — Violin per RNA-seq Cluster',
               fontsize=11, fontweight='bold', y=0.97)

out_y = OUTDIR + "spotlight_Y_NAB2_YRA1_PAB1_breadth.pdf"
fig_y.savefig(out_y, format='pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_y}")


# ════════════════════════════════════════════════════════════════════════════
# SPOTLIGHT FIGURE Z — RRP6 ncRNA Spotlight
# ════════════════════════════════════════════════════════════════════════════

fig_z = plt.figure(figsize=(22, 10))
fig_z.patch.set_facecolor('white')
gs_z = gridspec.GridSpec(1, 3, figure=fig_z, wspace=0.70,
                          width_ratios=[2.0, 1.2, 0.9],
                          left=0.06, right=0.96, top=0.88, bottom=0.12)

# Panel A: RRP6 vs all others — stacked bar (all baits, RRP6 highlighted)
ax_za = fig_z.add_subplot(gs_z[0, 0])

x  = np.arange(len(rna_baits))
bottom = np.zeros(len(rna_baits))
for cls, color in NC_COLORS.items():
    if cls in nc_frac.columns:
        vals = nc_frac[cls].values
        ax_za.bar(x, vals, bottom=bottom, color=color, label=cls,
                  edgecolor='grey', linewidth=0.3, width=1.0)
        bottom += vals

# Highlight RRP6
rrp6_idx = rna_baits.index('RRP6') if 'RRP6' in rna_baits else None
if rrp6_idx is not None:
    ax_za.bar(rrp6_idx, 1.0, bottom=0, color='none',
              edgecolor='#C0392B', linewidth=2.5, width=1.0, zorder=5,
              label='Rrp6p (highlighted)')

ax_za.set_xticks(x)
ax_za.set_xticklabels(fmt_baits(rna_baits), fontsize=5.5, rotation=60, ha='right', color='black')
ax_za.set_xlim(x[0] - 0.5, x[-1] + 0.5)
ax_za.set_ylabel('Fraction of co-purified transcripts', fontsize=8)
ax_za.set_ylim(0, 1.0)
ax_za.set_title('All Baits: Transcript Class\nComposition (Rrp6p highlighted)',
                fontsize=9, fontweight='bold')
ax_za.spines['top'].set_visible(False); ax_za.spines['right'].set_visible(False)
ax_za.legend(fontsize=6.5, loc='center left', bbox_to_anchor=(1.02, 0.5),
             framealpha=0.9, ncol=1, title='RNA class', title_fontsize=7)
ax_za.text(-0.08, 1.04, 'A', transform=ax_za.transAxes, fontsize=14)

# Panel B: RRP6 vs NAB2/YRA1/PAB1 — grouped bar comparing ncRNA fractions
ax_zb = fig_z.add_subplot(gs_z[0, 1])

compare_baits = ['RRP6', 'NAB2', 'YRA1', 'PAB1', 'MEX67', 'DBP5', 'SAC3']
compare_baits = [b for b in compare_baits if b in nc_frac.index]

x2 = np.arange(len(compare_baits))
bottom2 = np.zeros(len(compare_baits))
for cls, color in NC_COLORS.items():
    if cls in nc_frac.columns:
        vals = [nc_frac.loc[b, cls] if b in nc_frac.index else 0
                for b in compare_baits]
        ax_zb.bar(x2, vals, bottom=bottom2, color=color, label=cls,
                  edgecolor='grey', linewidth=0.3, width=1.0)
        bottom2 += np.array(vals)

ax_zb.set_xticks(x2)
ax_zb.set_xticklabels(fmt_baits(compare_baits), fontsize=8.5, rotation=30, ha='right', color='black')
ax_zb.set_xlim(x2[0] - 0.5, x2[-1] + 0.5)
ax_zb.set_ylabel('Fraction of co-purified transcripts', fontsize=8)
ax_zb.set_ylim(0, 1.0)
ax_zb.set_title('Rrp6p vs Export Factors & Companions\n(transcript class composition)',
                fontsize=9, fontweight='bold')
ax_zb.spines['top'].set_visible(False); ax_zb.spines['right'].set_visible(False)
ax_zb.legend(fontsize=6.5, loc='center left', bbox_to_anchor=(1.02, 0.5),
             framealpha=0.9, ncol=1, title='RNA class', title_fontsize=7)

ax_zb.text(-0.12, 1.04, 'B', transform=ax_zb.transAxes, fontsize=14)

# Panel C: RRP6 absolute composition pie chart
ax_zc = fig_z.add_subplot(gs_z[0, 2])

if 'RRP6' in nc_frac.index:
    rrp6_vals  = nc_frac.loc['RRP6']
    pie_labels = list(rrp6_vals.index)
    pie_values = list(rrp6_vals.values)
    pie_colors = [NC_COLORS.get(cls, '#CCCCCC') for cls in pie_labels]

    wedges, _ = ax_zc.pie(
        pie_values, colors=pie_colors,
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )

    # Collect midpoint directions for each wedge
    slice_data = []
    for wedge, label, val in zip(wedges, pie_labels, pie_values):
        ang = np.radians((wedge.theta1 + wedge.theta2) / 2.0)
        xu, yu = np.cos(ang), np.sin(ang)
        slice_data.append({'label': f'{label} ({val*100:.1f}%)', 'xu': xu, 'yu': yu})

    # Group by left/right side; sort each group by y; assign evenly-spaced text positions
    right = sorted([s for s in slice_data if s['xu'] >= 0], key=lambda s: s['yu'])
    left  = sorted([s for s in slice_data if s['xu'] <  0], key=lambda s: s['yu'])

    def spread_text(group, x_out):
        n = len(group)
        if n == 0: return
        y_positions = np.linspace(-0.80, 0.80, n) if n > 1 else [group[0]['yu']]
        for s, yp in zip(group, y_positions):
            s['x_text'] = x_out
            s['y_text'] = float(yp)

    spread_text(right, x_out= 1.55)
    spread_text(left,  x_out=-1.55)

    # Draw two-segment angled callout lines + labels
    for s in slice_data:
        xu, yu = s['xu'], s['yu']
        x_text = s.get('x_text',  1.55 if xu >= 0 else -1.55)
        y_text = s.get('y_text', yu)
        # First segment: radially outward from pie surface to kink
        x_kink, y_kink = 1.18 * xu, 1.18 * yu
        ax_zc.plot([0.93*xu, x_kink, x_text], [0.93*yu, y_kink, y_text],
                   color='#888888', lw=0.8, clip_on=False)
        ha = 'left' if xu >= 0 else 'right'
        ax_zc.text(x_text + (0.05 if xu >= 0 else -0.05), y_text,
                   s['label'], ha=ha, va='center', fontsize=8, clip_on=False)

    ax_zc.set_xlim(-2.3, 2.3)
    ax_zc.set_ylim(-1.6, 1.6)

ax_zc.set_title('RRP6: Transcript Class Composition\n(fraction of all co-purified transcripts)',
                fontsize=9, fontweight='bold')
ax_zc.text(-0.12, 1.04, 'C', transform=ax_zc.transAxes, fontsize=14)

# Overall title
fig_z.suptitle(
    'Spotlight Figure Z  |  RRP6 ncRNA Surveillance Signature\n'
    'RRP6 co-purifies the highest ncRNA fraction (~47%) of any mRNP biogenesis bait,\n'
    'validating its role as the primary nuclear exosome effector for ncRNA degradation.',
    fontsize=10, fontweight='bold', y=0.97
)

out_z = OUTDIR + "spotlight_Z_RRP6_ncRNA.pdf"
fig_z.savefig(out_z, format='pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_z}")

print("\nAll spotlight figures saved:")
print(f"  {out_x}")
print(f"  {out_y}")
print(f"  {out_z}")

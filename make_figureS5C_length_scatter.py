"""
Supplemental Figure 5C (formerly 5D; originally Figure 2C) — FPKM vs ORF length scatter
CBP80 (short-mRNA preference) vs GBP2 (broader distribution)
4 panels: full range (top) + zoomed (bottom) for each bait.
Raw FPKM loaded from coding RNA xlsx (replicate average).
Style: original layout, open circles, soft group colors, Helvetica.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import openpyxl, warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import apply_style, GROUP_COLORS, BAIT_GROUP_APMS, CLUSTER_COLORS, CLUSTER_ORDER, CLUSTER_LABELS
apply_style()

OUTDIR  = "output/"
DATADIR = "data/"
NC_DIR  = "data/ncRNA/"

# ── Column indices for CBP80 and GBP2 raw FPKM ────────────────────────────
# From NC_BAIT_COLS: CBP80=[2,3,4], GBP2=[35,36,37]
BAIT_COLS = {'CBP80': [2, 3, 4], 'GBP2': [35, 36, 37]}

# ── Load raw FPKM from coding RNA xlsx ────────────────────────────────────
print("Loading raw FPKM from coding RNA xlsx...")
coding_path = NC_DIR + "all_codingRNAs_WTPOs_and_Mutants.xlsx"
wb = openpyxl.load_workbook(coding_path, read_only=True, data_only=True)
ws = wb.active
rows = list(ws.iter_rows(values_only=True))
wb.close()

# First row = header
header = rows[0]
gene_col = 1   # column 1 is the gene/ORF name; column 0 is chromosome

fpkm_data = {'gene': []}
for bait in ['CBP80', 'GBP2']:
    fpkm_data[bait] = []

for row in rows[1:]:
    gene = row[gene_col]
    if not gene or not isinstance(gene, str):
        continue
    gene = gene.strip()
    # Keep only standard yeast ORF names (start with Y) to match dwell scores index
    if not gene.upper().startswith('Y'):
        continue
    fpkm_data['gene'].append(gene)
    for bait, cols in BAIT_COLS.items():
        vals = []
        for ci in cols:
            if ci < len(row):
                v = row[ci]
                try:
                    v = float(v)
                    if np.isfinite(v) and v >= 0:
                        vals.append(v)
                except (TypeError, ValueError):
                    pass
        fpkm_data[bait].append(np.mean(vals) if vals else np.nan)

df_fpkm = pd.DataFrame(fpkm_data).set_index('gene')
print(f"  Loaded {len(df_fpkm)} genes, CBP80 max={df_fpkm['CBP80'].max():.0f}, "
      f"GBP2 max={df_fpkm['GBP2'].max():.0f}")

# ── Load gene lengths and RNA cluster assignments ──────────────────────────
df_dwell  = pd.read_csv(DATADIR + "rna_dwell_scores.tsv",  sep='\t', index_col=0)
df_clust  = pd.read_csv(DATADIR + "rna_gene_clusters.tsv", sep='\t', index_col=0)
df_len    = df_dwell[['length']].copy()
df_len.index    = df_len.index.str.upper().str.strip()
df_fpkm.index   = df_fpkm.index.str.upper().str.strip()
df_clust.index  = df_clust.index.str.upper().str.strip()

df_merged = df_fpkm.join(df_len, how='inner').join(df_clust[['cluster']], how='left').dropna(subset=['length'])
df_merged['cluster'] = df_merged['cluster'].fillna(0).astype(int)
print(f"  After joining lengths: {len(df_merged)} genes")

# Color each gene point by its RNA cluster
def gene_color(cluster_id):
    return CLUSTER_COLORS.get(int(cluster_id), '#BBBBBB')

# ── Bait colors (still used for titles) ───────────────────────────────────
COLOR_CBP80 = GROUP_COLORS.get(BAIT_GROUP_APMS.get('CBP80', ''), '#8BDFE6')
COLOR_GBP2  = GROUP_COLORS.get(BAIT_GROUP_APMS.get('GBP2',  ''), '#FFCA7D')

# ── Figure: 2×2 grid ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 10))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(2, 2, figure=fig,
                         hspace=0.42, wspace=0.32,
                         left=0.09, right=0.97, top=0.93, bottom=0.08)

x        = df_merged['length'].values
y_cbp80  = df_merged['CBP80'].values
y_gbp2   = df_merged['GBP2'].values
clusters = df_merged['cluster'].values

panels = [
    # (ax, y_vals, bait_label, xlim, ylim, zoom_note)
    (gs[0, 0], y_cbp80, 'CBP80', (0, 15000), None,  'full range'),
    (gs[0, 1], y_gbp2,  'GBP2',  (0, 15000), None,  'full range'),
    (gs[1, 0], y_cbp80, 'CBP80', (0,  7000), (0, 700), 'zoomed'),
    (gs[1, 1], y_gbp2,  'GBP2',  (0,  7000), (0, 700), 'zoomed'),
]

bait_title_color = {'CBP80': COLOR_CBP80, 'GBP2': COLOR_GBP2}

for pi, (gsp, y_vals, bait, xlim, ylim, zoom) in enumerate(panels):
    ax = fig.add_subplot(gsp)

    # Restrict to xlim for scatter
    mask = (x >= xlim[0]) & (x <= xlim[1])
    if ylim is not None:
        mask = mask & (y_vals >= ylim[0]) & (y_vals <= ylim[1])
    xp, yp, cp = x[mask], y_vals[mask], clusters[mask]

    # Plot each cluster separately so they appear in legend
    for k in CLUSTER_ORDER:
        km = (cp == k)
        if km.sum() == 0:
            continue
        ax.scatter(xp[km], yp[km], s=8, c=CLUSTER_COLORS[k],
                   alpha=0.45, linewidths=0, rasterized=True,
                   label=CLUSTER_LABELS[k].replace('\n',' ') if pi == 0 else '_')
    # Unassigned genes
    um = (cp == 0)
    if um.sum() > 0:
        ax.scatter(xp[um], yp[um], s=6, c='#BBBBBB',
                   alpha=0.25, linewidths=0, rasterized=True,
                   label='Unassigned' if pi == 0 else '_')

    ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)

    ax.set_xlabel('ORF length (nucleotides)', fontsize=9)
    ax.set_ylabel('FPKM', fontsize=9)
    ax.set_title(f'"{bait}" — {zoom}', fontsize=10, fontweight='bold',
                 color=bait_title_color[bait])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Trend line per cluster (only on full-range panels for clarity)
    if ylim is None:
        for k in CLUSTER_ORDER:
            km = (cp == k)
            if km.sum() < 10:
                continue
            bins    = np.linspace(xlim[0], xlim[1], 20)
            bin_idx = np.digitize(xp[km], bins)
            bin_med = [np.median(yp[km][bin_idx == b]) for b in range(1, len(bins))
                       if np.sum(bin_idx == b) >= 5]
            bin_x   = [(bins[b-1]+bins[b])/2 for b in range(1, len(bins))
                        if np.sum(bin_idx == b) >= 5]
            if bin_med:
                ax.plot(bin_x, bin_med, color=CLUSTER_COLORS[k],
                        linewidth=1.4, alpha=0.8, zorder=3)

# Panel letter and cluster legend on first panel
axes_list = [fig.axes[i] for i in range(4)]
axes_list[0].text(-0.12, 1.05, 'C', transform=axes_list[0].transAxes,
                  fontsize=14, fontweight='bold')
cl_handles = [mpatches.Patch(color=CLUSTER_COLORS[k],
              label=CLUSTER_LABELS[k].replace('\n',' ')) for k in CLUSTER_ORDER
              if k in CLUSTER_COLORS]
cl_handles.append(mpatches.Patch(color='#BBBBBB', label='Unassigned'))
axes_list[0].legend(handles=cl_handles, fontsize=6.5, loc='upper right',
                    title='RNA cluster', title_fontsize=7, framealpha=0.85)

fig.suptitle(
    'Figure 2C  |  Preferential Binding of mRNP Factors to mRNAs of Different Length\n'
    'CBP80 (early, 5\'CBC) vs GBP2 (hnRNP/export adaptor) — raw FPKM vs ORF length',
    fontsize=10, fontweight='bold', y=0.98
)

out = OUTDIR + "figureS5C_length_scatter.pdf"
fig.savefig(out, format='pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")

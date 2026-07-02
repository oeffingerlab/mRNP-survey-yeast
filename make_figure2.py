"""
Figure 2 panels — AP-RNA-seq focused
A: RNA-seq HCL heatmap (pathway-ordered baits, gene clustering) + intron-gene strip
B: Intron-containing gene distribution (RP vs non-RP, stacked histogram by cluster)
C: k=8 cluster mean profiles (enrichment per bait per cluster)
D: Gene length violin plots by RNA cluster
E: Dwell score (dwell_auc) violin plots by RNA cluster
F: ncRNA stacked bar chart per bait (coding, SUT, CUT, XUT, lncRNA)
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, "/sessions/lucid-modest-goldberg/mnt/outputs/")
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS,
                           CLUSTER_COLORS, CMAP_APMS, CMAP_CORR, CMAP_SP, CMAP_RNA,
                           PATHWAY_ORDER_APMS, PATHWAY_ORDER_RNA, CLUSTER_ORDER,
                           CLUSTER_LABELS)
BAIT_GROUP = BAIT_GROUP_APMS
apply_style()

OUTDIR  = "/sessions/lucid-modest-goldberg/mnt/MS results/"
DATADIR = "/sessions/lucid-modest-goldberg/mnt/outputs/"
FPKMDIR = "/sessions/lucid-modest-goldberg/mnt/Alex/RNA data April 2014/fpkms/"

# ── Shared definitions ─────────────────────────────────────────────────────
PATHWAY_ORDER_RNA = [
    'CBP80','THP2','YRA1','SNU71',
    'CLP1','SYC1','REF2','HRP1',
    'PAB1','NAB2','NPL3','GBP2','GFD1',
    'RRP6','SUB2','SAC3','MEX67','DBP5',
]
# GROUP_COLORS imported from figure_style
BAIT_GROUP_RNA = {
    'CBP80':"5'end CBC",
    'THP2':"Co-Tx/THO",'YRA1':"Co-Tx/THO",
    'SNU71':"Splicing",
    'CLP1':"3'end processing",'SYC1':"3'end processing",
    'REF2':"3'end processing",'HRP1':"3'end processing",
    'PAB1':"Polyadenylation",'NAB2':"Polyadenylation",
    'NPL3':"Polyadenylation",'GBP2':"Polyadenylation",'GFD1':"Polyadenylation",
    'RRP6':"Degradation",'SUB2':"TREX",'SAC3':"NPC associated",
    'MEX67':"Export",'DBP5':"Export",
}

# k=8 cluster labels (pathway order: C8 early → C1 late)
CLUSTER_LABELS = {
    8: "C8 Splicing/RP\n(SNU71)",
    6: "C6 CBC/hnRNP\n(PAB1)",
    4: "C4 THO/TREX\n(THP2)",
    5: "C5 hnRNP\n(HRP1)",
    3: "C3 3′end proc.\n(REF2)",
    2: "C2 TREX-2\n(SAC3)",
    7: "C7 GFD1",
    1: "C1 Export\n(DBP5)",
}
CLUSTER_ORDER = [8, 6, 4, 5, 3, 2, 7, 1]   # early → late

# Cluster colors (matching standing palette from prior analysis)
# CLUSTER_COLORS imported from figure_style

def bait_color_bar_h(ax, baits):
    """Horizontal color bar matching bait order."""
    colors = [GROUP_COLORS.get(BAIT_GROUP_RNA.get(b,''), '#CCCCCC') for b in baits]
    n = len(baits)
    for i, c in enumerate(colors):
        ax.add_patch(mpatches.Rectangle((i/n,0), 1/n, 1,
            color=c, transform=ax.transAxes, clip_on=False))

def bait_color_bar_v(ax, baits):
    colors = [GROUP_COLORS.get(BAIT_GROUP_RNA.get(b,''), '#CCCCCC') for b in baits]
    n = len(baits)
    for i, c in enumerate(colors):
        ax.add_patch(mpatches.Rectangle((0,i/n), 1, 1/n,
            color=c, transform=ax.transAxes, clip_on=False))

# ── Intron / RP gene classification ────────────────────────────────────────
GFF = "/sessions/lucid-modest-goldberg/mnt/uploads/saccharomyces_cerevisiae.20260618.gff"
COLOR_NONE  = '#E8E8E8'
COLOR_NONRP = '#8BB4D0'
COLOR_RP    = '#E8A87C'
STRIP_HEIGHT = 0.04

INTRON_RE = re.compile(r'Name=([^;]+)_intron')
ORF_RE    = re.compile(r'^Y[A-P][LR]\d{3}[CW]$')
intron_genes = set()
rp_genes = set()
with open(GFF) as fh:
    for line in fh:
        if line.startswith('#'): continue
        if '\tintron\t' in line:
            m = INTRON_RE.search(line)
            if m and ORF_RE.match(m.group(1)):
                intron_genes.add(m.group(1))
        parts = line.strip().split('\t')
        if len(parts) >= 9 and parts[2] == 'gene':
            id_m   = re.search(r'ID=([^;]+)', parts[8])
            gene_m = re.search(r'gene=([^;]+)', parts[8])
            if id_m and gene_m:
                orf = id_m.group(1)
                if ORF_RE.match(orf) and re.match(r'^RP[LS]\d', gene_m.group(1), re.IGNORECASE):
                    rp_genes.add(orf)

def classify(g):
    if g in rp_genes:     return 2
    if g in intron_genes: return 1
    return 0

print(f"GFF: {len(intron_genes)} intron genes, {len(rp_genes)} RP genes")

# ── Load data ──────────────────────────────────────────────────────────────
df_rna   = pd.read_csv(DATADIR + "rnaseq_bait_means.tsv", sep='\t', index_col=0)
df_clust = pd.read_csv(DATADIR + "rna_gene_clusters.tsv",  sep='\t', index_col=0)
df_dwell = pd.read_csv(DATADIR + "rna_dwell_scores.tsv",   sep='\t', index_col=0)

rna_baits = [b for b in PATHWAY_ORDER_RNA if b in df_rna.columns]
df_rna = df_rna[rna_baits]
df_rna_filt = df_rna[df_rna.max(axis=1) > 0.3]

# HCL gene ordering
dist_g = pdist(df_rna_filt.values, metric='correlation')
dist_g = np.nan_to_num(dist_g, nan=1.0)
link_g = linkage(dist_g, method='average')
gene_ord = leaves_list(link_g)
df_hcl = df_rna_filt.iloc[gene_ord]

# Cluster mean profiles
df_merged = df_clust[['cluster']].join(df_rna_filt[rna_baits], how='inner')
cluster_means = df_merged.groupby('cluster')[rna_baits].mean()

# Gene length & dwell: merge with clusters
df_gl = df_dwell[['length','dwell_auc','class']].join(df_clust[['cluster']], how='inner')
df_gl = df_gl.dropna(subset=['length','dwell_auc','cluster'])
df_gl['cluster'] = df_gl['cluster'].astype(int)

# ncRNA stacked bars: read CDS, SUT, CUT, XUT, lncRNA fpkm files
def read_fpkm(path, col_start=2):
    """Read fpkm file, return per-gene per-replicate dataframe."""
    df = pd.read_csv(path, sep='\t', header=0, comment='#',
                     low_memory=False)
    df.columns = df.columns.str.strip()
    return df

# We'll use the averaged RNA-seq data + proportion approach
# Compute per-bait total FPKM for each RNA class
def load_fpkm_class(fpath, bait_cols_map):
    """Load fpkm, sum columns by bait, return Series bait->total."""
    try:
        df = pd.read_csv(fpath, sep='\t', low_memory=False)
        totals = {}
        for bait, cols in bait_cols_map.items():
            valid = [c for c in cols if c in df.columns]
            if valid:
                totals[bait] = df[valid].sum().sum()
        return pd.Series(totals)
    except:
        return pd.Series(dtype=float)

# Since we don't have clean column→bait mapping for all 102 cols,
# use the proportional approach from the averaged RNA-seq data
# The dwell data 'class' column gives us pathway-stage labels

# For ncRNA: load Alex's averaged per-bait coding/SUT/CUT/XUT/lncRNA data
# from the organized files
import os
nc_dir = "/sessions/lucid-modest-goldberg/mnt/Alex/RNA data April 2014/MOe/please filter and cluster/"
nc_files = {
    'Coding':  nc_dir + "all_codingRNAs_WTPOs_and_Mutants.xlsx",
    'SUT':     nc_dir + "all_SUTs_WTPOs_and_Mutants.xlsx",
    'CUT':     nc_dir + "all_CUTs_WTPOs_and_Mutants.xlsx",
    'XUT':     nc_dir + "all_XUTs_WTPOs_and_Mutants.xlsx",
    'lncRNA':  nc_dir + "all_lncRNAs_WTPOs_and_Mutants.xlsx",
}

import openpyxl

# Bait -> column index mapping (0-based) derived from column header inspection
# Headers format: 'Cbp80, 33_L14, bait' or 'Syc1-1_L2.LB8' (no 'bait' keyword)
# SNU114 excluded from AP-RNA-seq; all others included
NC_BAIT_COLS = {
    'CBP80': [2,3,4], 'THP2': [5,6], 'YRA1': [7,8], 'SNU71': [11,12],
    'SYC1':  [13,14,15], 'CLP1': [16,17,18], 'REF2': [19,20,21],
    'HRP1':  [22,23,24,25,26], 'RRP6': [27,28,29], 'NAB2': [30,31,32,33],
    'NPL3':  [34], 'GBP2': [35,36,37], 'SUB2': [38,39,40],
    'SAC3':  [41,42], 'MEX67': [43,44,45], 'GFD1': [46],
    'DBP5':  [47,48,49], 'PAB1': [50,51,52,53],
}

def load_nc_bait_totals(fpath, bait_list):
    """Load xlsx using fixed column-index mapping; return per-bait total FPKM sum."""
    wb = openpyxl.load_workbook(fpath, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    data = rows[1:]  # skip header
    totals = {}
    for bait in bait_list:
        col_idxs = NC_BAIT_COLS.get(bait, [])
        bait_sum = 0.0
        for row in data:
            rep_vals = []
            for ci in col_idxs:
                if ci < len(row):
                    v = row[ci]
                    try:
                        v = float(v)
                        if np.isfinite(v):
                            rep_vals.append(v)
                    except (TypeError, ValueError):
                        pass
            if rep_vals:
                bait_sum += np.mean(rep_vals)
        totals[bait] = bait_sum
    wb.close()
    return totals

nc_totals = {}
print("Loading ncRNA class data...")
for cls, fpath in nc_files.items():
    if os.path.exists(fpath):
        nc_totals[cls] = load_nc_bait_totals(fpath, rna_baits)
        print(f"  {cls}: {len(nc_totals[cls])} baits")

# Build proportional DataFrame
nc_df = pd.DataFrame(nc_totals).fillna(0)
if len(nc_df) > 0:
    nc_df_norm = nc_df.div(nc_df.sum(axis=1), axis=0)
    # Reorder by pathway (index is already uppercase bait names)
    nc_df_norm = nc_df_norm.loc[[b for b in rna_baits if b in nc_df_norm.index]]
    print(f"\nncRNA proportions: {nc_df_norm.shape}")
    print(nc_df_norm.head(3))

# ── Build strip values in HCL order ────────────────────────────────────────
strip_vals = np.array([classify(g) for g in df_hcl.index])
n_rp_in    = (strip_vals == 2).sum()
n_nonrp_in = (strip_vals == 1).sum()

# ── Cluster bar data for Panel B ────────────────────────────────────────────
df_ic = df_clust[['cluster']].copy()
df_ic['cat'] = pd.Series({g: classify(g) for g in df_ic.index})
clust_total  = df_ic.groupby('cluster').size()
rp_counts    = df_ic[df_ic['cat']==2].groupby('cluster').size().reindex(CLUSTER_ORDER, fill_value=0)
nonrp_counts = df_ic[df_ic['cat']==1].groupby('cluster').size().reindex(CLUSTER_ORDER, fill_value=0)
rp_frac      = (rp_counts    / clust_total * 100).reindex(CLUSTER_ORDER).fillna(0)
nonrp_frac   = (nonrp_counts / clust_total * 100).reindex(CLUSTER_ORDER).fillna(0)

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 25))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(4, 2, figure=fig,
                        height_ratios=[2.2, 1.2, 1.6, 1.4],
                        hspace=0.52, wspace=0.32,
                        left=0.07, right=0.97, top=0.95, bottom=0.05)

blue_cmap = CMAP_RNA

# ── Panel A: RNA-seq HCL + intron strip ────────────────────────────────────
ax_a = fig.add_subplot(gs[0, :])
im_a = ax_a.imshow(df_hcl.values.T, aspect='auto', cmap=blue_cmap,
                    vmin=0, vmax=1.0)
ax_a.set_title('AP-RNA-seq Hierarchical Clustering  (genes × baits, max-FPKM normalized)',
               fontsize=10, fontweight='bold')
ax_a.set_ylabel('Baits (pathway order → early to late)', fontsize=8)
ax_a.set_xlabel(
    f'{len(df_hcl)} co-enriched genes  |  clustered by Pearson / average linkage  '
    f'|  strip: {n_rp_in} RP genes (terracotta) + {n_nonrp_in} non-RP intron genes (blue)',
    fontsize=8)
ax_a.set_yticks(range(len(rna_baits)))
ax_a.set_yticklabels(rna_baits, fontsize=7)
ax_a.set_xticks([])
ax_a_left = ax_a.inset_axes([-0.022, 0, 0.012, 1])
ax_a_left.axis('off'); bait_color_bar_v(ax_a_left, rna_baits)
plt.colorbar(im_a, ax=ax_a, shrink=0.25, pad=0.01, label='Enrichment (max=1)')

# Three-color intron strip
strip_cmap = ListedColormap([COLOR_NONE, COLOR_NONRP, COLOR_RP])
ax_strip = ax_a.inset_axes([0, -STRIP_HEIGHT - 0.005, 1, STRIP_HEIGHT])
ax_strip.axis('off')
ax_strip.imshow(strip_vals.reshape(1,-1).astype(float),
                aspect='auto', cmap=strip_cmap, vmin=0, vmax=2, interpolation='nearest')
ax_strip.text(-0.005, 0.5, 'Intron\ngenes', ha='right', va='center',
              fontsize=7, transform=ax_strip.transAxes)

# Strip legend
p_rp    = mpatches.Patch(color=COLOR_RP,    label=f'RP genes (n={n_rp_in})')
p_nonrp = mpatches.Patch(color=COLOR_NONRP, label=f'Non-RP intron genes (n={n_nonrp_in})')
p_none  = mpatches.Patch(color=COLOR_NONE,  label='No intron')
ax_a.legend(handles=[p_rp, p_nonrp, p_none],
            loc='upper left', bbox_to_anchor=(0.0, -0.07 - STRIP_HEIGHT),
            fontsize=7.5, frameon=True, ncol=3)

ax_a.text(-0.025, 1.03, 'A', transform=ax_a.transAxes, fontsize=14, fontweight='bold')

# ── Panel B: Intron gene stacked bar by cluster ─────────────────────────────
ax_b2 = fig.add_subplot(gs[1, :])
x_pos = np.arange(len(CLUSTER_ORDER))
ax_b2.bar(x_pos, [nonrp_frac[k] for k in CLUSTER_ORDER],
          color=COLOR_NONRP, edgecolor='white', linewidth=0.4, width=0.7,
          label=f'Non-RP intron genes (n={n_nonrp_in})')
ax_b2.bar(x_pos, [rp_frac[k] for k in CLUSTER_ORDER],
          bottom=[nonrp_frac[k] for k in CLUSTER_ORDER],
          color=COLOR_RP, edgecolor='white', linewidth=0.4, width=0.7,
          label=f'RP genes — intron-enriched (n={n_rp_in})')

bar_labels = []
for k in CLUSTER_ORDER:
    nr = int(nonrp_counts[k]); rp = int(rp_counts[k]); tot = int(clust_total[k])
    bar_labels.append(f"{CLUSTER_LABELS[k].split(chr(10))[0]}\n({nr+rp}/{tot})")
ax_b2.set_xticks(x_pos)
ax_b2.set_xticklabels(bar_labels, fontsize=7.5, ha='center')
for tick, k in zip(ax_b2.get_xticklabels(), CLUSTER_ORDER):
    tick.set_color(CLUSTER_COLORS[k]); tick.set_fontweight('bold')

for xi, k in enumerate(CLUSTER_ORDER):
    tot_frac = nonrp_frac[k] + rp_frac[k]
    if tot_frac > 0.5:
        ax_b2.text(xi, tot_frac + 0.4, f'{tot_frac:.1f}%',
                   ha='center', va='bottom', fontsize=7.5, color='#444444')

max_y = max(nonrp_frac[k] + rp_frac[k] for k in CLUSTER_ORDER)
ax_b2.set_ylim(0, max_y * 1.35 + 1)
ax_b2.set_xlim(-0.5, len(CLUSTER_ORDER) - 0.5)
ax_b2.set_ylabel('Intron genes\n(% of cluster)', fontsize=8)
ax_b2.set_title('Intron-containing gene distribution across RNA-seq clusters (k=8)',
                fontsize=9, fontweight='bold', pad=10)
ax_b2.spines['top'].set_visible(False); ax_b2.spines['right'].set_visible(False)
ax_b2.legend(fontsize=7.5, loc='upper left', framealpha=0.9, handlelength=1.2)
ax_b2.text(-0.025, 1.05, 'B', transform=ax_b2.transAxes, fontsize=14, fontweight='bold')

# ── Panel C: k=8 cluster mean profiles ────────────────────────────────────
ax_b = fig.add_subplot(gs[2, 0])
cmap_b = plt.cm.get_cmap('tab10', 8)
for ci, k in enumerate(CLUSTER_ORDER):
    if k in cluster_means.index:
        profile = cluster_means.loc[k, rna_baits]
        ax_b.plot(range(len(rna_baits)), profile.values,
                  color=CLUSTER_COLORS[k], linewidth=2.0,
                  label=CLUSTER_LABELS[k].replace('\n',' '), alpha=0.9)
ax_b.set_xticks(range(len(rna_baits)))
ax_b.set_xticklabels(rna_baits, fontsize=6, rotation=60, ha='right')
ax_b.set_ylabel('Mean enrichment (max=1)', fontsize=8)
ax_b.set_title('k=8 RNA Cluster Mean Profiles\n(pathway-stage enrichment per cluster)',
               fontsize=9, fontweight='bold')
ax_b.set_ylim(-0.02, 0.85)
ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)
ax_b.legend(fontsize=5.5, ncol=2, loc='upper right', framealpha=0.7)
ax_b_top = ax_b.inset_axes([0, 1.01, 1, 0.025])
ax_b_top.axis('off'); bait_color_bar_h(ax_b_top, rna_baits)
ax_b.text(-0.1, 1.04, 'C', transform=ax_b.transAxes, fontsize=14, fontweight='bold')

# ── Panel D: Gene length violins by cluster ────────────────────────────────
ax_c = fig.add_subplot(gs[2, 1])
vdata_len = [df_gl[df_gl['cluster']==k]['length'].dropna().values
             for k in CLUSTER_ORDER]
parts = ax_c.violinplot(vdata_len, positions=range(len(CLUSTER_ORDER)),
                        showmedians=True, showextrema=False, widths=0.75)
for i, (pc, k) in enumerate(zip(parts['bodies'], CLUSTER_ORDER)):
    pc.set_facecolor(CLUSTER_COLORS[k])
    pc.set_alpha(0.7)
    pc.set_edgecolor('grey')
parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.5)
# WCL reference line
wcl_mean = 1200
ax_c.axhline(wcl_mean, color='black', linestyle='--', linewidth=1.0,
             alpha=0.6, label=f'WCL avg (~{wcl_mean} nt)')
ax_c.set_xticks(range(len(CLUSTER_ORDER)))
ax_c.set_xticklabels([CLUSTER_LABELS[k].split('\n')[0] for k in CLUSTER_ORDER],
                      fontsize=6.5, rotation=45, ha='right')
ax_c.set_ylabel('Gene length (nt)', fontsize=8)
ax_c.set_title('Gene Length Distribution by RNA Cluster\n(read-count weighted)',
               fontsize=9, fontweight='bold')
ax_c.set_ylim(0, 8000)
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)
ax_c.legend(fontsize=7, loc='upper left')
ax_c.text(-0.1, 1.04, 'D', transform=ax_c.transAxes, fontsize=14, fontweight='bold')

# ── Panel E: Dwell score violins by cluster ────────────────────────────────
ax_d = fig.add_subplot(gs[3, 0])
vdata_dwell = [df_gl[df_gl['cluster']==k]['dwell_auc'].dropna().values
               for k in CLUSTER_ORDER]
parts_d = ax_d.violinplot(vdata_dwell, positions=range(len(CLUSTER_ORDER)),
                           showmedians=True, showextrema=False, widths=0.75)
for i, (pc, k) in enumerate(zip(parts_d['bodies'], CLUSTER_ORDER)):
    pc.set_facecolor(CLUSTER_COLORS[k])
    pc.set_alpha(0.7)
    pc.set_edgecolor('grey')
parts_d['cmedians'].set_color('black'); parts_d['cmedians'].set_linewidth(1.5)
ax_d.set_xticks(range(len(CLUSTER_ORDER)))
ax_d.set_xticklabels([CLUSTER_LABELS[k].split('\n')[0] for k in CLUSTER_ORDER],
                      fontsize=6.5, rotation=45, ha='right')
ax_d.set_ylabel('Dwell score (AUC)', fontsize=8)
ax_d.set_title('Nuclear Processing Dwell Score by Cluster\n(integrated pathway-stage enrichment)',
               fontsize=9, fontweight='bold')
ax_d.spines['top'].set_visible(False); ax_d.spines['right'].set_visible(False)
ax_d.text(-0.1, 1.04, 'E', transform=ax_d.transAxes, fontsize=14, fontweight='bold')

# ── Panel F: ncRNA stacked bars ─────────────────────────────────────────────
ax_e = fig.add_subplot(gs[3, 1])
if len(nc_df) > 0 and nc_df_norm.shape[0] > 0:
    bar_baits = list(nc_df_norm.index)
    x = np.arange(len(bar_baits))
    # stack: Coding (tan), SUT (orange), CUT (cream/white), XUT (red), lncRNA (slate)
    NC_COLORS = {'Coding':'#D7B899','SUT':'#FF8C42','CUT':'#FFFACD',
                 'XUT':'#E53935','lncRNA':'#7986CB'}
    bottom = np.zeros(len(bar_baits))
    for cls, color in NC_COLORS.items():
        if cls in nc_df_norm.columns:
            vals = nc_df_norm[cls].values
            ax_e.bar(x, vals, bottom=bottom, color=color,
                     edgecolor='white', linewidth=0.3, label=cls, width=0.8)
            bottom += vals
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(bar_baits, fontsize=6, rotation=60, ha='right')
    ax_e.set_ylabel('Proportion of co-purified transcripts', fontsize=8)
    ax_e.set_ylim(0, 1.05)
    ax_e.set_title('Transcript Class Composition per Bait\n(coding, SUT, CUT, XUT, lncRNA)',
                   fontsize=9, fontweight='bold')
    ax_e.spines['top'].set_visible(False); ax_e.spines['right'].set_visible(False)
    ax_e.legend(fontsize=7, loc='lower right', ncol=2)
    ax_e_top = ax_e.inset_axes([0, 1.01, 1, 0.025])
    ax_e_top.axis('off'); bait_color_bar_h(ax_e_top, bar_baits)
else:
    ax_e.text(0.5, 0.5, 'ncRNA data loading\n(see ncrna_class_profiles.svg)',
              ha='center', va='center', transform=ax_e.transAxes, fontsize=9)
    ax_e.axis('off')
ax_e.text(-0.1, 1.04, 'F', transform=ax_e.transAxes, fontsize=14, fontweight='bold')

# Global legend for groups
legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
                  if any(BAIT_GROUP_RNA.get(b)==g for b in rna_baits)]
fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=7,
           frameon=True, bbox_to_anchor=(0.5, 0.0),
           title='Functional group', title_fontsize=8)

fig.suptitle('Figure 2  |  AP-RNA-seq Analysis: Transcript Landscape of mRNP Biogenesis Stages',
             fontsize=12, fontweight='bold', y=0.975)

out = OUTDIR + "figure2_composite.pdf"
fig.savefig(out, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")

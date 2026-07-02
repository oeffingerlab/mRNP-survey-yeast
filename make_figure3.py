"""
Figure 3 panels — Cross-correlation / Integration
A: Within- vs. between-module correlation per bait (four bootstrap-supported modules)
B: Multi-omics combined heatmap (bait x bait correlation, proteomics + transcriptomics)
C: CID boundary proximity violin plots by RNA cluster (k=8)
D: Promoter class distribution by RNA cluster (stacked bars)
E: Dwell score vs gene length scatter, colored by promoter class

Exclusions enforced:
  - MTR2 dropped from AP-MS baits (data quality)
  - SNU114 absent from AP-RNA-seq baits (mutant-only bait)
  - Poison/mutant AP-MS conditions excluded (separate experimental series)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, "/sessions/lucid-modest-goldberg/mnt/outputs/")
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS, fmt_bait, fmt_baits,
                           CLUSTER_COLORS, CMAP_APMS, CMAP_CORR, CMAP_SP, CMAP_RNA,
                           PATHWAY_ORDER_APMS, PATHWAY_ORDER_RNA, CLUSTER_ORDER,
                           CLUSTER_LABELS)
BAIT_GROUP = BAIT_GROUP_APMS
apply_style()

OUTDIR  = "/sessions/lucid-modest-goldberg/mnt/MS results/"
DATADIR = "/sessions/lucid-modest-goldberg/mnt/outputs/"

# ── Shared definitions ─────────────────────────────────────────────────────
PATHWAY_ORDER_APMS = [
    'CBP20','CBP80','THP2','YRA1','YRA2',
    'SNU71','SNU114','PRP42','PRP6',
    'RNA14','CLP1','YSH1','SYC1','CFT1','REF2','HRP1',
    'PAB1','PAP1','NAB2','NPL3','GBP2','HRB1','GFD1',
    'RRP6','SUB2','PML39','SAC3','THP1',
    'MEX67','DBP5','MLP1','MLP2','SHE2',
]
PATHWAY_ORDER_RNA = [
    'CBP80','THP2','YRA1','SNU71',
    'CLP1','SYC1','REF2','HRP1',
    'PAB1','NAB2','NPL3','GBP2','GFD1',
    'RRP6','SUB2','SAC3','MEX67','DBP5',
]
# GROUP_COLORS imported from figure_style
BAIT_GROUP = {
    'CBP20':"5'end CBC", 'CBP80':"5'end CBC",
    'THP2':"Co-Tx/THO", 'YRA1':"Co-Tx/THO", 'YRA2':"Co-Tx/THO",
    'SNU71':"Splicing", 'SNU114':"Splicing", 'PRP42':"Splicing", 'PRP6':"Splicing",
    'RNA14':"3'end processing", 'CLP1':"3'end processing",
    'YSH1':"3'end processing", 'SYC1':"3'end processing",
    'CFT1':"3'end processing", 'REF2':"3'end processing", 'HRP1':"3'end processing",
    'PAB1':"Polyadenylation", 'PAP1':"Polyadenylation",
    'NAB2':"Polyadenylation", 'NPL3':"Polyadenylation",
    'GBP2':"Polyadenylation", 'HRB1':"Polyadenylation", 'GFD1':"Polyadenylation",
    'RRP6':"Degradation",
    'SUB2':"TREX", 'PML39':"TREX",
    'SAC3':"NPC associated", 'THP1':"NPC associated",
    'MEX67':"Export", 'DBP5':"Export", 'MLP1':"Export", 'MLP2':"Export", 'SHE2':"Export",
}
CLUSTER_ORDER  = [8, 6, 4, 5, 3, 2, 7, 1]
CLUSTER_LABELS = {
    8: "C8\nSplicing/RP", 6: "C6\nCBC/hnRNP", 4: "C4\nTHO/TREX",
    5: "C5\nhnRNP",       3: "C3\n3'end proc.", 2: "C2\nTREX-2",
    7: "C7\nGFD1",        1: "C1\nExport",
}
# CLUSTER_COLORS imported from figure_style

# ── Load data ──────────────────────────────────────────────────────────────
df_apms = pd.read_csv(DATADIR + "apms_nuclear_matrix.tsv", sep='\t', index_col=0)
if 'MTR2' in df_apms.columns:
    df_apms = df_apms.drop(columns=['MTR2'])

df_rna   = pd.read_csv(DATADIR + "rnaseq_bait_means.tsv", sep='\t', index_col=0)
df_clust = pd.read_csv(DATADIR + "rna_gene_clusters.tsv",  sep='\t', index_col=0)
df_prom  = pd.read_csv(DATADIR + "promoter_annotated.tsv", sep='\t', index_col=0)
df_cid_g = pd.read_csv(DATADIR + "genes_cid_assigned.tsv", sep='\t', index_col=0)

# Shared baits for multi-omics (AP-MS baits that also appear in AP-RNA-seq)
apms_baits = [b for b in PATHWAY_ORDER_APMS if b in df_apms.columns]
rna_baits  = [b for b in PATHWAY_ORDER_RNA  if b in df_rna.columns]
shared_baits = [b for b in rna_baits if b in apms_baits]   # 17 baits

# ── Build bait-bait correlation matrices ──────────────────────────────────
# Proteomics: transpose so baits are rows → compute bait-bait Pearson
df_apms_baits = df_apms[apms_baits]
corr_prot = df_apms_baits.corr(method='pearson')   # (N_apms x N_apms)

# Transcriptomics: bait-bait Pearson on RNA-seq
df_rna_filt = df_rna[rna_baits]
df_rna_filt = df_rna_filt[df_rna_filt.max(axis=1) > 0.3]
corr_rna = df_rna_filt.corr(method='pearson')   # (18 x 18)

# Multi-omics: average the two correlation matrices for shared baits
corr_prot_s = corr_prot.loc[shared_baits, shared_baits]
corr_rna_s  = corr_rna.loc[shared_baits, shared_baits]
corr_multi  = (corr_prot_s + corr_rna_s) / 2.0

# Hierarchical clustering on combined matrix (1 - Pearson distance)
dist_multi = 1.0 - corr_multi.values
dist_multi = np.clip(dist_multi, 0, None)
np.fill_diagonal(dist_multi, 0)
link_multi = linkage(squareform(dist_multi), method='average')

# ── Fixed 4-module assignment (canonical, matches S7) ─────────────────────
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
CLUSTER_COLS = {
    1: '#DB2827',
    2: '#465EAB',
    3: '#56C4CB',
    4: '#0A7F40',
}


# (cluster bootstrap removed — using fixed CLUSTER_4 dict)

# ── CID + promoter merge ───────────────────────────────────────────────────
df_cid_g['cluster'] = df_cid_g['cluster'].astype('Int64')
df_cid_g = df_cid_g.dropna(subset=['dist_to_boundary','cluster'])

# Promoter: merge gene_class_1 with cluster assignments
df_pc = df_prom[['gene_class_1']].join(df_clust[['cluster']], how='inner')
df_pc = df_pc.dropna()
df_pc['cluster'] = df_pc['cluster'].astype(int)

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(3, 2, figure=fig,
                        height_ratios=[1.8, 1.6, 1.4],
                        hspace=0.45, wspace=0.30,
                        left=0.07, right=0.97, top=0.95, bottom=0.05)

warm_cmap = CMAP_CORR

# ── Panel A: Within- vs. between-module correlation per bait ─────────────
ax_a = fig.add_subplot(gs[0, 0])

intra_r, inter_r = [], []
for b in shared_baits:
    k     = CLUSTER_4.get(b, 0)
    same  = [b2 for b2 in shared_baits if b2 != b and CLUSTER_4.get(b2,0) == k]
    other = [b2 for b2 in shared_baits if CLUSTER_4.get(b2,0) != k]
    intra_r.append(np.mean([corr_multi.loc[b, b2] for b2 in same]) if same else 0)
    inter_r.append(np.mean([corr_multi.loc[b, b2] for b2 in other]) if other else 0)

x = np.arange(len(shared_baits))
cum_sizes = np.cumsum([0] + [len([b for b in shared_baits if CLUSTER_4.get(b,0)==k])
                              for k in [1,2,3,4]])
for ki, k in enumerate([1,2,3,4]):
    ax_a.axvspan(cum_sizes[ki]-0.5, cum_sizes[ki+1]-0.5,
                 color=CLUSTER_COLS[k], alpha=0.15, zorder=0)
    mid = (cum_sizes[ki] + cum_sizes[ki+1]) / 2
    ax_a.text(mid, 1.09, CLUSTER_NAMES[k], ha='center', fontsize=9,
              transform=ax_a.get_xaxis_transform())

ax_a.plot(x, intra_r, 'o-', color='#2E4057', linewidth=1.8, markersize=7,
          label='Mean within-module r', zorder=3)
ax_a.plot(x, inter_r, 's--', color='#AAAAAA', linewidth=1.2, markersize=5,
          label='Mean between-module r', zorder=3)
for i, b in enumerate(shared_baits):
    color = GROUP_COLORS.get(BAIT_GROUP.get(b,''), '#CCCCCC')
    ax_a.scatter(i, intra_r[i], s=85, color=color, zorder=4,
                 edgecolors='white', lw=1.2)

ax_a.set_xlim(-0.8, len(shared_baits)-0.2)
ax_a.set_ylim(-0.05, 1.05)
ax_a.set_xticks(x)
ax_a.set_xticklabels(fmt_baits(shared_baits), fontsize=9, rotation=45,
                      ha='right', color='black')
ax_a.set_ylabel('Mean Pearson r\n(combined AP-MS + AP-RNA-seq)', fontsize=9)
ax_a.set_title('Within- vs. Between-module Correlation per Bait\n'
               '(four bootstrap-supported modules)', fontsize=9, fontweight='bold')
ax_a.spines['top'].set_visible(False); ax_a.spines['right'].set_visible(False)
ax_a.legend(fontsize=8.5, loc='upper right', bbox_to_anchor=(0.99, 0.98))
ax_a.axhline(0, color='#DDDDDD', linewidth=0.8)
ax_a.text(-0.12, 1.04, 'A', transform=ax_a.transAxes, fontsize=16, fontweight='bold')

# ── Panel B: Multi-omics heatmap ──────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

# Reorder by module assignment (consistent with Panel A)
leaf_ord_names = []
for k in [1,2,3,4]:
    leaf_ord_names += [b for b in shared_baits if CLUSTER_4.get(b,0)==k]
corr_reord = corr_multi.loc[leaf_ord_names, leaf_ord_names]

im_b = ax_b.imshow(corr_reord.values, cmap=warm_cmap, vmin=-0.2, vmax=1.0,
                    aspect='auto')
ax_b.set_xticks(range(len(leaf_ord_names)))
ax_b.set_yticks(range(len(leaf_ord_names)))
ax_b.set_xticklabels(fmt_baits(leaf_ord_names), rotation=60, ha='right', fontsize=6.5)
ax_b.set_yticklabels(fmt_baits(leaf_ord_names), fontsize=6.5)

for lbl in ax_b.get_xticklabels() + ax_b.get_yticklabels():
    lbl.set_color('black')

plt.colorbar(im_b, ax=ax_b, shrink=0.6, pad=0.02, label='Pearson r (combined)')
ax_b.set_title('Multi-omics Bait-Bait Correlation\n(average of proteomics + transcriptomics)',
               fontsize=9, fontweight='bold')
ax_b.text(-0.15, 1.04, 'B', transform=ax_b.transAxes, fontsize=16, fontweight='bold')

# ── Panel C: CID boundary proximity by RNA cluster ─────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

vdata_cid = [df_cid_g[df_cid_g['cluster']==k]['dist_to_boundary'].dropna().values
             for k in CLUSTER_ORDER]

parts_c = ax_c.violinplot(vdata_cid, positions=range(len(CLUSTER_ORDER)),
                           showmedians=True, showextrema=False, widths=0.75)
for pc, k in zip(parts_c['bodies'], CLUSTER_ORDER):
    pc.set_facecolor(CLUSTER_COLORS[k])
    pc.set_alpha(0.7)
    pc.set_edgecolor('grey')
parts_c['cmedians'].set_color('black')
parts_c['cmedians'].set_linewidth(1.5)

ax_c.set_xticks(range(len(CLUSTER_ORDER)))
ax_c.set_xticklabels([CLUSTER_LABELS[k].replace('\n',' ') for k in CLUSTER_ORDER],
                      fontsize=6.5, rotation=45, ha='right')
ax_c.set_ylabel('Distance to nearest CID boundary (bp)', fontsize=8)
ax_c.set_title('Chromatin Domain Boundary Proximity by RNA Cluster\n(Hi-C CID boundaries, yeast)',
               fontsize=9, fontweight='bold')
ax_c.set_ylim(0, 5000)
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)

# Annotate with Kruskal-Wallis p-value
from scipy.stats import kruskal
kw_stat, kw_p = kruskal(*[d for d in vdata_cid if len(d) > 1])
ax_c.text(0.98, 0.97, f'KW p = {kw_p:.1e}', transform=ax_c.transAxes,
          ha='right', va='top', fontsize=7, style='italic')
ax_c.text(-0.1, 1.04, 'C', transform=ax_c.transAxes, fontsize=16, fontweight='bold')

# ── Panel D: Promoter class by RNA cluster (stacked bars) ──────────────────
ax_d = fig.add_subplot(gs[1, 1])

prom_counts = df_pc.groupby(['cluster','gene_class_1']).size().unstack(fill_value=0)
prom_frac = prom_counts.div(prom_counts.sum(axis=1), axis=0)
# Reorder clusters
prom_frac = prom_frac.reindex(CLUSTER_ORDER)

PROM_COLORS = {
    'TATA-containing': '#E53935',
    'TATA-less':       '#90A4AE',
}
x = np.arange(len(CLUSTER_ORDER))
bottom = np.zeros(len(CLUSTER_ORDER))
for pc_class, color in PROM_COLORS.items():
    if pc_class in prom_frac.columns:
        vals = prom_frac[pc_class].values
        ax_d.bar(x, vals, bottom=bottom, color=color, label=pc_class,
                 edgecolor='white', linewidth=0.4, width=0.75)
        bottom += vals

ax_d.set_xticks(x)
ax_d.set_xticklabels([CLUSTER_LABELS[k].replace('\n',' ') for k in CLUSTER_ORDER],
                      fontsize=6.5, rotation=45, ha='right')
ax_d.set_ylabel('Fraction of genes', fontsize=8)
ax_d.set_ylim(0, 1.05)
ax_d.set_title('Promoter Class Distribution by RNA Cluster\n(TATA-containing vs TATA-less)',
               fontsize=9, fontweight='bold')
ax_d.spines['top'].set_visible(False); ax_d.spines['right'].set_visible(False)
ax_d.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.9)

# Add cluster-specific n labels
for i, k in enumerate(CLUSTER_ORDER):
    n = prom_counts.loc[k].sum() if k in prom_counts.index else 0
    ax_d.text(i, 1.02, f'n={n}', ha='center', va='bottom', fontsize=5.5, color='#444')

ax_d.text(-0.1, 1.04, 'D', transform=ax_d.transAxes, fontsize=16, fontweight='bold')

# ── Panel E: Dwell score vs gene length, colored by promoter class ─────────
ax_e = fig.add_subplot(gs[2, :])

df_dwell = pd.read_csv(DATADIR + "rna_dwell_scores.tsv", sep='\t', index_col=0)
df_scatter = df_dwell[['dwell_auc','length']].join(df_prom[['gene_class_1']], how='inner')
df_scatter = df_scatter.dropna(subset=['dwell_auc','length','gene_class_1'])
df_scatter = df_scatter[df_scatter['length'] < 15000]   # trim extreme outliers

tata_yes = df_scatter[df_scatter['gene_class_1']=='TATA-containing']
tata_no  = df_scatter[df_scatter['gene_class_1']=='TATA-less']

ax_e.scatter(tata_no['length'], tata_no['dwell_auc'],
             c='#90A4AE', alpha=0.25, s=6, label='TATA-less', rasterized=True)
ax_e.scatter(tata_yes['length'], tata_yes['dwell_auc'],
             c='#E53935', alpha=0.35, s=8, label='TATA-containing', rasterized=True)

# Running median lines
for df_sub, color, label in [(tata_no, '#546E7A', 'TATA-less median'),
                               (tata_yes, '#B71C1C', 'TATA-containing median')]:
    df_sorted = df_sub.sort_values('length')
    # Use rolling window of 150 genes for smooth median line
    win = 150
    mids_list, vals_list = [], []
    for i in range(0, len(df_sorted) - win, win // 3):
        window = df_sorted.iloc[i:i+win]
        mids_list.append(window['length'].median())
        vals_list.append(window['dwell_auc'].median())
    ax_e.plot(mids_list, vals_list, color=color, linewidth=2.0, label=label)

ax_e.set_xlabel('Gene length (nt)', fontsize=9)
ax_e.set_ylabel('Dwell score (AUC)', fontsize=9)
ax_e.set_title('Nuclear Processing Dwell Score vs Gene Length\n(colored by promoter class — TATA-containing genes show higher dwell)',
               fontsize=9, fontweight='bold')
ax_e.set_xlim(0, 12000)
ax_e.spines['top'].set_visible(False); ax_e.spines['right'].set_visible(False)
ax_e.legend(fontsize=8, loc='lower right', framealpha=0.7)

# Mann-Whitney U test for dwell AUC by promoter class
from scipy.stats import mannwhitneyu
mw_stat, mw_p = mannwhitneyu(tata_yes['dwell_auc'], tata_no['dwell_auc'],
                               alternative='greater')
ax_e.text(0.98, 0.97, f'TATA-containing > TATA-less  (MW p = {mw_p:.2e})',
          transform=ax_e.transAxes, ha='right', va='top', fontsize=8, style='italic')
ax_e.text(-0.04, 1.02, 'E', transform=ax_e.transAxes, fontsize=16, fontweight='bold')
# Panel E dots are colored by promoter class — no unrelated functional-group legend here

fig.suptitle(
    'Figure 3  |  Multi-omics Integration: Protein-RNA Pathway Convergence and Chromatin Architecture',
    fontsize=12, fontweight='bold', y=0.975)

out = OUTDIR + "figure3_composite.pdf"
fig.savefig(out, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")

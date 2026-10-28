"""
Figure 3 panels — Cross-correlation / Integration
Portrait layout (8.5" × 11")

Letters as in the final Figure 3:
A: Multi-omics combined heatmap (SQUARE; baits ordered by the multi-omics dendrogram,
   which is drawn in Supplemental Figure 6A by make_figureS6A_dendrogram_AU.py)
B: CID boundary proximity violin plots
C: Promoter class distribution (x-axis colored by cluster)
D: Dwell score vs gene length scatter
(The dendrogram panel formerly drawn here as "A" is not part of Figure 3; removed 2026-10-28.)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS,
                           CLUSTER_COLORS, CMAP_APMS, CMAP_CORR, CMAP_SP, CMAP_RNA,
                           PATHWAY_ORDER_APMS, PATHWAY_ORDER_RNA, CLUSTER_ORDER,
                           CLUSTER_LABELS, fmt_baits, fmt_bait)
apply_style()

# Use correct BAIT_GROUP keys (matching GROUP_COLORS)
BAIT_GROUP = BAIT_GROUP_APMS

OUTDIR  = "output/"
DATADIR = "data/"

# ── Shared definitions ─────────────────────────────────────────────────────
PATHWAY_ORDER_APMS_L = [
    'CBP20','CBP80','THP2','YRA1','YRA2',
    'SNU71','SNU114','PRP42','PRP6',
    'RNA14','CLP1','YSH1','SYC1','CFT1','REF2','HRP1',
    'PAB1','PAP1','NAB2','NPL3','GBP2','HRB1','GFD1',
    'RRP6','SUB2','PML39','SAC3','THP1',
    'MEX67','DBP5','MLP1','MLP2','SHE2',
]
PATHWAY_ORDER_RNA_L = [
    'CBP80','THP2','YRA1','SNU71',
    'CLP1','SYC1','REF2','HRP1',
    'PAB1','NAB2','NPL3','GBP2','GFD1',
    'RRP6','SUB2','SAC3','MEX67','DBP5',
]
CLUSTER_ORDER = [1, 2, 3, 4, 5, 6, 7, 8]
CLUSTER_LABELS_SHORT = {1: "C1\nCBC/splicing/Npl3p", 2: "C2\nhnRNP/Poly(A)", 3: "C3\nTHO/TREX", 4: "C4\nHrp1p", 5: "C5\n3'end proc.", 6: "C6\nTREX-2", 7: "C7\nGfd1p/CBC", 8: "C8\nExport"}

# ── Load data ──────────────────────────────────────────────────────────────
df_apms = pd.read_csv(DATADIR + "apms_nuclear_matrix.tsv", sep='\t', index_col=0)
if 'MTR2' in df_apms.columns:
    df_apms = df_apms.drop(columns=['MTR2'])

df_rna   = pd.read_csv(DATADIR + "rnaseq_bait_means.tsv", sep='\t', index_col=0)
df_clust = pd.read_csv(DATADIR + "rna_gene_clusters.tsv",  sep='\t', index_col=0)
df_prom  = pd.read_csv(DATADIR + "promoter_annotated.tsv", sep='\t', index_col=0)
df_cid_g = pd.read_csv(DATADIR + "genes_cid_assigned.tsv", sep='\t', index_col=0)

apms_baits  = [b for b in PATHWAY_ORDER_APMS_L if b in df_apms.columns]
rna_baits   = [b for b in PATHWAY_ORDER_RNA_L  if b in df_rna.columns]
shared_baits = [b for b in rna_baits if b in apms_baits]

# ── Correlation matrices ───────────────────────────────────────────────────
corr_prot = df_apms[apms_baits].corr(method='pearson')
df_rna_filt = df_rna[rna_baits]
df_rna_filt = df_rna_filt[df_rna_filt.max(axis=1) > 0.3]
corr_rna    = df_rna_filt.corr(method='pearson')

corr_prot_s = corr_prot.loc[shared_baits, shared_baits]
corr_rna_s  = corr_rna.loc[shared_baits, shared_baits]
corr_multi  = (corr_prot_s + corr_rna_s) / 2.0

dist_multi = np.clip(1.0 - corr_multi.values, 0, None)
np.fill_diagonal(dist_multi, 0)
link_multi = linkage(squareform(dist_multi), method='average')

np.random.seed(42)
n_boot = 1000
X_prot = df_apms[apms_baits].loc[:, shared_baits].values
X_rna  = df_rna_filt.loc[:, shared_baits].values

clust_ids = fcluster(link_multi, 4, criterion='maxclust')
clust_id_map = dict(zip(shared_baits, clust_ids))

def boot_corr_combined(X_prot, X_rna, seed=None):
    rng = np.random.default_rng(seed)
    idx_p = rng.integers(0, X_prot.shape[0], X_prot.shape[0])
    idx_r = rng.integers(0, X_rna.shape[0],  X_rna.shape[0])
    cp = np.corrcoef(X_prot[idx_p].T)
    cr = np.corrcoef(X_rna[idx_r].T)
    return (cp + cr) / 2.0

pair_count = np.zeros((len(shared_baits), len(shared_baits)), dtype=float)
for i in range(n_boot):
    bc = boot_corr_combined(X_prot, X_rna, seed=i)
    dist_b = np.clip(1.0 - bc, 0, None)
    np.fill_diagonal(dist_b, 0)
    try:
        lk_b = linkage(squareform(dist_b), method='average')
        cl_b = fcluster(lk_b, 4, criterion='maxclust')
    except:
        continue
    for r in range(len(shared_baits)):
        for c in range(len(shared_baits)):
            if cl_b[r] == cl_b[c]:
                pair_count[r, c] += 1
print("Bootstrap done.")

# ── CID + promoter ─────────────────────────────────────────────────────────
# genes_cid_assigned.tsv carries the PRE-2026-07-16 cluster numbering; take the
# current C1–C8 labels from rna_gene_clusters.tsv instead (same genes, renumbered).
df_cid_g['cluster'] = df_clust['cluster'].reindex(df_cid_g.index)
df_cid_g['cluster'] = df_cid_g['cluster'].astype('Int64')
df_cid_g = df_cid_g.dropna(subset=['dist_to_boundary','cluster'])

df_pc = df_prom[['gene_class_1']].join(df_clust[['cluster']], how='inner')
df_pc = df_pc.dropna()
df_pc['cluster'] = df_pc['cluster'].astype(int)

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Portrait (8.5" × 11"), 2 cm margins ≈ 0.79"
# ════════════════════════════════════════════════════════════════════════════
# 2 cm / 8.5" = 0.235 in normalized coords for left/right
# 2 cm / 11" = 0.182 in normalized coords for top/bottom
fig = plt.figure(figsize=(8.5, 11))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(3, 2, figure=fig,
                        height_ratios=[1.8, 1.6, 1.4],
                        hspace=0.55, wspace=0.40,
                        left=0.12, right=0.88, top=0.93, bottom=0.07)

warm_cmap = CMAP_CORR

# Bait order for the heatmap: leaves of the multi-omics dendrogram (not drawn here)
dend = dendrogram(link_multi, labels=shared_baits, no_plot=True)
leaf_order = dend['leaves']

# ── Panel A (variable ax_b): Multi-omics heatmap (SQUARE) ─────────────────
ax_b = fig.add_subplot(gs[0, 0])

leaf_ord_names = [shared_baits[i] for i in leaf_order]
corr_reord = corr_multi.loc[leaf_ord_names, leaf_ord_names]

im_b = ax_b.imshow(corr_reord.values, cmap=warm_cmap, vmin=-0.2, vmax=1.0,
                    aspect='equal')   # SQUARE
ax_b.set_xticks(range(len(leaf_ord_names)))
ax_b.set_yticks(range(len(leaf_ord_names)))
# Black protein-convention names
ax_b.set_xticklabels(fmt_baits(leaf_ord_names), rotation=60, ha='right',
                      fontsize=5, color='black', fontweight='bold')
ax_b.set_yticklabels(fmt_baits(leaf_ord_names), fontsize=5,
                      color='black', fontweight='bold')

# Smaller colorbar (shrink reduced)
cb_b = plt.colorbar(im_b, ax=ax_b, shrink=0.35, pad=0.04,
                     label='Pearson r (combined)')
cb_b.ax.tick_params(labelsize=5)
cb_b.set_label('Pearson r', fontsize=5)
ax_b.set_title('Multi-omics Bait Correlation\n(proteomics + transcriptomics)',
               fontsize=7, fontweight='bold')
ax_b.text(-0.22, 1.06, 'A', transform=ax_b.transAxes, fontsize=16, fontweight='bold')

# ── Panel C: CID boundary proximity ───────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 1])

vdata_cid = [df_cid_g[df_cid_g['cluster']==k]['dist_to_boundary'].dropna().values
             for k in CLUSTER_ORDER]

parts_c = ax_c.violinplot(vdata_cid, positions=range(len(CLUSTER_ORDER)),
                           showmedians=True, showextrema=False, widths=0.75)
for pc, k in zip(parts_c['bodies'], CLUSTER_ORDER):
    pc.set_facecolor(CLUSTER_COLORS[k]); pc.set_alpha(0.7); pc.set_edgecolor('gray')
parts_c['cmedians'].set_color('black'); parts_c['cmedians'].set_linewidth(1.5)

ax_c.set_xticks(range(len(CLUSTER_ORDER)))
ax_c.set_xticklabels([CLUSTER_LABELS_SHORT[k].replace('\n',' ') for k in CLUSTER_ORDER],
                      fontsize=5.5, rotation=45, ha='right')
ax_c.set_ylabel('Distance to nearest\nCID boundary (bp)', fontsize=6)
ax_c.tick_params(axis='y', labelsize=5)   # smaller y-tick numbers
ax_c.set_title('Chromatin Domain Boundary\nProximity by RNA Cluster',
               fontsize=7, fontweight='bold')
ax_c.set_ylim(0, 5000)
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)

from scipy.stats import kruskal
kw_stat, kw_p = kruskal(*[d for d in vdata_cid if len(d) > 1])
ax_c.text(0.98, 0.97, f'KW p = {kw_p:.1e}', transform=ax_c.transAxes,
          ha='right', va='top', fontsize=6, style='italic')
ax_c.text(-0.20, 1.06, 'B', transform=ax_c.transAxes, fontsize=16, fontweight='bold')

# (cluster legend omitted: violins are labeled on the x-axis, as in the assembled figure)

# ── Panel D: Promoter class by RNA cluster ─────────────────────────────────
ax_d = fig.add_subplot(gs[1, :])

prom_counts = df_pc.groupby(['cluster','gene_class_1']).size().unstack(fill_value=0)
prom_frac   = prom_counts.div(prom_counts.sum(axis=1), axis=0).reindex(CLUSTER_ORDER)

PROM_COLORS = {'TATA-containing': '#E53935', 'TATA-less': '#90A4AE'}
x = np.arange(len(CLUSTER_ORDER))
bottom = np.zeros(len(CLUSTER_ORDER))
for pc_class, color in PROM_COLORS.items():
    if pc_class in prom_frac.columns:
        vals = prom_frac[pc_class].values
        ax_d.bar(x, vals, bottom=bottom, color=color, label=pc_class,
                 edgecolor='white', linewidth=0.4, width=0.75)
        bottom += vals

ax_d.set_xticks(x)
# X-axis labels COLORED by RNA cluster
xlbls = [CLUSTER_LABELS_SHORT[k].replace('\n',' ') for k in CLUSTER_ORDER]
ax_d.set_xticklabels(xlbls, fontsize=5.5, rotation=45, ha='right')
for tick, k in zip(ax_d.get_xticklabels(), CLUSTER_ORDER):
    tick.set_color('black')  # plain black cluster labels, as in the assembled figure

ax_d.set_ylabel('Fraction of genes', fontsize=6)
ax_d.tick_params(axis='y', labelsize=5)   # smaller y-tick numbers
ax_d.set_ylim(0, 1.05)
ax_d.set_xlim(-3, len(CLUSTER_ORDER) - 0.5)  # shift bars right within panel
ax_d.set_title('Promoter Class by RNA Cluster\n(TATA-containing vs TATA-less)',
               fontsize=7, fontweight='bold')
ax_d.spines['top'].set_visible(False); ax_d.spines['right'].set_visible(False)
ax_d.legend(fontsize=5.5, loc='center left', bbox_to_anchor=(1.05, 0.5),
            framealpha=0.9, title='Promoter class', title_fontsize=6)

for i, k in enumerate(CLUSTER_ORDER):
    n = prom_counts.loc[k].sum() if k in prom_counts.index else 0
    ax_d.text(i, 1.02, f'n={n}', ha='center', va='bottom', fontsize=4.5, color='#444')

ax_d.text(-0.08, 1.06, 'C', transform=ax_d.transAxes, fontsize=16, fontweight='bold')

# ── Panel E: Dwell score vs gene length ─────────────────────────────────────
ax_e = fig.add_subplot(gs[2, :])

df_dwell   = pd.read_csv(DATADIR + "rna_dwell_scores.tsv", sep='\t', index_col=0)
df_scatter = df_dwell[['dwell_auc','length']].join(df_prom[['gene_class_1']], how='inner')
df_scatter = df_scatter.dropna(subset=['dwell_auc','length','gene_class_1'])
df_scatter = df_scatter[df_scatter['length'] < 15000]

tata_yes = df_scatter[df_scatter['gene_class_1']=='TATA-containing']
tata_no  = df_scatter[df_scatter['gene_class_1']=='TATA-less']

ax_e.scatter(tata_no['length'], tata_no['dwell_auc'],
             c='#90A4AE', alpha=0.25, s=10, label='TATA-less', rasterized=True)
ax_e.scatter(tata_yes['length'], tata_yes['dwell_auc'],
             c='#E53935', alpha=0.35, s=12, label='TATA-containing', rasterized=True)

for df_sub, color, label in [(tata_no, '#546E7A', 'TATA-less median'),
                              (tata_yes, '#B71C1C', 'TATA-cont. median')]:
    df_sorted = df_sub.sort_values('length')
    win, mids_list, vals_list = 150, [], []
    for i in range(0, len(df_sorted) - win, win // 3):
        window = df_sorted.iloc[i:i+win]
        mids_list.append(window['length'].median())
        vals_list.append(window['dwell_auc'].median())
    ax_e.plot(mids_list, vals_list, color=color, linewidth=1.5, label=label)

ax_e.set_xlabel('Gene length (nt)', fontsize=7)   # smaller axis labels
ax_e.set_ylabel('Dwell score (AUC)', fontsize=7)
ax_e.set_title('Nuclear Processing Dwell Score vs Gene Length\n'
               '(TATA-containing genes show higher dwell)',
               fontsize=7, fontweight='bold')
ax_e.set_xlim(0, 12000)
ax_e.spines['top'].set_visible(False); ax_e.spines['right'].set_visible(False)
ax_e.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.01, 1.0), framealpha=0.9)
ax_e.tick_params(labelsize=6)

from scipy.stats import mannwhitneyu
mw_stat, mw_p = mannwhitneyu(tata_yes['dwell_auc'], tata_no['dwell_auc'],
                               alternative='greater')
ax_e.text(0.98, 0.97, f'TATA-cont. > TATA-less (MW p = {mw_p:.2e})',
          transform=ax_e.transAxes, ha='right', va='top', fontsize=6, style='italic')
ax_e.text(-0.05, 1.04, 'D', transform=ax_e.transAxes, fontsize=16, fontweight='bold')

# No suptitle — avoids text overlap
out = OUTDIR + "figure3_composite.pdf"
fig.savefig(out, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")

"""
Figure 2 panels B–E (figsize=22×12, matching Panel A width for assembly).
Letters as in the final Figure 2: B cluster mean profiles, C intron-containing genes per cluster,
D dwell score, E gene length (D/E swapped 2026-10-29). Panel A: make_figure2A_introns.py; panel F (non-coding RNA classes
relative to total RNA): make_figure2F_class_enrichment.py. The former transcript-class
composition panel is not part of the final figure and was removed (2026-10-28).
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

import sys, os
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (apply_style, GROUP_COLORS,
                           CLUSTER_COLORS, CMAP_RNA,
                           CLUSTER_ORDER, CLUSTER_LABELS, fmt_baits,
                           BAIT_GROUP_RNA as _BAIT_GROUP_RNA_BASE)
apply_style()

OUTDIR  = "output/"
DATADIR = "data/"

# Extend BAIT_GROUP_RNA for NAB4/SYC1 aliases
BAIT_GROUP_RNA = dict(_BAIT_GROUP_RNA_BASE)
BAIT_GROUP_RNA.setdefault('HRP1', "3'end processing")
BAIT_GROUP_RNA.setdefault('SYC1', "3'end processing")

PATHWAY_ORDER_RNA = [
    'CBP80','THP2','YRA1','SNU71',
    'CLP1','SYC1','REF2','HRP1',
    'PAB1','NAB2','NPL3','GBP2','GFD1',
    'RRP6','SUB2','SAC3','MEX67','DBP5',
]

CLUSTER_LABELS = {1: "C1\nCBC/splicing/Npl3p", 2: "C2\nhnRNP/Poly(A)", 3: "C3\nTHO/TREX", 4: "C4\nHrp1p", 5: "C5\n3'end proc.", 6: "C6\nTREX-2", 7: "C7\nGfd1p/CBC", 8: "C8\nExport"}
CLUSTER_ORDER = [1, 2, 3, 4, 5, 6, 7, 8]

COLOR_NONE  = '#E8E8E8'
COLOR_NONRP = '#8BB4D0'
COLOR_RP    = '#E8A87C'

def bait_color_bar_h(ax, baits):
    colors = [GROUP_COLORS.get(BAIT_GROUP_RNA.get(b,''), '#CCCCCC') for b in baits]
    n = len(baits)
    for i, c in enumerate(colors):
        ax.add_patch(mpatches.Rectangle((i/n, 0), 1/n, 1,
            color=c, transform=ax.transAxes, clip_on=False))

# ── GFF gene classification ───────────────────────────────────────────────────
GFF = "data/saccharomyces_cerevisiae.20260618.gff"
INTRON_RE = re.compile(r'Name=([^;]+)_intron')
ORF_RE    = re.compile(r'^Y[A-P][LR]\d{3}[CW](-[A-Z])?$')
intron_genes, rp_genes = set(), set()
with open(GFF) as fh:
    for line in fh:
        if line.startswith('#'): continue
        if '\tintron\t' in line:
            m = INTRON_RE.search(line)
            if m and ORF_RE.match(m.group(1)): intron_genes.add(m.group(1))
        p = line.strip().split('\t')
        if len(p) >= 9 and p[2] == 'gene':
            id_m = re.search(r'ID=([^;]+)', p[8])
            gn_m = re.search(r'gene=([^;]+)', p[8])
            if id_m and gn_m and ORF_RE.match(id_m.group(1)) and \
               re.match(r'^RP[LS]\d', gn_m.group(1), re.IGNORECASE):
                rp_genes.add(id_m.group(1))

def classify(g):
    if g in rp_genes and g in intron_genes: return 2   # intron-containing RP
    if g in intron_genes: return 1
    return 0

# ── Load data ─────────────────────────────────────────────────────────────────
df_rna   = pd.read_csv(DATADIR + "rnaseq_bait_means.tsv", sep='\t', index_col=0)
df_clust = pd.read_csv(DATADIR + "rna_gene_clusters.tsv",  sep='\t', index_col=0)
df_dwell = pd.read_csv(DATADIR + "rna_dwell_scores.tsv",   sep='\t', index_col=0)

rna_baits   = [b for b in PATHWAY_ORDER_RNA if b in df_rna.columns]
df_rna      = df_rna[rna_baits]
df_rna_filt = df_rna[df_rna.max(axis=1) > 0.3]

dist_g = pdist(df_rna_filt.values, metric='correlation')
dist_g = np.nan_to_num(dist_g, nan=1.0)
link_g = linkage(dist_g, method='average')
df_hcl = df_rna_filt.iloc[leaves_list(link_g)]

df_merged     = df_clust[['cluster']].join(df_rna_filt[rna_baits], how='inner')
cluster_means = df_merged.groupby('cluster')[rna_baits].mean()

df_gl = df_dwell[['length','dwell_auc','class']].join(df_clust[['cluster']], how='inner')
df_gl = df_gl.dropna(subset=['length','dwell_auc','cluster'])
df_gl['cluster'] = df_gl['cluster'].astype(int)

# Intron counts per cluster (Panel B)
df_ic = df_clust[['cluster']].copy()
df_ic['cat'] = pd.Series({g: classify(g) for g in df_ic.index})
clust_total  = df_ic.groupby('cluster').size()
rp_counts    = df_ic[df_ic['cat']==2].groupby('cluster').size().reindex(CLUSTER_ORDER, fill_value=0)
nonrp_counts = df_ic[df_ic['cat']==1].groupby('cluster').size().reindex(CLUSTER_ORDER, fill_value=0)
rp_frac      = (rp_counts    / clust_total * 100).reindex(CLUSTER_ORDER).fillna(0)
nonrp_frac   = (nonrp_counts / clust_total * 100).reindex(CLUSTER_ORDER).fillna(0)
# legend totals = sum of the per-cluster bar counts (same 2,434 clustered genes as Panel A)
n_rp_in    = int(rp_counts.sum())
n_nonrp_in = int(nonrp_counts.sum())

# ── ncRNA data ────────────────────────────────────────────────────────────────
import openpyxl
nc_dir = "data/ncRNA/"
nc_files = {
    'Coding': nc_dir+"all_codingRNAs_WTPOs_and_Mutants.xlsx",
    'SUT':    nc_dir+"all_SUTs_WTPOs_and_Mutants.xlsx",
    'CUT':    nc_dir+"all_CUTs_WTPOs_and_Mutants.xlsx",
    'XUT':    nc_dir+"all_XUTs_WTPOs_and_Mutants.xlsx",
    'lncRNA': nc_dir+"all_lncRNAs_WTPOs_and_Mutants.xlsx",
}
NC_BAIT_COLS = {
    'CBP80':[2,3,4],'THP2':[5,6],'YRA1':[7,8],'SNU71':[11,12],
    'SYC1':[13,14,15],'CLP1':[16,17,18],'REF2':[19,20,21],
    'HRP1':[22,23,24,25,26],'RRP6':[27,28,29],'NAB2':[30,31,32,33],
    'NPL3':[34],'GBP2':[35,36,37],'SUB2':[38,39,40],
    'SAC3':[41,42],'MEX67':[43,44,45],'GFD1':[46],
    'DBP5':[47,48,49],'PAB1':[50,51,52,53],
}
def load_nc(fpath, bait_list):
    wb = openpyxl.load_workbook(fpath, read_only=True, data_only=True)
    rows = list(wb.active.iter_rows(values_only=True))[1:]
    wb.close()
    out = {}
    for b in bait_list:
        idxs = NC_BAIT_COLS.get(b, [])
        s = 0.0
        for row in rows:
            vals = [float(row[i]) for i in idxs if i < len(row) and row[i] is not None
                    and str(row[i]).replace('.','',1).lstrip('-').isdigit()]
            if vals: s += np.mean(vals)
        out[b] = s
    return out

print("Loading ncRNA data...")
nc_totals = {}
for cls, fp in nc_files.items():
    if os.path.exists(fp):
        nc_totals[cls] = load_nc(fp, rna_baits)
nc_df = pd.DataFrame(nc_totals).fillna(0)
nc_df_norm = nc_df.div(nc_df.sum(axis=1), axis=0)
nc_df_norm = nc_df_norm.loc[[b for b in rna_baits if b in nc_df_norm.index]]

NC_COLORS = {'Coding':'#D7B899','SUT':'#FF8C42','CUT':'#FFFACD',
             'XUT':'#E53935','lncRNA':'#7986CB'}

# ════════════════════════════════════════════════════════════════════════════
# FIGURE — wide format matching Panel A (22" × 12")
# ════════════════════════════════════════════════════════════════════════════
FS = 2.4   # scale factor relative to original 8.5" design

fig = plt.figure(figsize=(22, 12))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(3, 2, figure=fig,
                        height_ratios=[1.2, 1.6, 1.4],
                        hspace=0.55, wspace=0.22,
                        left=0.06, right=0.84, top=0.95, bottom=0.06)

LBL = 32    # panel letter size
TTL = 16    # panel title size
AXL = 18    # axis label size
TKL = 15    # tick label size (C, F)
TKL_DE = 12 # tick label size (D, E — slightly smaller)
LGS = 11    # legend entry size
LGT = 12    # legend title size
ANT = 17    # annotation text size

# ── Panel C (variable ax_b): intron-containing genes per cluster ─────────────
ax_b = fig.add_subplot(gs[1, 0])
x_pos = np.arange(len(CLUSTER_ORDER))
ax_b.bar(x_pos, [nonrp_frac[k] for k in CLUSTER_ORDER],
         color=COLOR_NONRP, edgecolor='white', linewidth=0.4, width=0.65,
         label=f'Non-RP intron genes (n={n_nonrp_in})')
ax_b.bar(x_pos, [rp_frac[k] for k in CLUSTER_ORDER],
         bottom=[nonrp_frac[k] for k in CLUSTER_ORDER],
         color=COLOR_RP, edgecolor='white', linewidth=0.4, width=0.65,
         label=f'Intron-containing RP genes (n={n_rp_in})')

bar_labels = [f"{CLUSTER_LABELS[k].split(chr(10))[0]}\n({int(nonrp_counts[k])+int(rp_counts[k])}/{int(clust_total[k])})"
              for k in CLUSTER_ORDER]
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(bar_labels, fontsize=TKL, ha='center')
for tick, k in zip(ax_b.get_xticklabels(), CLUSTER_ORDER):
    tick.set_color(CLUSTER_COLORS[k]); tick.set_fontweight('bold')

for xi, k in enumerate(CLUSTER_ORDER):
    tot = nonrp_frac[k] + rp_frac[k]
    if tot > 0.5:
        ax_b.text(xi, tot + 0.4, f'{tot:.1f}%',
                  ha='center', va='bottom', fontsize=TKL, color='#444444')

max_y = max(nonrp_frac[k]+rp_frac[k] for k in CLUSTER_ORDER)
ax_b.set_ylim(0, max_y * 1.35 + 1)
ax_b.set_xlim(-0.5, len(CLUSTER_ORDER)-0.5)
ax_b.set_ylabel('Intron genes (% of cluster)', fontsize=AXL)
ax_b.set_title('Intron-containing gene distribution across RNA-seq clusters (k=8)',
               fontsize=TTL, fontweight='bold', pad=10)
ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)
ax_b.legend(fontsize=LGS+2, loc='upper center', framealpha=0.9, handlelength=1.2)
_lbl_b = ax_b.text(-0.08, 1.04, 'C', transform=ax_b.transAxes, fontsize=LBL, fontweight='bold')

# ── Panel B (variable ax_c): cluster mean profiles ──────────────────────────
ax_c = fig.add_subplot(gs[0, :])
for k in CLUSTER_ORDER:
    if k in cluster_means.index:
        ax_c.plot(range(len(rna_baits)), cluster_means.loc[k, rna_baits].values,
                  color=CLUSTER_COLORS[k], linewidth=2.5,
                  label=CLUSTER_LABELS[k].replace('\n',' '), alpha=0.9)
ax_c.set_xticks(range(len(rna_baits)))
ax_c.set_xticklabels(fmt_baits(rna_baits), fontsize=TKL, rotation=60, ha='right')
ax_c.set_ylabel('Mean enrichment (max=1)', fontsize=AXL)
ax_c.set_title('k=8 RNA Cluster Mean Profiles\n(pathway-stage enrichment per cluster)',
               fontsize=TTL, fontweight='bold')
ax_c.set_ylim(-0.02, 0.85)
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)
ax_c_top = ax_c.inset_axes([0, 1.01, 1, 0.025])
ax_c_top.axis('off'); bait_color_bar_h(ax_c_top, rna_baits)
ax_c.text(-0.018, 1.05, 'B', transform=ax_c.transAxes, fontsize=LBL, fontweight='bold')

# ── Panel D (half-width, right of C): dwell score ─────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
vdata_d = [df_gl[df_gl['cluster']==k]['dwell_auc'].dropna().values for k in CLUSTER_ORDER]
parts_d = ax_d.violinplot(vdata_d, positions=range(len(CLUSTER_ORDER)),
                           showmedians=True, showextrema=False, widths=0.75)
for pc, k in zip(parts_d['bodies'], CLUSTER_ORDER):
    pc.set_facecolor(CLUSTER_COLORS[k]); pc.set_alpha(0.7); pc.set_edgecolor('gray')
parts_d['cmedians'].set_color('black'); parts_d['cmedians'].set_linewidth(1.5)
ax_d.set_xticks(range(len(CLUSTER_ORDER)))
ax_d.set_xticklabels([CLUSTER_LABELS[k].split('\n')[0] for k in CLUSTER_ORDER],
                      fontsize=TKL_DE, rotation=45, ha='right')
ax_d.set_ylabel('Dwell score (AUC)', fontsize=AXL)
ax_d.set_title('Nuclear Processing Dwell Score by Cluster\n(integrated pathway-stage enrichment)',
               fontsize=TTL, fontweight='bold')
ax_d.spines['top'].set_visible(False); ax_d.spines['right'].set_visible(False)

# Shared cluster legend — right of Panel D, serves C, D and E
_clust_h = [mpatches.Patch(color=CLUSTER_COLORS[k],
    label=CLUSTER_LABELS[k].replace('\n',' ')) for k in CLUSTER_ORDER]
ax_d.legend(handles=_clust_h, fontsize=LGS, ncol=1,
    loc='upper left', bbox_to_anchor=(1.03, 1.0), framealpha=0.9,
    title='RNA cluster', title_fontsize=LGT, handlelength=1.0, labelspacing=0.35)
ax_d.text(-0.08, 1.04, 'D', transform=ax_d.transAxes, fontsize=LBL, fontweight='bold')

# ── Panel E (half-width, bottom left; F sits to its right in the assembled figure): gene length ─────────────────────────────────────────
ax_e = fig.add_subplot(gs[2, 0])
vdata_len = [df_gl[df_gl['cluster']==k]['length'].dropna().values for k in CLUSTER_ORDER]
parts = ax_e.violinplot(vdata_len, positions=range(len(CLUSTER_ORDER)),
                        showmedians=True, showextrema=False, widths=0.75)
for pc, k in zip(parts['bodies'], CLUSTER_ORDER):
    pc.set_facecolor(CLUSTER_COLORS[k]); pc.set_alpha(0.7); pc.set_edgecolor('gray')
parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.5)
ax_e.axhline(1200, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
ax_e.annotate('WCL avg\n~1200 nt',
              xy=(len(CLUSTER_ORDER)-1, 1200),
              xytext=(len(CLUSTER_ORDER)-1.5, 1200+1400),
              fontsize=ANT, fontweight='bold', color='#333333',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                        edgecolor='#888888', linewidth=1.2),
              arrowprops=dict(arrowstyle='->', color='#666666', lw=1.0))
ax_e.set_xticks(range(len(CLUSTER_ORDER)))
ax_e.set_xticklabels([CLUSTER_LABELS[k].split('\n')[0] for k in CLUSTER_ORDER],
                      fontsize=TKL_DE, rotation=45, ha='right')
ax_e.set_ylabel('Gene length (nt)', fontsize=AXL)
ax_e.set_title('Gene Length Distribution by RNA Cluster\n(read-count weighted)',
               fontsize=TTL, fontweight='bold')
ax_e.set_ylim(0, 8000)
ax_e.spines['top'].set_visible(False); ax_e.spines['right'].set_visible(False)
ax_e.text(-0.08, 1.04, 'E', transform=ax_e.transAxes, fontsize=LBL, fontweight='bold')

out = OUTDIR + "figure2_composite.pdf"
fig.savefig(out, bbox_inches='tight', facecolor='white')
# Intron panel alone (manuscript Figure 2C), for placement in the assembled figure
_bb = ax_b.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
fig.savefig(OUTDIR + "figure2C_introns.pdf", bbox_inches=_bb.expanded(1.02, 1.04), facecolor='white')
plt.close()
print(f"Saved: {out}")

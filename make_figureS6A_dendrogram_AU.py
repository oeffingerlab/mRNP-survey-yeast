"""
Supplemental Figure 6A with multiscale-bootstrap AU values at internal nodes.

Reproduces Panel A of make_figureS3.py exactly (combined AP-MS + AP-RNA-seq bait-bait
Pearson correlation, 18 shared baits, 1 - r distance, average linkage, per-leaf
functional-module markers) and labels each internal node with its AU (%) from
multiomics_module_AU.tsv (output of multiomics_bootstrap_au.py, ten scales x 1,200
resamples). The root has no AU. Panel geometry and fonts match the panel A cell of
make_figureS3.py so it can be placed at 100% in the Illustrator file.

Inputs : data/apms_nuclear_matrix.tsv, data/rnaseq_bait_means.tsv, multiomics_module_AU.tsv
Output : output/figureS6A_dendrogram_AU.pdf
"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, dendrogram
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import apply_style, fmt_bait, PATHWAY_ORDER_RNA, PATHWAY_ORDER_APMS
apply_style()

DATADIR, OUTDIR = "data/", "output/"
os.makedirs(OUTDIR, exist_ok=True)

def find(name):
    for p in [name, DATADIR + name] + glob.glob("**/" + name, recursive=True):
        if os.path.exists(p):
            return p
    sys.exit("missing " + name)

# ── Data and linkage (identical to make_figureS3.py) ──────────────────────
df_apms = pd.read_csv(DATADIR + "apms_nuclear_matrix.tsv", sep='\t', index_col=0).drop(columns=['MTR2'], errors='ignore')
df_rna = pd.read_csv(DATADIR + "rnaseq_bait_means.tsv", sep='\t', index_col=0)
rna_baits = [b for b in PATHWAY_ORDER_RNA if b in df_rna.columns]
apms_baits = [b for b in PATHWAY_ORDER_APMS if b in df_apms.columns]
shared = [b for b in rna_baits if b in apms_baits]
df_rna_filt = df_rna[rna_baits]
df_rna_filt = df_rna_filt[df_rna_filt.max(axis=1) > 0.3]
corr_apms = df_apms[shared].corr(method='pearson').loc[shared, shared]
corr_rna = df_rna_filt[shared].corr(method='pearson').loc[shared, shared]
corr_comb = ((corr_apms + corr_rna) / 2.0).loc[shared, shared]
dist = np.clip(1.0 - corr_comb.values, 0, None); np.fill_diagonal(dist, 0)
Z = linkage(dist[np.triu_indices(len(shared), k=1)], method='average')

CLUSTER_4 = {'CBP80': 1, 'THP2': 1, 'YRA1': 1,
             'SNU71': 2, 'SYC1': 2, 'REF2': 2, 'CLP1': 2, 'HRP1': 2,
             'NAB2': 3, 'NPL3': 3, 'GBP2': 3, 'PAB1': 3, 'GFD1': 3, 'RRP6': 3,
             'SAC3': 4, 'MEX67': 4, 'DBP5': 4, 'SUB2': 4}
CLUSTER_COLS = {1: '#DB2827', 2: '#465EAB', 3: '#56C4CB', 4: '#0A7F40'}

# ── AU values keyed by member set ────────────────────────────────────────
au = pd.read_csv(find("multiomics_module_AU.tsv"), sep='\t')
AU = {frozenset(m.split(';')): v for m, v in zip(au['members'], au['AU'])}

# ── Panel geometry = panel A cell of make_figureS3.py (22 x 26 in figure) ─
cell_w = 22 * (0.97 - 0.06) / (2 + 0.32)          # one GridSpec column
cell_h = 26 * (0.96 - 0.04) / (2 + 0.38)          # one GridSpec row
ax_w, ax_h = cell_w, cell_h * 4 / 5                # dendrogram = top 4/5 of the cell
fig_w, fig_h = ax_w + 1.6, ax_h + 2.4
fig = plt.figure(figsize=(fig_w, fig_h)); fig.patch.set_facecolor('white')
ax = fig.add_axes([1.2 / fig_w, 1.6 / fig_h, ax_w / fig_w, ax_h / fig_h])

dend = dendrogram(Z, labels=shared, ax=ax, orientation='top', color_threshold=0,
                  above_threshold_color='black', leaf_font_size=7)
ax.set_xticklabels([fmt_bait(t.get_text()) for t in ax.get_xticklabels()])
for lbl in ax.get_xticklabels():
    lbl.set_color('black'); lbl.set_fontweight('bold')

leaf_names = [shared[i] for i in dend['leaves']]
ylim = ax.get_ylim(); span = ylim[1] - ylim[0]
bar_y = ylim[0] - 0.05 * span; sq_h = 0.035 * span
ax.set_ylim(bar_y - 0.03 * span, ylim[1])
for i, nm in enumerate(leaf_names):
    ax.add_patch(mpatches.Rectangle((i * 10 + 5 - 2.6, bar_y), 5.2, sq_h,
                 facecolor=CLUSTER_COLS[CLUSTER_4[nm]], edgecolor='white', linewidth=0.6,
                 transform=ax.transData, clip_on=False))

# ── Node x positions and member sets, then AU labels ─────────────────────
n = len(shared)
xpos = {dend['leaves'][i]: i * 10 + 5 for i in range(n)}
members = {i: {shared[i]} for i in range(n)}
labeled, missing = 0, []
parent_h = {}
for k, (a, b, h, _) in enumerate(Z):
    parent_h[int(a)] = h; parent_h[int(b)] = h
for k, (a, b, h, _) in enumerate(Z):
    node = n + k; a, b = int(a), int(b)
    xpos[node] = (xpos[a] + xpos[b]) / 2
    members[node] = members[a] | members[b]
for k, (a, b, h, _) in enumerate(Z):
    node = n + k
    if len(members[node]) == n:
        continue                                   # root: no AU
    key = frozenset(members[node])
    if key not in AU:
        missing.append(sorted(key)); continue
    # above the bar, unless the parent bar sits just above (then below the bar)
    below = parent_h.get(node, 1e9) - h < 0.03
    ax.text(xpos[node], h - 0.006 if below else h + 0.006, f"{AU[key]:.1f}",
            ha='center', va='top' if below else 'bottom', fontsize=7, color='black',
            bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none'))
    labeled += 1
assert not missing, missing
print(f"labeled {labeled} internal nodes")

ax.set_title('Multi-omics Bait Dendrogram\n'
             '(AP-MS + AP-RNA-seq combined Pearson, average linkage)',
             fontsize=11, fontweight='bold')
ax.text(1.0, 1.0, 'Node values: multiscale-bootstrap AU (%)', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=7)
ax.set_ylabel('Distance (1 – Pearson)', fontsize=10)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.text(-0.08, 1.04, 'A', transform=ax.transAxes, fontsize=16, fontweight='bold')

fig.savefig(OUTDIR + "figureS6A_dendrogram_AU.pdf")
fig.savefig(OUTDIR + "figureS6A_dendrogram_AU.png", dpi=200)
print("wrote", OUTDIR + "figureS6A_dendrogram_AU.pdf")

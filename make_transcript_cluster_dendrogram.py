"""
Transcript-cluster relationships: centroid dendrogram of the eight AP-RNA-seq
transcript clusters C1–C8 with bootstrap support (Supplemental Figure 4B).

Reproduces `Fig2_transcript_cluster_dendrogram_panel.pdf` (built inline
2026-10-01, §23J) as a saved, reproducible script.

Method
  Data      data/rna_gene_clusters.tsv — 2,434 transcripts x 18 baits + cluster
            label C1–C8 (clusters from make_rna_gene_clusters.py: Ward linkage on
            1 − r, k = 8).
  Centroids mean 18-bait enrichment profile per cluster (8 x 18).
  Tree      hierarchical clustering of the centroids, Pearson correlation
            distance (1 − r), average linkage.
  Support   feature bootstrap: resample the 18 baits with replacement (B = 2000,
            numpy default_rng(1)), re-cluster, and report for each internal node
            the % of bootstrap trees containing the same clade (BP %).
            Expected: two principal branches {C8,C5,C6} and {C3,C4,C7,C1,C2}
            both 98 %; {C5,C6} 61; {C4,C7,C1,C2} 88; {C7,C1,C2} 58; {C1,C2} 50.

Output    output/transcript_cluster_dendrogram.pdf (+ .png), panel letter via --letter
Usage     python3 make_transcript_cluster_dendrogram.py [--letter B]
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figure_style import apply_style, CLUSTER_COLORS, CLUSTER_LABELS
apply_style()
plt.rcParams['pdf.fonttype'] = 42

DATADIR = os.path.join(HERE, "data")
OUTDIR = os.path.join(HERE, "output")
os.makedirs(OUTDIR, exist_ok=True)
LETTER = sys.argv[sys.argv.index("--letter") + 1] if "--letter" in sys.argv else "B"
B = 2000
SEED = 1

df = pd.read_csv(os.path.join(DATADIR, "rna_gene_clusters.tsv"), sep='\t', index_col=0)
cent = df.groupby('cluster').mean()          # 8 clusters x 18 baits
X = cent.values
labs = list(cent.index)
n = len(labs)


def tree(M):
    return linkage(pdist(M, metric='correlation'), method='average')


def clades(Z):
    """Map linkage row index -> frozenset of member cluster ids (proper clades only)."""
    members = {i: frozenset([labs[i]]) for i in range(n)}
    out = {}
    for i, (a, b, _, _) in enumerate(Z):
        members[n + i] = members[int(a)] | members[int(b)]
        if len(members[n + i]) < n:
            out[i] = members[n + i]
    return out


Z = tree(X)
ref = clades(Z)

rng = np.random.default_rng(SEED)
support = {i: 0 for i in ref}
for _ in range(B):
    idx = rng.integers(0, X.shape[1], X.shape[1])
    boot = set(clades(tree(X[:, idx])).values())
    for i, c in ref.items():
        support[i] += c in boot
bp = {i: round(100 * v / B) for i, v in support.items()}

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 4.1))
fig.patch.set_facecolor('white')
d = dendrogram(Z, ax=ax, labels=[f"C{k}" for k in labs], color_threshold=0,
               above_threshold_color='#333333', link_color_func=lambda k: '#333333')
for coll in ax.collections:
    coll.set_linewidth(1.5)

# node support labels: find each internal node's bar (top of its U)
leaf_x = {lab: 5 + 10 * i for i, lab in enumerate(d['ivl'])}
for icoord, dcoord in zip(d['icoord'], d['dcoord']):
    h = dcoord[1]
    row = int(np.argmin(np.abs(Z[:, 2] - h)))
    if row in bp:
        ax.text(icoord[2] + 1, h + 0.015, f"{bp[row]}", ha='left', va='bottom',
                fontsize=7.5, fontweight='bold')

ticks = ax.get_xticks()
ax.set_xticklabels([])
for t, lab in zip(ticks, d['ivl']):
    k = int(lab[1:])
    ax.text(t, -0.02, CLUSTER_LABELS[k], transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=8, fontweight='bold', color=CLUSTER_COLORS[k])
ax.tick_params(axis='x', length=0)
ax.tick_params(axis='y', labelsize=7)
ax.set_ylabel('Pearson distance (1 - r)', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Transcript-cluster relationships (centroid dendrogram,\n'
             'Pearson / average linkage; node values = bootstrap support %)', fontsize=8.5)
ax.text(-0.11, 1.06, LETTER, transform=ax.transAxes, fontsize=14, fontweight='bold')

fig.savefig(os.path.join(OUTDIR, "transcript_cluster_dendrogram.pdf"), bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(OUTDIR, "transcript_cluster_dendrogram.png"), dpi=150, bbox_inches='tight', facecolor='white')

print("leaf order:", ' '.join(d['ivl']))
for i, c in sorted(ref.items(), key=lambda kv: -Z[kv[0], 2]):
    print(f"  {{{', '.join(f'C{k}' for k in sorted(c))}}}  height {Z[i, 2]:.3f}  BP {bp[i]}%")

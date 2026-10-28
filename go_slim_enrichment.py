"""
GO-slim biological-process enrichment of AP-RNA-seq co-purified transcripts
(Supplemental Figure 4C, per bait; 4D, per transcript cluster).

Method
  Annotation  data/go_slim_mapping.tab — SGD GO-slim gene mapping
              (https://downloads.yeastgenome.org/curation/literature/go_slim_mapping.tab),
              biological-process aspect (P) only; the root/'other'/'not_yet_annotated'
              terms are excluded.
  Background  all coding ORFs in data/rnaseq_bait_means.tsv that carry at least one
              GO-slim P annotation (the AP-RNA-seq gene set, n = 2,452 before annotation).
  Gene sets   --by cluster : the eight transcript clusters C1–C8 (data/rna_gene_clusters.tsv)
              --by bait    : per bait, genes with mean max-normalized enrichment > 0.3
  Test        one-sided Fisher's exact test (over-representation) per set × term;
              Benjamini–Hochberg FDR across all tests of the run.
              Fold enrichment = (k / n) / (K / N).
  Terms shown terms with FDR < 0.05 and fold enrichment >= 1.5 in at least one set
              (all terms are written to the TSV).

Output   output/go_slim_enrichment_<by>.tsv (all set × term statistics)
         output/go_slim_enrichment_<by>.pdf/.png (vector heatmap: fold enrichment,
         capped at 5×; dot = FDR < 0.05)
Usage    python3 go_slim_enrichment.py [--by cluster|bait] [--letter C|D]  (default: C per bait, D per cluster = Supp Fig 4C, 4D)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figure_style import apply_style, CLUSTER_ORDER, CLUSTER_LABELS, PATHWAY_ORDER_RNA, fmt_bait
apply_style()
plt.rcParams['pdf.fonttype'] = 42

DATADIR = os.path.join(HERE, "data")
OUTDIR = os.path.join(HERE, "output")
os.makedirs(OUTDIR, exist_ok=True)
BY = sys.argv[sys.argv.index("--by") + 1] if "--by" in sys.argv else "cluster"
LETTER = sys.argv[sys.argv.index("--letter") + 1] if "--letter" in sys.argv else ("C" if BY == "bait" else "D")
THRESH = 0.3
FDR_MAX, FOLD_MIN, CAP = 0.05, 1.5, 5.0
EXCLUDE = {'biological_process', 'other', 'not_yet_annotated'}


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


# ── Annotation ────────────────────────────────────────────────────────────
cols = ['orf', 'gene', 'sgdid', 'aspect', 'term', 'goid', 'ftype']
go = pd.read_csv(os.path.join(DATADIR, "go_slim_mapping.tab"), sep='\t', header=None,
                 names=cols, dtype=str)
go = go[(go.aspect == 'P') & (~go.term.isin(EXCLUDE))]
rna = pd.read_csv(os.path.join(DATADIR, "rnaseq_bait_means.tsv"), sep='\t', index_col=0)
universe = sorted(set(rna.index) & set(go.orf))
go = go[go.orf.isin(universe)]
term_genes = go.groupby('term').orf.apply(set)
N = len(universe)

# ── Gene sets ─────────────────────────────────────────────────────────────
if BY == 'cluster':
    cl = pd.read_csv(os.path.join(DATADIR, "rna_gene_clusters.tsv"), sep='\t', index_col=0)['cluster']
    sets = {CLUSTER_LABELS[k].replace('\n', ' '): set(cl[cl == k].index) & set(universe)
            for k in CLUSTER_ORDER}
else:
    baits = [b for b in PATHWAY_ORDER_RNA if b in rna.columns]
    sets = {fmt_bait(b): set(rna.index[rna[b] > THRESH]) & set(universe) for b in baits}

# ── Tests ─────────────────────────────────────────────────────────────────
rows = []
for sname, S in sets.items():
    n = len(S)
    for term, T in term_genes.items():
        K = len(T); k = len(S & T)
        p = fisher_exact([[k, n - k], [K - k, N - n - K + k]], alternative='greater')[1]
        fold = (k / n) / (K / N) if n and K else np.nan
        rows.append(dict(set=sname, term=term, k=k, n=n, K=K, N=N, fold=fold, p=p))
res = pd.DataFrame(rows)
res['fdr'] = bh(res.p.values)
res.sort_values(['set', 'p']).to_csv(os.path.join(OUTDIR, f"go_slim_enrichment_{BY}.tsv"),
                                    sep='\t', index=False)

sig = res[(res.fdr < FDR_MAX) & (res.fold >= FOLD_MIN)]
set_names = list(sets)
# order terms by the set in which they are most enriched, then by fold (diagonal layout)
best = sig.loc[sig.groupby('term').fold.idxmax()]
best['rank'] = best.set.map({s: i for i, s in enumerate(set_names)})
terms = best.sort_values(['rank', 'fold'], ascending=[True, False]).term.tolist()
print(f"universe N = {N}; {len(term_genes)} GO-slim P terms; sets: "
      + ", ".join(f"{s} (n={len(sets[s])})" for s in set_names))
print(f"{len(sig)} significant set×term pairs (FDR<{FDR_MAX}, fold>={FOLD_MIN}); {len(terms)} terms shown")
for s in set_names:
    top = sig[sig.set == s].sort_values('p').head(4)
    print(f"  {s:22s} " + "; ".join(f"{r.term} {r.fold:.1f}x q={r.fdr:.1e}" for r in top.itertuples()))

# ── Figure (vector cells via pcolormesh) ──────────────────────────────────
if terms:
    F = res.pivot(index='set', columns='term', values='fold').loc[set_names, terms].fillna(0)
    Q = res.pivot(index='set', columns='term', values='fdr').loc[set_names, terms]
    fig, ax = plt.subplots(figsize=(max(6, 0.32 * len(terms) + 2.5), 0.32 * len(set_names) + 2.4))
    m = ax.pcolormesh(np.clip(F.values, 0, CAP), cmap='YlOrRd', vmin=0, vmax=CAP,
                      edgecolors='white', linewidth=0.4)
    yy, xx = np.where(Q.values < FDR_MAX)
    ax.scatter(xx + 0.5, yy + 0.5, s=6, color='black', zorder=3)
    ax.set_xticks(np.arange(len(terms)) + 0.5)
    ax.set_xticklabels(terms, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(np.arange(len(set_names)) + 0.5)
    ax.set_yticklabels(set_names, fontsize=7.5)
    ax.invert_yaxis()
    cb = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label('Fold enrichment (cap 5×)', fontsize=7.5); cb.ax.tick_params(labelsize=7)
    ax.set_title('GO-slim biological-process enrichment '
                 f"({'transcript cluster' if BY == 'cluster' else 'per bait, enrichment > 0.3'}; "
                 'Fisher, BH FDR; dot = FDR < 0.05)', fontsize=8.5, fontweight='bold')
    ax.text(-0.02, 1.04, LETTER, transform=ax.transAxes, ha='right', fontsize=14, fontweight='bold')
    for ext, kw in (('pdf', {}), ('png', {'dpi': 150})):
        fig.savefig(os.path.join(OUTDIR, f"go_slim_enrichment_{BY}.{ext}"),
                    bbox_inches='tight', facecolor='white', **kw)

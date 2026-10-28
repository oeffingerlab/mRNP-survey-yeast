"""
AP-RNA-seq transcript clusters C1–C8 — reproducible regeneration.

Recovers exactly the cluster assignments in data/rna_gene_clusters.tsv
(all 2,434 genes, gene-by-gene identical; verified 2026-10-08).

Method
  Input   data/rnaseq_bait_means.tsv — per-gene, per-bait mean of the
          replicate-level max-normalized FPKM enrichment (18 analyzed baits).
  Filter  genes with enrichment > 0.3 in at least one of the 18 baits
          (n = 2,434 coding ORFs).
  Cluster hierarchical clustering, Ward linkage on Pearson correlation
          distance (1 − r) between gene profiles across the 18 baits
          (scipy: linkage(pdist(X, 'correlation'), 'ward')),
          tree cut into k = 8 clusters (fcluster, criterion='maxclust').
  Labels  scipy's fcluster ids are the ORIGINAL (June 2026) numbering; they are
          mapped to the current early→late numbering C1–C8 (renumbering of
          2026-07-16) with OLD_TO_NEW below.

Notes
  - Bait column order is the one stored in rna_gene_clusters.tsv; it does not
    affect the result (distances are computed across all 18 baits).
  - k = 8 is not a silhouette optimum (silhouette on 1 − r falls monotonically
    from k = 2: 0.37 to k = 8: 0.22); k = 8 was chosen to resolve pathway stages.

Usage
  python3 make_rna_gene_clusters.py            # regenerate + verify
  python3 make_rna_gene_clusters.py --write    # also write data/rna_gene_clusters_regenerated.tsv
"""
import sys
import os
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

HERE = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(HERE, "data")

BAITS = ['CBP80', 'NAB2', 'HRP1', 'NPL3', 'GBP2', 'PAB1', 'SNU71', 'SUB2', 'THP2',
         'CLP1', 'SYC1', 'REF2', 'SAC3', 'GFD1', 'YRA1', 'MEX67', 'DBP5', 'RRP6']
THRESHOLD = 0.3
K = 8
# fcluster id (original numbering) -> current cluster label C1–C8 (renumbered 2026-07-16)
OLD_TO_NEW = {1: 8, 2: 6, 3: 5, 4: 3, 5: 4, 6: 2, 7: 7, 8: 1}


def regenerate():
    means = pd.read_csv(os.path.join(DATADIR, "rnaseq_bait_means.tsv"), sep="\t", index_col=0)
    X = means[BAITS]
    X = X[(X > THRESHOLD).any(axis=1)]
    Z = linkage(pdist(X.values, metric="correlation"), method="ward")
    raw = fcluster(Z, K, criterion="maxclust")
    out = X.copy()
    out["cluster"] = [OLD_TO_NEW[int(c)] for c in raw]
    return out


def main():
    out = regenerate()
    print(f"genes clustered: {len(out)}")
    print("cluster sizes C1–C8:", [int((out['cluster'] == k).sum()) for k in range(1, K + 1)])
    ref_path = os.path.join(DATADIR, "rna_gene_clusters.tsv")
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path, sep="\t", index_col=0)["cluster"]
        same_genes = set(ref.index) == set(out.index)
        identical = same_genes and (ref.loc[out.index] == out["cluster"]).all()
        print(f"identical to rna_gene_clusters.tsv: {identical}")
        if not identical:
            sys.exit("MISMATCH with committed cluster file")
    if "--write" in sys.argv:
        p = os.path.join(DATADIR, "rna_gene_clusters_regenerated.tsv")
        out.to_csv(p, sep="\t")
        print("wrote", p)


if __name__ == "__main__":
    main()

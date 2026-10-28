"""
Mlp1p (nuclear basket) AP-RNA-seq vs. the 18-bait AP-RNA-seq survey.

Compares the survey with the Mlp1p AP-RNA-seq dataset of Bensidoun et al. (2022)
Mol Cell 82:3856-3871 (basket-containing vs. basket-less NPCs). Used for the
Discussion paragraph on the nuclear basket checkpoint.

Inputs (published data, not redistributed; pass paths on the command line):
  --counts  DESeq2 normalized-count table (genes x samples; columns AP<X>_replicat<n> ...)
            A = basket-containing Mlp1p purification, B = basket-less purification,
            E = IgG control for A, F = GBP control for B (C, D, G not used here)
  --deseq   DESeq2 result table for A vs B, no header:
            gene, baseMean, log2FoldChange (A/B), lfcSE, stat, pvalue, padj

Survey input: data/rna_gene_clusters.tsv (per-bait max-normalized enrichment + cluster C1-C8)

Measures
  1. Mlp1p association = mean(A) - mean(E) per gene (background-subtracted basket signal)
  2. Basket preference  = DESeq2 log2 fold change, basket-containing / basket-less
Tests
  - Spearman correlation of each bait's enrichment with 1 and 2 (shared genes)
  - per cluster: log2 fold change of the cluster vs. all other clusters, two-sided Mann-Whitney U

Writes
  data/mlp1_bait_correlations.tsv
  data/mlp1_cluster_basket_preference.tsv

Usage
  python mlp1_basket_comparison.py --counts <normalized_counts.tabular> --deseq <DESeq2_AP_A_vs_AP_B.tabular>
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def load_counts(path):
    c = pd.read_csv(path, sep="\t", index_col=0)
    names = []
    for col in c.columns:
        m = re.match(r"AP(\w)_replicat(\d)", col)
        if m is None:
            raise ValueError(f"unexpected column name: {col}")
        names.append(m.group(1) + m.group(2))
    c.columns = names
    return c


def condition_mean(c, k):
    cols = [x for x in c.columns if x[0] == k]
    return c[cols].mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--deseq", required=True)
    args = ap.parse_args()

    counts = load_counts(args.counts)
    basket_signal = condition_mean(counts, "A") - condition_mean(counts, "E")

    de = pd.read_csv(args.deseq, sep="\t", header=None, index_col=0,
                     names=["gene", "baseMean", "log2FC", "lfcSE", "stat", "pvalue", "padj"])
    log2fc = de["log2FC"].dropna()

    survey = pd.read_csv(DATA / "rna_gene_clusters.tsv", sep="\t", index_col=0)
    baits = [b for b in survey.columns if b != "cluster"]

    shared = survey.index.intersection(basket_signal.index)
    shared_fc = survey.index.intersection(log2fc.index)
    print(f"genes shared with survey: {len(shared)} (with a fold change: {len(shared_fc)})")

    rows = []
    for b in baits:
        r1 = stats.spearmanr(basket_signal.loc[shared], survey.loc[shared, b])
        r2 = stats.spearmanr(log2fc.loc[shared_fc], survey.loc[shared_fc, b])
        rows.append({"bait": b,
                     "rho_basket_signal": r1.statistic, "p_basket_signal": r1.pvalue,
                     "rho_basket_vs_basketless_log2FC": r2.statistic, "p_log2FC": r2.pvalue})
    bait_tab = pd.DataFrame(rows).sort_values("rho_basket_vs_basketless_log2FC", ascending=False)

    rows = []
    for k in sorted(survey["cluster"].unique()):
        g = survey.index[survey["cluster"] == k].intersection(shared_fc)
        rest = shared_fc.difference(g)
        mw = stats.mannwhitneyu(log2fc.loc[g], log2fc.loc[rest], alternative="two-sided")
        rows.append({"cluster": f"C{k}", "n": len(g),
                     "median_log2FC": log2fc.loc[g].median(),
                     "mannwhitney_p_vs_other_clusters": mw.pvalue})
    clus_tab = pd.DataFrame(rows)

    bait_tab.to_csv(DATA / "mlp1_bait_correlations.tsv", sep="\t", index=False, float_format="%.4g")
    clus_tab.to_csv(DATA / "mlp1_cluster_basket_preference.tsv", sep="\t", index=False, float_format="%.4g")
    with pd.option_context("display.width", 160):
        print(bait_tab.round(3).to_string(index=False))
        print(clus_tab.to_string(index=False))


if __name__ == "__main__":
    main()

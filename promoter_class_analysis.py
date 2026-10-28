#!/usr/bin/env python3
"""
promoter_class_analysis.py
==========================
Promoter class correlation with nuclear mRNA maturation pathway kinetics.

Cross-references yeast promoter architecture data (Rhee & Pugh 2012, Nature 483:295-301)
with AP-RNAseq bait enrichment profiles and pathway dwell scores to test whether
TATA-box / TFIID-SAGA promoter class predicts nuclear mRNA processing kinetics.

Usage
-----
python promoter_class_analysis.py \
    --rnaseq   rnaseq_bait_means.tsv \
    --dwell    rna_dwell_scores.tsv \
    --clusters rna_gene_clusters.tsv \
    --rhee     Rhee_SuppData1_revised.xls \
    --gff      data/saccharomyces_cerevisiae.20260618.gff \
    --outdir   results/

Outputs
-------
- promoter_annotated.tsv          : per-gene promoter class + dwell + cluster
- promoter_class_analysis.svg     : 4-panel figure
- promoter_gene_classes.tsv       : gene length quartile and gene class labels

References
----------
Rhee HS, Pugh BF (2012) Comprehensive genome-wide protein-DNA interactions
  detected at single-nucleotide resolution. Nature 483:295-301.
Basehoar AD, Zanton SJ, Pugh BF (2004) Identification and distinct regulation
  of yeast TATA box-containing genes. Cell 116:699-709.

Author  : mRNP Maturation Project (Oeffinger Lab)
Date    : June 2026
License : MIT
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec


# ── Constants ──────────────────────────────────────────────────────────────────

BAITS = [
    "CBP80", "NAB2", "HRP1", "NPL3", "GBP2", "PAB1", "SNU71", "SUB2", "THP2",
    "CLP1", "SYC1", "REF2", "SAC3", "GFD1", "YRA1", "MEX67", "DBP5", "RRP6",
]

STAGES = [
    "CBC", "hnRNP", "hnRNP", "hnRNP", "hnRNP", "hnRNP", "Splicing",
    "THO", "THO", "3'proc", "3'proc", "3'proc", "TREX-2", "TREX-2",
    "Export", "Export", "Export", "Surv.",
]

STAGE_COLORS = {
    "CBC": "#2D6BB5", "hnRNP": "#5AAE45", "Splicing": "#E07B2A",
    "THO": "#8ABEDE", "3'proc": "#CC2222", "TREX-2": "#7030A0",
    "Export": "#C04010", "Surv.": "#606060",
}

FOUR_CLASSES = [
    "TATA-less / TAF1-enriched",
    "TATA-less / TAF1-depleted",
    "TATA-containing / TAF1-depleted",
    "TATA-containing / TAF1-enriched",
]
FOUR_LABELS = [
    "TATA-less\nTAF1-enr.\n(TFIID)",
    "TATA-less\nTAF1-dep.\n(intermediate)",
    "TATA-cont.\nTAF1-dep.\n(SAGA)",
    "TATA-cont.\nTAF1-enr.\n(unusual)",
]
FOUR_COLORS = ["#4472C4", "#70AD47", "#D62728", "#FF7F0E"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_rnaseq(path: str) -> pd.DataFrame:
    """Load pre-computed bait-mean AP-RNAseq enrichment matrix."""
    return pd.read_csv(path, sep="\t", index_col=0)


def load_dwell(path: str) -> pd.DataFrame:
    """Load per-gene pathway dwell scores."""
    return pd.read_csv(path, sep="\t", index_col=0)


def load_clusters(path: str) -> pd.DataFrame:
    """Load gene cluster assignments (k=8 Ward linkage)."""
    return pd.read_csv(path, sep="\t", index_col=0)


def load_rhee(path: str) -> pd.DataFrame:
    """Load Rhee & Pugh 2012 Supplementary Data 1."""
    df = pd.read_excel(path)
    df = df.set_index("gene_id")
    return df[["gene_class_1", "gene_class_2", "TBP", "Pol II", "mismatch"]]


def load_rp_gene_ids(gff_path: str) -> set:
    """
    Ribosomal-protein genes from the SGD GFF (same rule as make_figure2A_introns.py):
    nuclear ORFs (Y[A-P][LR]nnn[CW], incl. -A/-B suffixed ORFs) whose standard gene name starts with RPL or RPS.
    (Previously taken from the lab ABACUS export; switched 2026-10-08.)
    """
    rp = set()
    with open(gff_path) as fh:
        for line in fh:
            if line.startswith('##FASTA'):
                break
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            id_m = re.search(r'ID=([^;]+)', parts[8])
            gene_m = re.search(r'gene=([^;]+)', parts[8])
            if id_m and gene_m and re.match(r'^Y[A-P][LR]\d{3}[CW](-[A-Z])?$', id_m.group(1)) \
                    and re.match(r'^RP[LS]\d', gene_m.group(1), re.IGNORECASE):
                rp.add(id_m.group(1))
    return rp


# ── Annotation ─────────────────────────────────────────────────────────────────

def build_annotation(rna: pd.DataFrame, dwell: pd.DataFrame,
                     clusters: pd.DataFrame, rhee: pd.DataFrame,
                     rp_ids: set) -> pd.DataFrame:
    """
    Build unified per-gene annotation table combining:
    - Transcript length quartile (Q1–Q4)
    - Rhee & Pugh promoter class (TATA/TAF1 status)
    - Gene category label (Ribosomal protein / Other)
    - Pathway dwell score and cluster assignment
    """
    ann = dwell[["length", "breadth", "dwell_auc", "peak_stage", "class"]].copy()

    # Length quartile
    ann["length_q"] = pd.qcut(
        ann["length"], q=4,
        labels=["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"],
    )

    # Cluster
    ann = ann.join(clusters[["cluster"]], how="left")

    # Gene category
    ann["gene_class"] = "Other"
    ann.loc[ann.index.isin(rp_ids), "gene_class"] = "Ribosomal protein"

    # Rhee & Pugh promoter class
    ann = ann.join(rhee[["gene_class_1", "gene_class_2", "TBP", "mismatch"]], how="left")
    ann["tata_taf"] = (
        ann["gene_class_1"].fillna("") + " / " + ann["gene_class_2"].fillna("")
    )

    return ann


# ── Statistics ─────────────────────────────────────────────────────────────────

def run_statistics(ann: pd.DataFrame, rna: pd.DataFrame) -> dict:
    """
    Compute all pairwise Mann-Whitney U tests for dwell score across the
    four promoter classes. Bonferroni-corrected p-values for 6 comparisons.
    Returns dict of (class_i, class_j) -> (U, p_raw, p_corrected).
    """
    dwell_by_class = {
        fc: ann[ann["tata_taf"] == fc]["dwell_auc"].dropna().values
        for fc in FOUR_CLASSES
    }
    n_comparisons = 6
    results = {}
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for i, j in pairs:
        u, p = stats.mannwhitneyu(
            dwell_by_class[FOUR_CLASSES[i]],
            dwell_by_class[FOUR_CLASSES[j]],
            alternative="two-sided",
        )
        results[(FOUR_CLASSES[i], FOUR_CLASSES[j])] = (u, p, min(p * n_comparisons, 1.0))

    # Kruskal-Wallis across all four
    kw = stats.kruskal(*dwell_by_class.values())
    results["kruskal_wallis"] = kw

    # Length quartile KW
    q_data = [
        ann[ann["length_q"] == q]["dwell_auc"].dropna().values
        for q in ["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"]
    ]
    results["kruskal_length_q"] = stats.kruskal(*q_data)

    return results


def print_summary(ann: pd.DataFrame, rna: pd.DataFrame, stat_results: dict):
    """Print key numerical results to stdout."""
    print("=" * 60)
    print("PROMOTER CLASS ANALYSIS — SUMMARY")
    print("=" * 60)

    print("\nGene counts per promoter class:")
    for fc in FOUR_CLASSES:
        n = (ann["tata_taf"] == fc).sum()
        print(f"  {fc}: n={n}")

    print("\nDwell score per class (mean ± std):")
    for fc in FOUR_CLASSES:
        d = ann[ann["tata_taf"] == fc]["dwell_auc"].dropna()
        print(f"  {fc}: {d.mean():.3f} ± {d.std():.3f}  (n={len(d)})")

    print("\nPairwise Mann-Whitney (Bonferroni-corrected, 6 tests):")
    for (ci, cj), (u, p, pc) in stat_results.items():
        if isinstance(ci, str):
            sig = "***" if pc < 0.001 else "**" if pc < 0.01 else "*" if pc < 0.05 else "n.s."
            print(f"  {ci[:20]} vs {cj[:20]}: p_corr={pc:.2e} {sig}")

    kw = stat_results.get("kruskal_wallis")
    if kw:
        print(f"\nKruskal-Wallis across 4 classes: H={kw.statistic:.1f}, p={kw.pvalue:.2e}")


# ── Figures ────────────────────────────────────────────────────────────────────

def make_figure(ann: pd.DataFrame, rna: pd.DataFrame, stat_results: dict,
                outpath: str):
    """Generate 4-panel figure: heatmap / bar / violin / scatter."""

    def get_profile(class_label):
        genes = [g for g in ann[ann["tata_taf"] == class_label].index if g in rna.index]
        return rna.loc[genes, BAITS].mean(), len(genes)

    profiles = {fc: get_profile(fc) for fc in FOUR_CLASSES}
    tl_genes = [g for g in ann[ann["gene_class_1"] == "TATA-less"].index if g in rna.index]
    tc_genes = [g for g in ann[ann["gene_class_1"] == "TATA-containing"].index if g in rna.index]
    tl_profile = rna.loc[tl_genes, BAITS].mean()
    tc_profile = rna.loc[tc_genes, BAITS].mean()

    fig = plt.figure(figsize=(17, 14), facecolor="white")
    gs_fig = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38,
                               left=0.07, right=0.97, top=0.95, bottom=0.07)

    # ── Panel A: Heatmap ────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs_fig[0, 0])
    mat = np.array([profiles[fc][0].values for fc in FOUR_CLASSES])
    im = ax_a.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=0.10, vmax=0.50,
                     interpolation="nearest", rasterized=True)
    ax_a.set_xticks(range(len(BAITS)))
    ax_a.set_xticklabels(BAITS, rotation=45, ha="right", fontsize=8.5)
    ax_a.set_yticks(range(4))
    ax_a.set_yticklabels(FOUR_LABELS, fontsize=8.5)
    for i, color in enumerate(FOUR_COLORS):
        ax_a.add_patch(mpatches.FancyBboxPatch(
            (-0.7, i - 0.45), 0.25, 0.9, boxstyle="square,pad=0",
            facecolor=color, edgecolor="none", clip_on=False, zorder=5))
    plt.colorbar(im, ax=ax_a, shrink=0.6, pad=0.02, label="Mean enrichment")
    for i, stage in enumerate(STAGES):
        ax_a.add_patch(mpatches.FancyBboxPatch(
            (i - 0.48, 3.52), 0.96, 0.20, boxstyle="square,pad=0",
            facecolor=STAGE_COLORS[stage], edgecolor="none", clip_on=False, zorder=5))
    prev, s0 = None, 0
    for i, stage in enumerate(STAGES + ["END"]):
        if stage != prev and prev is not None:
            ax_a.text((s0 + i - 1) / 2, 3.82, prev, ha="center", va="bottom",
                      fontsize=7, color=STAGE_COLORS[prev], fontweight="bold")
            s0 = i
        prev = stage
    ax_a.set_xlim(-0.5, len(BAITS) - 0.5)
    ax_a.set_title("A.  Promoter class × pathway enrichment (Rhee & Pugh 2012)",
                   fontsize=10, fontweight="bold", loc="left", pad=22)

    # ── Panel B: TATA-less vs TATA-containing bars ──────────────────────────
    ax_b = fig.add_subplot(gs_fig[0, 1])
    x = np.arange(len(BAITS))
    w = 0.38
    ax_b.bar(x - w / 2, tl_profile.values, w,
             label=f"TATA-less (n={len(tl_genes)})",
             color="#4472C4", alpha=0.85, edgecolor="none")
    ax_b.bar(x + w / 2, tc_profile.values, w,
             label=f"TATA-containing (n={len(tc_genes)})",
             color="#D62728", alpha=0.85, edgecolor="none")
    for i, stage in enumerate(STAGES):
        ax_b.axvspan(i - 0.5, i + 0.5, ymin=0, ymax=0.025,
                     facecolor=STAGE_COLORS[stage], alpha=0.85, zorder=0)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(BAITS, rotation=45, ha="right", fontsize=8.5)
    ax_b.set_ylabel("Mean enrichment")
    ax_b.set_title("B.  TATA-less vs TATA-containing enrichment profiles",
                   fontsize=10, fontweight="bold", loc="left")
    ax_b.legend(fontsize=8.5, loc="upper right")
    ax_b.set_ylim(0, 0.60)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    for i, bait in enumerate(BAITS):
        diff = tc_profile[bait] - tl_profile[bait]
        if abs(diff) > 0.055:
            sign = "+" if diff > 0 else ""
            col = "#D62728" if diff > 0 else "#4472C4"
            ax_b.text(i, max(tl_profile[bait], tc_profile[bait]) + 0.018,
                      f"{sign}{diff:.2f}", ha="center", va="bottom",
                      fontsize=7, color=col, fontweight="bold")

    # ── Panel C: Violin plot of dwell scores ────────────────────────────────
    ax_c = fig.add_subplot(gs_fig[1, 0])
    dwell_data = [ann[ann["tata_taf"] == fc]["dwell_auc"].dropna().values
                  for fc in FOUR_CLASSES]
    parts = ax_c.violinplot(dwell_data, positions=range(4),
                            showmeans=False, showmedians=False, showextrema=False)
    for pc, col in zip(parts["bodies"], FOUR_COLORS):
        pc.set_facecolor(col)
        pc.set_alpha(0.65)
        pc.set_edgecolor("none")
    for i, d in enumerate(dwell_data):
        q25, q50, q75 = np.percentile(d, [25, 50, 75])
        ax_c.vlines(i, q25, q75, lw=5, color="white", zorder=3)
        ax_c.vlines(i, q25, q75, lw=3, color="black", zorder=4, alpha=0.7)
        ax_c.scatter(i, q50, color="white", s=28, zorder=5)
        ax_c.scatter(i, np.mean(d), color=FOUR_COLORS[i], s=45, zorder=6,
                     edgecolors="black", linewidths=0.8, marker="D")
        ax_c.text(i, min(max(d), 6.55) * 1.005,
                  f"n={len(d)}\nμ={np.mean(d):.2f}",
                  ha="center", va="bottom", fontsize=7.5, color=FOUR_COLORS[i])
    # Significance bracket (TFIID vs SAGA — most contrasting pair)
    key = (FOUR_CLASSES[0], FOUR_CLASSES[2])
    _, p_raw, p_corr = stat_results[key]
    sig_label = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*"
    ax_c.plot([0, 0, 2, 2], [6.10, 6.28, 6.28, 6.10], lw=1.2, color="black")
    ax_c.text(1, 6.30, sig_label, ha="center", va="bottom", fontsize=10)
    ax_c.text(1, 6.45, f"p={p_corr:.1e}", ha="center", fontsize=7.5, color="gray")
    ax_c.set_xticks(range(4))
    ax_c.set_xticklabels(FOUR_LABELS, fontsize=8.5)
    ax_c.set_ylabel("Pathway dwell score (AUC, 0–8 stages)")
    ax_c.set_title("C.  Pathway dwell time by promoter class (Rhee & Pugh 2012)",
                   fontsize=10, fontweight="bold", loc="left")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.set_ylim(1.0, 7.0)
    overall_mean = ann["dwell_auc"].mean()
    ax_c.axhline(overall_mean, color="gray", lw=1, ls="--", alpha=0.6, zorder=0)
    ax_c.text(3.45, overall_mean + 0.07, "overall\nmean", ha="right",
              fontsize=7, color="gray")

    # ── Panel D: CBC vs CLP1 scatter ────────────────────────────────────────
    ax_d = fig.add_subplot(gs_fig[1, 1])
    for fc, col, lbl in zip(FOUR_CLASSES, FOUR_COLORS, FOUR_LABELS):
        genes = [g for g in ann[ann["tata_taf"] == fc].index if g in rna.index]
        if len(genes) > 5:
            ax_d.scatter(rna.loc[genes, "CBP80"].values,
                         rna.loc[genes, "CLP1"].values,
                         c=col, alpha=0.22, s=12,
                         label=f"{lbl.replace(chr(10), ' ')} n={len(genes)}", zorder=2)
    for fc, col in zip(FOUR_CLASSES, FOUR_COLORS):
        genes = [g for g in ann[ann["tata_taf"] == fc].index if g in rna.index]
        if len(genes) > 5:
            ax_d.scatter(rna.loc[genes, "CBP80"].mean(), rna.loc[genes, "CLP1"].mean(),
                         c=col, s=120, edgecolors="black", linewidths=1.5, zorder=5, marker="D")
    ax_d.axvline(rna["CBP80"].median(), color="gray", lw=0.8, ls="--", alpha=0.5)
    ax_d.axhline(rna["CLP1"].median(), color="gray", lw=0.8, ls="--", alpha=0.5)
    ax_d.text(0.72, 0.72, "late-pathway\n(SAGA-like)", ha="center", va="center",
              fontsize=8, color="#D62728", alpha=0.7, transform=ax_d.transAxes)
    ax_d.text(0.28, 0.28, "early-pathway\n(TFIID-like)", ha="center", va="center",
              fontsize=8, color="#4472C4", alpha=0.7, transform=ax_d.transAxes)
    ax_d.set_xlabel("CBP80 enrichment (early pathway)")
    ax_d.set_ylabel("CLP1 enrichment (3' processing)")
    ax_d.set_title("D.  CBC vs 3' processing enrichment by promoter class",
                   fontsize=10, fontweight="bold", loc="left")
    ax_d.legend(fontsize=7, loc="upper right", markerscale=1.5, framealpha=0.85)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Figure saved: {outpath}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rnaseq",   required=True, help="rnaseq_bait_means.tsv")
    p.add_argument("--dwell",    required=True, help="rna_dwell_scores.tsv")
    p.add_argument("--clusters", required=True, help="rna_gene_clusters.tsv")
    p.add_argument("--rhee",     required=True, help="Rhee_SuppData1_revised.xls")
    p.add_argument("--gff",      default="data/saccharomyces_cerevisiae.20260618.gff",
                   help="SGD GFF (RP gene list)")
    p.add_argument("--outdir",   default=".",   help="Output directory (default: .)")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    rna      = load_rnaseq(args.rnaseq)
    dwell    = load_dwell(args.dwell)
    clusters = load_clusters(args.clusters)
    rhee     = load_rhee(args.rhee)
    rp_ids   = load_rp_gene_ids(args.gff)

    print("Building annotation table...")
    ann = build_annotation(rna, dwell, clusters, rhee, rp_ids)
    ann.to_csv(outdir / "promoter_annotated.tsv", sep="\t")
    print(f"  Annotated {len(ann)} genes; "
          f"{ann['gene_class_1'].notna().sum()} with Rhee & Pugh promoter class")

    print("Running statistics...")
    stat_results = run_statistics(ann, rna)
    print_summary(ann, rna, stat_results)

    print("Generating figure...")
    make_figure(ann, rna, stat_results, str(outdir / "promoter_class_analysis.svg"))

    print("\nDone. Output files written to:", outdir)


if __name__ == "__main__":
    main()

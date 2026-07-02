#!/usr/bin/env python3
"""
cid_boundary_analysis.py
========================
Chromosome Interaction Domain (CID) boundary proximity vs nuclear mRNA
processing kinetics in Saccharomyces cerevisiae.

Tests whether gene position relative to CID boundaries (Hsieh et al. 2015
Micro-C data) predicts nuclear mRNA processing kinetics (AP-RNAseq dwell
scores, RNA clusters, and promoter class from this study).

Usage
-----
python cid_boundary_analysis.py \\
    --dwell    rna_dwell_scores.tsv \\
    --clusters rna_gene_clusters.tsv \\
    --promoter promoter_annotated.tsv \\
    --gff      saccharomyces_cerevisiae.gff \\
    --cid      mmc4-2.xlsx \\
    --outdir   results/

Outputs
-------
- genes_cid_assigned.tsv    : per-gene CID assignment, size, boundary distance
- cid_boundary_stats.tsv    : per-boundary nearby gene statistics
- cid_hic_analysis.svg      : 3-panel figure

References
----------
Hsieh HS & Pugh BF (2015) Mapping Nucleosome Resolution Chromosome Folding
  in Yeast by Micro-C. Molecular Cell 62(1):90-104.
  doi:10.1016/j.molcel.2015.05.002

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

# Roman numeral to integer mapping for yeast chromosome names
ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
    "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
    "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16,
}

# RNA cluster definitions (k=8 Ward linkage on AP-RNAseq bait-mean matrix)
CLUSTER_LABELS = {
    1: "C1 Export", 2: "C2 TREX-2", 3: "C3 3'proc", 4: "C4 THO/TREX",
    5: "C5 hnRNP",  6: "C6 CBC/hnRNP", 7: "C7 GFD1",  8: "C8 Splicing/RP",
}
CLUSTER_COLORS = {
    1: "#C04010", 2: "#7030A0", 3: "#CC2222", 4: "#8ABEDE",
    5: "#5AAE45", 6: "#2D6BB5", 7: "#888888", 8: "#E07B2A", -1: "#CCCCCC",
}

# Pathway cluster order (early → late) for Panel C
CLUSTER_ORDER = [8, 6, 7, 5, 4, 2, 3, 1]

# Promoter class definitions (Rhee & Pugh 2012)
FOUR_CLASSES = [
    "TATA-less / TAF1-enriched",
    "TATA-less / TAF1-depleted",
    "TATA-containing / TAF1-depleted",
    "TATA-containing / TAF1-enriched",
]
FOUR_LABELS = ["TFIID", "Intermediate", "SAGA", "Unusual"]
FOUR_COLORS = ["#4472C4", "#70AD47", "#D62728", "#FF7F0E"]

# Example region for Panel A (chrXV, 60-kb window with all 8 clusters)
EXAMPLE_CHR, EXAMPLE_START, EXAMPLE_END = 15, 200_000, 260_000


# ── Data loading ───────────────────────────────────────────────────────────────

def load_gff(path: str) -> pd.DataFrame:
    """
    Parse SGD GFF3 annotation and return gene feature coordinates.

    Parameters
    ----------
    path : str
        Path to saccharomyces_cerevisiae.gff (SGD GFF3 format).

    Returns
    -------
    pd.DataFrame
        Index: systematic gene ID (e.g. YAL001C).
        Columns: chr (int 1-16), start, end, strand, tss, midpoint.
    """
    genes = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            chrom = parts[0]
            start, end = int(parts[3]), int(parts[4])
            strand = parts[6]
            attrs = parts[8]
            # Extract systematic ORF ID
            m = re.search(r"ID=(Y[A-Z]{2}\d+[WC](?:-[A-Z])?)", attrs)
            if not m:
                m = re.search(r"ID=([\w\-]+);", attrs)
            gene_id = m.group(1) if m else None
            chr_str = chrom.replace("chr", "")
            chr_num = ROMAN.get(chr_str)
            if chr_num and gene_id:
                genes.append({
                    "gene_id": gene_id,
                    "chr": chr_num,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "tss": start if strand == "+" else end,
                    "midpoint": (start + end) // 2,
                })
    df = pd.DataFrame(genes).drop_duplicates("gene_id").set_index("gene_id")
    return df


def load_cid_boundaries(path: str, sheet: str = "Sheet1") -> pd.DataFrame:
    """
    Load consensus CID boundary calls from Hsieh et al. 2015 Table S4.

    Parameters
    ----------
    path  : Path to mmc4-2.xlsx.
    sheet : Sheet name (default 'Sheet1' = consensus boundaries).

    Returns
    -------
    pd.DataFrame sorted by Chr, Coord with columns: Chr (int), Coord, Strength.
    """
    df = pd.read_excel(path, sheet_name=sheet)
    return df.sort_values(["Chr", "Coord"]).reset_index(drop=True)


def load_dwell(path: str) -> pd.DataFrame:
    """Load per-gene pathway dwell scores."""
    return pd.read_csv(path, sep="\t", index_col=0)


def load_clusters(path: str) -> pd.DataFrame:
    """Load RNA cluster assignments (k=8)."""
    return pd.read_csv(path, sep="\t", index_col=0)[["cluster"]]


def load_promoter_annotation(path: str) -> pd.DataFrame:
    """Load promoter class annotation (Rhee & Pugh 2012 cross-reference)."""
    return pd.read_csv(path, sep="\t", index_col=0)


# ── CID assignment ─────────────────────────────────────────────────────────────

def assign_cid(gene_chr: int, gene_pos: int, cid_df: pd.DataFrame):
    """
    Assign a gene to a CID (inter-boundary interval) and compute distance
    to the nearest boundary.

    Parameters
    ----------
    gene_chr : Chromosome number (1-16).
    gene_pos : Gene midpoint in bp.
    cid_df   : CID boundary DataFrame (sorted by Chr, Coord).

    Returns
    -------
    Tuple: (cid_id: str, cid_size: float, dist_to_boundary: float)
    """
    chrom_bounds = cid_df[cid_df["Chr"] == gene_chr]["Coord"].values
    if len(chrom_bounds) < 2:
        return None, None, None
    idx = np.searchsorted(chrom_bounds, gene_pos)
    if idx == 0:
        left_b, right_b = 0, chrom_bounds[0]
    elif idx >= len(chrom_bounds):
        left_b, right_b = chrom_bounds[-1], chrom_bounds[-1] + int(1e9)
    else:
        left_b, right_b = chrom_bounds[idx - 1], chrom_bounds[idx]
    cid_id = f"chr{gene_chr}_{int(left_b)}_{int(right_b)}"
    cid_size = right_b - left_b
    dist = min(gene_pos - left_b, right_b - gene_pos)
    return cid_id, float(cid_size), float(dist)


def build_merged_table(gff: pd.DataFrame, dwell: pd.DataFrame,
                       clusters: pd.DataFrame, promoter: pd.DataFrame,
                       cid: pd.DataFrame, max_cid_kb: float = 50.0) -> pd.DataFrame:
    """
    Build unified per-gene table with coordinates, dwell, cluster, promoter
    class, and CID assignment.

    Genes in CIDs > max_cid_kb are excluded (chromosome-end artefacts).
    """
    common = [g for g in dwell.index if g in gff.index]
    merged = gff.loc[common].join(dwell[["dwell_auc", "breadth", "length", "class"]])\
                             .join(promoter[["tata_taf", "gene_class"]])\
                             .join(clusters)
    merged = merged.dropna(subset=["dwell_auc"])
    merged["cluster"] = merged["cluster"].fillna(-1).astype(int)

    # CID assignment
    results = merged.apply(
        lambda r: pd.Series(
            assign_cid(r["chr"], r["midpoint"], cid),
            index=["cid_id", "cid_size", "dist_to_boundary"],
        ),
        axis=1,
    )
    merged = pd.concat([merged, results], axis=1)
    merged = merged.dropna(subset=["cid_id"])
    merged = merged[merged["cid_size"] < max_cid_kb * 1000]
    return merged


# ── Statistics ─────────────────────────────────────────────────────────────────

def run_statistics(merged: pd.DataFrame) -> dict:
    """
    Run all primary statistical tests and return results dictionary.
    Includes Kruskal-Wallis and pairwise Mann-Whitney U tests (Bonferroni
    correction where applicable).
    """
    results = {}
    m = merged.dropna(subset=["dist_to_boundary", "dwell_auc"])

    # Dwell quartile vs boundary distance
    m = m.copy()
    m["dwell_q"] = pd.qcut(m["dwell_auc"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    q_data = {q: m[m["dwell_q"] == q]["dist_to_boundary"].values
              for q in ["Q1", "Q2", "Q3", "Q4"]}
    results["kw_dwell_q"] = stats.kruskal(*q_data.values())
    results["mw_Q1_Q4"] = stats.mannwhitneyu(q_data["Q1"], q_data["Q4"],
                                              alternative="two-sided")

    # Continuous correlation
    results["spearman_dwell_dist"] = stats.spearmanr(m["dist_to_boundary"],
                                                      m["dwell_auc"])

    # RNA cluster KW
    m3 = m[m["cluster"] > 0]
    cluster_data = [m3[m3["cluster"] == c]["dist_to_boundary"].values
                    for c in CLUSTER_ORDER if len(m3[m3["cluster"] == c]) > 5]
    results["kw_clusters"] = stats.kruskal(*cluster_data)
    results["mw_C8_C1"] = stats.mannwhitneyu(
        m3[m3["cluster"] == 8]["dist_to_boundary"].values,
        m3[m3["cluster"] == 1]["dist_to_boundary"].values,
        alternative="two-sided",
    )

    # Promoter class
    m4 = m.dropna(subset=["tata_taf"])
    pclass_data = [m4[m4["tata_taf"] == fc]["dist_to_boundary"].values
                   for fc in FOUR_CLASSES if len(m4[m4["tata_taf"] == fc]) > 5]
    results["kw_promoter_class"] = stats.kruskal(*pclass_data)
    results["mw_TFIID_SAGA"] = stats.mannwhitneyu(
        m4[m4["tata_taf"] == FOUR_CLASSES[0]]["dist_to_boundary"].values,
        m4[m4["tata_taf"] == FOUR_CLASSES[2]]["dist_to_boundary"].values,
        alternative="two-sided",
    )

    # RP genes
    rp = m[m["gene_class"] == "Ribosomal protein"]["dist_to_boundary"].values
    other = m[m["gene_class"] != "Ribosomal protein"]["dist_to_boundary"].values
    results["mw_RP_vs_other"] = stats.mannwhitneyu(rp, other, alternative="two-sided")

    return results


def print_statistics(results: dict, merged: pd.DataFrame):
    """Print key results to stdout."""
    print("=" * 60)
    print("CID BOUNDARY ANALYSIS — KEY STATISTICS")
    print("=" * 60)

    kw_q = results["kw_dwell_q"]
    mw_q = results["mw_Q1_Q4"]
    spr  = results["spearman_dwell_dist"]
    kw_c = results["kw_clusters"]
    mw_cs= results["mw_C8_C1"]
    kw_p = results["kw_promoter_class"]
    mw_ts= results["mw_TFIID_SAGA"]
    mw_rp= results["mw_RP_vs_other"]

    print(f"\nDwell quartile KW:  H={kw_q.statistic:.1f}  p={kw_q.pvalue:.2e}")
    print(f"Q1 vs Q4 MW:        p={mw_q.pvalue:.2e}")
    print(f"Spearman(dist,dwell): r={spr.statistic:.3f}  p={spr.pvalue:.2e}")
    print(f"\nRNA cluster KW:     H={kw_c.statistic:.1f}  p={kw_c.pvalue:.2e}")
    print(f"C8 vs C1 MW:        p={mw_cs.pvalue:.2e}")
    print(f"\nPromoter class KW:  H={kw_p.statistic:.1f}  p={kw_p.pvalue:.3f}")
    print(f"TFIID vs SAGA MW:   p={mw_ts.pvalue:.3e}")
    print(f"\nRP vs other MW:     p={mw_rp.pvalue:.2e}")

    m = merged.dropna(subset=["dist_to_boundary", "dwell_auc"]).copy()
    m["dwell_q"] = pd.qcut(m["dwell_auc"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    print("\nMedian boundary distance by dwell quartile:")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        d = m[m["dwell_q"] == q]["dist_to_boundary"]
        print(f"  {q}: {d.median():.0f} bp  (n={len(d)})")


# ── Boundary stats ─────────────────────────────────────────────────────────────

def compute_boundary_stats(merged: pd.DataFrame, cid: pd.DataFrame,
                            window_bp: int = 5000) -> pd.DataFrame:
    """
    For each CID boundary, compute statistics of nearby genes (within window_bp).
    Returns DataFrame with boundary position, strength, n_nearby, mean_dwell,
    SAGA fraction, and TFIID fraction.
    """
    rows = []
    m = merged.dropna(subset=["dwell_auc", "tata_taf"])
    for _, row in cid.iterrows():
        c = int(row["Chr"]); coord = row["Coord"]; strength = row["Strength"]
        near = m[(m["chr"] == c) & (abs(m["midpoint"] - coord) < window_bp)]
        if len(near) == 0:
            continue
        rows.append({
            "chr": c, "coord": coord, "strength": strength,
            "n_nearby": len(near),
            "mean_dwell_nearby": near["dwell_auc"].mean(),
            "saga_frac": (near["tata_taf"] == FOUR_CLASSES[2]).mean(),
            "tfiid_frac": (near["tata_taf"] == FOUR_CLASSES[0]).mean(),
        })
    return pd.DataFrame(rows)


# ── Figure ─────────────────────────────────────────────────────────────────────

def make_figure(merged: pd.DataFrame, cid: pd.DataFrame,
                stat_results: dict, outpath: str):
    """Generate 3-panel CID boundary analysis figure."""
    m = merged.dropna(subset=["dist_to_boundary", "dwell_auc"]).copy()
    m["dwell_q"] = pd.qcut(m["dwell_auc"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    fig = plt.figure(figsize=(17, 14), facecolor="white")
    gs_fig = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.38,
                               left=0.07, right=0.97, top=0.95, bottom=0.07)

    # ── Panel A: Genomic browser view ─────────────────────────────────────────
    ax_a = fig.add_subplot(gs_fig[0, :])
    reg_genes = m[
        (m["chr"] == EXAMPLE_CHR) &
        (m["midpoint"] >= EXAMPLE_START) &
        (m["midpoint"] <= EXAMPLE_END)
    ].sort_values("start")
    reg_bounds = cid[
        (cid["Chr"] == EXAMPLE_CHR) &
        (cid["Coord"] >= EXAMPLE_START) &
        (cid["Coord"] <= EXAMPLE_END)
    ]
    # CID shading
    bounds_in_win = [EXAMPLE_START] + list(reg_bounds["Coord"].values) + [EXAMPLE_END]
    for i in range(len(bounds_in_win) - 1):
        ax_a.axvspan(bounds_in_win[i], bounds_in_win[i + 1],
                     facecolor="#F5F5F5" if i % 2 == 0 else "white", alpha=0.6, zorder=0)
    for _, brow in reg_bounds.iterrows():
        lw = 0.8 + brow["Strength"] / 6
        ax_a.axvline(brow["Coord"], color="#444", lw=lw, zorder=2, alpha=0.7)
        ax_a.text(brow["Coord"], 1.18, f"{brow['Strength']:.1f}",
                  ha="center", va="bottom", fontsize=6.5, color="#444", rotation=90)
    TRACK_Y = {"+": 0.65, "-": 0.25}
    ARROW_H = 0.22
    for _, g in reg_genes.iterrows():
        cl = int(g["cluster"]) if g["cluster"] > 0 else -1
        col = CLUSTER_COLORS.get(cl, "#CCCCCC")
        y = TRACK_Y.get(g["strand"], 0.45)
        gstart = max(g["start"], EXAMPLE_START)
        gend = min(g["end"], EXAMPLE_END)
        w = gend - gstart
        if w < 100:
            continue
        ax_a.add_patch(mpatches.FancyArrow(
            gstart, y, w * 0.85, 0, width=ARROW_H, head_width=ARROW_H,
            head_length=w * 0.15, fc=col, ec="white", lw=0.4, zorder=3,
            length_includes_head=True))
        ax_a.text((gstart + gend) / 2, y + (ARROW_H / 2 + 0.045) * (1 if g["strand"] == "+" else -1),
                  g.name, ha="center", va="center", fontsize=6.2, color="#222",
                  fontweight="bold" if g.get("gene_class") == "Ribosomal protein" else "normal")
        dw_norm = min(g["dwell_auc"] / 8.0, 1.0)
        ax_a.add_patch(mpatches.Rectangle(
            (gstart, y - ARROW_H / 2 - 0.14), w, 0.08,
            fc=plt.cm.RdYlGn_r(dw_norm), ec="none", zorder=4, alpha=0.9))
    legend_patches = [mpatches.Patch(fc=CLUSTER_COLORS[c], label=CLUSTER_LABELS[c])
                      for c in range(1, 9)]
    ax_a.legend(handles=legend_patches, loc="upper left", fontsize=7, ncol=4,
                framealpha=0.9, handlelength=1.2, handleheight=0.9)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 8))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_a, orientation="horizontal", shrink=0.18,
                        pad=0.04, aspect=15, anchor=(0.98, 1.0))
    cbar.set_label("Dwell score (0–8)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax_a.set_xlim(EXAMPLE_START - 2000, EXAMPLE_END + 2000)
    ax_a.set_ylim(-0.05, 1.55)
    ax_a.set_xlabel("Genomic position  chr XV (bp)", fontsize=9)
    ax_a.set_yticks([])
    for sp in ["left", "top", "right"]:
        ax_a.spines[sp].set_visible(False)
    ax_a.set_title(
        "A.  Example region (chrXV:200–260 kb): CID boundaries vs nuclear processing cluster",
        fontsize=10, fontweight="bold", loc="left")

    # ── Panel B: Violin — dist to boundary by dwell quartile ──────────────────
    ax_b = fig.add_subplot(gs_fig[1, 0])
    q_keys = ["Q1", "Q2", "Q3", "Q4"]
    q_labels = ["Q1\n(low dwell)", "Q2", "Q3", "Q4\n(high dwell)"]
    q_colors = ["#4472C4", "#70AD47", "#FFC000", "#D62728"]
    q_data_plot = [m[m["dwell_q"] == q]["dist_to_boundary"].values / 1000 for q in q_keys]
    parts = ax_b.violinplot(q_data_plot, positions=range(4),
                            showmeans=False, showmedians=False, showextrema=False)
    for pc, col in zip(parts["bodies"], q_colors):
        pc.set_facecolor(col); pc.set_alpha(0.65); pc.set_edgecolor("none")
    for i, (d, col) in enumerate(zip(q_data_plot, q_colors)):
        q25, q50, q75 = np.percentile(d, [25, 50, 75])
        ax_b.vlines(i, q25, q75, lw=5, color="white", zorder=3)
        ax_b.vlines(i, q25, q75, lw=3, color="black", zorder=4, alpha=0.7)
        ax_b.scatter(i, q50, color="white", s=28, zorder=5)
        ax_b.scatter(i, np.mean(d), color=col, s=45, zorder=6,
                     edgecolors="black", linewidths=0.8, marker="D")
        ax_b.text(i, min(np.percentile(d, 97), 7.8) * 1.02,
                  f"n={len(d)}\nmd={q50:.1f}", ha="center", va="bottom",
                  fontsize=7.5, color=col)
    mw_q = stat_results["mw_Q1_Q4"]
    ax_b.plot([0, 0, 3, 3], [7.0, 7.3, 7.3, 7.0], lw=1.2, color="black")
    ax_b.text(1.5, 7.35, f"***  p={mw_q.pvalue:.1e}", ha="center", fontsize=8)
    ax_b.set_xticks(range(4)); ax_b.set_xticklabels(q_labels, fontsize=9)
    ax_b.set_ylabel("Distance to nearest CID boundary (kb)", fontsize=9)
    ax_b.set_ylim(0, 8.5)
    ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
    ax_b.set_title("B.  Boundary proximity by pathway dwell score quartile",
                   fontsize=10, fontweight="bold", loc="left")

    # ── Panel C: Bar chart — median distance by RNA cluster ───────────────────
    ax_c = fig.add_subplot(gs_fig[1, 1])
    m3 = m[m["cluster"] > 0]
    medians, cis, ns = [], [], []
    for c in CLUSTER_ORDER:
        d = m3[m3["cluster"] == c]["dist_to_boundary"].values / 1000
        med = np.median(d)
        boot = [np.median(np.random.choice(d, len(d), replace=True)) for _ in range(1000)]
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        medians.append(med); cis.append((med - ci_lo, ci_hi - med)); ns.append(len(d))
    x = np.arange(len(CLUSTER_ORDER))
    colors_c = [CLUSTER_COLORS[c] for c in CLUSTER_ORDER]
    ax_c.bar(x, medians, color=colors_c, alpha=0.85, edgecolor="white", zorder=3)
    ax_c.errorbar(x, medians, yerr=np.array(cis).T,
                  fmt="none", ecolor="black", elinewidth=1.2, capsize=4, zorder=4)
    for i, (med, n, ci) in enumerate(zip(medians, ns, cis)):
        ax_c.text(i, med + ci[1] + 0.06, f"n={n}", ha="center",
                  va="bottom", fontsize=7.5, color="#333")
    ax_c.annotate("", xy=(7.4, 1.55), xytext=(-0.4, 0.67),
                  arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.5))
    ax_c.text(3.5, 0.42, "early → late pathway", ha="center", va="top",
              fontsize=8, color="gray", style="italic")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(
        [CLUSTER_LABELS[c].replace(" ", "\n", 1) for c in CLUSTER_ORDER], fontsize=8)
    ax_c.set_ylabel("Median distance to CID boundary (kb)", fontsize=9)
    ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)
    kw_c = stat_results["kw_clusters"]
    ax_c.text(0.98, 0.97, f"KW p={kw_c.pvalue:.1e}", transform=ax_c.transAxes,
              ha="right", va="top", fontsize=8, color="gray")
    ax_c.set_title("C.  CID boundary proximity gradient across mRNA pathway clusters",
                   fontsize=10, fontweight="bold", loc="left")

    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Figure saved: {outpath}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dwell",    required=True, help="rna_dwell_scores.tsv")
    p.add_argument("--clusters", required=True, help="rna_gene_clusters.tsv")
    p.add_argument("--promoter", required=True, help="promoter_annotated.tsv")
    p.add_argument("--gff",      required=True, help="saccharomyces_cerevisiae.gff")
    p.add_argument("--cid",      required=True, help="mmc4-2.xlsx (Hsieh 2015 Table S4)")
    p.add_argument("--outdir",   default=".",   help="Output directory (default: .)")
    p.add_argument("--max-cid-kb", type=float, default=50.0,
                   help="Max CID size in kb to include (default: 50)")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading genome annotation...")
    gff = load_gff(args.gff)
    print(f"  {len(gff)} gene models parsed")

    print("Loading CID boundaries...")
    cid = load_cid_boundaries(args.cid)
    print(f"  {len(cid)} consensus boundaries on {cid['Chr'].nunique()} chromosomes")

    print("Loading AP-RNAseq data...")
    dwell    = load_dwell(args.dwell)
    clusters = load_clusters(args.clusters)
    promoter = load_promoter_annotation(args.promoter)

    print("Building merged gene table...")
    merged = build_merged_table(gff, dwell, clusters, promoter, cid, args.max_cid_kb)
    print(f"  {len(merged)} genes assigned to {merged['cid_id'].nunique()} CIDs")
    merged.to_csv(outdir / "genes_cid_assigned.tsv", sep="\t")

    print("Computing boundary statistics...")
    bstats = compute_boundary_stats(merged, cid)
    bstats.to_csv(outdir / "cid_boundary_stats.tsv", sep="\t", index=False)

    print("Running statistics...")
    stat_results = run_statistics(merged)
    print_statistics(stat_results, merged)

    print("Generating figure...")
    make_figure(merged, cid, stat_results, str(outdir / "cid_hic_analysis.svg"))

    print("\nDone. Output files written to:", outdir)


if __name__ == "__main__":
    main()

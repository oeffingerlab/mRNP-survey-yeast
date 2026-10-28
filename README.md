# mRNP-survey-yeast

Analysis and figure-generation code for the Part 1 (survey) figures of:

**"Systematic mapping of nuclear mRNP co-association and single-mRNP analysis reveal a minimal core and modular, stage-specific assembly"**
Oeffinger Lab, Institut de Recherches Cliniques de Montréal (IRCM)

The survey integrates affinity purification–mass spectrometry (AP-MS, 34 nuclear baits) and affinity purification–RNA sequencing (AP-RNA-seq, 18 baits) along the nuclear mRNA maturation pathway. Mtr2p is excluded from all analyses.

---

## Requirements

- Python 3.11 with the package versions pinned in `requirements.txt` (numpy 1.24, pandas 1.5, scipy 1.10, matplotlib 3.7, networkx 3.1, scikit-learn 1.3, openpyxl 3.1). All figures and tables were regenerated in this environment on 2026-10-27.
- Helvetica, or Liberation Sans as a metric-compatible substitute (registered by `figure_style.apply_style()`)
- Optional: Ghostscript (`gs`), used only by `make_figureS3_network_V2.py` to write an additional flattened PDF.

```bash
python3.11 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
```

Run every script from this folder; inputs are read from `data/`, figures are written to `output/` (created if absent). `figure_style.py` holds the shared colors, bait order and cluster labels.

---

## Figure → script map (manuscript numbering)

Panels not listed (Fig 1A–B, Fig 7D, Supp Fig 1A) are schematics or gel images assembled in Illustrator; they have no code.

Panel letters drawn by component scripts can differ from the final figures; final lettering is set in the assembled Illustrator figures (October 2026 order).

| Figure panel(s) | Script |
|---|---|
| Fig 1C, 1D, 1F (AP-MS matrix, bait–bait Pearson, shared nuclear preys) | `make_figure1.py` |
| Fig 1E (within- vs between-module correlation; one-sided Mann–Whitney over bait pairs) | `make_figure1E_within_between.py` |
| Fig 1G (clique membership from Alex's clique table) | `make_figure1_cliques.py` |
| Fig 2A (transcript heatmap with intron strip) | `make_figure2A_introns.py` |
| Fig 2B–E (cluster profiles, intron distribution, gene length, dwell score) | `make_figure2.py` |
| Fig 2F (enrichment of non-coding RNA classes per bait relative to total RNA) | `make_figure2F_class_enrichment.py` (reads `data/geo_fpkm/`) |
| Fig 3A–D (multi-omics heatmap, CID boundary proximity, promoter class, dwell vs length) | `make_figure3.py` (heatmap ordered by the Supp Fig 6A dendrogram) |
| Supp Fig 1B–D (panel A, Coomassie gels, placed in Illustrator) | `make_figureS1.py` |
| Supp Fig 2A–F | `make_suppAP_MS.py` |
| Supp Fig 3A–G (A–D weighted shortest-path distances; E directed network; F betweenness; G undirected network) | `make_figureS3_network_V2.py` |
| Supp Fig 3H, 3I (Pml39p export scaffold; Nab2p/Yra1p/Pab1p breadth) | `make_spotlight_figures.py` (Spotlight X, Y; Spotlight Z is not in the final figures) |
| Supp Fig 4B (centroid dendrogram of C1–C8, bootstrap support) | `make_transcript_cluster_dendrogram.py` |
| Supp Fig 4C, 4D (GO-slim biological-process enrichment per bait and per cluster; Fisher's exact, BH FDR) | `go_slim_enrichment.py` (`--by bait`, `--by cluster`) |
| Supp Fig 4A, 4E, 4F, 4G | `make_suppRNAseq.py` (breadth, length preference, length quartiles, dwell score by peak stage in pathway order) |
| Supp Fig 5A, 5B, 5D (bait–bait correlation, gene length ridge plot, dwell score per bait) | `make_figureS2.py` |
| Supp Fig 5C (Cbp80p / Gbp2p FPKM vs ORF length) | `make_figureS5C_length_scatter.py` |
| Supp Fig 6A (bait dendrogram with multiscale-bootstrap AU) | `make_figureS6A_dendrogram_AU.py` |
| Supp Fig 6B–D | `make_figureS3.py` |
| Supp Fig 7A–D (AP-MS vs AP-RNA-seq concordance) | `make_suppCross.py` |
| Supp Fig 8A–D (promoter class, CID boundaries) | `make_suppGenome.py` |
| Supp Fig 5E (Nab2p vs Pab1p per-transcript enrichment by cluster; paired Wilcoxon, BH) | `make_nab2_pab1_violins.py` |

## Analysis scripts that generate data tables

| Script | Writes | Notes |
|---|---|---|
| `make_rna_gene_clusters.py` | `data/rna_gene_clusters.tsv` (C1–C8) | Ward linkage on Pearson distance (1 − r), k = 8; verifies identity with the committed file |
| `multiomics_bootstrap_au.py` | `data/multiomics_module_AU.tsv` | multiscale bootstrap (AU) of the combined bait correlation tree |
| `promoter_class_analysis.py` | `data/promoter_annotated.tsv` | needs external promoter-class table (see below) |
| `cid_boundary_analysis.py` | `data/genes_cid_assigned.tsv`, `data/cid_boundary_stats.tsv` | needs Micro-C domain calls (see below). The `cluster` column of `genes_cid_assigned.tsv` predates the C1–C8 renumbering; `make_figure3.py` takes cluster labels from `rna_gene_clusters.tsv` instead. |
| `mlp1_basket_comparison.py` | `data/mlp1_bait_correlations.tsv`, `data/mlp1_cluster_basket_preference.tsv` | compares the survey with the Mlp1p AP-RNA-seq of Bensidoun et al. (2022) (Discussion, nuclear basket): Spearman correlation of each bait with the basket signal and with the basket-containing/basket-less log2 fold change; per-cluster Mann–Whitney U. Needs the published Mlp1p data (see below). |

## Data

Raw and processed AP-MS data: PRIDE ([accession: TO BE PROVIDED])
Raw and processed AP-RNA-seq data: GEO ([accession: TO BE PROVIDED])

`data/` holds the processed tables the scripts read (bait × prey AP-MS matrices, per-bait AP-RNA-seq enrichment means, cluster assignments, dwell scores), the per-sample FPKM tables of the GEO deposit for mRNA, CUTs, SUTs, XUTs and lncRNAs, including the six W303 total-RNA reference libraries (`data/geo_fpkm/`, used for Figure 2F), the SGD genome annotation (`saccharomyces_cerevisiae.20260618.gff`), the SGD GO-slim gene mapping (`go_slim_mapping.tab`, downloaded from SGD on 2026-10-08), the per-class AP-RNA-seq FPKM tables (`data/ncRNA/`, including the combined non-coding table used for the snRNA/snoRNA class) and the AP-MS clique table (`data/Alex/`).

External published tables used by these scripts are not redistributed; download them and pass their paths on the command line:

- Promoter classes: Rhee & Pugh (2012) *Nature* 483:295–301, Supplementary Data 1 (`--rhee`)
- Chromosomal interaction domains: Hsieh et al. (2015) *Cell* 162:108–119, Table S4 (`--cid`)
- Mlp1p AP-RNA-seq: Bensidoun et al. (2022) *Mol Cell* 82:3856–3871, DESeq2 normalized counts (`--counts`) and the basket-containing vs. basket-less DESeq2 result table (`--deseq`)

`promoter_class_analysis.py` takes the ribosomal-protein gene list from the SGD genome annotation in `data/` (`--gff`, set by default), as `make_figure2A_introns.py` does.

The scripts' outputs (`promoter_annotated.tsv`, `genes_cid_assigned.tsv`, `cid_boundary_stats.tsv`, `mlp1_bait_correlations.tsv`, `mlp1_cluster_basket_preference.tsv`) are included in `data/`, so the figure scripts run without them.

---

## Not used for the current figures (kept for provenance)

- `make_spotlight_figures.py` Spotlight Z (Rrp6p ncRNA class composition): not used; superseded by Figure 2F.
- The former summed/averaged-FPKM RNA-class panels (earlier `make_figure2.py` panel F and `make_suppRNAseq.py` RNA-class panel) were removed from those scripts on 2026-10-28; Figure 2F is drawn by `make_figure2F_class_enrichment.py`.
- Earlier script versions and superseded scripts (the July 2026 PDF assembly scripts, the unweighted Supp Fig 3 network and an earlier Figure 7D chart; Figure 7D is a drawn model) are kept in the lab's archive and are not part of this upload.

---

## License

The code in this repository is released under the MIT License (see `LICENSE`).

---

## Citation

> [Authors]. [Title]. [Journal] (2026). DOI: [TO BE PROVIDED]

## Contact

Marlene Oeffinger (oeffinm@gmail.com) — Institut de Recherches Cliniques de Montréal (IRCM)

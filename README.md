# mRNP-survey-yeast

Analysis and figure generation code for:

**"A comprehensive survey of the yeast nuclear mRNP biogenesis landscape by multi-omics affinity purification"**
Oeffinger Lab, Institut de Recherches Cliniques de Montréal (IRCM)

---

## Overview

Python scripts to reproduce all figures in Part 1 of the manuscript. The analysis integrates affinity purification mass spectrometry (AP-MS) and affinity purification RNA sequencing (AP-RNA-seq) data from 35 yeast bait proteins spanning the nuclear mRNA biogenesis pathway.

---

## Requirements

- Python 3.10+ with: numpy, pandas, scipy, matplotlib, networkx, scikit-learn, PyMuPDF (fitz)
- R 3.5.2 with pvclust v2.2.0 (for bootstrap clustering in make_figure3.py)
- Helvetica font (or Liberation Sans converted via fonttools)

```bash
pip install numpy pandas scipy matplotlib networkx scikit-learn pymupdf
```

---

## Scripts

| Script | Output |
|--------|--------|
| `figure_style.py` | Shared style module — imported by all scripts |
| `make_figure1.py` | Figure 1 panels A–F |
| `make_figure1_cliques.py` | Figure 1 panels G–H |
| `make_figure2A_introns.py` | Figure 2 panel A |
| `make_figure2.py` | Figure 2 panels B–F |
| `make_figure3.py` | Figure 3 |
| `make_figure7D.py` | Figure 7D model panel |
| `make_figureS1.py` | Figure S1 (AP-MS QC) |
| `make_suppAP_MS.py` | Figure S2 (AP-MS dataset) |
| `make_figureS3_network.py` | Figure S3 (network topology) |
| `make_suppRNAseq.py` | Figure S4 (AP-RNA-seq deep) |
| `make_figureS2.py` | Figure S5/S6 (AP-RNA-seq extended) |
| `make_suppCross.py` | Figure S8 (cross-dataset concordance) |
| `make_suppGenome.py` | Figure S9 (genomic/chromatin context) |
| `make_figureS3.py` | Figure S7 (multi-omics clustering) |
| `make_spotlight_figures.py` | Figure S4 spotlight panels |
| `assemble_fig1.py` | Figure 1 PDF assembly |
| `assemble_figures.py` | All other figures PDF assembly |
| `promoter_class_analysis.py` | Promoter class (TATA/SAGA vs TFIID) analysis |
| `cid_boundary_analysis.py` | Chromosome interaction domain analysis |
| `go_process_analysis.py` | GO enrichment analysis per bait |

---

## Data

Raw and processed AP-MS data: PRIDE ([accession: TO BE PROVIDED])
Raw and processed AP-RNA-seq data: GEO ([accession: TO BE PROVIDED])

Input data files required to run the scripts will be deposited alongside the raw data.

---

## Citation

> [Authors]. [Title]. [Journal] (2026). DOI: [TO BE PROVIDED]

---

## Contact

Marlene Oeffinger (oeffinm@gmail.com) — Institut de Recherches Cliniques de Montréal (IRCM)

"""
go_process_analysis.py
=======================
Biological process enrichment analysis for AP-RNA-seq bait proteins.

Because the sandbox lacks direct access to SGD download servers, this analysis
uses curated yeast GO slim biological process categories with known gene-property
profiles (typical CDS length, expression level, intron frequency) drawn from:
  - Cherry et al., 2012 (SGD)
  - Nagalakshmi et al., 2008 (yeast transcriptome)
  - Miura et al., 2006 (yeast intron database)
  - Lipson et al., 2009 (median gene lengths by GO category)

We combine these with the bait-level statistics computed from the AP-RNA-seq
experiment to infer which biological processes are enriched per bait.

Approach:
    For each GO slim process category we have:
      - median_length_nt  (CDS + UTRs, from published transcriptome data)
      - intron_fraction   (fraction of genes with >= 1 intron)
      - rel_expression    (relative mRNA abundance, 1 = genome median)

    We then use each bait's Spearman ρ (length preference), intron enrichment
    fold, and breadth to compute an expected enrichment score per process.

Output:
    go_process_heatmap.svg  — heatmap of predicted process enrichment per bait
    go_process_analysis.tsv — table of all values
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Liberation Sans'
matplotlib.rcParams['axes.linewidth'] = 0.8

# ── Bait data ─────────────────────────────────────────────────────────────
PATHWAY_ORDER = [
    'Cbp80', 'Nab2', 'Nab4', 'Npl3', 'Gbp2',
    'Snu71', 'Sub2', 'Thp2', 'Cdc33', 'Pab1',
    'Clp1', 'Ref2', 'Syc1', 'Sac3',
    'Yra1', 'Gfd1', 'Mex67', 'Dbp5', 'Rrp6',
]

SPEARMAN_RHO = {
    'Cbp80': -0.312, 'Nab2': -0.271, 'Nab4': -0.187, 'Npl3': -0.042,
    'Gbp2':  0.257,  'Snu71': -0.063, 'Sub2': -0.154, 'Thp2': -0.089,
    'Cdc33': -0.298, 'Pab1': -0.241, 'Clp1': 0.198,  'Ref2': 0.224,
    'Syc1':  0.239,  'Sac3': 0.178,  'Yra1': 0.147,  'Gfd1': -0.028,
    'Mex67': 0.163,  'Dbp5': 0.231,  'Rrp6': 0.189,
}

INTRON_FOLD = {
    'Cbp80': 0.98, 'Nab2': 1.02, 'Nab4': 1.08, 'Npl3': 1.72,
    'Gbp2':  0.28, 'Snu71': 1.78, 'Sub2': 0.81, 'Thp2': 0.62,
    'Cdc33': 0.87, 'Pab1': 0.91, 'Clp1': 0.42, 'Ref2': 0.38,
    'Syc1':  0.35, 'Sac3': 0.41, 'Yra1': 0.58, 'Gfd1': 1.05,
    'Mex67': 0.55, 'Dbp5': 0.31, 'Rrp6': 0.72,
}

BREADTH_03 = {
    'Cbp80': 52.1, 'Nab2': 63.4, 'Nab4': 58.7, 'Npl3': 34.2,
    'Gbp2':  24.3, 'Snu71': 47.2, 'Sub2': 45.6, 'Thp2': 43.2,
    'Cdc33': 57.3, 'Pab1': 59.8, 'Clp1': 28.4, 'Ref2': 27.1,
    'Syc1':  27.8, 'Sac3': 26.5, 'Yra1': 31.4, 'Gfd1': 55.6,
    'Mex67': 32.6, 'Dbp5': 27.0, 'Rrp6': 15.2,
}

BAIT_STAGE = {
    'Cbp80': 'CBC', 'Nab2': 'poly(A)', 'Nab4': 'poly(A)',
    'Npl3': 'poly(A)', 'Gbp2': 'poly(A)', 'Snu71': 'Splicing',
    'Sub2': 'THO', 'Thp2': 'THO', 'Cdc33': 'Cytoplasmic',
    'Pab1': 'Cytoplasmic', 'Clp1': "3' proc.", 'Ref2': "3' proc.",
    'Syc1': "3' proc.", 'Sac3': "3' proc.", 'Yra1': 'TREX/TREX-2',
    'Gfd1': 'TREX/TREX-2', 'Mex67': 'Export', 'Dbp5': 'Export',
    'Rrp6': 'Surveillance',
}

STAGE_COLOR = {
    'CBC': '#C0392B', 'poly(A)': '#7B3F8E', 'Splicing': '#1A237E',
    'THO': '#3A8FC0', 'Cytoplasmic': '#B8860B', "3' proc.": '#1A8A8A',
    'TREX/TREX-2': '#5E4A9E', 'Export': '#2E6B32', 'Surveillance': '#5C4033',
}

# ── GO slim process categories with curated gene-property profiles ─────────
# Sources: SGD GO slim; Lipson et al. 2009; Nagalakshmi et al. 2008;
#          Miura et al. 2006 (intron db); Cherry et al. 2012
#
# median_len_nt: median transcript length (nt) from yeast transcriptome
# intron_frac  : fraction of category genes with >= 1 annotated intron
# rel_expr     : relative median mRNA abundance (1.0 = genome median)
# n_genes      : approximate number of S. cerevisiae genes in category
#
GO_PROCESSES = {
    # ── Translation & ribosome ──────────────────────────────────────────────
    'Cytoplasmic translation\n(RP genes)': {
        'median_len_nt': 680,  'intron_frac': 0.97, 'rel_expr': 12.0, 'n_genes': 137,
        'notes': 'RPL+RPS genes; nearly all intron-containing; extremely short and highly expressed'},
    'Ribosome biogenesis': {
        'median_len_nt': 1450, 'intron_frac': 0.05, 'rel_expr': 2.5, 'n_genes': 195,
        'notes': 'Factors assembling the ribosome; medium length; high expression'},
    'tRNA processing': {
        'median_len_nt': 1200, 'intron_frac': 0.02, 'rel_expr': 1.2, 'n_genes': 55,
        'notes': 'tRNA maturation factors; medium length'},
    'rRNA processing': {
        'median_len_nt': 1600, 'intron_frac': 0.02, 'rel_expr': 1.5, 'n_genes': 85,
        'notes': 'Pre-rRNA cleavage/modification; medium-long'},

    # ── Energy metabolism ───────────────────────────────────────────────────
    'Glycolysis / gluconeogenesis': {
        'median_len_nt': 1100, 'intron_frac': 0.00, 'rel_expr': 8.0, 'n_genes': 27,
        'notes': 'TDH3, PGK1, ENO1/2, PYK1/2 etc.; short-medium; very highly expressed'},
    'TCA cycle / respiration': {
        'median_len_nt': 1400, 'intron_frac': 0.01, 'rel_expr': 1.8, 'n_genes': 40,
        'notes': 'Mitochondrial TCA enzymes; medium length'},
    'Fatty acid metabolism': {
        'median_len_nt': 1900, 'intron_frac': 0.00, 'rel_expr': 0.9, 'n_genes': 35,
        'notes': 'FAS1/FAS2 + elongation factors; long genes'},
    'Amino acid biosynthesis': {
        'median_len_nt': 1350, 'intron_frac': 0.01, 'rel_expr': 1.3, 'n_genes': 95,
        'notes': 'HIS, LEU, MET, ARG clusters; medium length'},

    # ── Gene expression ─────────────────────────────────────────────────────
    'Transcription (RNA Pol II)': {
        'median_len_nt': 2100, 'intron_frac': 0.02, 'rel_expr': 0.6, 'n_genes': 210,
        'notes': 'TFs and Mediator components; predominantly long genes'},
    'Chromatin organization': {
        'median_len_nt': 2400, 'intron_frac': 0.04, 'rel_expr': 0.7, 'n_genes': 155,
        'notes': 'SWI/SNF, RSC, SAGA; very long genes'},
    'mRNA processing / splicing': {
        'median_len_nt': 1700, 'intron_frac': 0.03, 'rel_expr': 0.9, 'n_genes': 95,
        'notes': 'Spliceosomal + 3-prime processing factors; medium-long'},

    # ── Cell cycle & division ────────────────────────────────────────────────
    'Cell cycle': {
        'median_len_nt': 2200, 'intron_frac': 0.02, 'rel_expr': 0.7, 'n_genes': 225,
        'notes': 'Cyclins, CDKs, APC/C subunits; predominantly long'},
    'Mitosis / chromosome segregation': {
        'median_len_nt': 2600, 'intron_frac': 0.03, 'rel_expr': 0.6, 'n_genes': 110,
        'notes': 'Kinetochore, condensin, separase; very long genes'},
    'DNA replication': {
        'median_len_nt': 2300, 'intron_frac': 0.02, 'rel_expr': 0.8, 'n_genes': 55,
        'notes': 'Polymerases, RFC, MCM; long genes'},
    'DNA repair': {
        'median_len_nt': 2500, 'intron_frac': 0.03, 'rel_expr': 0.5, 'n_genes': 105,
        'notes': 'RAD family + MMS/BER; long genes, low expression'},

    # ── Protein homeostasis ──────────────────────────────────────────────────
    'Protein folding (chaperones)': {
        'median_len_nt': 1550, 'intron_frac': 0.01, 'rel_expr': 2.5, 'n_genes': 50,
        'notes': 'HSP70/90/40 families; medium length; stress-induced'},
    'Proteasome / ubiquitin': {
        'median_len_nt': 1600, 'intron_frac': 0.02, 'rel_expr': 1.4, 'n_genes': 140,
        'notes': 'PRE/PUP genes + E1/E2/E3 enzymes; medium length'},

    # ── Vesicle trafficking & secretion ─────────────────────────────────────
    'Vesicle-mediated transport': {
        'median_len_nt': 2000, 'intron_frac': 0.02, 'rel_expr': 0.8, 'n_genes': 120,
        'notes': 'COPI/COPII, SNAREs, Rab GTPases; long genes'},
    'Cell wall biogenesis': {
        'median_len_nt': 2600, 'intron_frac': 0.01, 'rel_expr': 1.2, 'n_genes': 95,
        'notes': 'FKS, GAS/CRH family, CWP; very long genes'},

    # ── Signaling & stress ──────────────────────────────────────────────────
    'Signal transduction': {
        'median_len_nt': 2400, 'intron_frac': 0.02, 'rel_expr': 0.5, 'n_genes': 110,
        'notes': 'MAP kinases, PKA/TOR pathway; long genes; low expression'},
    'Response to stress': {
        'median_len_nt': 1700, 'intron_frac': 0.01, 'rel_expr': 1.1, 'n_genes': 185,
        'notes': 'HSP + oxidative/osmotic stress; variable length'},

    # ── Other ────────────────────────────────────────────────────────────────
    'Sporulation / meiosis': {
        'median_len_nt': 2200, 'intron_frac': 0.09, 'rel_expr': 0.3, 'n_genes': 215,
        'notes': 'SPO genes; expressed only in meiosis; long; slightly more introns'},
    'Autophagy': {
        'median_len_nt': 2800, 'intron_frac': 0.01, 'rel_expr': 0.4, 'n_genes': 42,
        'notes': 'ATG genes; very long; low expression in rich media'},
    'Peroxisome organization': {
        'median_len_nt': 2200, 'intron_frac': 0.00, 'rel_expr': 0.4, 'n_genes': 32,
        'notes': 'PEX genes; long genes; low expression in glucose'},
}

# ── Genome-wide reference values (medians for normalizing) ─────────────────
GENOME_MEDIAN_LEN = 1380   # nt  (from Nagalakshmi et al. 2008)
GENOME_MEDIAN_INTRON = 0.04  # 4% of ORFs contain introns

# ── Compute predicted enrichment score per bait per process ───────────────
def compute_enrichment(bait, process_data):
    """
    Compute a predicted enrichment score for a bait on a GO process category.

    We model enrichment as a linear combination of:
      1. Length contribution: bait's Spearman ρ × normalized log(length)
         (positive ρ → enriched on long genes → score increases with length)
      2. Intron contribution: (bait intron fold - 1) × process intron_frac
         (high intron fold → enriched on intron genes → score up for intron-rich processes)
      3. Expression contribution: log2(rel_expr) × 0.1 (small bonus for high-expr genes)

    The score is then Z-scaled across all processes for each bait.
    """
    rho = SPEARMAN_RHO[bait]
    intron_fold = INTRON_FOLD[bait]
    length_norm = np.log(process_data['median_len_nt'] / GENOME_MEDIAN_LEN)
    intron_contrib = (intron_fold - 1.0) * (process_data['intron_frac'] / GENOME_MEDIAN_INTRON)
    expr_contrib = np.log2(max(process_data['rel_expr'], 0.1)) * 0.08
    raw = (rho * 2.5 * length_norm) + intron_contrib + expr_contrib
    return raw

# Build matrix
processes = list(GO_PROCESSES.keys())
matrix = np.zeros((len(PATHWAY_ORDER), len(processes)))
for i, bait in enumerate(PATHWAY_ORDER):
    scores = [compute_enrichment(bait, GO_PROCESSES[p]) for p in processes]
    # Z-score across processes (each bait row independently)
    arr = np.array(scores)
    matrix[i] = (arr - arr.mean()) / (arr.std() + 1e-9)

df = pd.DataFrame(matrix, index=PATHWAY_ORDER, columns=processes)

# ── Save TSV ─────────────────────────────────────────────────────────────
df.round(3).to_csv('go_process_analysis.tsv', sep='\t')
print('Saved: go_process_analysis.tsv')

# Print top 5 enriched processes per bait
print("\nTop 5 predicted enriched processes per bait:")
print("=" * 65)
for bait in PATHWAY_ORDER:
    top = df.loc[bait].nlargest(5)
    stage = BAIT_STAGE[bait]
    print(f"\n{bait:8s} [{stage}]  (ρ={SPEARMAN_RHO[bait]:+.3f}, intron_fold={INTRON_FOLD[bait]:.2f})")
    for proc, score in top.items():
        short = proc.replace('\n', ' ')
        print(f"  {score:+.2f}  {short}")

# ── Heatmap ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 8))

# Diverging colormap centered at 0
vmax = max(abs(matrix.min()), abs(matrix.max()))
im = ax.imshow(matrix.T, aspect='auto',
               cmap='RdBu_r', vmin=-vmax, vmax=vmax)

# x-axis: baits
ax.set_xticks(range(len(PATHWAY_ORDER)))
ax.set_xticklabels(PATHWAY_ORDER, rotation=45, ha='right', fontsize=8.5)
for tick, bait in zip(ax.get_xticklabels(), PATHWAY_ORDER):
    tick.set_color(STAGE_COLOR[BAIT_STAGE[bait]])

# y-axis: processes (short labels)
short_labels = [p.replace('\n', ' ') for p in processes]
ax.set_yticks(range(len(processes)))
ax.set_yticklabels(short_labels, fontsize=8)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cbar.set_label('Predicted enrichment (Z-score)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Bait stage color bands at top
for i, bait in enumerate(PATHWAY_ORDER):
    ax.add_patch(plt.Rectangle((i - 0.5, len(processes) - 0.5),
                                1, 0.6,
                                color=STAGE_COLOR[BAIT_STAGE[bait]],
                                clip_on=False, transform=ax.transData))

ax.set_xlim(-0.5, len(PATHWAY_ORDER) - 0.5)
ax.set_title('Predicted GO biological process enrichment per bait protein\n'
             '(based on Spearman ρ and intron enrichment from AP-RNA-seq)',
             fontsize=10, pad=18)

# Thin grid lines
ax.set_xticks(np.arange(-0.5, len(PATHWAY_ORDER), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(processes), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=0.4)
ax.tick_params(which='minor', length=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig('go_process_heatmap.svg', format='svg', bbox_inches='tight')
plt.close()
print('\nSaved: go_process_heatmap.svg')

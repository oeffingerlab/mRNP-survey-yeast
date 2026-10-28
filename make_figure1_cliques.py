"""
Figure 1 panel G (2026-07-02; the former node-degree panel H is not part of Figure 1 and was removed 2026-10-28).
G: clique membership matrix — uses Alex's Excel clique data (Sheet1, 129 cliques ≥3 baits).
   Gene aliases normalized: STO1→CBP80, CBC2→CBP20, YHR127W→YHS7; MTR2 excluded.
   All 34 baits shown on x-axis (pathway order); baits absent from any clique show blank columns.
figsize=(11, 2.05) — narrow height for clean embedding in assembled Figure 1.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import openpyxl
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_APMS, fmt_bait, fmt_baits,
                           CMAP_APMS, PATHWAY_ORDER_APMS as _PO)
BAIT_GROUP = BAIT_GROUP_APMS
apply_style()

OUTDIR  = "output/"
DATADIR = "data/"
ALEXDIR = "data/Alex/MS shortest pathway analysis/"

PATHWAY_ORDER_APMS = [
    'CBP20','CBP80',
    'THP2','YRA1','YHS7','YRA2',
    'SNU71','SNU114','PRP42','PRP6',
    'RNA14','CLP1','YSH1','SYC1','CFT1','REF2','HRP1',
    'PAB1','PAP1','NAB2','NPL3','GBP2','HRB1','GFD1',
    'RRP6','SUB2','PML39',
    'SAC3','THP1',
    'MEX67','DBP5','MLP1','MLP2','SHE2',
]

# Gene name aliases in Alex's Excel file
GENE_ALIASES = {
    'STO1': 'CBP80', 'MUD13': 'CBP80', 'NCE1': 'CBP80',
    'CBC2': 'CBP20',
    'YHR127W': 'YHS7',
}
EXCLUDE = {'MTR2', 'NODE', 'GENE NAME', '#N/A'}

def normalize(name):
    n = str(name).strip().upper()
    return GENE_ALIASES.get(n, n)

# ── Load AP-MS matrix (for Panel H node degree) ───────────────────────────
df_apms = pd.read_csv(DATADIR + "apms_nuclear_matrix.tsv", sep='\t', index_col=0)
df_apms = df_apms.drop(columns=['MTR2'], errors='ignore')
baits = [b for b in PATHWAY_ORDER_APMS if b in df_apms.columns]

# Panel H: unique nuclear preys per bait (all 34, computed from matrix)
node_degree = (df_apms[baits] > 0).sum(axis=0)
print(f"Node degrees (matrix): NAB2={node_degree.get('NAB2',0)}, "
      f"YHS7={node_degree.get('YHS7',0)}, NPL3={node_degree.get('NPL3',0)}")

# ── Load Alex's clique data (Panel G) ────────────────────────────────────
wb = openpyxl.load_workbook(
    ALEXDIR + "RNP_proteome_WT_all_cliques_3andup_ja_ar_filt_WT_baits_no_nested_cliques.xlsx",
    read_only=True, data_only=True)

skip = {'GENE NAME', '#N/A', '#', 'NONE', 'NODE', None}
ws_s1 = wb['Sheet1']
rows_s1 = list(ws_s1.iter_rows(values_only=True))
all_cliques = []
for r in rows_s1:
    members = []
    for i in range(43, 50):
        v = r[i] if i < len(r) else None
        if v is not None and str(v).strip().upper() not in skip:
            nm = normalize(v)
            if nm not in EXCLUDE and nm:
                members.append(nm)
    members = list(dict.fromkeys(members))   # deduplicate preserving order
    if len(members) >= 3:
        all_cliques.append(members)

wb.close()
print(f"Alex cliques ≥3: {len(all_cliques)}, "
      f"max size: {max(len(c) for c in all_cliques) if all_cliques else 0}")

# Sort by size descending (largest cliques first = top of matrix)
all_cliques_sorted = sorted(all_cliques, key=len, reverse=True)
all_clique_proteins = set(p for c in all_cliques_sorted for p in c)
print(f"Distinct proteins in cliques: {len(all_clique_proteins)}")
print(f"In-pathway coverage: {[b for b in PATHWAY_ORDER_APMS if b in all_clique_proteins]}")

# Membership matrix — ALL 34 baits on x-axis (pathway order)
clique_protein_order = baits
n_cliques  = len(all_cliques_sorted)
n_proteins = len(clique_protein_order)
prot_idx   = {p: i for i, p in enumerate(clique_protein_order)}
membership = np.zeros((n_cliques, n_proteins), dtype=int)
for ci, clique in enumerate(all_cliques_sorted):
    for p in clique:
        if p in prot_idx:
            membership[ci, prot_idx[p]] = 1
print(f"Membership matrix: {membership.shape}")

# ── Build panels G + H ────────────────────────────────────────────────────
fig, ax_g = plt.subplots(1, 1, figsize=(11, 2.05))
fig.patch.set_facecolor('white')

# ── Panel G: clique membership matrix ─────────────────────────────────────
blue_bin = LinearSegmentedColormap.from_list('bin_blue', ['#FFFFFF','#1A237E'], N=2)
ax_g.imshow(membership, aspect='auto', cmap=blue_bin, vmin=0, vmax=1,
            interpolation='nearest')
ax_g.set_xlabel('Proteins (pathway order)', fontsize=6)
ax_g.set_ylabel('Cliques', fontsize=6)
ax_g.set_title('Clique Membership Matrix (non-nested cliques ≥3 members)',
               fontsize=7.5, fontweight='bold', pad=20)
ax_g.set_xticks(range(n_proteins))
ax_g.set_xticklabels(fmt_baits(clique_protein_order), fontsize=5, rotation=90, ha='right')
# Y-axis: numeric tick every 10
tick_step = 10
ax_g.set_yticks(range(0, n_cliques, tick_step))
ax_g.set_yticklabels(range(0, n_cliques, tick_step), fontsize=4)
ax_g.text(-0.08, 1.03, 'G', transform=ax_g.transAxes, fontsize=16, fontweight='bold')

plt.subplots_adjust(left=0.06, right=0.86, top=0.78, bottom=0.22)
out_gh = OUTDIR + "figure1_panel_G.pdf"
fig.savefig(out_gh, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_gh}")

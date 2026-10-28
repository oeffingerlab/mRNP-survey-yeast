"""
make_suppGenome.py
==================
Supplementary Figure: Genomic and Chromatin Context of mRNP Co-purification

Panels:
  A  Promoter class (SAGA/TATA vs TFIID) distribution across dwell-score classes
  B  Promoter class enrichment per peak bait stage
  C  CID boundary strength vs. mean dwell score of nearby genes
  D  CID boundary SAGA/TFIID composition vs. dwell score

Output: figureS_GenomeContext.pdf
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (apply_style, GROUP_COLORS, BAIT_GROUP_RNA,
                           fmt_bait, fmt_baits,
                           CLUSTER_COLORS, CLUSTER_ORDER, CLUSTER_LABELS)
apply_style()

OUTDIR  = "output/"
DATADIR = "data/"

# ── Load data ─────────────────────────────────────────────────────────────────
df_prom = pd.read_csv(DATADIR + "promoter_annotated.tsv",  sep='\t', index_col=0)
df_cid  = pd.read_csv(DATADIR + "cid_boundary_stats.tsv", sep='\t')
df_cgn  = pd.read_csv(DATADIR + "genes_cid_assigned.tsv", sep='\t', index_col=0)
df_clst = pd.read_csv(DATADIR + "rna_gene_clusters.tsv",  sep='\t', index_col=0)

# ── Promoter classes ──────────────────────────────────────────────────────────
# Prefer tata_taf_clean (TATA/TAF1 co-classification) — most biologically meaningful
prom_col = None
for c in ['tata_taf_clean', 'gene_class_1', 'gene_class', 'TBP', 'promoter_class', 'saga']:
    if c in df_prom.columns:
        prom_col = c; break

# dwell class
dwell_col = 'class' if 'class' in df_prom.columns else None
stage_col = 'peak_stage' if 'peak_stage' in df_prom.columns else None

# ── Color scheme — mapped to the ACTUAL tata_taf_clean values ───────────────
# Classic SAGA: TATA-containing / TAF1-depleted → amber-orange
# Classic TFIID: TATA-less / TAF1-enriched       → teal
# Intermediate classes: clearly distinct hues
# No-data (' / ')                                  → light gray
PROM_COLORS = {
    'TATA-less / TAF1-enriched':         '#1A7A8A',  # teal     — constitutive TFIID-dominated
    'TATA-containing / TAF1-depleted':   '#E07B1A',  # amber    — classic SAGA / stress-responsive
    'TATA-less / TAF1-depleted':         '#8E44AD',  # purple   — TATA-less but SAGA-like
    'TATA-containing / TAF1-enriched':   '#2E8B57',  # sea-green — unusual TATA+TFIID
    ' / ':                               '#C8CDD0',  # lt gray  — no classification
    # Legacy / fallback keys
    'SAGA':    '#E07B1A',
    'TATA':    '#8E44AD',
    'TFIID':   '#1A7A8A',
    'Other':   '#7D8D9A',
    'Unknown': '#C8CDD0',
}

# Short display labels for long tata_taf_clean strings
SHORT_LABELS = {
    'TATA-less / TAF1-enriched':        'TATA-less\nTAF1-enriched',
    'TATA-containing / TAF1-depleted':  'TATA-cont.\nTAF1-depleted',
    'TATA-less / TAF1-depleted':        'TATA-less\nTAF1-depleted',
    'TATA-containing / TAF1-enriched':  'TATA-cont.\nTAF1-enriched',
    ' / ':                              'Not\nclassified',
}
def short_label(s):
    return SHORT_LABELS.get(str(s), str(s))

def prom_color(label):
    s = str(label)
    if s in PROM_COLORS:          # exact match first
        return PROM_COLORS[s]
    for k in PROM_COLORS:         # substring fallback
        if k.lower() in s.lower():
            return PROM_COLORS[k]
    return '#C8CDD0'

# ════════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    height_ratios=[1, 1],
    hspace=0.46, wspace=0.44,   # extra wspace so Panel A legend fits between columns
    left=0.08, right=0.93, top=0.93, bottom=0.07,
)

# ── Panel A: Promoter class × dwell class ────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

if dwell_col and prom_col:
    cross = pd.crosstab(df_prom[dwell_col], df_prom[prom_col])
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

    DWELL_ORDER_PREF = [
        'Universal (6-8 stages)', 'Broad (4-5 stages)',
        'Early-specific', 'Processing-specific',
        'Export/surveillance-specific', 'Late-pathway (3+ consec.)',
        'Stage-specific (other)', 'Unenriched',
    ]
    dwell_order = [c for c in DWELL_ORDER_PREF if c in cross_pct.index]
    # Add any remaining classes not in the preferred list
    dwell_order += [c for c in cross_pct.index if c not in dwell_order]

    x_a = np.arange(len(dwell_order))
    bottom_a = np.zeros(len(dwell_order))
    for col in cross_pct.columns:
        vals = cross_pct.loc[dwell_order, col].fillna(0).values
        ax_a.bar(x_a, vals, bottom=bottom_a, width=0.7,
                 color=prom_color(col), edgecolor='gray', linewidth=0.3, label=col)
        bottom_a += vals
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(dwell_order, fontsize=8, rotation=35, ha='right')
    ax_a.set_ylabel('% of genes', fontsize=8)
    ax_a.set_ylim(0, 100)
    ax_a.set_title('Promoter class distribution\nacross nuclear dwell-score classes',
                   fontsize=9)
    handles_a = [mpatches.Patch(color=prom_color(c), label=c)
                 for c in cross_pct.columns]
    ax_a.legend(handles=handles_a, fontsize=7.5,
                loc='center left', bbox_to_anchor=(1.02, 0.5),
                framealpha=0.9, title='Promoter class', title_fontsize=7.5)
else:
    # Fallback: use TBP column if present
    tbp_col = 'TBP' if 'TBP' in df_prom.columns else None
    if tbp_col and dwell_col:
        cross = pd.crosstab(df_prom[dwell_col], df_prom[tbp_col].fillna('Unknown'))
        cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
        dwell_order = list(cross_pct.index)
        x_a = np.arange(len(dwell_order))
        bottom_a = np.zeros(len(dwell_order))
        for col in cross_pct.columns:
            vals = cross_pct.loc[dwell_order, col].fillna(0).values
            ax_a.bar(x_a, vals, bottom=bottom_a, width=0.7,
                     color=prom_color(str(col)), edgecolor='gray',
                     linewidth=0.3, label=str(col))
            bottom_a += vals
        ax_a.set_xticks(x_a)
        ax_a.set_xticklabels(dwell_order, fontsize=8, rotation=35, ha='right')
        ax_a.set_ylabel('% of genes', fontsize=8)
        ax_a.set_ylim(0, 100)
        ax_a.set_title('Promoter TBP usage\nacross dwell-score classes', fontsize=9)
        ax_a.legend(fontsize=7.5, loc='center left', bbox_to_anchor=(1.02, 0.5),
                    framealpha=0.9, title='TBP binding', title_fontsize=7.5)
    else:
        ax_a.text(0.5, 0.5, 'Promoter class data\nnot available',
                  ha='center', va='center', transform=ax_a.transAxes)
        ax_a.axis('off')

ax_a.spines['top'].set_visible(False); ax_a.spines['right'].set_visible(False)
ax_a.text(-0.12, 1.06, 'A', transform=ax_a.transAxes, fontsize=14, fontweight='bold')

# ── Panel B: Dwell score by promoter class (violins) ─────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

if prom_col and 'dwell_auc' in df_prom.columns:
    prom_classes = df_prom[prom_col].dropna().unique()
    dwell_by_prom = {pc: df_prom[df_prom[prom_col] == pc]['dwell_auc'].dropna().values
                     for pc in prom_classes
                     if len(df_prom[df_prom[prom_col] == pc]) > 5
                     and str(pc).strip() != '/'}   # exclude no-data category
    prom_order = sorted(dwell_by_prom.keys(), key=lambda p: np.median(dwell_by_prom[p]),
                        reverse=True)
    vdata_b = [dwell_by_prom[p] for p in prom_order]
    pcolors_b = [prom_color(p) for p in prom_order]
    if len(vdata_b) > 0:
        parts_b = ax_b.violinplot(vdata_b, positions=range(len(prom_order)),
                                   showmedians=True, showextrema=False, widths=0.65)
        for pc, col in zip(parts_b['bodies'], pcolors_b):
            pc.set_facecolor(col); pc.set_alpha(0.7); pc.set_edgecolor('gray')
        parts_b['cmedians'].set_color('black'); parts_b['cmedians'].set_linewidth(1.5)
        ax_b.set_xticks(range(len(prom_order)))
        ax_b.set_xticklabels([short_label(p) for p in prom_order],
                              fontsize=8, rotation=35, ha='right')
        # Mann-Whitney U between first two groups
        if len(prom_order) >= 2:
            u, p = stats.mannwhitneyu(dwell_by_prom[prom_order[0]],
                                       dwell_by_prom[prom_order[1]],
                                       alternative='two-sided')
            ax_b.set_title(f'Dwell score by promoter class\n'
                           f'(top-2 MWU p = {p:.2e})', fontsize=9)
        else:
            ax_b.set_title('Dwell score by promoter class', fontsize=9)
elif 'TBP' in df_prom.columns and 'dwell_auc' in df_prom.columns:
    tbp_vals = df_prom['TBP'].dropna().unique()
    dwell_by_tbp = {t: df_prom[df_prom['TBP'] == t]['dwell_auc'].dropna().values
                    for t in tbp_vals if len(df_prom[df_prom['TBP'] == t]) > 5}
    prom_order = sorted(dwell_by_tbp.keys(),
                        key=lambda t: np.median(dwell_by_tbp[t]), reverse=True)
    vdata_b = [dwell_by_tbp[t] for t in prom_order]
    pcolors_b = [prom_color(str(t)) for t in prom_order]
    if vdata_b:
        parts_b = ax_b.violinplot(vdata_b, positions=range(len(prom_order)),
                                   showmedians=True, showextrema=False, widths=0.65)
        for pc, col in zip(parts_b['bodies'], pcolors_b):
            pc.set_facecolor(col); pc.set_alpha(0.7); pc.set_edgecolor('gray')
        parts_b['cmedians'].set_color('black'); parts_b['cmedians'].set_linewidth(1.5)
        ax_b.set_xticks(range(len(prom_order)))
        ax_b.set_xticklabels([str(t) for t in prom_order],
                              fontsize=8, rotation=35, ha='right')
    ax_b.set_title('Dwell score by TBP binding class', fontsize=9)
else:
    ax_b.text(0.5, 0.5, 'Data not available', ha='center', va='center',
              transform=ax_b.transAxes); ax_b.axis('off')

ax_b.set_ylabel('Dwell score (AUC)', fontsize=8)
ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)
ax_b.text(-0.14, 1.06, 'B', transform=ax_b.transAxes, fontsize=14, fontweight='bold')

# ── Panel C: CID boundary strength vs. mean dwell score ──────────────────────
ax_c = fig.add_subplot(gs[1, 0])
df_cid_c = df_cid.dropna(subset=['strength', 'mean_dwell_nearby'])
ax_c.scatter(df_cid_c['strength'], df_cid_c['mean_dwell_nearby'],
             c='#5B8DB8', s=12, alpha=0.5, linewidths=0, rasterized=True)
# Regression
if len(df_cid_c) > 5:
    sl, ic, r, p, se = stats.linregress(df_cid_c['strength'],
                                         df_cid_c['mean_dwell_nearby'])
    x_fit = np.linspace(df_cid_c['strength'].min(), df_cid_c['strength'].max(), 100)
    ax_c.plot(x_fit, sl*x_fit + ic, 'k--', linewidth=1.2,
              label=f'r = {r:.2f}  p = {p:.2e}')
    ax_c.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
ax_c.set_xlabel('CID boundary insulation strength', fontsize=8)
ax_c.set_ylabel('Mean dwell score of nearby genes', fontsize=8)
ax_c.set_title('CID boundary strength vs. local mRNP dwell score\n'
               '(each point = one Micro-C domain boundary)', fontsize=9)
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)
ax_c.text(-0.12, 1.06, 'C', transform=ax_c.transAxes, fontsize=14, fontweight='bold')

# ── Panel D: CID SAGA/TFIID fraction vs. nearby dwell score ──────────────────
ax_d = fig.add_subplot(gs[1, 1])
if 'saga_frac' in df_cid.columns and 'mean_dwell_nearby' in df_cid.columns:
    df_cid_d = df_cid.dropna(subset=['saga_frac', 'tfiid_frac', 'mean_dwell_nearby'])
    scatter_d = ax_d.scatter(df_cid_d['saga_frac'], df_cid_d['mean_dwell_nearby'],
                             c=df_cid_d['tfiid_frac'], cmap='RdBu_r',
                             vmin=0, vmax=1, s=12, alpha=0.6,
                             linewidths=0, rasterized=True)
    plt.colorbar(scatter_d, ax=ax_d, label='Fraction TFIID-regulated genes nearby',
                 shrink=0.7)
    ax_d.set_xlabel('Fraction SAGA-regulated genes nearby', fontsize=8)
    ax_d.set_ylabel('Mean dwell score of nearby genes', fontsize=8)
    ax_d.set_title('CID boundary promoter composition vs. dwell score\n'
                   '(SAGA fraction on x; TFIID fraction as color)', fontsize=9)
    if len(df_cid_d) > 5:
        sl, ic, r, p, _ = stats.linregress(df_cid_d['saga_frac'],
                                             df_cid_d['mean_dwell_nearby'])
        ax_d.set_xlabel(f'SAGA fraction  (r = {r:.2f}, p = {p:.2e})', fontsize=8)
else:
    # Fallback: dwell score per RNA cluster
    df_cl = df_clst[['cluster']].join(
        pd.read_csv(DATADIR + "rna_dwell_scores.tsv", sep='\t', index_col=0)[['dwell_auc']],
        how='inner').dropna()
    df_cl['cluster'] = df_cl['cluster'].astype(int)
    vdata_d = [df_cl[df_cl['cluster'] == k]['dwell_auc'].values for k in CLUSTER_ORDER]
    parts_d = ax_d.violinplot(vdata_d, positions=range(len(CLUSTER_ORDER)),
                               showmedians=True, showextrema=False, widths=0.7)
    for pc, k in zip(parts_d['bodies'], CLUSTER_ORDER):
        pc.set_facecolor(CLUSTER_COLORS[k]); pc.set_alpha(0.7); pc.set_edgecolor('gray')
    parts_d['cmedians'].set_color('black'); parts_d['cmedians'].set_linewidth(1.5)
    ax_d.set_xticks(range(len(CLUSTER_ORDER)))
    ax_d.set_xticklabels([CLUSTER_LABELS[k].split('\n')[0] for k in CLUSTER_ORDER],
                          fontsize=7, rotation=40, ha='right')
    ax_d.set_ylabel('Dwell score (AUC)', fontsize=8)
    ax_d.set_title('Dwell score by RNA cluster (k=8)', fontsize=9)

ax_d.spines['top'].set_visible(False); ax_d.spines['right'].set_visible(False)
ax_d.text(-0.14, 1.06, 'D', transform=ax_d.transAxes, fontsize=14, fontweight='bold')

fig.suptitle('Supplementary Figure  |  Genomic and Chromatin Context of mRNP Co-purification\n'
             '(promoter class, dwell scores, CID boundaries, Micro-C domain context)',
             fontsize=11, fontweight='bold', y=0.975)

out = OUTDIR + "figureS_GenomeContext.pdf"
fig.savefig(out, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"Saved: {out}")

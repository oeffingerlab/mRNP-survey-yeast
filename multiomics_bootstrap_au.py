#!/usr/bin/env python3
"""
Multi-omics bait clustering — robustness by multiscale bootstrap (pvclust-style AU).
Reproduces the 18-bait AVERAGED-correlation analysis behind Figure 3A / Supplemental Figure 6A.

Method (matches the figures): compute a bait-bait Pearson correlation independently for
AP-MS (over the nuclear preys) and AP-RNA-seq (over co-enriched transcripts), average the two
matrices, cluster (1 - r distance, average linkage). Robustness = multiscale bootstrap AU
(Shimodaira 2004), resampling the features of BOTH datasets and re-averaging.

Inputs (in ./data or same dir): apms_nuclear_matrix.tsv, rnaseq_bait_means.tsv
Output: data/multiomics_module_AU.tsv
Run:  python3 multiomics_bootstrap_au.py
"""
import os
import sys, glob
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import norm

def find(name):
    for p in [name, "data/"+name] + glob.glob("**/"+name, recursive=True):
        try: open(p); return p
        except: pass
    sys.exit("missing "+name)

apms = pd.read_csv(find("apms_nuclear_matrix.tsv"), sep='\t', index_col=0).drop(columns=['gene_name','MTR2'], errors='ignore')
rna  = pd.read_csv(find("rnaseq_bait_means.tsv"),  sep='\t', index_col=0)
shared = [b for b in rna.columns if b in apms.columns]; n = len(shared)
P = apms[shared].values                      # nuclear preys x 18
R = rna[rna.max(axis=1) > 0.3][shared].values  # co-enriched transcripts x 18
print(f"{n} shared baits | {P.shape[0]} AP-MS preys | {R.shape[0]} transcripts")

def nc(X):
    c = np.corrcoef(X.T); return np.nan_to_num(c, nan=0.0)
def link_avg(P, R):
    cm = (nc(P) + nc(R)) / 2.0
    D = 1 - cm; np.fill_diagonal(D, 0); D = (D + D.T) / 2
    return linkage(squareform(D, checks=False), 'average')
def edge_sets(Z):
    _, nodes = to_tree(Z, rd=True); out = set()
    for nd in nodes:
        if not nd.is_leaf():
            out.add(frozenset(shared[i] for i in nd.pre_order(lambda x: x.id)))
    return out

rng = np.random.default_rng(11)
Z0 = link_avg(P, R)
obs = [s for s in edge_sets(Z0) if 2 <= len(s) < n]
scales = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]; NB = 1200
bp = {fs: {} for fs in obs}
for r in scales:
    npm, nrm = max(2,int(round(r*P.shape[0]))), max(2,int(round(r*R.shape[0]))); cnt = {fs:0 for fs in obs}
    for _ in range(NB):
        es = edge_sets(link_avg(P[rng.integers(0,P.shape[0],npm)], R[rng.integers(0,R.shape[0],nrm)]))
        for fs in obs:
            if fs in es: cnt[fs] += 1
    for fs in obs: bp[fs][r] = min(max(cnt[fs]/NB, 1e-4), 1-1e-4)

def au_bp(bpr):  # pvclust two-parameter AU model
    rs = np.array(sorted(bpr)); y = norm.ppf(1 - np.array([bpr[r] for r in rs]))
    A = np.column_stack([np.sqrt(rs), 1/np.sqrt(rs)])
    w = NB*norm.pdf(y)**2/np.clip(np.array([bpr[r]*(1-bpr[r]) for r in rs]),1e-6,None)
    W = np.diag(w); a,b = np.linalg.lstsq(A.T@W@A, A.T@W@y, rcond=None)[0]
    return (1-norm.cdf(a-b))*100, (1-norm.cdf(a+b))*100

cut = fcluster(Z0, 4, 'maxclust')
mods4 = set(frozenset([shared[j] for j in range(n) if cut[j]==c]) for c in set(cut))
rows = []
for fs in sorted(obs, key=lambda s: -len(s)):
    au, bpv = au_bp(bp[fs])
    rows.append({"n_baits":len(fs), "AU":round(au,1), "BP":round(bpv,1),
                 "is_4cut_module": fs in mods4, "members":";".join(sorted(fs))})
out = pd.DataFrame(rows).sort_values(["n_baits","AU"], ascending=[False,False])
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"), exist_ok=True)
out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "multiomics_module_AU.tsv"), sep='\t', index=False)
print(out.to_string(index=False))
print("\nwrote data/multiomics_module_AU.tsv")

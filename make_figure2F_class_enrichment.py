"""Figure 2F: enrichment of non-coding RNA classes per bait relative to total RNA.
Inputs: the GEO FPKM tables (data/geo_fpkm/FPKM_{mRNA_CDS,CUT,SUT,XUT,lncRNA}.txt; 2014 TopHat/Cufflinks), which include
six W303 wild-type total-RNA libraries; current SGD GFF for classes in the cds table.
Per feature: replicate-mean FPKM per bait and for W303 total RNA. Per class: median over features detected (>0) in at least
one bait or total-RNA sample. Enrichment of a class with a bait, relative to total RNA:
log2[(class median / mRNA median)_bait / (class median / mRNA median)_total RNA]; 0 = same proportion as in total RNA."""
import os, argparse, pandas as pd, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument("--fpkm",default=os.path.join(HERE,"data","geo_fpkm")+os.sep, help="folder with the GEO FPKM tables (FPKM_<class>.txt)")
ap.add_argument("--gff",default=os.path.join(HERE,"data","saccharomyces_cerevisiae.20260618.gff"))
ap.add_argument("--letter",default="F")
args=ap.parse_args(); FP=args.fpkm; GFF=args.gff
os.makedirs(os.path.join(HERE,"output"),exist_ok=True)
SAMPLES={'CBP80':['33_L14','34_L15','35_L16'],'THP2':['31_L12','32_L13'],'YRA1':['DS-80152_GCCAAT','DS-80165_TGACCA'],
 'SNU71':['37_L18','38_L19'],'SYC1':['Syc1-1_L2.LB8','Syc1-2_L2.LB9','Syc1-3_L2.LB10'],'CLP1':['Clp1-1_L3.LB14','Clp1-2_L3.LB15','Clp1-3_L3.LB16'],
 'REF2':['Ref2-1_L3.LB11','Ref2-2_L3.LB12','Ref2-3_L3.LB13'],'HRP1':['28_L9','29_L10','Nab4-1b_L8.LB23','Nab4-2_L8.LB25_1','Nab4-3_L8.LB27_1'],
 'RRP6':['Rrp6-1_L8.LB1_1','Rrp6-2_L8.LB2_1','Rrp6-3_L8.LB3_1'],'NAB2':['1_L1','2_L2','3_L3','DS-80217_TAGCTT'],'NPL3':['Npl3-1_L3.LB21'],
 'GBP2':['Gbp2-1_L2.LB5','Gbp2-2_L2.LB6','Gbp2-3_L2.LB7'],'SUB2':['25_L6','26_L7','27_L8'],'SAC3':['Sac3-1_L7.LB14_1','Sac3-2_L7.LB15_1'],
 'MEX67':['DS-80146_TTAGGC','DS-80171_ATCACG','DS-80230_CGATGT'],'GFD1':['30_L11'],'DBP5':['Dbp5-1_L2.LB2','Dbp5-2_L2.LB3','Dbp5-3_L2.LB4'],
 'PAB1':['Pab1-1_L8.LB20','Pab1-2_L8.LB21','Pab1-3_L8.LB22_1','DS-80192_GATCAG'],
 'TOTAL':['1A_L1.D701_501','1B_L1.D702_501','2A_L1.D703_501','2B_L1.D704_501','3A_L1.D705_501','3B_L1.D706_501']}
ORDER=['CBP80','THP2','YRA1','SNU71','CLP1','SYC1','REF2','HRP1','PAB1','NAB2','NPL3','GBP2','GFD1','RRP6','SUB2','SAC3','MEX67','DBP5']
def load(f):
    """Read a GEO-format FPKM table: chr, gene, then one column per sample headed 'Bait [lane label]'."""
    d=pd.read_csv(f,sep='\t',index_col=1).drop(columns=['chr'])
    d.columns=[c.split('[')[1].rstrip(']') for c in d.columns]
    return d.apply(pd.to_numeric,errors='coerce')
def means(d): return pd.DataFrame({b:d[s].mean(axis=1) for b,s in SAMPLES.items()})
typ={}
for line in open(GFF):
    if line.startswith('#') or line.count('\t')<8: continue
    f=line.rstrip('\n').split('\t'); a=dict(x.split('=',1) for x in f[8].split(';') if '=' in x)
    for k in ('ID','Name','gene'):
        for v in a.get(k,'').split(','):
            v=v.replace('gene:','').strip()
            if v and v not in typ: typ[v]=f[2]
cds=load(FP+'FPKM_mRNA_CDS.txt'); g=pd.Series(cds.index); t=g.map(lambda x: typ.get(x,''))
orf=t.isin(['gene','transposable_element_gene','pseudogene','blocked_reading_frame']).values
snrna=((t=='snRNA_gene')|g.isin(['LSR1','snR6','snR7-L','snR7-S','snR14','snR19'])).values
snorna=((t=='snoRNA_gene')|g.str.match(r'^snR')).values & ~snrna
C={'mRNA':means(cds[orf]),'snRNA':means(cds[snrna]),'snoRNA':means(cds[snorna])}
for k,f in [('CUT','CUT'),('SUT','SUT'),('XUT','XUT'),('lncRNA','lncRNA')]: C[k]=means(load(FP+f'FPKM_{f}.txt'))
med={k:v[v.max(axis=1)>0].median() for k,v in C.items()}
M=pd.DataFrame(med)                      # rows: baits + TOTAL
ratio=np.log2(M.div(M['mRNA'],axis=0))   # class vs mRNA within each sample group
enr=(ratio-ratio.loc['TOTAL']).loc[ORDER].drop(columns='mRNA')
print('n features:',{k:int((v.max(axis=1)>0).sum()) for k,v in C.items()})
print('\nclass/mRNA (log2) in total RNA:\n',ratio.loc['TOTAL'].round(2).to_string())
print('\nenrichment vs total RNA (log2):\n',enr.round(2).to_string())
enr.to_csv(os.path.join(HERE,'output','figure2F_class_enrichment_vs_totalRNA_log2.tsv'),sep='\t')

# ── plot ──
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica','Arial','DejaVu Sans'],'pdf.fonttype':42})
GROUP_COLORS={"CBC":"#C93A3B","Co-Txp/THO/TREX":"#338FBF","Splicing":"#3D477E","3'end processing":"#56C4CB",
 "Surveillance":"#A87247","TREX-2/NPC":"#78C5A3","Export/pore":"#67BA94"}
GROUP={'CBP80':"CBC",'THP2':"Co-Txp/THO/TREX",'YRA1':"Co-Txp/THO/TREX",'SUB2':"Co-Txp/THO/TREX",'SNU71':"Splicing",
 'CLP1':"3'end processing",'SYC1':"3'end processing",'REF2':"3'end processing",'HRP1':"3'end processing",'PAB1':"3'end processing",
 'NAB2':"3'end processing",'NPL3':"3'end processing",'GBP2':"3'end processing",'GFD1':"3'end processing",'RRP6':"Surveillance",
 'SAC3':"TREX-2/NPC",'MEX67':"Export/pore",'DBP5':"Export/pore"}
E=enr
rows=['CUT','SUT','XUT','snRNA','snoRNA']; M=E[rows].T
baits=list(M.columns); lab=[b[0]+b[1:].lower()+'p' for b in baits]
fig=plt.figure(figsize=(7.2,2.7))
ax=fig.add_axes([0.11,0.30,0.76,0.50]); strip=fig.add_axes([0.11,0.805,0.76,0.035]); cax=fig.add_axes([0.89,0.30,0.012,0.50])
v=4; im=ax.imshow(M.values.clip(-v,v),cmap='RdBu_r',vmin=-v,vmax=v,aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        x=M.values[i,j]; ax.text(j,i,f'{x:.1f}',ha='center',va='center',fontsize=6,color='white' if abs(x)>2.5 else 'black')
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab,rotation=60,ha='right',fontsize=7.5)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows,fontsize=7.5)
strip.imshow([[matplotlib.colors.to_rgb(GROUP_COLORS[GROUP[b]]) for b in baits]],aspect='auto'); strip.axis('off')
cb=fig.colorbar(im,cax=cax,extend='both'); cb.ax.tick_params(labelsize=6.5); cb.set_label('log$_2$ enrichment\nvs total RNA',fontsize=6.5)
fig.suptitle('Co-purification of non-coding RNA classes relative to total RNA, per bait',fontsize=8.5,fontweight='bold',y=0.97)
fig.text(0.01,0.95,args.letter,fontsize=12,fontweight='bold')
out=os.path.join(HERE,'output','figure2F_class_enrichment_vs_totalRNA.pdf'); fig.savefig(out); print('wrote',out)

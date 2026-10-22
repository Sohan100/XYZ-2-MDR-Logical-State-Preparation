import sys
sys.path.insert(0,'paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
u=C.usable(th)
x=u[~u.decoder.str.contains(':')].copy()
lab=x.noise.map(C.xyz2_label)
x['base'],x['sched'],x['L']=[l[0] for l in lab],[l[1] for l in lab],[l[2] for l in lab]
rows=[]
for (b,r),g in x.groupby(['base','rounds']):
    if r not in ['d','1','2','3','5','10']: continue
    n=0; n2=0; tot=0
    for s,gs in g.groupby('sched'):
        best={L:gs[gs.L==L].pth.max() for L in 'XY' if (gs.L==L).any()}
        if len(best)==2:
            w=min(best,key=best.get); n+=(gs.L==w).sum(); n2+=(gs.L==w).sum()
        else:
            n2+=len(gs)
        tot+=len(gs)
    rows.append((b,r,n,n2,tot))
df=pd.DataFrame(rows,columns=['base','r','weak_pool','weak_or_single','all'])
xt=df[df.base.isin(['h2_p','helios_p'])]
print(xt.to_string())
nx=df[~df.base.isin(['h2_p','helios_p'])]
print(len(nx), nx.weak_pool.min(), nx.weak_pool.median(), nx.weak_pool.max(), nx.weak_or_single.min(), nx.weak_or_single.max())
print(sorted(df.base.unique()))

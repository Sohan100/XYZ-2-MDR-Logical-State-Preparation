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
    n=0; tot=0
    for s,gs in g.groupby('sched'):
        best={L:gs[gs.L==L].pth.max() for L in 'XY' if (gs.L==L).any()}
        if len(best)==2:
            w=min(best,key=best.get); n+=(gs.L==w).sum()
        tot+=len(gs)
    rows.append((b,r,n,tot))
df=pd.DataFrame(rows,columns=['base','r','weak_pool','all'])
print(df.groupby('r').weak_pool.describe())
print(df[df.base.isin(['sd6','si1000','phen','phen_b10','em3','biased100'])].to_string())
c=u[u.decoder.str.contains(':')].copy()
c['code']=c.decoder.str.split('-').str[0]
print(c.groupby(['noise','rounds']).size().describe())

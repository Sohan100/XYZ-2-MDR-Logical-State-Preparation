import sys
sys.path.insert(0,'paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
t=pd.read_csv('docs/data/campaign/competitors_table.csv'); t['rounds']=t.rounds.astype(str)
t=t[t.ratio.notna()]
u=C.usable(th)
x=u[~u.decoder.str.contains(':')].copy()
lab=x.noise.map(C.xyz2_label)
x['base'],x['sched'],x['L']=[l[0] for l in lab],[l[1] for l in lab],[l[2] for l in lab]
xt=('h2_p','helios_p')
res=[]
for row in t.itertuples():
    g=x[(x.base==row.noise)&(x.rounds==row.rounds)]
    n=0
    for s,gs in g.groupby('sched'):
        best={L:gs[gs.L==L].pth.max() for L in 'XY' if (gs.L==L).any()}
        if len(best)==2:
            w=min(best,key=best.get); n+=(gs.L==w).sum()
        else: n+=len(gs)
    # competitor candidates: weaker-basis usable fits per code over hardware
    srcs=[row.noise]+C.COMPETITOR_HARDWARE.get(row.noise,[])
    c=u[u.decoder.str.contains(':')&u.noise.isin(srcs)&(u.rounds==row.rounds)].copy()
    c['code']=c.decoder.str.split('-').str[0]; c['b']=c.decoder.str.split(':').str[0].str.split('-').str[1]
    nc=0
    for (code,nz),gc in c.groupby(['code','noise']):
        if code=='honeycomb' and row.noise not in ('em3','hyb','hyb_lr2'): pass
        best={b:gc[gc.b==b].pth.max() for b in 'XZ' if (gc.b==b).any()}
        if len(best)==2:
            w=min(best,key=best.get); nc+=(gc.b==w).sum()
    res.append((row.noise,row.rounds,n,nc,row.noise in xt, row.xyz2_dec))
d=pd.DataFrame(res,columns=['noise','r','nx','nc','xt','dec'])
print(d[~d.xt].nx.describe(), d[d.xt].nx.describe())
print(d[~d.xt].nc.describe())
print(d.groupby('noise').nc.agg(['min','max']))
# decoder used
dec=d.dec.str.split(',').str[0]
print(dec.value_counts())

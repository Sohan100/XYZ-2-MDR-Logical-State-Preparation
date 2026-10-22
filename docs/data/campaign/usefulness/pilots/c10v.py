import sys, pandas as pd, numpy as np
sys.path.insert(0,'paper/analysis')
from competitor_report import usable
th=usable(pd.read_csv('docs/data/campaign/thresholds.csv'))
th=th[th.rounds.astype(str)=='1']
def g(n,lab):
    r=th[(th.noise==n)&(th.decoder==lab)]
    return None if r.empty else (100*r.pth.iloc[0],100*r.err.iloc[0],100*r.stat.iloc[0],r.nu.iloc[0])
models=['hyb','hyb_lr2','sd6_lr2','sd6_lr3','si1000_lr2','phen','sd6_il','phen_b10']
hw={'hyb_lr2':['em3','sd6'],'hyb':['em3','sd6'],'sd6_lr2':['sd6'],'sd6_lr3':['sd6'],'si1000_lr2':['si1000'],'sd6_il':['sd6']}
for m in models:
    x=th[(th.noise==m+'@depth6-X')]
    if x.empty: print(m,'none');continue
    b=x.loc[x.pth.idxmax()]; xv,xe,xs=100*b.pth,100*b.err,100*b.stat
    comps=th[th.noise.isin(hw.get(m,[m]))&th.decoder.str.contains(':')]
    best={}
    for _,c in comps.iterrows():
        k=c.decoder.split(':')[0]
        if k not in best or c.pth>best[k][0]: best[k]=(c.pth,c.err,c.stat,c.decoder,c.noise)
    zs=[]
    for k,(p,e,s,d,n) in sorted(best.items()):
        zs.append((k,round(100*p,3),round((xv-100*p)/np.hypot(xe,100*e),2),round((xv-100*p)/np.hypot(xs,100*s),2),d.split(':')[1]))
    print(m,b.decoder,round(xv,3),round(xe,3),'nu',round(b.nu,2)); 
    for z in zs: print('   ',z)

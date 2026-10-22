import pandas as pd, numpy as np
B='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
th=pd.read_csv(B+'thresholds.csv'); th['rounds']=th.rounds.astype(str)
u=th[th.pth.notna()&(th.flag.fillna('')!='drift')&(th.dmax>=11)&(th.method=='fss')]
x=u[~u.decoder.str.contains(':')].copy()
def lab(n):
    if '@' not in n: return n,'depth6','X'
    b,r=n.split('@',1); s,L=r.rsplit('-',1); return b,s,L
L=x.noise.map(lab); x['base']=[a[0] for a in L]; x['sched']=[a[1] for a in L]; x['L']=[a[2] for a in L]
c=u[u.decoder.str.contains(':')].copy()
c['code']=c.decoder.str.split('-').str[0]; c['basis']=c.decoder.str.split(':').str[0].str.split('-').str[1]
out=[]
R=['d','1','2','3','5','10']
HW={"hyb": ["sd6", "em3"], "hyb_lr2": ["sd6", "em3"], "sd6_lr2": ["sd6"], "sd6_lr3": ["sd6"],"si1000_lr2": ["si1000"], "sd6_il": ["sd6"]}
for (b,r),g in x[x.rounds.isin(R)].groupby(['base','rounds']):
    nweak=0; ntot=len(g)
    for s,gs in g.groupby('sched'):
        best={l:gs[gs.L==l].pth.max() for l in 'XY' if (gs.L==l).any()}
        if len(best)==2:
            l=min(best,key=best.get); nweak+=(gs.L==l).sum()
    src=[b]+HW.get(b,[])
    gc=c[(c.rounds==r)&c.noise.isin(src)]
    ncw=0
    for (n,code),gg in gc.groupby(['noise','code']):
        best={l:gg[gg.basis==l].pth.max() for l in 'XZ' if (gg.basis==l).any()}
        if len(best)==2:
            l=min(best,key=best.get); ncw+=(gg.basis==l).sum()
    out.append(dict(base=b,r=r,xyz_weak_pool=nweak,xyz_total=ntot,comp_weak_pool=ncw,comp_total=len(gc)))
d=pd.DataFrame(out)
print(d.describe().round(1).to_string())
print(d[d.r=='d'].to_string())

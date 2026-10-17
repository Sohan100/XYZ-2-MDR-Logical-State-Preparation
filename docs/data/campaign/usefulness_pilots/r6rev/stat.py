import sys, numpy as np, pandas as pd
sys.path.insert(0,'/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/paper/analysis')
import competitor_report as C
th=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
th['rounds']=th.rounds.astype(str)
comp=th[th.decoder.str.contains(':')]
cv=C.code_values(comp)
xs=C.xyz2_values(th)
# need stat errors: re-find rows
def statof(noise,rk,code_row):
    pass
# map xyz2 entries to stat
u=C.usable(th)
def xstat(x):
    sched,L=x.basis.split()[0],x.basis.split()[1][2]
    name=x.noise if (sched=='depth6' and L=='X') else f"{x.noise}@{sched}-{L}"
    r=u[(u.noise==name)&(u.rounds==x.rounds)&(u.decoder==x.decoder)]
    return float(r.stat.iloc[0])
def cstat(c,noise,rk):
    srcn=c.noise
    r=u[(u.noise==srcn)&(u.rounds==rk)&(u.decoder==f"{c.code}-{c.basis.split(',')[0]}:{c.decoder}")]
    return float(r.stat.iloc[0])
for n,rk in [('si1000','1'),('si1000_lr2','1'),('si1000_lr2','2'),('si1000_lr2','3'),('si1000_lr2','5'),('si1000_lr2','10'),('sd6_il','2'),('sd6_il','5'),('sd6_il','10'),('h2_p_noxt','3'),('si1000','2'),('si1000','5'),('si1000','3'),('si1000','10'),('sd6_il','1')]:
    x=xs[(xs.noise==n)&(xs.rounds==rk)].iloc[0]
    xst=xstat(x)
    cf=C.competitor_for(cv,n,rk)
    zz=[]
    for c,r in cf.items():
        if c=='honeycomb': continue
        try: s=cstat(r,n,rk)
        except Exception as e: s=np.nan
        zz.append((c,round((x.pth-r.pth)/np.hypot(xst,s),2)))
    print(n,rk,x.decoder,x.basis,round(xst*100,4),zz)

import sys, numpy as np, pandas as pd
sys.path.insert(0,'/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/paper/analysis')
import competitor_report as C
th=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
th['rounds']=th.rounds.astype(str)
comp=th[th.decoder.str.contains(':')]
xyz=C.xyz2_values(th); cv=C.code_values(comp)
rows=[]
for (noise,rk) in sorted(set(zip(xyz.noise,xyz.rounds))):
    x=xyz[(xyz.noise==noise)&(xyz.rounds==rk)].iloc[0]
    if pd.isna(x.get('pth')): continue
    cf=C.competitor_for(cv,noise,rk)
    zs={}
    for c,r in cf.items():
        if c=='honeycomb' and noise not in ('hyb','hyb_lr2','em3'): pass
        zs[c]=(x.pth-r.pth)/np.hypot(x.err,r.err)
    if not cf: continue
    best=max(cf,key=lambda c:cf[c].pth)
    zmin=min(zs.values()); zbest=zs[best]
    lab='loss' if zmin<=-3 else ('win' if zbest>=3 else 'tie')
    rows.append(dict(noise=noise,r=rk,x=x.pth*100,xe=x.err*100,dec=x.decoder,basis=x.basis,best=best,bp=cf[best].pth*100,ratio=x.pth/cf[best].pth,zmin=zmin,zbest=zbest,lab=lab))
t=pd.DataFrame(rows)
pd.set_option('display.width',250); pd.set_option('display.max_rows',400)
print(t.to_string())
print(t.lab.value_counts())

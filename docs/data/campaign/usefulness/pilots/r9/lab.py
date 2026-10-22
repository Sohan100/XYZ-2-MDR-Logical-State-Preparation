import sys, numpy as np, pandas as pd
sys.path.insert(0,'/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/paper/analysis')
import competitor_report as C
th=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
th['rounds']=th['rounds'].astype(str)
comp=th[th.decoder.str.contains(':')]
xyz=C.xyz2_values(th[~th.decoder.str.contains(":")]); cv=C.code_values(comp)
rows=[]
for (noise,rk),x in xyz.groupby(['noise','rounds']):
    x=x.iloc[0]
    if pd.isna(x.get('pth')): continue
    cf=C.competitor_for(cv,noise,rk)
    if not cf: continue
    zs={}
    for code,r in cf.items():
        if code=='honeycomb' and noise not in ('hyb','hyb_lr2','em3'): continue
        zs[code]=((x.pth-r.pth)/np.hypot(x.err,r.err), r.pth/ x.pth)
    if not zs: continue
    best=max(zs, key=lambda c: cf[c].pth)
    zmin=min(v[0] for v in zs.values())
    lab='loss' if zmin<=-3 else ('win' if zs[best][0]>=3 else 'tie')
    rows.append((noise,rk,round(x.pth*100,4),round(x.err*100,4),x.decoder,x.basis,best,round(cf[best].pth*100,4),round(x.pth/cf[best].pth,3),round(zmin,2),lab))
df=pd.DataFrame(rows,columns='noise r xyz err dec basis best bestv ratio zmin lab'.split())
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
print(df.to_string())
print(df.lab.value_counts())

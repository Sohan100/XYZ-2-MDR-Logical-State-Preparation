import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv')
th['rounds']=th['rounds'].astype(str)
comp_src=th[th.decoder.str.contains(':')]
cv=C.code_values(comp_src)
xv=C.xyz2_values(th)
rows=[]
for (noise,rk),x in xv.groupby(['noise','rounds']):
    x=x.iloc[0]
    if pd.isna(x.get('pth')): continue
    cf=C.competitor_for(cv,noise,rk)
    if not cf: continue
    zs={c:(x.pth-r.pth)/np.hypot(x.err,r.err) for c,r in cf.items()}
    best=max(cf,key=lambda c:cf[c].pth)
    zmin=min(zs.values()); zbest=zs[best]
    lab='loss' if zmin<=-3 else ('win' if zbest>=3 else 'tie')
    rows.append(dict(noise=noise,r=rk,x=x.pth*100,xe=x.err*100,dec=x.decoder,basis=x.basis,best=best,bp=cf[best].pth*100,ratio=x.pth/cf[best].pth,zmin=zmin,zbest=zbest,lab=lab))
t=pd.DataFrame(rows)
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
print(t.to_string())
print(t.lab.value_counts())

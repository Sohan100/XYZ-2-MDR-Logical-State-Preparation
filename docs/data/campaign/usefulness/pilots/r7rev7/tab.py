import sys, numpy as np, pandas as pd
sys.path.insert(0,'paper/analysis')
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv')
src=sys.argv[1] if len(sys.argv)>1 else 'th'
comp_th = th if src=='th' else pd.read_csv('docs/data/campaign/competitors.csv')
xyz=C.xyz2_values(th[~th.decoder.str.contains(':')])
comp=C.code_values(comp_th[comp_th.decoder.str.contains(':')])
rows=[]
for (noise,rk) in sorted(set(zip(xyz.noise,xyz.rounds))):
    x=xyz[(xyz.noise==noise)&(xyz.rounds==rk)].iloc[0]
    if pd.isna(x.get('pth')): continue
    cs=C.competitor_for(comp,noise,rk)
    cs={k:v for k,v in cs.items() if 'pth' in v and pd.notna(v.pth)}
    if not cs: continue
    zs={k:(x.pth-v.pth)/np.hypot(x.err,v.err) for k,v in cs.items()}
    best=max(cs,key=lambda k:cs[k].pth)
    zb=zs[best]
    lab='loss' if min(zs.values())<=-3 else ('win' if zb>=3 else 'tie')
    rows.append(dict(noise=noise,r=rk,ratio=x.pth/cs[best].pth,best=best,zbest=zb,zmin=min(zs.values()),lab=lab,dec=x.decoder,basis=x.basis))
df=pd.DataFrame(rows)
print(len(df), df.lab.value_counts().to_dict(), (df.ratio<1).sum())
pd.set_option('display.width',200); pd.set_option('display.max_rows',300)
print(df[df.lab!='loss'].to_string())
df.to_csv(sys.argv[2] if len(sys.argv)>2 else '/dev/null',index=False)

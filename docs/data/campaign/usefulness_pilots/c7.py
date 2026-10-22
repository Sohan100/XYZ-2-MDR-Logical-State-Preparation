import sys
sys.path.insert(0,'/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/paper/analysis')
import competitor_report as C, pandas as pd, numpy as np
th=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
cv=C.code_values(th[th.decoder.str.contains(":")]); xv=C.xyz2_values(th)
for n in ['biased10','biased100','purez']:
  for r in ['d','1','2','3','5','10']:
    x=xv[(xv.noise==n)&(xv.rounds==r)]; c=cv[(cv.noise==n)&(cv.rounds==r)&(cv.code=='css')]
    if len(x)==0 or len(c)==0: print(n,r,'missing'); continue
    x=x.iloc[0]; c=c.iloc[0]
    z=(x.pth-c.pth)/np.hypot(x.err,c.err)
    # stat only
    u=C.usable(th)
    print(n,r,f"x {x.pth*100:.4f}({x.err*1e6:.0f}) {x.decoder} {x.basis} css {c.pth*100:.4f}({c.err*1e6:.0f}) {c.decoder} {c.basis} ratio {x.pth/c.pth:.3f} z {z:.2f} mw {x.mwpm*100:.3f} {c.mwpm*100:.3f} {x.mwpm/c.mwpm:.3f}")

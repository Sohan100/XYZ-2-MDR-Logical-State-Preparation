import sys, numpy as np, pandas as pd
sys.path.insert(0,'/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/paper/analysis')
import competitor_report as C
pd.set_option('display.width',250)
th=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
th['rounds']=th.rounds.astype(str)
NO=['biased10','biased100','purez']
c=th[th.decoder.str.startswith('css-')&th.noise.isin(NO)]
p=c[c.method!='fss']
print(p[['noise','rounds','decoder','method','pth','err','stat','dmin','dmax','flag']].sort_values(['noise','rounds','decoder']).to_string())
cv=C.code_values(th[th.decoder.str.contains(':')]); cv=cv[cv.code=='css']
xv=C.xyz2_values(th)
for rk in ['d','1','2','3','5','10']:
  for n in NO:
    a=xv[(xv.noise==n)&(xv.rounds==rk)].iloc[0]; b=cv[(cv.noise==n)&(cv.rounds==rk)].iloc[0]
    cand=p[(p.noise==n)&(p.rounds==rk)&(p.dmin==9)&(p.dmax==11)&(p.flag.fillna('')!='drift')&p.pth.notna()&p.decoder.str.startswith('css-X')]
    # need weaker basis: css-Z equivalent? show
    best=cand.sort_values('pth').iloc[-1] if len(cand) else None
    s=f"{rk} {n} fss {b.pth*100:.4f}({b.err*100:.4f})"
    if best is not None and best.pth>b.pth:
      z=(a.pth-best.pth)/np.hypot(a.err,best.err)
      s+=f" -> {best.decoder} {best.pth*100:.4f}({best.err*100:.4f}) z {z:.2f} (zfss {(a.pth-b.pth)/np.hypot(a.err,b.err):.2f})"
    print(s)

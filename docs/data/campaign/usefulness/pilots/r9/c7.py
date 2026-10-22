import sys
sys.path.insert(0,'paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv')
print(th.columns.tolist())
xyz=th[~th.decoder.str.contains(':')]
comp=th[th.decoder.str.contains(':')]
for noise in ['biased10','biased100','purez']:
    x=C.xyz2_values(xyz[(xyz.decoder=='bm')])
    x=x[(x.noise==noise)&(x.rounds=='d')]
    cs=comp[comp.decoder.str.endswith(':bm')]
    cv=C.code_values(cs)
    c=cv[(cv.noise==noise)&(cv.rounds=='d')&(cv.code=='css')]
    p1,e1=float(x.pth.iloc[0]),float(x.err.iloc[0]); p2,e2=float(c.pth.iloc[0]),float(c.err.iloc[0])
    print(noise,x.basis.iloc[0],p1,e1,c.basis.iloc[0],p2,e2,p1/p2,(p1-p2)/np.hypot(e1,e2))

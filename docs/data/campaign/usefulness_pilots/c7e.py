import sys,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
th=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
th['rounds']=th['rounds'].astype(str)
NO=['biased10','biased100','purez']
isx=~th.decoder.str.contains(':')
c=C.code_values(th[~isx])
u=C.usable(th)
for n in NO:
  cr=c[(c.noise==n)&(c.rounds=='d')&(c.code=='css')].iloc[0]
  for dec in sorted(th[isx].decoder.unique()):
    x=C.xyz2_values(th[isx&(th.decoder==dec)])
    xr=x[(x.noise==n)&(x.rounds=='d')]
    if len(xr)==0 or pd.isna(xr.pth.iloc[0]): continue
    xr=xr.iloc[0]
    print(n,dec,xr.basis,f"{xr.pth*100:.4f}({xr.err*1e6:.0f})",f"z={(xr.pth-cr.pth)/np.hypot(xr.err,cr.err):.2f}")

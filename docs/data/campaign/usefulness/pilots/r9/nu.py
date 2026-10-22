import sys, numpy as np, pandas as pd
sys.path.insert(0,'/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/paper/analysis')
import competitor_report as C
th=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
th['rounds']=th['rounds'].astype(str)
u=C.usable(th).copy()
M=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','sd6_lr2','sd6_lr3','si1000_lr2','sd6_il']
u['base']=u.noise.map(lambda n: n.split('@')[0])
u['isc']=u.decoder.str.contains(':')
# competitors: for lr/il models, competitors are those of sd6/si1000 - include competitor rows of base models only (they exist only for base noise)
x=u[(~u.isc)&u.base.isin(M)]
c=u[(u.isc)&u.noise.isin(M)]
for r in ['1','2','3','4','5','6','7','8','9','10','11','15','21','d']:
    print(r, 'xyz', round(x[x.rounds==r].nu.median(),3), len(x[x.rounds==r]), 'comp', round(c[c.rounds==r].nu.median(),3) if len(c[c.rounds==r]) else None, len(c[c.rounds==r]))
xr=x[~x.rounds.isin(['1','2','3','5','10','d'])]
print('xyz-only medians range', xr.groupby('rounds').nu.median().min(), xr.groupby('rounds').nu.median().max())

import sys; sys.path.insert(0,'paper/analysis')
import pandas as pd, numpy as np
import competitor_report as cr
th=pd.read_csv('docs/data/campaign/thresholds.csv')
cp=pd.read_csv('docs/data/campaign/competitors.csv')
x=cr.xyz2_values(th); c=cr.code_values(cp)
ties=[];wins=[];loss=0;over1=[]
for _,r in x.iterrows():
    comp=cr.competitor_for(c,r.noise,r.rounds)
    if not comp or pd.isna(r.get('pth')): continue
    zs={k:(r.pth-v.pth)/np.hypot(r.err,v.err) for k,v in comp.items()}
    best=max(comp.values(),key=lambda v:v.pth)
    zb=(r.pth-best.pth)/np.hypot(r.err,best.err)
    if min(zs.values())<=-3: loss+=1
    elif zb>=3: wins.append((r.noise,r.rounds))
    else: ties.append((r.noise,r.rounds,round(zb,2)))
    if r.pth>best.pth: over1.append((r.noise,r.rounds,r.pth/best.pth))
print(loss,len(ties),wins); print(ties); print(over1)

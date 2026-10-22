import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
comp=C.code_values(th[th.decoder.str.contains(':')])
xt=th[~th.decoder.str.contains(':')]
sched=xt.noise.map(lambda n:C.xyz2_label(n)[1])
def lab(xv,stat=False):
    out={}
    for _,x in xv.iterrows():
        if pd.isna(x.get('pth')): continue
        cf=C.competitor_for(comp,x.noise,x.rounds)
        if not cf: continue
        zs={c:(x.pth-r.pth)/np.hypot(x.err,r.err) for c,r in cf.items()}
        best=max(cf,key=lambda c:cf[c].pth)
        out[(x.noise,x.rounds)]=('loss' if min(zs.values())<=-3 else ('win' if zs[best]>=3 else 'tie'), round(min(zs.values()),2))
    return out
full=lab(C.xyz2_values(xt))
sym=lab(C.xyz2_values(xt[(sched=='both')&xt.decoder.isin(['mwpm','corr_gauge','bm'])]))
import collections
print('full',collections.Counter(v[0] for v in full.values()),len(full))
print('sym',collections.Counter(v[0] for v in sym.values()),len(sym))
print('sym ties',sorted(k for k,v in sym.items() if v[0]=='tie'))
for k in sorted(set(full)&set(sym)):
    if full[k][0]!=sym[k][0]: print('flip',k,full[k],sym[k])

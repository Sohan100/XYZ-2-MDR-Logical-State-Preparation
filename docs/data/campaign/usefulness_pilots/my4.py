import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
u=C.usable(th)
def g(n,d): r=u[(u.noise==n)&(u.decoder==d)&(u.rounds=='1')].iloc[0]; return r
a=g('si1000','corr_gauge')
for d in ['css-Z:bm','xzzx-X:corr','xzzx-Z:bm','xy-Z:bm']:
    b=g('si1000',d); print(d,'z %.2f zstat %.2f'%((a.pth-b.pth)/np.hypot(a.err,b.err),(a.pth-b.pth)/np.hypot(a.stat,b.stat)))
a=g('sd6_lr3@depth6-X','bm'); b=g('sd6','xzzx-Z:corr'); print('lr3 bm z %.2f'%((a.pth-b.pth)/np.hypot(a.err,b.err)))
p=u[(u.noise.str.startswith('phen_b10'))&(u.rounds=='1')&(~u.decoder.str.contains(':'))]
print('phen_b10 xyz2 min/max', p.pth.min()*100, p.pth.max()*100, len(p))
print(u[(u.noise=='phen_b10')&(u.rounds=='1')&(u.decoder.str.startswith('xzzx'))][['decoder','pth','err']].to_string())
a=g('em3','cfe_tn')
for d in ['css-X:bm','xy-Z:bm']:
    b=g('em3',d); print(d, a.pth/b.pth)
a=g('hyb_lr2@depth6-X','tnml'); b=g('em3','honeycomb-Z:mwpm'); print('hyb_lr2 ratio',a.pth/b.pth)

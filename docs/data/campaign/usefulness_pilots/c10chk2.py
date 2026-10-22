import sys, pandas as pd, numpy as np
sys.path.insert(0,'paper/analysis')
from competitor_report import usable
th=pd.read_csv('docs/data/campaign/thresholds.csv')
u=usable(th)
r1=u[u.rounds.astype(str)=='1']
g=r1[(r1.noise=='helios_p')&r1.decoder.str.contains(':')]
print('helios comp nu',g.nu.min(),g.nu.max(),len(g))
def row(noise,dec): 
    x=r1[(r1.noise==noise)&(r1.decoder==dec)]; assert len(x)==1,(noise,dec,len(x)); return x.iloc[0]
def z(a,b): return (a.pth-b.pth)/np.hypot(a.err,b.err),(a.pth-b.pth)/np.hypot(a.stat,b.stat), a.pth/b.pth
def comps(noise):
    g=r1[(r1.noise==noise)&r1.decoder.str.contains(':')]
    return g
# SI1000 matched corr_gauge
a=row('si1000@depth6-X','corr_gauge')
for d in ['css-Z:bm','xzzx-X:corr','xzzx-Z:bm','xy-Z:bm']:
    print('si1000 cg vs',d, z(a,row('si1000',d)))
# all comps per basis best for si1000 vs corr_gauge
# sd6 vs each competitor basis best
for noise,x in [('sd6','cfe_tn'),('helios_p_noxt','tnml'),('h2_p_noxt','tnml')]:
    a=row(noise+'@depth6-X',x)
    g=comps(noise).copy(); g['cb']=g.decoder.str.split(':').str[0]
    best=g.sort_values('pth').groupby('cb').tail(1).sort_values('pth')
    print(noise, x, a.pth, a.err)
    for _,b in best.iterrows(): print('   ',b.decoder, round(b.pth*100,3), round(b.err*100,3), [round(v,2) for v in z(a,b)])
# sd6_lr3
a=row('sd6_lr3@depth6-X','tnml')
g=comps('sd6').copy(); g['cb']=g.decoder.str.split(':').str[0]
best=g.sort_values('pth').groupby('cb').tail(1).sort_values('pth')
for _,b in best.iterrows(): print(' lr3 vs',b.decoder, round(b.pth*100,3), [round(v,2) for v in z(a,b)])
# sd6_il r1
print(r1[r1.noise.str.startswith('sd6_il')][['noise','decoder','pth','err','nu']].sort_values('pth').tail(6))
a=r1[r1.noise=='sd6_il@depth6-X'].sort_values('pth').iloc[-1]; print('sd6_il best',a.decoder,a.pth,a.err,a.nu)
for _,b in best.iterrows(): print(' il vs',b.decoder, round(b.pth*100,3), [round(v,2) for v in z(a,b)])
# phen r1
print(r1[r1.noise.str.startswith('phen@')][['noise','decoder','pth','err','nu']].sort_values('pth').tail(5))
a=r1[r1.noise=='phen@depth6-X'].sort_values('pth').iloc[-1]; print('phen best',a.decoder,a.pth,a.err,a.nu)
g=comps('phen').copy(); g['cb']=g.decoder.str.split(':').str[0]
best=g.sort_values('pth').groupby('cb').tail(1).sort_values('pth')
for _,b in best.iterrows(): print(' phen vs',b.decoder, round(b.pth*100,3), round(b.err*100,3), [round(v,2) for v in z(a,b)])
# sd6_lr2 depth6-Y
print(r1[r1.noise=='sd6_lr2@depth6-Y'][['decoder','pth','err','nu']].sort_values('pth').tail(3))
a=row('sd6_lr2@depth6-Y','tnml'); print('lr2 Y vs xzzx-Z corr', z(a,row('sd6','xzzx-Z:corr')))
g=comps('sd6').copy(); g['cb']=g.decoder.str.split(':').str[0]
best=g.sort_values('pth').groupby('cb').tail(1)
print('lr2 Y vs all', [(b.decoder, round(z(a,b)[0],2)) for _,b in best.iterrows()])
# hyb_lr2 vs honeycomb X bm em3
a=row('hyb_lr2@depth6-X','tnml'); print('hyb_lr2 vs hcX', z(a,row('em3','honeycomb-X:bm')), z(a,row('em3','honeycomb-Z:mwpm')))

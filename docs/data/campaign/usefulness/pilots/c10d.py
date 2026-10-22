import pandas as pd, numpy as np
t=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
u=t[t.pth.notna() & (t.flag.fillna("")!="drift") & (t.dmax>=11) & (t.method=="fss")]
print(sorted(u.rounds.astype(str).unique()))
rd=u[u.rounds.astype(str)=='d']
for m in ['sd6','si1000','helios_p_noxt','h2_p_noxt','sd6_il','hyb_lr2','biased10','em3','biased100','purez']:
    kx = m if m in rd.noise.values and not m.startswith(('sd6_il','hyb')) else m+'@depth6-X'
    gx=rd[(rd.noise==kx)&~rd.decoder.str.contains(':')]; gy=rd[(rd.noise==m+'@depth6-Y')&~rd.decoder.str.contains(':')]
    if len(gx)==0 or len(gy)==0: print(m,'missing',len(gx),len(gy)); continue
    bx=gx.sort_values('pth').iloc[-1]; by=gy.sort_values('pth').iloc[-1]
    print(m, kx, bx.decoder, '%.3f(%.0f)'%(bx.pth*100,bx.err*1e5), by.decoder,'%.3f(%.0f)'%(by.pth*100,by.err*1e5), 'z %.3f'%((bx.pth-by.pth)/np.hypot(bx.err,by.err)))
r1=u[u.rounds.astype(str)=='1']
print(r1[(r1.noise=='phen_b10')][['decoder','pth','err','nu']].sort_values('pth').to_string())

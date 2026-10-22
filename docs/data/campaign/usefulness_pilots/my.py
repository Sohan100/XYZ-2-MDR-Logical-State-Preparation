import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
u=C.usable(th)
comp=u[u.decoder.str.contains(':')].copy()
comp['cb']=comp.decoder.str.split(':').str[0]
for noise in ['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','em3','hyb','hyb_lr2','sd6_lr2','sd6_lr3','si1000_lr2','helios_p','h2_p','phen_b10','phen','sd6_il']:
    x=u[((u.noise==noise)|(u.noise==noise+'@depth6-X'))&(u.rounds=='1')&(~u.decoder.str.contains(':'))]
    if len(x)==0: print(noise,'no xyz2 depth6-X usable'); continue
    b=x.sort_values('pth').iloc[-1]
    srcs=[noise]+C.COMPETITOR_HARDWARE.get(noise,[])
    c=comp[(comp.rounds=='1')&comp.noise.isin(srcs)]
    best=c.sort_values('pth').groupby('cb').tail(1)
    zs=sorted([((b.pth-r.pth)/np.hypot(b.err,r.err),(b.pth-r.pth)/np.hypot(b.stat,r.stat), r.noise,r.decoder,round(r.pth*100,3),round(r.err*100,3),round(r.nu,2)) for _,r in best.iterrows()],key=lambda z:z[0])
    print(noise, b.noise, b.decoder, round(b.pth*100,3),round(b.err*100,3),'nu',round(b.nu,2))
    for z in zs: print('   z=%.2f zstat=%.2f'%z[:2], z[2:])

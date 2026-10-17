import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
u=C.usable(th)
comp=u[u.decoder.str.contains(':')].copy()
parts=comp.decoder.map(C.split); comp['code']=[p[0] for p in parts]; comp['basis']=[p[1] for p in parts]; comp['dec']=[p[2] for p in parts]
for noise in ['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','em3','hyb','hyb_lr2','sd6_lr2','sd6_lr3','si1000_lr2','helios_p','h2_p']:
    x=u[((u.noise==noise)|(u.noise==noise+'@depth6-X'))&(u.rounds=='1')&(~u.decoder.str.contains(':'))]
    if len(x)==0: print(noise,'no xyz2 depth6-X'); continue
    b=x.sort_values('pth').iloc[-1]
    srcs=[noise]+C.COMPETITOR_HARDWARE.get(noise,[])
    c=comp[(comp.rounds=='1')&comp.noise.isin(srcs)]
    best=c.groupby(['code','basis']).apply(lambda g:g.sort_values('pth').iloc[-1])
    zs=[( (b.pth-r.pth)/np.hypot(b.err,r.err), r.noise,r.code,r.basis,r.dec,r.pth*100,r.err*100) for _,r in best.iterrows()]
    zmin=min(zs)
    print(noise, b.decoder, round(b.pth*100,3),round(b.err*100,3),'nu',round(b.nu,2),'| min z',round(zmin[0],2),zmin[1:5],round(zmin[5],3),round(zmin[6],3),'| max comp',round(max(z[5] for z in zs),3))

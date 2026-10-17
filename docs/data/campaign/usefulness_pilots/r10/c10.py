import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv',dtype={'rounds':str})
u=C.usable(th)
comp=u[u.decoder.str.contains(':')].copy()
p=comp.decoder.map(C.split); comp['code']=[a[0] for a in p]; comp['basis']=[a[1] for a in p]
M=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','em3','hyb','hyb_lr2','sd6_lr2','sd6_lr3','si1000_lr2','helios_p','h2_p']
rk=sys.argv[1] if len(sys.argv)>1 else '1'
for m in M:
    x=u[((u.noise==m)|(u.noise==m+"@depth6-X"))&(u.rounds==rk)&(~u.decoder.str.contains(":"))]
    if len(x)==0: print(m,'none'); continue
    b=x.sort_values('pth').iloc[-1]
    srcs=[m]+C.COMPETITOR_HARDWARE.get(m,[])
    c=comp[(comp.noise.isin(srcs))&(comp.rounds==rk)]
    best=c.sort_values('pth').groupby(['noise','code','basis']).tail(1)
    zs=sorted([( (b.pth-r.pth)/np.hypot(b.err,r.err), r.noise,r.code,r.basis,r.decoder.split(':')[1],round(100*r.pth,3)) for r in best.itertuples()])
    lab='loss' if zs[0][0]<=-3 else 'tie'
    print(f"{m:15s} {b.decoder:10s} {100*b.pth:.3f}({100*b.err:.3f}) nu={b.nu:.2f} {lab} zmin={zs[0][0]:.2f} vs {zs[0][1:]} ; zmax={zs[-1][0]:.2f}")

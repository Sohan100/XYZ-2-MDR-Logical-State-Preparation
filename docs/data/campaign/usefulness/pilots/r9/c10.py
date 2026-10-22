import sys, numpy as np, pandas as pd
sys.path.insert(0,'/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/paper/analysis')
import competitor_report as C
th=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
th['rounds']=th['rounds'].astype(str)
u=C.usable(th)
r1=u[u.rounds=='1']
comp=r1[r1.decoder.str.contains(':')].copy()
parts=comp.decoder.map(C.split); comp['code']=[p[0] for p in parts]; comp['basis']=[p[1] for p in parts]; comp['dec']=[p[2] for p in parts]
xyz=r1[~r1.decoder.str.contains(':')].copy()
models=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','em3','hyb','hyb_lr2','sd6_lr2','sd6_lr3','si1000_lr2','helios_p','h2_p','phen','phen_b10','sd6_il']
for m in models:
    lx=xyz[xyz.noise==m] if m in xyz.noise.values else xyz.iloc[:0]
    # depth6 L_X series name: bare noise
    lx=xyz[xyz.noise==m]
    if not len(lx): print(m,'no depth6-X'); continue
    b=lx.sort_values('pth').iloc[-1]
    srcs=[m]+C.COMPETITOR_HARDWARE.get(m,[])
    g=comp[comp.noise.isin(srcs)]
    out=[]
    for (code,basis),gb in g.groupby(['code','basis']):
        r=gb.sort_values('pth').iloc[-1]
        z=(b.pth-r.pth)/np.hypot(b.err,r.err)
        out.append(f"{code}-{basis}:{r.dec}@{r.noise} {r.pth*100:.3f}({r.err*1e5:.0f}) z{z:+.2f} R{b.pth/r.pth:.3f}")
    zmin=min(float(s.split(' z')[1].split()[0]) for s in out) if out else None
    print(f"{m}: {b.decoder} {b.pth*100:.3f}({b.err*1e5:.0f}) nu{b.nu:.2f} zmin {zmin}")
    for s in out: print('   ',s)

import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
cth=th[th.decoder.str.contains(':')]
comp=C.code_values(cth)
# attach stat to comp rows
u=C.usable(cth).copy(); parts=u.decoder.map(C.split); u['code']=[p[0] for p in parts]; u['basis']=[p[1] for p in parts]; u['dec']=[p[2] for p in parts]
xt=th[~th.decoder.str.contains(':')]
xv=C.xyz2_values(xt)
ux=C.usable(xt)
rows=[('si1000','1'),('si1000','2'),('si1000','5'),('si1000','3'),('si1000','10')]+[('si1000_lr2',r) for r in ['1','2','3','5','10']]+[('h2_p_noxt','3'),('sd6_il','2'),('sd6_il','5'),('sd6_il','10'),('sd6_il','1')]
for n,r in rows:
    x=xv[(xv.noise==n)&(xv.rounds==r)].iloc[0]
    # find stat of xyz2 fit
    lab=x.basis.split()[0]; L=x.basis.split()[1][-1]
    nm=n if (lab=='depth6' and L=='X') else f'{n}@{lab}-{L}'
    xs=ux[(ux.noise.isin([nm]))&(ux.rounds==r)&(ux.decoder==x.decoder)]
    if len(xs)==0 and lab=='depth6' and L=='X': xs=ux[(ux.noise==n+'@depth6-X')&(ux.rounds==r)&(ux.decoder==x.decoder)]
    xst=float(xs.stat.iloc[0])
    cf=C.competitor_for(comp,n,r)
    zs=[]
    for c,cr in cf.items():
        src=cr.noise
        m=u[(u.noise==src)&(u.rounds==r)&(u.code==c)&(u.basis==cr.basis.split(',')[0])&(u.dec==cr.decoder)]
        cst=float(m.stat.iloc[0])
        zs.append(((x.pth-cr.pth)/np.hypot(xst,cst),c))
    print(n,r,x.decoder,x.basis,'stat-only zmin',round(min(zs)[0],2),min(zs)[1])

from lab import *
u=C.usable(th)
def st(p): r=u[u.pth==p]; return float(r.stat.iloc[0])
xf=C.xyz2_values(xth)
T={}
for _,a in xf.iterrows():
    rs=C.competitor_for(comp,a.noise,a.rounds)
    rs={k:v for k,v in rs.items() if pd.notna(v.get('pth',np.nan))}
    if not rs or pd.isna(a.pth): continue
    top=max(rs,key=lambda k:rs[k].pth); v=rs[top]
    R=a.pth/v.pth
    e=R*np.hypot(a.err/a.pth,v.err/v.pth); es=R*np.hypot(st(a.pth)/a.pth,st(v.pth)/v.pth)
    T[(a.noise,a.rounds)]=(R,e,es)
for m in ['sd6','si1000','biased100','purez','helios_p_noxt','h2_p_noxt','biased10']:
    for r1,r2 in [('1','d'),('1','2'),('3','1'),('3','d')]:
        if (m,r1) in T and (m,r2) in T:
            a,b=T[(m,r1)],T[(m,r2)]
            print(m,r1,r2,'%.3f(%.3f) %.3f(%.3f) z=%.2f zs=%.2f'%(a[0],a[1],b[0],b[1],(a[0]-b[0])/np.hypot(a[1],b[1]),(a[0]-b[0])/np.hypot(a[2],b[2])))
# nu medians
M=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','sd6_lr2','sd6_lr3','si1000_lr2','sd6_il']
uu=u.copy()
uu['base']=uu.noise.map(lambda s:s.split('@')[0])
uu['isx']=~uu.decoder.str.contains(':')
uu=uu[uu.base.isin(M)]
for r in ['1','2','3','5','10','d']:
    g=uu[uu.rounds==r]
    print(r, round(g[g.isx].nu.median(),3), round(g[~g.isx].nu.median(),3))
g=uu[uu.isx & ~uu.rounds.isin(['1','2','3','5','10','d'])]
print(g.groupby('rounds').nu.median().agg(['min','max']))
print(g.rounds.unique())

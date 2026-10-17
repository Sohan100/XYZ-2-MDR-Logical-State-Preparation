from lab import *
xf=C.xyz2_values(xth)
cc=comp[comp.code=='css']
u=C.usable(th)
def st(p): r=u[u.pth==p]; return float(r.stat.iloc[0])
for m in ['biased10','biased100','purez']:
    for r in ['1','2','3','5','10','d']:
        a=xf[(xf.noise==m)&(xf.rounds==r)].iloc[0]; c=cc[(cc.noise==m)&(cc.rounds==r)].iloc[0]
        z=(a.pth-c.pth)/np.hypot(a.err,c.err); zs=(a.pth-c.pth)/np.hypot(st(a.pth),st(c.pth))
        print(m,r,'%.3f(%.0f) %s vs css %.3f(%.0f) %s ratio %.3f z %.2f zs %.2f'%(a.pth*100,a.err*1e5,a.decoder,c.pth*100,c.err*1e5,c.decoder,a.pth/c.pth,z,zs))
# mwpm matched at all r
for m in ['biased10','biased100','purez']:
    for r in ['1','2','3','5','10','d']:
        xm=xth[(xth.noise.str.startswith(m))&(xth.rounds==r)&(xth.decoder=='mwpm')]
        xm=C.usable(xm)
        # xyz2 mwpm: per schedule weaker basis then best schedule
        xv=C.xyz2_values(xm)
        cm=C.code_values(C.usable(cth[(cth.noise==m)&(cth.rounds==r)&cth.decoder.str.endswith(':mwpm')]))
        cm=cm[cm.code=='css']
        if len(xv) and len(cm):
            a=xv.iloc[0]; c=cm.iloc[0]
            print('mwpm',m,r,'%.3f(%.1f) vs %.3f(%.1f) ratio %.3f z %.1f'%(a.pth*100,a.err*1e3*100/10,c.pth*100,c.err*1e3*100/10,a.pth/c.pth,(a.pth-c.pth)/np.hypot(a.err,c.err)))

from lab import *
xf=C.xyz2_values(xth)
u=C.usable(cth)
for m in ['biased10','biased100','purez']:
    out=[]
    for r in ['1','2','3','5','10','d']:
        a=xf[(xf.noise==m)&(xf.rounds==r)].iloc[0]
        xy=comp[(comp.noise==m)&(comp.rounds==r)&(comp.code=='xy')].iloc[0]
        # xzzx mwpm weaker basis
        mw=[u[(u.noise==m)&(u.rounds==r)&(u.decoder==f'xzzx-{b}:mwpm')] for b in 'XZ']
        w=min([g.iloc[0] for g in mw if len(g)],key=lambda q:q.pth)
        out.append('r=%s xy %.1f (%s %.3f) xzzxmwpm %.3f z %.2f'%(r,(a.pth-xy.pth)/np.hypot(a.err,xy.err),xy.decoder,xy.pth*100,w.pth*100,(a.pth-w.pth)/np.hypot(a.err,w.err)))
    print(m); print('\n'.join(out))

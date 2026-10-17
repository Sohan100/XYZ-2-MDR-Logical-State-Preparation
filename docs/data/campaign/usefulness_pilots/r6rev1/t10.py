from lab import *
u=C.usable(th)
print(sorted(set(n for n in th.noise.unique() if n.startswith('hyb') or n.startswith('sd6_lr') or n.startswith('si1000_lr'))))
M=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','em3','hyb','hyb_lr2','sd6_lr2','sd6_lr3','si1000_lr2','helios_p','h2_p']
for m in M:
    nm=m if m in ('sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','em3','helios_p','h2_p') else m
    g=u[((u.noise==m)|(u.noise==m+'@depth6-X'))&(u.rounds=='1')&~u.decoder.str.contains(':')]
    if not len(g): print(m,'none'); continue
    b=g.sort_values('pth').iloc[-1]
    srcs=[m]+C.COMPETITOR_HARDWARE.get(m,[])
    cu=u[u.noise.isin(srcs)&(u.rounds=='1')&u.decoder.str.contains(':')].copy()
    cu['cb']=cu.decoder.str.split(':').str[0]
    best=cu.sort_values('pth').groupby('cb').tail(1)
    zs=[( (b.pth-r.pth)/np.hypot(b.err,r.err), r.cb, r.noise, r.pth*100) for _,r in best.iterrows()]
    zmin=min(zs); zmax=max(zs)
    print(m,'%s %.3f(%.0f) nu=%.2f'%(b.decoder,b.pth*100,b.err*1e5,b.nu),'zmin %.2f %s %s %.3f'%zmin,'zmax %.2f %s'%zmax[:2], 'lab', 'loss' if zmin[0]<=-3 else ('win' if all(z[0]>=3 for z in zs) else 'tie'))

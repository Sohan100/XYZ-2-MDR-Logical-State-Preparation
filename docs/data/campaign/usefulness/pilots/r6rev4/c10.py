from lab import *
u=C.usable(th)
M=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','em3','hyb','hyb_lr2','sd6_lr2','sd6_lr3','si1000_lr2','helios_p','h2_p','phen','phen_b10','sd6_il']
for m in M:
    x=u[(u.noise==m)&(u.rounds=='1')&(~u.decoder.str.contains(':'))]
    if len(x)==0: print(m,'none'); continue
    b=x.sort_values('pth').iloc[-1]
    srcs=[m]+C.COMPETITOR_HARDWARE.get(m,[])
    cs=u[(u.noise.isin(srcs))&(u.rounds=='1')&(u.decoder.str.contains(':'))].copy()
    cs['cb']=cs.decoder.str.split(':').str[0]
    best=cs.sort_values('pth').groupby('cb').tail(1)
    zs=[( r.cb, r.noise, round(r.pth*100,3), round((b.pth-r.pth)/np.hypot(b.err,r.err),2)) for _,r in best.iterrows()]
    zmin=min(z[3] for z in zs)
    print(m, b.decoder, round(b.pth*100,3), round(b.err*100,3), 'nu',round(b.nu,2), 'zmin',zmin, 'label','loss' if zmin<=-3 else 'tie/win', sorted(zs,key=lambda z:z[3])[:2])

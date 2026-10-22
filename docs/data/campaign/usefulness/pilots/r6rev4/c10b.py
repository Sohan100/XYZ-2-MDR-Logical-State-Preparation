from lab import *
u=C.usable(th)
M=['hyb','hyb_lr2','sd6_lr2','sd6_lr3','si1000_lr2','phen','phen_b10','sd6_il']
for m in M:
    x=u[(u.noise.isin([m+'@depth6-X'] if m not in ('phen','phen_b10') else [m+'@depth6-X']))&(u.rounds=='1')]
    if len(x)==0: print(m,'none'); continue
    b=x.sort_values('pth').iloc[-1]
    srcs=[m]+C.COMPETITOR_HARDWARE.get(m,[])
    cs=u[(u.noise.isin(srcs))&(u.rounds=='1')&(u.decoder.str.contains(':'))].copy()
    cs['cb']=cs.decoder.str.split(':').str[0]
    best=cs.sort_values('pth').groupby('cb').tail(1)
    zs=[( r.cb, r.noise, round(r.pth*100,3), round((b.pth-r.pth)/np.hypot(b.err,r.err),2)) for _,r in best.iterrows()]
    zmin=min(z[3] for z in zs)
    print(m, b.decoder, round(b.pth*100,3), round(b.err*100,3), 'nu',round(b.nu,2), 'zmin',zmin, 'label','loss' if zmin<=-3 else 'tie/win', sorted(zs,key=lambda z:z[3])[:3])
    print('   ',x.sort_values('pth')[['decoder','pth','err','nu']].tail(4).values.tolist())

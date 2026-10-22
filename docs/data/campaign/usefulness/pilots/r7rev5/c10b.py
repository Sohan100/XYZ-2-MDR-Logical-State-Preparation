from th import *
u=C.usable(TH); u=u[u.rounds.astype(str)=='1']
for m in ['hyb','hyb_lr2','sd6_lr2','sd6_lr3','si1000_lr2','phen','phen_b10','sd6_il']:
    x=u[(u.noise==m+'@depth6-X')&(~u.decoder.str.contains(':'))]
    if len(x)==0: print(m,'none'); continue
    b=x.sort_values('pth').iloc[-1]
    srcs=[m]+C.COMPETITOR_HARDWARE.get(m,[])
    c=u[u.noise.isin(srcs)&u.decoder.str.contains(':')].copy()
    c['cb']=c.decoder.str.split(':').str[0]
    best=c.sort_values('pth').groupby(['cb']).tail(1)
    zs=[(r.cb,r.noise,r.decoder.split(':')[1],round(r.pth*100,3),round((b.pth-r.pth)/np.hypot(b.err,r.err),2), round((b.pth-r.pth)/np.hypot(b.stat,r.stat),2)) for r in best.itertuples()]
    zmin=min(z[4] for z in zs)
    print(m,b.decoder,round(b.pth*100,3),round(b.err*100,3),round(b.nu,2),'min z',zmin, 'LOSS' if zmin<=-3 else 'TIE')
    print('   ',sorted(zs,key=lambda z:z[4]))
    print('   top3',x.sort_values('pth')[['decoder','pth','err','nu']].tail(3).values.tolist())

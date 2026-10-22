from lib import *
u=C.usable(th)
models=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','em3','hyb','hyb_lr2','sd6_lr2','sd6_lr3','si1000_lr2','helios_p','h2_p','phen','phen_b10','sd6_il']
for n in models:
    x=u[(u.noise.isin([n,n+'@depth6-X']))&(u.rounds=='1')&~u.decoder.str.contains(':')]
    if not len(x): print(n,'no usable'); continue
    b=x.sort_values('pth').iloc[-1]
    srcs=[n]+C.COMPETITOR_HARDWARE.get(n,[])
    cc=u[(u.noise.isin(srcs))&(u.rounds=='1')&u.decoder.str.contains(':')].copy()
    cc['cb']=cc.decoder.str.split(':').str[0]
    best=cc.sort_values('pth').groupby('cb').tail(1)
    zs={r.cb+'@'+r.noise+':'+r.decoder.split(':')[1]:(round(r.pth*100,3),round(r.err*100,3),round((b.pth-r.pth)/np.hypot(b.err,r.err),2),round(b.pth/r.pth,3)) for r in best.itertuples()}
    mn=min(v[2] for k,v in zs.items() if not k.startswith('honey'))
    mnall=min(v[2] for v in zs.values())
    lab='loss' if mnall<=-3 else 'tie'
    print(n,b.decoder,round(b.pth*100,3),round(b.err*100,3),'nu',round(b.nu,2),lab,'min(no honey)',mn,'minall',mnall)
    for k,v in sorted(zs.items(), key=lambda kv: kv[1][2]): print('    ',k,v)

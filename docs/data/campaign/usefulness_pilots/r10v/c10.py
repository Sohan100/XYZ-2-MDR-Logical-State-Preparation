exec(open('/tmp/claude-108819/-pscratch-sd-s-sohan100/b9bb5492-172d-4149-9bf3-b4e9300d223c/scratchpad/r10v/lib.py').read())
u=C.usable(th); u=u[u.rounds=='1']
models=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','em3','hyb','hyb_lr2','sd6_lr2','sd6_lr3','si1000_lr2','helios_p','h2_p','phen','phen_b10','sd6_il']
for m in models:
    x=u[u.noise.isin([m,m+'@depth6-X'])&~u.decoder.str.contains(':')]
    if not len(x): print(m,'no xyz2'); continue
    b=x.sort_values('pth').iloc[-1]
    srcs=[m]+C.COMPETITOR_HARDWARE.get(m,[])
    c=u[u.noise.isin(srcs)&u.decoder.str.contains(':')]
    c=c.assign(cb=c.decoder.str.split(':').str[0])
    best=c.sort_values('pth').groupby(['noise','cb']).tail(1)
    zs=[( (b.pth-r.pth)/np.hypot(b.err,r.err), r.noise, r.decoder, r.pth*100,r.err*100) for _,r in best.iterrows()]
    zs.sort(); zs=[z for z in zs if 'honeycomb' not in z[2] or m in ('em3','hyb','hyb_lr2')]
    zmin=zs[0]
    print(f"{m}: {b.decoder} {b.pth*100:.3f}({b.err*100:.3f}) nu={b.nu:.2f}; min z {zmin[0]:.2f} vs {zmin[1]} {zmin[2]} {zmin[3]:.3f}; max z {zs[-1][0]:.2f} {zs[-1][2]}")

exec(open('/tmp/claude-108819/-pscratch-sd-s-sohan100/b9bb5492-172d-4149-9bf3-b4e9300d223c/scratchpad/r10v/lib.py').read())
def ok(t,pairs):
    m=t.pth.notna()&(t.flag.fillna('')!='drift')
    if pairs: return t[m&(((t.method=='fss')&(t.dmax>=11))|((t.method=='pair')))]
    return t[m&(t.method=='fss')&(t.dmax>=11)]
for pairs in [False,True]:
  u=ok(th,pairs)
  for rk in ['1','2','3','5','10']:
    g=u[(u.rounds==rk)]
    xs={}
    for s in ['depth6','both']:
        bas={}
        for L in 'XY':
            nm=['phen@%s-%s'%(s,L)]+(['phen'] if (s,L)==('depth6','X') else [])
            gg=g[g.noise.isin(nm)&~g.decoder.str.contains(':')]
            if len(gg): bas[L]=gg.sort_values('pth').iloc[-1]
        if len(bas)==2: xs[s]=min(bas.values(),key=lambda r:r.pth)
    comp={}
    gc=g[(g.noise=='phen')&g.decoder.str.contains(':')&~g.decoder.str.startswith('honeycomb')]
    for code in ['css','xzzx','xy']:
        bas={}
        for B in 'XZ':
            gg=gc[gc.decoder.str.startswith(f'{code}-{B}:')]
            if len(gg): bas[B]=gg.sort_values('pth').iloc[-1]
        if len(bas)==2: comp[code]=min(bas.values(),key=lambda r:r.pth)
    best=max(xs.values(),key=lambda r:r.pth)
    out=[]
    for s,x in xs.items():
        zz=[(x.pth-c.pth)/np.hypot(x.err,c.err) for c in comp.values()]
        out.append((s,x.decoder,round(x.pth*100,3),round(max(zz,key=abs),2), s==('depth6' if best is xs.get('depth6') else 'both')))
    print('pairs' if pairs else 'fss',rk,out,{k:(v.decoder,round(v.pth*100,3)) for k,v in comp.items()})

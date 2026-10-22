from tab import *
NO=["sd6","si1000","biased10","biased100","purez","em3","helios_p_noxt","h2_p_noxt","helios_p","h2_p","hyb","hyb_lr2","sd6_lr2","sd6_lr3","si1000_lr2","phen","phen_b10","sd6_il"]
SYMX=['mwpm','corr_gauge','bm']; SYMC=['mwpm','corr','bm']
rows=[]
for n in NO:
  for rk in ['1','2','3','5','10','d']:
    x=xyz(n,rk); xs=xyz(n,rk,SYMX,['both'])
    if x is None: continue
    cs={c:comp(n,rk,c) for c in ['css','xzzx','xy']}
    cs={c:v for c,v in cs.items() if v is not None}
    csy={c:comp(n,rk,c,SYMC) for c in ['css','xzzx','xy']}
    csy={c:v for c,v in csy.items() if v is not None}
    if not cs: continue
    top=max(cs, key=lambda c: cs[c].pth); b=cs[top]
    zt=(x.pth-b.pth)/np.hypot(x.err,b.err); zs=(x.pth-b.pth)/np.hypot(x.stat,b.stat)
    row=dict(n=n,r=rk,x=round(x.pth*100,3),xe=round(x.err*100,3),xdec=x.noise+':'+x.decoder,top=top,c=round(b.pth*100,3),ce=round(b.err*100,3),ratio=round(x.pth/b.pth,3),zt=round(zt,2),zs=round(zs,2))
    if xs is not None and csy:
        ts=max(csy, key=lambda c: csy[c].pth); bs=csy[ts]
        row.update(xsym=round(xs.pth*100,3), xsyme=round(xs.err*100,3), xsymdec=xs.noise+':'+xs.decoder, topsym=ts, rsym=round(xs.pth/bs.pth,3), zsym=round((xs.pth-bs.pth)/np.hypot(xs.err,bs.err),2))
    rows.append(row)
R=pd.DataFrame(rows)
print(R.to_string())
R.to_csv('z.csv',index=False)

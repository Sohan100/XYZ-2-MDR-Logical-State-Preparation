import pandas as pd, numpy as np
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/'
th=pd.read_csv(R+'docs/data/campaign/thresholds.csv')
u=th[th.pth.notna() & (th.flag.fillna('')!='drift') & (th.dmax>=11) & (th.method=='fss')].copy()
x=u[~u.decoder.str.contains(':')].copy()
def lab(n):
    if '@' not in n: return n,'depth6','X'
    b,r=n.split('@',1); s,L=r.rsplit('-',1); return b,s,L
x[['base','sched','L']]=pd.DataFrame(x.noise.map(lab).tolist(),index=x.index)
co=u[u.decoder.str.contains(':')].copy()
co['code']=co.decoder.str.split('-').str[0]
co['basis']=co.decoder.str.split('-').str[1].str.split(':').str[0]
co['dec']=co.decoder.str.split(':').str[1]
CH={"hyb": ["sd6", "em3"], "hyb_lr2": ["sd6", "em3"], "sd6_lr2": ["sd6"], "sd6_lr3": ["sd6"],"si1000_lr2": ["si1000"], "sd6_il": ["sd6"]}
def xyzval(base,rk):
    g=x[(x.base==base)&(x.rounds==rk)]
    cands=[];single=[]
    for s,gs in g.groupby('sched'):
        best={L:gs[gs.L==L].sort_values('pth').iloc[-1] for L in 'XY' if (gs.L==L).any()}
        if len(best)==2:
            L,r=min(best.items(),key=lambda kv:kv[1].pth); cands.append((r.pth,r.err,r.stat,s,L,r.decoder,r.nu,r.chi2))
        else:
            single+=[(r.pth,r.err,r.stat,s,L,r.decoder,r.nu,r.chi2) for L,r in best.items()]
    return max(cands) if cands else (max(single) if single else None)
def compval(noise,rk,code):
    best=None
    for n in [noise]+CH.get(noise,[]):
        g=co[(co.noise==n)&(co.rounds==rk)&(co.code==code)]
        b={B:g[g.basis==B].sort_values('pth').iloc[-1] for B in 'XZ' if (g.basis==B).any()}
        if len(b)==2:
            B,r=min(b.items(),key=lambda kv:kv[1].pth)
            if best is None or r.pth>best[0]: best=(r.pth,r.err,r.stat,n,B,r.dec)
    return best
t=pd.read_csv(R+'docs/data/campaign/competitors_table.csv')
t=t[t.ratio.notna()]
rows=[]
for _,r in t.iterrows():
    xv=xyzval(r.noise,str(r.rounds))
    cs={c:compval(r.noise,str(r.rounds),c) for c in ['css','xzzx','xy','honeycomb']}
    cs={c:v for c,v in cs.items() if v}
    top=max(cs,key=lambda c:cs[c][0])
    zt=(xv[0]-cs[top][0])/np.hypot(xv[1],cs[top][1]); zs=(xv[0]-cs[top][0])/np.hypot(xv[2],cs[top][2])
    rows.append(dict(noise=r.noise,r=r.rounds,xyz=round(100*xv[0],3),xe=round(100*xv[1],3),xs=round(100*xv[2],3),xd=f'{xv[5]},{xv[3]}{xv[4]}',nu=round(xv[6],2),top=top,comp=round(100*cs[top][0],3),ce=round(100*cs[top][1],3),ratio=round(xv[0]/cs[top][0],3),tabratio=round(r.ratio,3),zt=round(zt,2),zs=round(zs,2)))
df=pd.DataFrame(rows)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
print(df[(df.zt.abs()<3)].to_string())
print(df[(df.ratio-df.tabratio).abs()>1e-3])

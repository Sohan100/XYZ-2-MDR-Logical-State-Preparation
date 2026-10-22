import pandas as pd, numpy as np, sys
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
XD=set(sys.argv[1].split(',')) if len(sys.argv)>1 else {'mwpm','corr_gauge','bm'}
SC=set(sys.argv[2].split(',')) if len(sys.argv)>2 else {'both'}
CD=set(sys.argv[3].split(',')) if len(sys.argv)>3 else {'mwpm','corr','bm'}
def xyzval(base,rk):
    g=x[(x.base==base)&(x.rounds==rk)&x.decoder.isin(XD)&x.sched.isin(SC)]
    cands=[]
    for s,gs in g.groupby('sched'):
        best={L:gs[gs.L==L].sort_values('pth').iloc[-1] for L in 'XY' if (gs.L==L).any()}
        if len(best)==2:
            L,r=min(best.items(),key=lambda kv:kv[1].pth); cands.append((r.pth,r.err,s,L,r.decoder))
    return max(cands) if cands else None
def compval(noise,rk,code):
    best=None
    for n in [noise]+CH.get(noise,[]):
        g=co[(co.noise==n)&(co.rounds==rk)&(co.code==code)&co.dec.isin(CD)]
        b={B:g[g.basis==B].sort_values('pth').iloc[-1] for B in 'XZ' if (g.basis==B).any()}
        if len(b)==2:
            B,r=min(b.items(),key=lambda kv:kv[1].pth)
            if best is None or r.pth>best[0]: best=(r.pth,r.err,n,B,r.dec)
    return best
t=pd.read_csv(R+'docs/data/campaign/competitors_table.csv')
t=t[t.ratio.notna()]
rows=[]
for _,r in t.iterrows():
    xv=xyzval(r.noise,str(r.rounds))
    cs={c:compval(r.noise,str(r.rounds),c) for c in ['css','xzzx','xy','honeycomb']}
    cs={c:v for c,v in cs.items() if v}
    if not xv or not cs: continue
    top=max(cs,key=lambda c:cs[c][0])
    z=(xv[0]-cs[top][0])/np.hypot(xv[1],cs[top][1])
    rows.append(dict(noise=r.noise,r=r.rounds,xyz=xv[0],xe=xv[1],xd=f'{xv[4]},{xv[2]}{xv[3]}',orig=r.xyz2,drop=1-xv[0]/r.xyz2,top=top,comp=cs[top][0],ce=cs[top][1],cd=f'{cs[top][4]},{cs[top][3]},{cs[top][2]}',ratio=xv[0]/cs[top][0],z=z,origratio=r.ratio))
df=pd.DataFrame(rows)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
print(df.round(4).to_string())

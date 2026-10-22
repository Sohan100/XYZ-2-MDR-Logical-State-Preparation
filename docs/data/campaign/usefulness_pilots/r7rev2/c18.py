import sys, numpy as np, pandas as pd
sys.path.insert(0,'paper/analysis')
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
cp=th[th.decoder.str.contains(':')]
xy=th[~th.decoder.str.contains(':')]
def lab(th): return th.noise.map(C.xyz2_label)
def values(pair_ok, sched=None):
    def U(df):
        u=df[df.pth.notna() & (df.flag.fillna('')!='drift') & (df.dmax>=11)]
        return u if pair_ok else u[u.method=='fss']
    # competitor per code weaker basis best per basis
    for rk in ['1','2','3','5','10','d']:
        c=U(cp[(cp.noise=='phen')&(cp.rounds==rk)]).copy()
        c['code']=c.decoder.str.split('-').str[0]; c['basis']=c.decoder.str.split('-').str[1].str.split(':').str[0]
        comps={}
        for code,g in c.groupby('code'):
            if code=='honeycomb': continue
            b={B:g[g.basis==B].sort_values('pth').iloc[-1] for B in 'XZ' if (g.basis==B).any()}
            if len(b)==2: comps[code]=min(b.values(),key=lambda r:r.pth)
        x=U(xy[(xy.rounds==rk)]).copy()
        l=lab(x); x['base']=[a[0] for a in l]; x['sched']=[a[1] for a in l]; x['L']=[a[2] for a in l]
        x=x[x.base=='phen']
        cands=[]
        for s,g in x.groupby('sched'):
            b={L:g[g.L==L].sort_values('pth').iloc[-1] for L in 'XY' if (g.L==L).any()}
            if len(b)==2: cands.append((s,min(b.values(),key=lambda r:r.pth)))
        if not cands: continue
        best=max(cands,key=lambda t:t[1].pth)
        for s,r in cands:
            zs={k:(r.pth-v.pth)/np.hypot(r.err,v.err) for k,v in comps.items()}
            mz=max(zs.values(),key=abs)
            tag='best' if s==best[0] else 'other'
            print(rk,'pairs' if pair_ok else 'fss',tag,s,r.noise,r.decoder,r.method,round(r.pth*100,3),round(r.err*100,3),{k:round(v,2) for k,v in zs.items()})
for p in [False,True]: values(p)

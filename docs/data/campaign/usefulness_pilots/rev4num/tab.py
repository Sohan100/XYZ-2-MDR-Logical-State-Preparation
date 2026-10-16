import pandas as pd, numpy as np, sys
sys.path.insert(0,'/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/paper/analysis')
pd.set_option('display.width',250); pd.set_option('display.max_rows',500); pd.set_option('display.max_columns',40)
T=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
T['rounds']=T.rounds.astype(str)
U=T[T.pth.notna() & (T.flag.fillna('')!='drift') & (T.dmax>=11) & (T.method=='fss')].copy()
COMP_HW = {"hyb": ["sd6", "em3"], "hyb_lr2": ["sd6", "em3"], "sd6_lr2": ["sd6"], "sd6_lr3": ["sd6"],
           "si1000_lr2": ["si1000"], "sd6_il": ["sd6"], "sd6_il_noidle": ["sd6_noidle"]}
def lab(n):
    if '@' not in n: return n,'depth6','X'
    b,r=n.split('@',1); s,L=r.rsplit('-',1); return b,s,L
X=U[~U.decoder.str.contains(':')].copy()
l=X.noise.map(lab); X['base']=[a[0] for a in l]; X['sched']=[a[1] for a in l]; X['L']=[a[2] for a in l]
C=U[U.decoder.str.contains(':')].copy()
sp=C.decoder.str.split(':'); C['cb']=[a[0] for a in sp]; C['dec']=[a[1] for a in sp]
C['code']=C.cb.str.split('-').str[0]; C['basis']=C.cb.str.split('-').str[1]
def xyz(base, rk, decs=None, scheds=None):
    g=X[(X.base==base)&(X.rounds==rk)]
    if decs: g=g[g.decoder.isin(decs)]
    if scheds: g=g[g.sched.isin(scheds)]
    best=None
    for s,gs in g.groupby('sched'):
        bb={L:gs[gs.L==L].sort_values('pth').iloc[-1] for L in 'XY' if (gs.L==L).any()}
        if len(bb)==2:
            L,r=min(bb.items(), key=lambda kv: kv[1].pth)
            if best is None or r.pth>best.pth: best=r
    return best
def comp(noise, rk, code, decs=None):
    srcs=[noise]+COMP_HW.get(noise,[])
    g=C[(C.noise.isin(srcs))&(C.rounds==rk)&(C.code==code)]
    if decs: g=g[g.dec.isin(decs)]
    best=None
    for n,gn in g.groupby('noise'):
        bb={b:gn[gn.basis==b].sort_values('pth').iloc[-1] for b in 'XZ' if (gn.basis==b).any()}
        if len(bb)==2:
            r=min(bb.values(), key=lambda r: r.pth)
            if best is None or r.pth>best.pth: best=r
    return best

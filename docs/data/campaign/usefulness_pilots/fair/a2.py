import pandas as pd, numpy as np
B='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
th=pd.read_csv(B+'thresholds.csv'); th['rounds']=th.rounds.astype(str)
t=pd.read_csv(B+'competitors_table.csv'); t['rounds']=t.rounds.astype(str)
u=th[th.pth.notna()&(th.flag.fillna('')!='drift')&(th.dmax>=11)&(th.method=='fss')]
x=u[~u.decoder.str.contains(':')].copy()
def lab(n):
    if '@' not in n: return n,'depth6','X'
    b,r=n.split('@',1); s,L=r.rsplit('-',1); return b,s,L
L=x.noise.map(lab); x['base']=[a[0] for a in L]; x['sched']=[a[1] for a in L]; x['L']=[a[2] for a in L]
def val(g, decs=None, scheds=None):
    if decs is not None: g=g[g.decoder.isin(decs)]
    if scheds is not None: g=g[g.sched.isin(scheds)]
    best=[]; nfits=[]
    for s,gs in g.groupby('sched'):
        b={l:gs[gs.L==l].sort_values('pth').iloc[-1] for l in 'XY' if (gs.L==l).any()}
        if len(b)==2:
            l,r=min(b.items(),key=lambda kv:kv[1].pth); best.append((r.pth,r.err,s,l,r.decoder,r.nu,r.chi2, len(gs[gs.L==l])))
    if not best: return None
    return max(best,key=lambda z:z[0])
M=['mwpm','corr_gauge','bm']
rows=[]
for _,row in t[t.rounds.isin(['d','1','2','3','5','10'])].iterrows():
    g=x[(x.base==row.noise)&(x.rounds==row.rounds)]
    full=val(g); both=val(g,scheds=['both']); m=val(g,decs=M); mb=val(g,decs=M,scheds=['both'])
    bo=row.best_other
    if not isinstance(bo,str): continue
    ob=row[bo]
    f=lambda v: np.nan if v is None else v[0]
    rows.append(dict(noise=row.noise,r=row.rounds,full=f(full)*100,both=f(both)*100,matched=f(m)*100,matched_both=f(mb)*100,other=ob*100,bo=bo,
       rf=f(full)/ob, rmb=f(mb)/ob, nfit_full=None if full is None else full[7], fulldec=None if full is None else f'{full[4]} {full[2]} {full[3]} nu{full[5]:.2f} chi{full[6]:.2f}'))
d=pd.DataFrame(rows)
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
print(d.round(3).to_string())
d['drop_both']=1-d.both/d.full; d['drop_mb']=1-d.matched_both/d.full
print(d.groupby('r')[['drop_both','drop_mb']].describe().round(3).to_string())

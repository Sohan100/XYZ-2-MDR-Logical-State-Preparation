import pandas as pd, numpy as np
t=pd.read_csv('docs/data/campaign/thresholds.csv')
t=t[(t.method=='fss')&(t.dmax>=11)&(t.flag.isna())]
def split(row):
    n=row.noise; dec=row.decoder
    if '@' in n: base,sch=n.split('@')
    else: base,sch=n,None
    if ':' in dec:
        cb,d=dec.split(':'); code,b=cb.split('-'); return base,code,b,d
    else:
        if sch is None: sch='depth6-X'
        s,b=sch.split('-'); return base,'xyz2-'+s,b,dec
r=t.apply(split,axis=1,result_type='expand'); r.columns=['base','code','basis','dec']
t=pd.concat([t,r],axis=1)
g=t.loc[t.groupby(['base','rounds','code','basis']).pth.idxmax()][['base','rounds','code','basis','dec','pth','err','nu']]
g['pth']*=100; g['err']*=100
import sys
for (b,rd),x in g.groupby(['base','rounds']):
    s=' | '.join(f"{c}-{bb}:{p:.3f}({e:.3f})" for c,bb,p,e in zip(x.code,x.basis,x.pth,x.err))
    print(b,rd,s)

import pandas as pd, numpy as np
pe = pd.read_csv('docs/data/campaign/eq/points_eps.csv')
pe = pe[pe.model.isin(['biased100','purez'])]
def cmp(code, d1, d2, basis=None):
    a = pe[(pe.code==code)&(pe.dec==d1)]; b = pe[(pe.code==code)&(pe.dec==d2)]
    m = a.merge(b, on=['model','p','d','basis','rkey'], suffixes=('_1','_2'))
    m = m[(m.errors_1>=30)&(m.errors_2>=30)]
    if basis: m = m[m.basis==basis]
    r = m.eps_1/m.eps_2
    return len(r), r.min(), r.max(), r.median(), m.loc[r.idxmin(), ['model','p','d','basis','rkey']].tolist() if len(r) else None
print(cmp('xyz2','corr_gauge','bm'))
print(cmp('xzzx','corr','bm','X'))
print(cmp('xzzx','corr','bm','Z'))
# XZZX corr X>Z count
x = pe[(pe.code=='xzzx')&(pe.dec=='corr')]
m = x[x.basis=='X'].merge(x[x.basis=='Z'], on=['model','p','d','rkey'], suffixes=('_x','_z'))
m = m[(m.errors_x>0)&(m.errors_z>0)]
print(len(m), (m.eps_x>m.eps_z).sum())
print(m[m.eps_x<=m.eps_z][['model','p','d','rkey','eps_x','eps_z','errors_x','errors_z']].assign(r=lambda t: t.eps_z/t.eps_x))

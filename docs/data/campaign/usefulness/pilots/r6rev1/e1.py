import pandas as pd, numpy as np
E='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/eq/'
pe=pd.read_csv(E+'points_eps.csv')
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
def pt(model,code,dec,basis,d,p,rkey='3d'):
    g=pe[(pe.model==model)&(pe.code==code)&(pe.dec==dec)&(pe.basis==basis)&(pe.d==d)&(np.isclose(pe.p,p))&(pe.rkey==rkey)]
    return g
for m in ['biased100','purez']:
  for code,dec,bases in [('xyz2','corr',['X','Y']),('xzzx','corr',['X','Z'])]:
    for b in bases:
      for d in [5,7,9,11]:
        g=pt(m,code,dec,b,d,1e-3)
        if len(g): r=g.iloc[0]; print(m,code,dec,b,d,'%.4g [%.4g,%.4g] err=%d upper=%s'%(r.eps,r.eps_lo,r.eps_hi,r.errors,r.upper))
print(pe.dec.unique(), pe.code.unique())

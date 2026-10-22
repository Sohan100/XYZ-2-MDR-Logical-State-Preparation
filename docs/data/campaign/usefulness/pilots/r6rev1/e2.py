import pandas as pd, numpy as np
E='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/eq/'
pe=pd.read_csv(E+'points_eps.csv')
def pt(model,code,dec,basis,d,p,rkey='3d'):
    g=pe[(pe.model==model)&(pe.code==code)&(pe.dec==dec)&(pe.basis==basis)&(pe.d==d)&(np.isclose(pe.p,p))&(pe.rkey==rkey)]
    return g.iloc[0] if len(g) else None
for m in ['biased100','purez']:
  for p in [1e-3,2e-3,3e-3]:
    for b in ['X','Y']:
      for dec in ['corr_gauge','bm']:
        for d in [3,5,7,9,11]:
          r=pt(m,'xyz2',dec,b,d,p)
          if r is not None: print(m,p,dec,b,d,'%.4g [%.4g,%.4g] err=%d up=%s'%(r.eps,r.eps_lo,r.eps_hi,r.errors,r.upper))
# steps
def step(m,code,dec,b,p,d1,d2):
    a=pt(m,code,dec,b,d1,p); c=pt(m,code,dec,b,d2,p)
    return a.eps/c.eps, a.eps_lo/c.eps_hi, a.eps_hi/c.eps_lo
for m,b in [('biased100','X'),('biased100','Y'),('purez','Y'),('purez','X')]:
    try: print('xyz2',m,b,step(m,'xyz2','corr_gauge',b,1e-3,7,9))
    except Exception as e: print(e)
for m in ['biased100','purez']:
    print('xzzx',m,step(m,'xzzx','corr','X',1e-3,7,9))
    for p in [2e-3]:
        for b in ['X','Y']:
            try: print('2e-3 xyz2',m,b,step(m,'xyz2','corr_gauge',b,p,7,9))
            except Exception as e: print('no',m,b)
        print('2e-3 xzzx',m,step(m,'xzzx','corr','X',p,7,9))

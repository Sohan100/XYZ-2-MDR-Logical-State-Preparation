import csv
rows=list(csv.DictReader(open('docs/data/campaign/eq/points_eps.csv')))
def get(model,code,basis,dec,d,p,rkey='3d'):
    out=[r for r in rows if r['model']==model and r['code']==code and r['basis']==basis and r['dec']==dec and int(r['d'])==d and abs(float(r['p'])-p)<1e-9 and r['rkey']==rkey]
    return out
for p in (1e-3,2e-3,3e-3):
  for d in (3,5,7,9,11):
    x=get('sd6','xyz2','X','corr',d,p); y=get('sd6','xyz2','Y','corr',d,p)
    if not x: x=get('sd6','xyz2','X','corr_gauge',d,p)
    if not y: y=get('sd6','xyz2','Y','corr_gauge',d,p)
    def f(r): 
        if not r: return None
        r=r[0]; return (float(r['eps']) if r['eps'] else None, r['eps_lo'], r['eps_hi'], r['errors'], r['dec'])
    print(p,d,'X',f(x),'Y',f(y))

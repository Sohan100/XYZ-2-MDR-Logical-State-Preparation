import csv, collections
f='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/eq/points_eps.csv'
D=collections.defaultdict(dict)
for r in csv.DictReader(open(f)):
    if r['rkey']!='3d': continue
    key=(r['model'],r['p'],r['code'],r['dec'])
    D[key][(int(r['d']),r['basis'])]=(float(r['eps'] or 'nan'),int(r['errors']))
for m in ['biased100','purez','sd6']:
  for p in ['0.001','0.002','0.003']:
    print('==',m,p)
    for code,dec in [('xyz2','corr_gauge'),('xyz2','bm'),('xzzx','corr'),('css','corr')]:
        k=(m,p,code,dec)
        if k not in D: continue
        s=' '.join(f"d{d}{b}:{e:.3g}({n})" for (d,b),(e,n) in sorted(D[k].items()))
        print(code,dec,s)

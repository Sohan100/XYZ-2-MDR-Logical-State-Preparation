import pandas as pd, numpy as np
df=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/eq/points_eps.csv')
df=df[(df.rkey=='3d')&df.model.isin(['biased100','purez'])]
print(df.code.unique(), df.dec.unique(), df.fam.unique())
for (m,p),g in df.groupby(['model','p']):
  print('=====',m,p)
  for dec in sorted(g.dec.unique()):
    for code in ['xyz2','xzzx']:
      h=g[(g.code==code)&(g.dec==dec)]
      for d,hh in h.groupby('d'):
        s=' '.join(f"{r.basis}:{r.eps:.3e}({r.errors}/{r.shots})" for r in hh.itertuples())
        print(dec,code,d,s)

import pandas as pd, numpy as np
pe = pd.read_csv('docs/data/campaign/eq/points_eps.csv')
pe = pe[(pe.rkey=='3d')&pe.model.isin(['biased100','purez'])&(pe.dec=='mwpm')&(pe.errors>0)]
w = pe.groupby(['model','code','p','d']).eps.max().unstack('code')
w = w.dropna()
w = w.reset_index(); w=w[(w.d>=5)&(w.d<=11)]
w['r']=w.xyz2/w.xzzx
print(w.to_string())
print(w.r.min(), w.r.max())

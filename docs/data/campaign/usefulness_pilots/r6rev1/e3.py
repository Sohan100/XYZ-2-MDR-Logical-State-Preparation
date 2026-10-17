import pandas as pd, numpy as np
E='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/eq/'
pe=pd.read_csv(E+'points_eps.csv')
g=pe[(pe.code=='xzzx')&(pe.model.isin(['biased100','purez']))&(pe.rkey=='3d')]
print(g[['model','p','dec','basis','d','eps','eps_lo','eps_hi','errors','upper']].sort_values(['model','p','dec','basis','d']).to_string())
pa=pd.read_csv(E+'pairs.csv')
print(pa[pa.model.isin(['biased100','purez'])&(pa.other=='xzzx')][['model','p','mode','n_xyz2','n_other','basis_xyz2','basis_other','eps_xyz2','err_xyz2','eps_other','err_other','upper_xyz2','upper_other','ratio','ratio_lo','ratio_hi']].to_string())

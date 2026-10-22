import pandas as pd, numpy as np
pe = pd.read_csv('docs/data/campaign/eq/points_eps.csv')
for m in ['biased100','purez']:
    q = pe[(pe.model==m)&(pe.rkey=='3d')&(pe.code.isin(['xyz2','xzzx']))]
    print(q[['code','dec','basis','d','p','n_qubits','errors','eps','eps_lo','eps_hi']].sort_values(['code','dec','p','d','basis']).to_string())

import pandas as pd, numpy as np
pe = pd.read_csv('docs/data/campaign/eq/points_eps.csv')
print(pe.dec.unique(), pe.code.unique(), pe.rkey.unique())
q = pe[(pe.model=='sd6')&(pe.rkey=='3d')&(pe.code=='xyz2')]
print(q[['dec','basis','d','p','n_qubits','errors','eps','eps_lo','eps_hi']].sort_values(['dec','p','d','basis']).to_string())

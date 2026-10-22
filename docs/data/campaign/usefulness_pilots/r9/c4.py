import pandas as pd, numpy as np
from scipy.stats import chi2, norm
pe = pd.read_csv('docs/data/campaign/eq/points_eps.csv')
print(pe.columns.tolist())
q = pe[(pe.model=='sd6')&(pe.rkey=='3d')&(pe.dec=='corr')]
print(q[['code','basis','d','p','n_qubits','errors','eps','eps_lo','eps_hi']].sort_values(['code','p','d','basis']).to_string())
# poisson bounds
p3 = norm.sf(3)
up2 = chi2.ppf(1-p3, 2*(2+1))/2
lo300 = chi2.ppf(p3, 2*300)/2
print(up2, lo300, 693.43*(lo300/300)*(2/up2), 693.43*(2/up2))

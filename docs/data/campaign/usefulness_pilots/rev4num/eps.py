import pandas as pd, sys
pd.set_option('display.width',260); pd.set_option('display.max_rows',1000); pd.set_option('display.max_columns',40)
E=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/eq/points_eps.csv')
E=E[E.rkey=='3d']
m=sys.argv[1]
d=E[E.model==m][['model','p','code','basis','dec','noise','d','n_qubits','shots','errors','eps','eps_lo','eps_hi','upper']].copy()
d['eps']=d.eps.map(lambda v:'%.3e'%v)
print(d.sort_values(['p','code','dec','basis','d']).to_string())

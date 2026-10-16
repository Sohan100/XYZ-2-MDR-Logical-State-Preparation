import pandas as pd, numpy as np
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/'
th=pd.read_csv(R+'docs/data/campaign/thresholds.csv')
u=th[th.pth.notna() & (th.flag.fillna('')!='drift') & (th.dmax>=11) & (th.method=='fss')].copy()
co=u[u.decoder.str.contains(':')].copy()
co['code']=co.decoder.str.split('-').str[0]
co['basis']=co.decoder.str.split('-').str[1].str.split(':').str[0]
g=co.groupby(['noise','rounds','code']).basis.agg(lambda s:''.join(sorted(set(s))))
print(g[g.str.len()<2])
# all competitor rows for phen_b10 r=10 and h2_p d
a=th[th.decoder.str.contains(':')&(((th.noise=='phen_b10')&(th.rounds=='10'))|((th.noise=='h2_p')&(th.rounds.isin(['d','5']))))]
print(a[['noise','decoder','rounds','method','pth','err','nu','chi2','dmax','flag']].to_string())

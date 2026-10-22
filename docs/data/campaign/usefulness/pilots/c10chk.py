import sys, pandas as pd, numpy as np
sys.path.insert(0,'paper/analysis')
from competitor_report import usable
th=pd.read_csv('docs/data/campaign/thresholds.csv')
u=usable(th)
r1=u[u.rounds.astype(str)=='1']
def show(noise, pat=None):
    g=r1[r1.noise==noise]
    if pat: g=g[g.decoder.str.contains(pat)]
    print(g[['noise','decoder','pth','err','stat','nu']].sort_values('pth').to_string())
def z(a,b):
    return (a.pth-b.pth)/np.hypot(a.err,b.err), (a.pth-b.pth)/np.hypot(a.stat,b.stat)
# phen_b10 xyz2
g=r1[r1.noise.str.startswith('phen_b10@')]
print('phen_b10 xyz2 usable r1 range', g.pth.min(), g.pth.max(), len(g)); print(g.sort_values('pth').head(3)[['noise','decoder','pth','err']])
print(th[(th.noise=='phen_b10@depth6-Y')&(th.rounds.astype(str)=='1')&(th.decoder=='seq_match')][['method','pth','flag']])
# h2_p competitors
g=r1[(r1.noise=='h2_p')]
print('h2_p comp nu', g.nu.min(), g.nu.max(), g.nu.median(), len(g))
print(g[['decoder','pth','err','stat','nu']].sort_values('nu').to_string())
g=r1[(r1.noise=='helios_p')]
print('helios_p comp nu', g.nu.min(), g.nu.max(), len(g))

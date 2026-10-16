import pandas as pd, numpy as np
B='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
th=pd.read_csv(B+'thresholds.csv'); th['rounds']=th.rounds.astype(str)
for n in ['phen_b10','phen']:
  for r in ['d','1']:
    g=th[th.noise.str.startswith(n+'@')|(th.noise==n)]
    g=g[(g.rounds==r)&(~g.decoder.str.contains(':'))&(g.method=='fss')]
    pv=g.pivot_table(index='decoder',columns='noise',values='pth')*100
    print(n,r); print(pv.round(3).to_string())
    fl=g.pivot_table(index='decoder',columns='noise',values='flag',aggfunc=lambda s: ''.join(map(str,s.fillna('')))) if False else None

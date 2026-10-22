import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
p=th[(th.noise.str.startswith('phen_b10'))&(th.rounds=='1')&(~th.decoder.str.contains(':'))]
print(p.sort_values('pth')[['noise','decoder','method','pth','dmax','flag']].head(8).to_string())
u=C.usable(p); print(u.sort_values('pth')[['noise','decoder','pth','err']].head(4).to_string())

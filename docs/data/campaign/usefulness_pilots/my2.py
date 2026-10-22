import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
u=C.usable(th)
x=u[(u.rounds=='1')&(~u.decoder.str.contains(':'))].copy()
x['base']=x.noise.str.split('@').str[0]
x['ser']=x.noise.map(lambda n: n.split('@')[1] if '@' in n else 'depth6-X')
for b,g in x.groupby('base'):
    s=g.sort_values('pth').groupby('ser').tail(1)
    print(b, ' | '.join('%s %s %.3f(%.0f) nu%.2f'%(r.ser,r.decoder,r.pth*100,r.err*1e5,r.nu) for _,r in s.iterrows()))
print('rows with hw_ at r=1 in main th:', len(th[(th.noise.str.startswith('hw'))&(th.rounds=='1')]))
print(sorted(set(th[th.noise.str.startswith('hw')].rounds)))

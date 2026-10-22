import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
print(sorted(th.rounds.unique()))
u=C.usable(th)
x=u[(~u.decoder.str.contains(':'))].copy()
x['base']=x.noise.str.split('@').str[0]
x['ser']=x.noise.map(lambda n: n.split('@')[1] if '@' in n else 'depth6-X')
for m in ['sd6','si1000','helios_p_noxt','h2_p_noxt','sd6_il','hyb_lr2','biased10','em3','biased100','purez']:
    g=x[(x.base==m)&(x.rounds=='d')]
    a=g[g.ser=='depth6-X'].sort_values('pth').iloc[-1]; b=g[g.ser=='depth6-Y'].sort_values('pth').iloc[-1]
    print(m,a.decoder,'%.3f(%.0f)'%(a.pth*100,a.err*1e5),b.decoder,'%.3f(%.0f)'%(b.pth*100,b.err*1e5),'z %.2f'%((a.pth-b.pth)/np.hypot(a.err,b.err)))
# extra checks
def show(n,dec,r='1'):
    t=u[(u.noise==n)&(u.decoder==dec)&(u.rounds==r)]
    for _,q in t.iterrows(): print(n,dec,r,'%.4f err %.4f stat %.4f nu %.2f drift %s flag %s'%(q.pth*100,q.err*100,q.stat*100,q.nu,q.drift,q.flag))
show('si1000','corr_gauge'); show('sd6','bm'); show('sd6','mwpm'); show('sd6_lr3@depth6-X','bm'); show('sd6_lr3@depth6-X','seq_soft'); show('si1000_lr2@depth6-X','bm')
show('helios_p_noxt','css-Z:mwpm')
s=u[(u.noise=='sd6_lr3@depth6-X')&(u.rounds=='1')].sort_values('pth'); print(s[['decoder','pth','err','nu']].to_string())
for n in ['helios_p','h2_p']:
  s=th[(th.noise==n)&(th.rounds=='1')]; print(s[['decoder','method','pth','err','stat','nu','dmax','flag']].to_string())

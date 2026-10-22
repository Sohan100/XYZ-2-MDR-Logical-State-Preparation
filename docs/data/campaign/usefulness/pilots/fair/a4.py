import pandas as pd, numpy as np
B='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
th=pd.read_csv(B+'thresholds.csv'); th['rounds']=th.rounds.astype(str)
u=th[th.pth.notna()&(th.flag.fillna('')!='drift')&(th.dmax>=11)&(th.method=='fss')]
c=u[u.decoder.str.contains(':')].copy()
c['cb']=c.decoder.str.split(':').str[0]
for n in ['sd6','si1000','biased10','biased100','purez','em3','helios_p_noxt','h2_p_noxt','phen','phen_b10']:
    g=c[(c.noise==n)&(c.rounds=='1')]
    b=g.sort_values('pth').groupby('cb').tail(1).sort_values('cb')
    s=' '.join(f"{r.cb}:{r.decoder.split(':')[1]} {100*r.pth:.3f}({100*r.err:.3f})" for _,r in b.iterrows())
    x=u[(u.noise==n)&(u.rounds=='1')&(~u.decoder.str.contains(':'))].sort_values('pth').tail(1)
    xs=' '.join(f"{r.decoder} {100*r.pth:.3f}({100*r.err:.3f})" for _,r in x.iterrows())
    print(n,'| XYZ2 depth6-LX best:',xs,'|',s)

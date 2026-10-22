import pandas as pd, numpy as np
th=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
th=th[th.rounds.astype(str)=='1']
u=th[th.pth.notna() & (th.flag.fillna('')!='drift') & (th.dmax>=11) & (th.method=='fss')]
for m in ['helios_p_noxt','h2_p_noxt','biased10','biased100','purez']:
    cs=u[(u.noise==m) & u.decoder.str.contains(':')].copy()
    cs['cb']=cs.decoder.str.split(':').str[0]
    print(m, '  '.join(f"{cb} {g.sort_values('pth').iloc[-1].decoder.split(':')[1]} {g.pth.max()*100:.3f}" for cb,g in cs.groupby('cb')))
import pandas as pd, numpy as np
t=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
u=t[t.pth.notna() & (t.flag.fillna("")!="drift") & (t.dmax>=11) & (t.method=="fss")]
r1=u[u.rounds.astype(str)=='1']
CH={"hyb": ["sd6", "em3"], "hyb_lr2": ["sd6", "em3"], "sd6_lr2": ["sd6"], "sd6_lr3": ["sd6"],"si1000_lr2": ["si1000"]}
models=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','em3','hyb','hyb_lr2','sd6_lr2','sd6_lr3','si1000_lr2','helios_p','h2_p','phen_b10']
for m in models:
    key = m if m in r1.noise.values and m not in CH else m+'@depth6-X'
    g=r1[(r1.noise==key)&(~r1.decoder.str.contains(':'))]
    if m=='phen_b10':
        g=r1[(r1.noise.isin(['phen_b10','phen_b10@depth6-X']))&(~r1.decoder.str.contains(':'))]
    if len(g)==0: print(m,'none'); continue
    b=g.sort_values('pth').iloc[-1]
    hw=CH.get(m,[m])
    c=r1[(r1.noise.isin(hw))&(r1.decoder.str.contains(':'))]
    zs=[]
    for (n,cb),cg in c.groupby([c.noise,c.decoder.str.split(':').str[0]]):
        bb=cg.sort_values('pth').iloc[-1]
        z=(b.pth-bb.pth)/np.hypot(b.err,bb.err); zs.append((z,n,bb.decoder,bb.pth*100,bb.err*100,b.pth/bb.pth,(b.pth-bb.pth)/np.hypot(b.stat,bb.stat)))
    zs.sort()
    lab='loss' if zs[0][0]<=-3 else ('win' if min(z[0] for z in zs)>=3 else 'tie')
    print(f"{m} [{key}] {b.decoder} {b.pth*100:.3f}({b.err*1e5:.0f}) nu {b.nu:.2f} -> {lab}")
    for z in zs: print('   %+.2f %s %s %.3f(%.0f) ratio %.3f zstat %+.2f'%(z[0],z[1],z[2],z[3],z[4]*1000,z[5],z[6]))

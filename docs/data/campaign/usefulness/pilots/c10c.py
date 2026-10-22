import pandas as pd, numpy as np
t=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
u=t[t.pth.notna() & (t.flag.fillna("")!="drift") & (t.dmax>=11) & (t.method=="fss")]
r1=u[u.rounds.astype(str)=='1']
pd.set_option('display.width',250)
cols=['noise','decoder','pth','err','stat','nu','dmax','drift','flag']
def show(q): print(q[cols].sort_values('pth').to_string())
show(r1[r1.noise=='sd6_lr3@depth6-X'])
show(r1[r1.noise=='si1000_lr2@depth6-X'])
cg=r1[(r1.noise=='si1000')&(r1.decoder=='corr_gauge')].iloc[0]
for c in ['css-Z:bm','xzzx-X:corr','xzzx-Z:bm','xy-Z:bm']:
    b=r1[(r1.noise=='si1000')&(r1.decoder==c)].iloc[0]; print(c,(cg.pth-b.pth)/np.hypot(cg.err,b.err), cg.pth*100,cg.err*100, cg.pth/b.pth)
bm=r1[(r1.noise=='si1000_lr2@depth6-X')&(r1.decoder=='bm')].iloc[0]
for c in r1[(r1.noise=='si1000')&r1.decoder.str.contains(':')].itertuples():
    print('lr2 bm vs',c.decoder,(bm.pth-c.pth)/np.hypot(bm.err,c.err))
sd=r1[r1.noise=='sd6']
for d in ['bm','mwpm']:
    x=sd[sd.decoder==d].iloc[0]; print(d, x.pth/0.02417, x.pth/0.02489, x.pth/0.02441,x.pth/0.02408)
show(r1[r1.noise.isin(['helios_p','h2_p'])])
show(r1[r1.noise.str.startswith('phen_b10')&~r1.decoder.str.contains(':')])
show(r1[(r1.noise=='helios_p_noxt')&r1.decoder.str.contains(':')])
show(r1[(r1.noise=='em3')|(r1.noise.str.startswith('em3@'))&~r1.decoder.str.contains(':')].query('~decoder.str.contains(":")',engine='python'))

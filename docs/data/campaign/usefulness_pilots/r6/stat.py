import sys; sys.path.insert(0,'paper/analysis')
import pandas as pd, numpy as np, competitor_report as C
pd.set_option('display.width',250); pd.set_option('display.max_rows',400)
th=pd.read_csv('docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
comp_th=th[th.decoder.str.contains(':')]; xyz_th=th[~th.decoder.str.contains(':')]
# need stat: rebuild values with stat column by temporarily using err=stat
def withstat(df):
    d=df.copy(); d['err']=d['stat']; return d
comp=C.code_values(comp_th); compS=C.code_values(withstat(comp_th))
X=C.xyz2_values(xyz_th); XS=C.xyz2_values(withstat(xyz_th))
rows=[]
for (_,a),(_,b) in zip(X.iterrows(),XS.iterrows()):
    if pd.isna(a.pth): continue
    rs=C.competitor_for(comp,a.noise,a.rounds); rsS=C.competitor_for(compS,a.noise,a.rounds)
    if not rs: continue
    z={k:(a.pth-v.pth)/np.hypot(a.err,v.err) for k,v in rs.items()}
    zs={k:(b.pth-v.pth)/np.hypot(b.err,v.err) for k,v in rsS.items()}
    rows.append(dict(noise=a.noise,r=a.rounds,zmin=min(z.values()),zsmin=min(zs.values()),arg=min(zs,key=zs.get)))
D=pd.DataFrame(rows)
D.to_csv(sys.argv[1]+'/stat.csv',index=False)
M=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','sd6_lr2','sd6_lr3','si1000_lr2','sd6_il']
d=D[D.noise.isin(M)&D.r.isin(['1','2','3','5','10','d'])]
print(d[d.zmin>-3.5].to_string())
print(d[d.zmin<=-3.5].zsmin.max(), d[d.zmin<=-3.5].zsmin.min(), d[d.zmin<=-3.5].sort_values('zsmin',ascending=False).head(3).to_string())

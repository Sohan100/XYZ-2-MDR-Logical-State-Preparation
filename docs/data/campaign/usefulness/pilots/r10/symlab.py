import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv')
xyz_th=th[~th.decoder.str.contains(':')]
lab=xyz_th.noise.map(C.xyz2_label); sched=lab.map(lambda x:x[1])
sym_th=xyz_th[(sched=='both')&xyz_th.decoder.isin(['mwpm','corr_gauge','bm'])]
xyz=C.xyz2_values(sym_th); comp=C.code_values(th[th.decoder.str.contains(':')])
rows=[]
for (n,rk),g in xyz.groupby(['noise','rounds']):
    x=g.iloc[0]
    if pd.isna(x.get('pth')): continue
    cs={k:v for k,v in C.competitor_for(comp,n,rk).items() if pd.notna(v.get('pth',np.nan))}
    if not cs: continue
    zs={k:(x.pth-v.pth)/np.hypot(x.err,v.err) for k,v in cs.items()}
    top=max(cs,key=lambda k:cs[k].pth); zmin=min(zs.values())
    l='loss' if zmin<=-3 else ('win' if zs[top]>=3 else 'tie')
    rows.append(dict(noise=n,r=rk,ratio=x.pth/cs[top].pth,top=top,ztop=zs[top],zmin=zmin,lab=l,pth=x.pth,err=x.err,dec=x.decoder))
t=pd.DataFrame(rows); t.to_csv('sym_labels.csv',index=False)
print(len(t),t.lab.value_counts().to_dict())
print(t[t.lab=='tie'][['noise','r','ratio','zmin','ztop']].to_string())
print(t[t.r=='d'][['noise','r','ratio','top','zmin']].to_string())

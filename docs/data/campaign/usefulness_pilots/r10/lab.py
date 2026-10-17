import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv')
compsrc = sys.argv[1] if len(sys.argv)>1 else 'th'
xyz=C.xyz2_values(th)
if compsrc=='th':
    comp=C.code_values(th[th.decoder.str.contains(':')])
else:
    comp=C.code_values(pd.read_csv(R+'/docs/data/campaign/competitors.csv'))
rows=[]
for (n,rk),g in xyz.groupby(['noise','rounds']):
    x=g.iloc[0]
    if pd.isna(x.get('pth')): continue
    cs=C.competitor_for(comp,n,rk)
    cs={k:v for k,v in cs.items() if pd.notna(v.get('pth',np.nan))}
    if not cs: continue
    zs={k:(x.pth-v.pth)/np.hypot(x.err,v.err) for k,v in cs.items()}
    top=max(cs,key=lambda k:cs[k].pth)
    zmin=min(zs.values())
    lab='loss' if zmin<=-3 else ('win' if zs[top]>=3 else 'tie')
    rows.append(dict(noise=n,r=rk,ratio=x.pth/cs[top].pth,top=top,ztop=zs[top],zmin=zmin,lab=lab,xdec=x.decoder,xb=x.basis))
t=pd.DataFrame(rows)
t.to_csv('labels_%s.csv'%compsrc,index=False)
print(len(t), t.lab.value_counts().to_dict())

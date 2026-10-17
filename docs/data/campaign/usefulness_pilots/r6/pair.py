import sys
sys.path.insert(0,'paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
pd.set_option('display.width',250); pd.set_option('display.max_rows',400)
th=pd.read_csv('docs/data/campaign/thresholds.csv')
th['rounds']=th.rounds.astype(str)
comp_th=th[th.decoder.str.contains(':')].copy()
xyz_th=th[~th.decoder.str.contains(':')]
dec=comp_th.decoder.str.split(':').str[1]
adm=(comp_th.method=='pair')&(comp_th.dmin==9)&(comp_th.dmax==11)&(comp_th.flag.fillna('')!='drift')&dec.isin(['bposd','tesseract'])
print('admitted',adm.sum(), 'fss bposd/tess', ((comp_th.method=='fss')&dec.isin(['bposd','tesseract'])).sum())
c2=comp_th.copy(); c2.loc[adm,'method']='fss'
comp=C.code_values(comp_th); compP=C.code_values(c2)
X=C.xyz2_values(xyz_th)
rows=[]
for _,a in X.iterrows():
    if pd.isna(a.pth): continue
    r0=C.competitor_for(comp,a.noise,a.rounds); r1=C.competitor_for(compP,a.noise,a.rounds)
    if not r0: continue
    t0=max(r0.values(),key=lambda q:q.pth); t1=max(r1.values(),key=lambda q:q.pth)
    z0=min((a.pth-v.pth)/np.hypot(a.err,v.err) for v in r0.values())
    z1=min((a.pth-v.pth)/np.hypot(a.err,v.err) for v in r1.values())
    zt0=(a.pth-t0.pth)/np.hypot(a.err,t0.err); zt1=(a.pth-t1.pth)/np.hypot(a.err,t1.err)
    L0='loss' if z0<=-3 else ('win' if zt0>=3 else 'tie'); L1='loss' if z1<=-3 else ('win' if zt1>=3 else 'tie')
    rows.append(dict(noise=a.noise,r=a.rounds,top0=t0.code,v0=t0.pth*100,top1=t1.code,v1=t1.pth*100,e1=t1.err*100,dec1=t1.decoder,b1=t1.basis,ratio0=a.pth/t0.pth,ratio1=a.pth/t1.pth,z0=z0,z1=z1,L0=L0,L1=L1))
D=pd.DataFrame(rows)
ch=D[(D.v1>D.v0+1e-9)|(D.L0!=D.L1)]
print(ch.to_string())

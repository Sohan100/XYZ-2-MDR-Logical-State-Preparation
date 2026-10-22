import sys
sys.path.insert(0,'paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
comp_th=th[th.decoder.str.contains(':')]
xyz_th=th[~th.decoder.str.contains(':')]
for dec_x,dec_c in (('bm','bm'),('mwpm','mwpm'),('corr_gauge','corr')):
    xv=C.xyz2_values(xyz_th[xyz_th.decoder==dec_x])
    cv=C.code_values(comp_th[comp_th.decoder.str.endswith(':'+dec_c)])
    for n in ('biased10','biased100','purez'):
        for r in ('d','1','2','3','5','10'):
            a=xv[(xv.noise==n)&(xv.rounds==r)]
            b=cv[(cv.noise==n)&(cv.rounds==r)&(cv.code=='css')]
            if len(a)==0 or len(b)==0 or pd.isna(b.pth.iloc[0]): print(dec_x,n,r,'missing');continue
            a=a.iloc[0]; b=b.iloc[0]
            print(dec_x,n,r,round(a.pth*100,4),round(a.err*100,4),a.basis,'css',round(b.pth*100,4),round(b.err*100,4),b.basis,'ratio',round(a.pth/b.pth,3),'z',round((a.pth-b.pth)/np.hypot(a.err,b.err),2))

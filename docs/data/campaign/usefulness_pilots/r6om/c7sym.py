import sys
sys.path.insert(0,'paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
comp_th=th[th.decoder.str.contains(':')]; xyz_th=th[~th.decoder.str.contains(':')]
comp=C.code_values(comp_th)
lab=xyz_th.noise.map(C.xyz2_label); sched=lab.map(lambda x:x[1])
for name,sub in [('full',xyz_th),('sym',xyz_th[(sched=='both')&xyz_th.decoder.isin(['mwpm','corr_gauge','bm'])]),('md',xyz_th[xyz_th.decoder.isin(['mwpm','corr_gauge','bm'])])]:
    xs=C.xyz2_values(sub)
    for n in ['biased10','biased100','purez']:
        a=xs[(xs.noise==n)&(xs.rounds=='d')]
        if len(a)==0: continue
        a=a.iloc[0]
        rs=C.competitor_for(comp,n,'d')
        for k,q in rs.items():
            if 'css' in str(k).lower():
                z=(a.pth-q.pth)/np.hypot(a.err,q.err)
                print(name,n,a.decoder,a.basis,round(a.pth*100,3),'vs',k,round(q.pth*100,3),round(q.err*100,3),'ratio',round(a.pth/q.pth,3),'z',round(z,1))

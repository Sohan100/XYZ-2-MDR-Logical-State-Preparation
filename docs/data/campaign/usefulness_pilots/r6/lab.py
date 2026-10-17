import sys
sys.path.insert(0,'paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv')
th['rounds']=th.rounds.astype(str)
comp_th=th[th.decoder.str.contains(':')]
xyz_th=th[~th.decoder.str.contains(':')]
comp=C.code_values(comp_th)
lab=xyz_th.noise.map(C.xyz2_label)
sched=lab.map(lambda x:x[1])
sym_th=xyz_th[(sched=='both')&xyz_th.decoder.isin(['mwpm','corr_gauge','bm'])]
def labels(xv, errcol='err'):
    out=[]
    for _,a in xv.iterrows():
        rs=C.competitor_for(comp,a.noise,a.rounds)
        rs={k:v for k,v in rs.items() if k!='honeycomb' or a.noise in ('hyb','hyb_lr2','em3') or True}
        if not rs or pd.isna(a.get('pth')): continue
        zs={k:(a.pth-v.pth)/np.hypot(a.err,v.err) for k,v in rs.items()}
        top=max(rs,key=lambda k:rs[k].pth)
        zmin=min(zs.values())
        L='loss' if zmin<=-3 else ('win' if zs[top]>=3 else 'tie')
        out.append(dict(noise=a.noise,r=a.rounds,x=a.pth*100,xe=a.err*100,dec=a.decoder,basis=a.basis,top=top,tv=rs[top].pth*100,ratio=a.pth/rs[top].pth,ztop=zs[top],zmin=zmin,arg=min(zs,key=zs.get),L=L))
    return pd.DataFrame(out)
pd.set_option('display.width',250); pd.set_option('display.max_rows',400)
T=labels(C.xyz2_values(xyz_th)); S=labels(C.xyz2_values(sym_th))
print(T.L.value_counts(), len(T)); print(S.L.value_counts(), len(S))
T.to_csv(sys.argv[1]+'/T.csv',index=False); S.to_csv(sys.argv[1]+'/S.csv',index=False)

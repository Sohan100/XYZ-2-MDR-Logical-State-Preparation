import sys
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv')
th['rounds']=th.rounds.astype(str)
comp_th=th[th.decoder.str.contains(':')]
xyz_th=th[~th.decoder.str.contains(':')]
comp=C.code_values(comp_th)
lab=xyz_th.noise.map(C.xyz2_label)
sched=lab.map(lambda x:x[1])
sym_th=xyz_th[(sched=='both')&xyz_th.decoder.isin(['mwpm','corr_gauge','bm'])]
def rows(xv, comp, errcol='err'):
    out=[]
    for _,a in xv.iterrows():
        rs=C.competitor_for(comp,a.noise,a.rounds)
        rs={k:v for k,v in rs.items() if k!='honeycomb' or a.noise in ('hyb','hyb_lr2','em3') or True}
        if not rs or pd.isna(a.get('pth')): continue
        zs={k:(a.pth-v.pth)/np.hypot(a.err,v.err) for k,v in rs.items()}
        top=max(rs.values(),key=lambda q:q.pth)
        zmin=min(zs.values()); ztop=(a.pth-top.pth)/np.hypot(a.err,top.err)
        labl='loss' if zmin<=-3 else ('win' if ztop>=3 else 'tie')
        out.append(dict(noise=a.noise,r=a.rounds,x=a.pth*100,xe=a.err*100,dec=a.decoder,basis=a.basis,top=top.code,tp=top.pth*100,te=top.err*100,ratio=a.pth/top.pth,zmin=zmin,zmincode=min(zs,key=zs.get),ztop=ztop,label=labl))
    return pd.DataFrame(out)
if __name__=='__main__':
    pd.set_option('display.width',250); pd.set_option('display.max_rows',400)
    full=rows(C.xyz2_values(xyz_th),comp)
    sym=rows(C.xyz2_values(sym_th),comp)
    print(full.label.value_counts(), len(full))
    print(sym.label.value_counts(), len(sym))
    full.to_csv('full.csv',index=False); sym.to_csv('sym.csv',index=False)

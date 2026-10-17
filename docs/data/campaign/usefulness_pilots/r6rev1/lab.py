import sys
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
cth=th[th.decoder.str.contains(':')]; xth=th[~th.decoder.str.contains(':')]
comp=C.code_values(cth)
lab=xth.noise.map(C.xyz2_label); sched=lab.map(lambda x:x[1])
sym_th=xth[(sched=='both')&xth.decoder.isin(['mwpm','corr_gauge','bm'])]
def labels(xv, comp, statonly=False, src=th):
    out=[]
    for _,a in xv.iterrows():
        n,r=a.noise,a.rounds
        if pd.isna(a.get('pth')): continue
        rs=C.competitor_for(comp,n,r)
        rs={k:v for k,v in rs.items() if pd.notna(v.get('pth',np.nan)) and (k!='honeycomb' or n in ('hyb','hyb_lr2'))}
        # The table includes honeycomb always; keep all to mimic table
        rs2=C.competitor_for(comp,n,r)
        rs2={k:v for k,v in rs2.items() if pd.notna(v.get('pth',np.nan))}
        if not rs2: continue
        ea=a.err
        zs={}
        for k,v in rs2.items():
            ev=v.err
            if statonly:
                # stat columns
                pass
            zs[k]=(a.pth-v.pth)/np.hypot(ea,ev)
        top=max(rs2,key=lambda k:rs2[k].pth)
        zmin=min(zs.values()); ztop=zs[top]
        lbl='loss' if zmin<=-3 else ('win' if ztop>=3 else 'tie')
        out.append(dict(noise=n,rounds=r,x=a.pth*100,xe=a.err*100,dec=a.decoder,basis=a.basis,top=top,topv=rs2[top].pth*100,ratio=a.pth/rs2[top].pth,zmin=zmin,argmin=min(zs,key=zs.get),ztop=ztop,label=lbl))
    return pd.DataFrame(out)

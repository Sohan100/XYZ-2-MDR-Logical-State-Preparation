import sys
sys.path.insert(0,'paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
cc=pd.read_csv('docs/data/campaign/competitors.csv'); cc['rounds']=cc.rounds.astype(str)
xyz_th=th[~th.decoder.str.contains(':')]
lab=xyz_th.noise.map(C.xyz2_label); sched=lab.map(lambda x:x[1])
sym_th=xyz_th[(sched=='both')&xyz_th.decoder.isin(['mwpm','corr_gauge','bm'])]
def labels(xth, compsrc, errcol='err'):
    comp=C.code_values(compsrc)
    xv=C.xyz2_values(xth)
    out=[]
    for r in xv.itertuples():
        if pd.isna(r.pth): continue
        rs=C.competitor_for(comp,r.noise,r.rounds)
        rs={k:v for k,v in rs.items() if pd.notna(v.get('pth',np.nan))}
        if not rs: continue
        zs={k:(r.pth-v.pth)/np.hypot(r.err,v.err) for k,v in rs.items()}
        top=max(rs,key=lambda k:rs[k].pth)
        zmin=min(zs.values())
        if zs[top]>=3: L='win'
        elif zmin<=-3: L='loss'
        else: L='tie'
        out.append((r.noise,r.rounds,round(r.pth/rs[top].pth,3),round(zs[top],2),round(zmin,2),L))
    return pd.DataFrame(out,columns=['noise','r','ratio','ztop','zmin','label'])
for name,xth in (('table',xyz_th),('sym',sym_th)):
  for src,cs in (('thr',th[th.decoder.str.contains(':')]),('comp',cc)):
    d=labels(xth,cs)
    print(name,src,len(d),d.label.value_counts().to_dict())
    if src=='thr':
        print(d[d.label!='loss'].to_string())

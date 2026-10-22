import sys, numpy as np, pandas as pd
sys.path.insert(0,'paper/analysis')
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
cpf=pd.read_csv('docs/data/campaign/competitors.csv'); cpf['rounds']=cpf.rounds.astype(str)
xy=th[~th.decoder.str.contains(':')]
def statmap(df):
    return {(round(p,12)):s for p,s in zip(df.pth,df.stat)}
for name,comp_src in [('thr',th[th.decoder.str.contains(':')]),('cpf',cpf)]:
    sm=statmap(comp_src); smx=statmap(xy)
    cv=C.code_values(comp_src); xv=C.xyz2_values(xy)
    res=[]
    for _,r in xv.iterrows():
        cs=C.competitor_for(cv,r.noise,r.rounds)
        cs={k:v for k,v in cs.items() if 'pth' in v and pd.notna(v.get('pth'))}
        if not cs or pd.isna(r.get('pth')): continue
        sx=smx[round(r.pth,12)]
        zs={k:(r.pth-v.pth)/np.hypot(sx,sm[round(v.pth,12)]) for k,v in cs.items()}
        best=max(cs,key=lambda k:cs[k].pth)
        lab='win' if zs[best]>=3 else ('loss' if min(zs.values())<=-3 else 'tie')
        res.append((r.noise,r.rounds,lab,zs[best],min(zs.values())))
    L=pd.DataFrame(res,columns=['noise','r','lab','zbest','zmin'])
    print(name,L.lab.value_counts().to_dict())
    print(L[L.lab!='loss'].to_string())

import sys, numpy as np, pandas as pd
sys.path.insert(0,'paper/analysis')
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
cpf=pd.read_csv('docs/data/campaign/competitors.csv'); cpf['rounds']=cpf.rounds.astype(str)
xy=th[~th.decoder.str.contains(':')]
def label(xv, comp, stat=False, exclude_honey=False):
    res=[]
    for _,r in xv.iterrows():
        cs=C.competitor_for(comp, r.noise, r.rounds)
        if exclude_honey: cs={k:v for k,v in cs.items() if k!='honeycomb'}
        cs={k:v for k,v in cs.items() if 'pth' in v and pd.notna(v.get('pth'))}
        if not cs or pd.isna(r.get('pth')): continue
        zs={}
        for k,v in cs.items():
            if stat:
                ea=r.stat_; eb=v.stat_
            else:
                ea=r.err; eb=v.err
            zs[k]=(r.pth-v.pth)/np.hypot(ea,eb)
        best=max(cs,key=lambda k:cs[k].pth)
        if zs[best]>=3: lab='win'
        elif min(zs.values())<=-3: lab='loss'
        else: lab='tie'
        res.append((r.noise,r.rounds,lab,r.pth/cs[best].pth,zs[best],min(zs.values()),best))
    return pd.DataFrame(res,columns=['noise','r','lab','ratio','zbest','zmin','best'])
def add_stat(df):
    # code_values/xyz2_values drop stat; map back
    return df
# include stat column by computing via merges: simpler: patch functions to carry stat
def xyz2_values_stat(th):
    v=C.xyz2_values(th)
    return v
for name,comp_src in [('competitors.csv',cpf),('thresholds.csv',th[th.decoder.str.contains(':')])]:
    cv=C.code_values(comp_src); xv=C.xyz2_values(xy)
    L=label(xv,cv)
    print(name, L.lab.value_counts().to_dict(), 'rows',len(L), 'nominal lower', (L.ratio<1).sum())
    if name=='competitors.csv':
        print(L[L.lab!='loss'].sort_values(['noise','r']).to_string())
# symmetric
xs=xy[xy.decoder.isin(['mwpm','corr_gauge','bm'])]
xs=xs[xs.noise.str.contains('@both')]
cs=cpf[cpf.decoder.str.split(':').str[1].isin(['mwpm','corr','bm'])]
L=label(C.xyz2_values(xs),C.code_values(cs))
print('sym', L.lab.value_counts().to_dict(), len(L))
print(L[L.lab!='loss'].sort_values(['noise','r']).to_string())

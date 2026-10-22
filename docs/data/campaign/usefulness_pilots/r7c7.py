import sys, numpy as np, pandas as pd
sys.path.insert(0,'paper/analysis')
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv')
def xyz(th,decs):
    t=th[~th.decoder.str.contains(':')]
    t=t[t.decoder.isin(decs)]
    return C.xyz2_values(t)
def comp(th,decs):
    t=th[th.decoder.str.contains(':')]
    t=t[t.decoder.map(lambda s:s.split(':')[1]).isin(decs)]
    return C.code_values(t)
for model in ['biased10','biased100','purez']:
  for xd,cd in [(['corr_gauge'],['corr']),(['bm'],['bm']),(['mwpm','corr_gauge','bm'],['mwpm','corr','bm']),(['corr_gauge','mwpm'],['corr','mwpm'])]:
    x=xyz(th,xd); c=comp(th,cd)
    xr=x[(x.noise==model)&(x.rounds=='d')]
    cr=c[(c.noise==model)&(c.rounds=='d')&(c.code=='css')]
    if len(xr) and len(cr):
        a=xr.iloc[0]; b=cr.iloc[0]
        print(model,xd,cd,a.pth*100,a.err*100,a.basis,a.decoder,'css',b.pth*100,b.err*100,b.basis,b.decoder,'ratio %.3f z %.2f'%(a.pth/b.pth,(a.pth-b.pth)/np.hypot(a.err,b.err)))

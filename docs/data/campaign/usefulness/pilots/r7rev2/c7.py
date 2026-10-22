import sys, numpy as np, pandas as pd
sys.path.insert(0,'paper/analysis')
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
xy=th[~th.decoder.str.contains(':')]
cp=th[th.decoder.str.contains(':')]
def sub(x, decs): return x[x.decoder.isin(decs)]
for noise in ['biased10','biased100','purez']:
    for decs_x, decs_c, lab in [(['bm'],['css-X:bm','css-Z:bm'],'bm/bm'),(['mwpm','corr_gauge','bm'],None,'sym')]:
        xv=C.xyz2_values(sub(xy,decs_x))
        r=xv[(xv.noise==noise)&(xv.rounds=='d')]
        cc=cp[cp.decoder.str.startswith('css-')]
        if decs_c: cc=cc[cc.decoder.isin(decs_c)]
        else: cc=cc[cc.decoder.str.split(':').str[1].isin(['mwpm','corr','bm'])]
        cv=C.code_values(cc)
        c=cv[(cv.noise==noise)&(cv.rounds=='d')]
        if len(r) and len(c):
            a,ea=r.pth.iloc[0],r.err.iloc[0]; b,eb=c.pth.iloc[0],c.err.iloc[0]
            print(noise,lab,r.basis.iloc[0],r.decoder.iloc[0],a,ea,'css',c.basis.iloc[0],c.decoder.iloc[0],b,eb,'ratio',a/b,'z',(a-b)/np.hypot(ea,eb))

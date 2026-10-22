import sys, numpy as np, pandas as pd
sys.path.insert(0,'/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/paper/analysis')
import competitor_report as C
th=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
th['rounds']=th.rounds.astype(str)
NO=['biased10','biased100','purez']
cv=C.code_values(th[th.decoder.str.contains(':')]); cv=cv[cv.code=='css']
x=th[~th.decoder.str.contains(':')]
sets={d:[d] for d in ['bm','bp_corr','bp_full','cfe','cfe0','seq_soft','seq_match','seq_erasure','corr_links','tesseract']}
sets['nonBP']=['mwpm','corr_gauge','corr_links','seq_match','seq_erasure','tesseract']
for k,s in sets.items():
  xv=C.xyz2_values(x[x.decoder.isin(s)])
  out=[]
  for n in NO:
    a=xv[(xv.noise==n)&(xv.rounds=='d')]; b=cv[(cv.noise==n)&(cv.rounds=='d')].iloc[0]
    if len(a)==0 or pd.isna(a.iloc[0].get('pth',np.nan)): out.append('--'); continue
    a=a.iloc[0]; out.append(f"{a.pth*100:.3f}({a.err*100:.3f}) r{a.pth/b.pth:.3f} z{(a.pth-b.pth)/np.hypot(a.err,b.err):.1f} [{a.basis}]")
  print(k,' | '.join(out))

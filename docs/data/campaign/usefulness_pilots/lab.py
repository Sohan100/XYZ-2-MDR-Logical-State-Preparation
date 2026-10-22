import sys,math
sys.path.insert(0,'paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv')
for compfile in ('docs/data/campaign/competitors.csv','docs/data/campaign/thresholds.csv'):
 for errcol in ('err','stat'):
  cth=pd.read_csv(compfile)
  xth=th.copy()
  cth=cth[cth.decoder.str.contains(":")].copy(); 
  cth['err']=cth[errcol]; xth['err']=xth[errcol]
  comp=C.code_values(cth); xyz=C.xyz2_values(xth)
  W=T=L=0; rows=0; ties=[]; nonneg=0; below=0
  for (noise,rk),g in xyz.groupby(['noise','rounds']):
    x=g.iloc[0]
    if pd.isna(x.get('pth')): continue
    cs=C.competitor_for(comp,noise,rk)
    cs={k:v for k,v in cs.items() if pd.notna(v.get('pth'))}
    if not cs: continue
    rows+=1
    zs={k:(x.pth-v.pth)/math.hypot(x.err,v.err) for k,v in cs.items()}
    top=max(cs,key=lambda k:cs[k].pth)
    if x.pth<cs[top].pth: below+=1
    if min(zs.values())<=-3: L+=1
    elif zs[top]>=3: W+=1
    else: T+=1; ties.append((noise,rk,round(zs[top],2)))
  print(compfile.split('/')[-1],errcol,'rows',rows,'below',below,'W',W,'T',T,'L',L)
  if errcol=='err' and 'competitors' in compfile: print(sorted(ties))

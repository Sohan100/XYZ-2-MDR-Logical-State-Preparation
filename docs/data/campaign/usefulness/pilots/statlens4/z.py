import sys
sys.path.insert(0,'/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/paper/analysis')
import pandas as pd, numpy as np
import competitor_report as cr
th=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
th['rounds']=th['rounds'].astype(str)
x=cr.xyz2_values(th)
cth=pd.read_csv("/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/competitors.csv"); cth["rounds"]=cth.rounds.astype(str); comp=cr.code_values(cth[cth.decoder.str.contains(":")]); thall=pd.concat([th,cth],ignore_index=True)
# attach stat to xyz2 rows: find matching row
def find_stat(noise, rk, pth):
    g=thall[(thall.rounds==rk)&(abs(thall.pth-pth)<1e-12)]
    return g.iloc[0]
out=[]
for _,r in x.iterrows():
    if pd.isna(r.get('pth')): continue
    s=find_stat(r.noise,r.rounds,r.pth)
    co=cr.competitor_for(comp,r.noise,r.rounds)
    best=None
    for code,cr_ in co.items():
        if pd.isna(cr_.get('pth')): continue
        if best is None or cr_.pth>best[1].pth: best=(code,cr_)
    if best is None: continue
    cs=find_stat(r.noise,r.rounds,best[1].pth)
    zt=(r.pth-best[1].pth)/np.hypot(r.err,best[1].err)
    zs=(r.pth-best[1].pth)/np.hypot(s.stat,cs.stat)
    out.append((r.noise,r.rounds,r.pth*100,r.err*100,best[0],best[1].pth*100,best[1].err*100,r.pth/best[1].pth,zt,zs,s.nu,s.chi2,cs.nu,cs.chi2))
df=pd.DataFrame(out,columns='noise r x xe comp c ce ratio zt zs xnu xchi cnu cchi'.split())
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
print(df.round(3).to_string())
df.to_csv('/tmp/claude-108819/-pscratch-sd-s-sohan100/b9bb5492-172d-4149-9bf3-b4e9300d223c/scratchpad/statlens4/z.csv',index=False)

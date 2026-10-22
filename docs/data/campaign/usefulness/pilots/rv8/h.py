import sys, pandas as pd, numpy as np
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
t=pd.read_csv(R+'/docs/data/campaign/thresholds.csv')
t['rounds']=t['rounds'].astype(str)
U=C.usable(t)
def q(noise,dec=None,r='d',usable=True,method=None):
    d=U if usable else t
    s=d[(d.noise==noise)&(d.rounds==str(r))]
    if dec: s=s[s.decoder==dec]
    if method: s=s[s.method==method]
    return s[['noise','decoder','rounds','method','pth','err','stat','syst','nu','chi2','dmin','dmax','drift','flag']]
def z(a,ea,b,eb): return (a-b)/np.hypot(ea,eb)

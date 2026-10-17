import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
xyz=C.xyz2_values(th); comp=C.code_values(th[th.decoder.str.contains(":")])
def q(noise, rk, dec=None):
    g=th[(th.noise==noise)|(th.noise.str.startswith(noise+'@'))|(th.decoder.str.contains(':')&(th.noise==noise))]
    g=g[g.rounds==str(rk)]
    if dec: g=g[g.decoder.str.contains(dec)]
    return g

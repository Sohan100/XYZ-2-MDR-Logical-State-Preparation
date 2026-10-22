import sys, pandas as pd, numpy as np
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv')
def q(noise_prefix, rounds, dec=None, method=None):
    t=th[th.noise.str.startswith(noise_prefix)&(th.rounds.astype(str)==str(rounds))]
    if dec: t=t[t.decoder.str.contains(dec)]
    if method: t=t[t.method==method]
    return t[['noise','decoder','rounds','method','pth','err','stat','syst','nu','chi2','dmin','dmax','drift','flag']]
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
if __name__=='__main__':
    print(q(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else None, sys.argv[4] if len(sys.argv)>4 else None).to_string())

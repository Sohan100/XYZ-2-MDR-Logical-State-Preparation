import pandas as pd, sys, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',500); pd.set_option('display.max_columns',30)
T=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
T['rounds']=T.rounds.astype(str)
def q(noise_prefix, rounds=None, dec=None, method=None, exact=False):
    m = (T.noise==noise_prefix) if exact else T.noise.str.split('@').str[0].eq(noise_prefix)
    d=T[m]
    if rounds is not None: d=d[d.rounds==str(rounds)]
    if dec is not None: d=d[d.decoder.str.contains(dec)]
    if method is not None: d=d[d.method==method]
    d=d.copy()
    for c in ['pth','err','stat','syst','cross_first','cross_last']: d[c]=d[c]*100
    return d[['noise','decoder','rounds','method','pth','err','stat','syst','nu','chi2','dmin','dmax','cross_first','cross_last','drift','flag']].round(4)
if __name__=='__main__':
    a=sys.argv
    r=a[2] if len(a)>2 and a[2]!='-' else None
    dec=a[3] if len(a)>3 and a[3]!='-' else None
    meth=a[4] if len(a)>4 else None
    print(q(a[1],r,dec,meth).sort_values(['noise','decoder']).to_string())

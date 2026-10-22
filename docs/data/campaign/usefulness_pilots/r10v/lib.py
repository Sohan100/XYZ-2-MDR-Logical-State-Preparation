import sys, numpy as np, pandas as pd
sys.path.insert(0,'paper/analysis')
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv')
th['rounds']=th.rounds.astype(str)
def sub(th, xdec=None, cdec=None):
    t=th.copy()
    isx=~t.decoder.str.contains(':')
    if xdec is not None: t=t[~isx | t.decoder.isin(xdec)]
    isx=~t.decoder.str.contains(':')
    if cdec is not None: t=t[isx | t.decoder.str.split(':').str[1].isin(cdec)]
    return t

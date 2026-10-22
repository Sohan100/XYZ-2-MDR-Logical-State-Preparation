import pandas as pd, numpy as np
th=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
th['rounds']=th.rounds.astype(str)
comp=th[th.decoder.str.contains(':')].copy()
x=th[~th.decoder.str.contains(':')].copy()
# competitor tesseract/bposd
c=comp.copy(); c['dec']=c.decoder.str.split(':').str[1]
for dec in ['tesseract','bposd','bm','corr','mwpm']:
    g=c[c.dec==dec]
    print(dec, 'n',len(g),'method',g.method.value_counts().to_dict(),'dmax',g.dmax.value_counts().to_dict())
g=x[x.decoder=='tesseract']
print('xyz2 tesseract', g.method.value_counts().to_dict(), 'fss dmax21', ((g.method=='fss')&(g.dmax==21)).sum(), 'fss nodrift dmax>=11', ((g.method=='fss')&(g.dmax>=11)&(g.flag.fillna('')!='drift')).sum())
# tnml / cfe_tn rounds
for dec in ['tnml','cfe_tn']:
    g=x[x.decoder==dec]; print(dec, g.rounds.value_counts().to_dict())
for dec in ['cfe','cfe0']:
    g=x[x.decoder==dec]; print(dec, g.rounds.value_counts().to_dict())

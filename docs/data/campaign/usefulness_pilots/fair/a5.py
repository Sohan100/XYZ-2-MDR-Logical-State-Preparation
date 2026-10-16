import pandas as pd, numpy as np, itertools
B='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
c=pd.read_csv(B+'competitors.csv'); c['rounds']=c.rounds.astype(str)
print(c.columns.tolist()); print(len(c))
c=c[(c.method=='fss')&c.pth.notna()&(c.flag.fillna('')!='drift')]
c['cb']=c.decoder.str.split(':').str[0]; c['dec']=c.decoder.str.split(':').str[1]
for n in ['sd6','si1000','phen']:
    g=c[c.noise==n]
    for m in [('css-X','xzzx-X'),('css-X','xzzx-Z'),('css-Z','xzzx-X'),('css-Z','xzzx-Z'),('css-X','xy-X'),('css-X','xy-Z'),('css-Z','xy-X'),('css-Z','xy-Z')]:
        a=g[g.cb==m[0]].set_index(['rounds','dec']); b=g[g.cb==m[1]].set_index(['rounds','dec'])
        j=a.join(b,lsuffix='_a',rsuffix='_b',how='inner')
        if len(j)==0: continue
        pt=(j.pth_a-j.pth_b)/np.hypot(j.err_a,j.err_b)
        ps=(j.pth_a-j.pth_b)/np.hypot(j.stat_a,j.stat_b)
        print(n,m,len(j),'rms tot %.2f stat %.2f'%(np.sqrt((pt**2).mean()),np.sqrt((ps**2).mean())))

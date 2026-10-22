import sys, pandas as pd, numpy as np
sys.path.insert(0,'paper/analysis')
from competitor_report import usable, xyz2_label
th=pd.read_csv('docs/data/campaign/thresholds.csv')
u=usable(th)
r1=u[u.rounds.astype(str)=='1'].copy()
r1['isx']=~r1.decoder.str.contains(':')
lab=[xyz2_label(n) if x else (n,None,None) for n,x in zip(r1.noise,r1.isx)]
r1['base']=[l[0] for l in lab]; r1['sl']=[ (l[1] or '')+'-'+(l[2] or '') for l in lab]
X=r1[r1.isx]; C=r1[~r1.isx].copy(); C['cb']=C.decoder.str.split(':').str[0]
def xs(base,sl='depth6-X'): return X[(X.base==base)&(X.sl==sl)]
def row(base,dec,sl='depth6-X'):
    x=xs(base,sl); x=x[x.decoder==dec]; assert len(x)==1,(base,dec,sl,len(x)); return x.iloc[0]
def crow(noise,dec):
    x=C[(C.noise==noise)&(C.decoder==dec)]; assert len(x)==1; return x.iloc[0]
def z(a,b): return round((a.pth-b.pth)/np.hypot(a.err,b.err),2),round((a.pth-b.pth)/np.hypot(a.stat,b.stat),2), round(a.pth/b.pth,3)
def bestc(noise):
    g=C[C.noise==noise]; return g.sort_values('pth').groupby('cb').tail(1).sort_values('pth')
g=X[X.base=='phen_b10']; print('phen_b10 XYZ2 usable r1', g.pth.min(), g.pth.max(), len(g)); print(g.sort_values('pth')[['noise','decoder','pth','err']].head(3)); print(g.sort_values('pth')[['noise','decoder','pth','err']].tail(2))
print('dups phen_b10 bare vs @depth6-X'); print(g[g.sl=='depth6-X'][['noise','decoder','pth','err']].sort_values('decoder').to_string())
a=row('si1000','corr_gauge')
for d in ['css-Z:bm','xzzx-X:corr','xzzx-Z:bm','xy-Z:bm']: print('si1000 cg vs',d, z(a,crow('si1000',d)))
a=row('si1000','cfe_tn')
for _,b in bestc('si1000').iterrows(): print('si1000 cfe_tn vs',b.decoder, z(a,b))
for noise,x in [('sd6','cfe_tn'),('helios_p_noxt','tnml'),('h2_p_noxt','tnml')]:
    a=row(noise,x); print(noise,x,a.pth,a.err)
    for _,b in bestc(noise).iterrows(): print('   ',b.decoder, round(b.pth*100,3), round(b.err*100,3), z(a,b))
a=row('sd6_lr3','tnml')
for _,b in bestc('sd6').iterrows(): print(' lr3 vs',b.decoder, round(b.pth*100,3), z(a,b))
g=xs('sd6_il').sort_values('pth'); a=g.iloc[-1]; print('sd6_il best',a.decoder,a.pth,a.err,a.nu, a.dmax)
for _,b in bestc('sd6').iterrows(): print(' il vs',b.decoder, round(b.pth*100,3), z(a,b))
g=xs('phen').sort_values('pth'); print(g[['noise','decoder','pth','err','nu']].tail(4)); a=g.iloc[-1]
for _,b in bestc('phen').iterrows(): print(' phen vs',b.decoder, round(b.pth*100,3), round(b.err*100,3), z(a,b))
print('phen all xyz2 series max', X[X.base=='phen'].groupby('sl').pth.max())
g=xs('sd6_lr2','depth6-Y').sort_values('pth'); print(g[['decoder','pth','err','nu']].tail(3)); a=g.iloc[-1]
for _,b in bestc('sd6').iterrows(): print(' lr2Y vs',b.decoder, z(a,b))
a=row('hyb_lr2','tnml'); print('hyb_lr2 vs', z(a,crow('em3','honeycomb-X:bm')), z(a,crow('em3','honeycomb-Z:mwpm')))

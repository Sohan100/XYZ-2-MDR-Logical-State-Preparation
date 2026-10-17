from lab import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',400)
# stat lookup
u=C.usable(th)
def statof(noise_series, dec, pth):
    r=u[(u.pth==pth)]
    return float(r.stat.iloc[0]) if len(r) else np.nan
xf=C.xyz2_values(xth)
M=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','sd6_lr2','sd6_lr3','si1000_lr2','sd6_il']
R=['1','2','3','5','10','d']
rows=[]
for _,a in xf.iterrows():
    if a.noise not in M or a.rounds not in R: continue
    rs=C.competitor_for(comp,a.noise,a.rounds)
    sa=statof(None,None,a.pth)
    zs={k:(a.pth-v.pth)/np.hypot(sa,statof(None,None,v.pth)) for k,v in rs.items() if pd.notna(v.get('pth',np.nan))}
    zt={k:(a.pth-v.pth)/np.hypot(a.err,v.err) for k,v in rs.items() if pd.notna(v.get('pth',np.nan))}
    rows.append(dict(noise=a.noise,rounds=a.rounds,zstat=min(zs.values()),zerr=min(zt.values())))
D=pd.DataFrame(rows)
D['lab']=np.where(D.zerr<=-3,'loss','tie')
print(D.sort_values('zerr').to_string())

from lab import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',400)
M=['sd6','sd6_lr2','sd6_lr3','sd6_il','si1000','si1000_lr2','biased10','biased100','purez','helios_p_noxt','h2_p_noxt']
compnh=comp[comp.code!='honeycomb']
def rows2(xv,comp,col='err'):
    out=[]
    for _,a in xv.iterrows():
        if a.noise not in M or a.rounds not in ['1','2','3','5','10','d']: continue
        rs=C.competitor_for(comp,a.noise,a.rounds)
        if not rs: continue
        def e(r): 
            if col=='err': return r.err
            return th[(th.noise==r.noise)&(th.rounds==a.rounds)&(th.decoder==f"{r.code}-{r.basis.split(',')[0]}:{r.decoder}")].stat.iloc[0]
        ax=a.err if col=='err' else xyz_th[(xyz_th.noise==a.noise+('' if 'depth6 L_X' in a.basis and False else ''))].stat.iloc[0]
        zs={k:(a.pth-v.pth)/np.hypot(a.err,v.err) for k,v in rs.items()}
        top=max(rs.values(),key=lambda q:q.pth)
        zmin=min(zs.values())
        ztop=(a.pth-top.pth)/np.hypot(a.err,top.err)
        out.append(dict(noise=a.noise,r=a.rounds,x=round(a.pth*100,3),xe=round(a.err*100,3),dec=a.decoder,basis=a.basis,top=top.code,ratio=round(a.pth/top.pth,3),zmin=round(zmin,2),zc=min(zs,key=zs.get),label='loss' if zmin<=-3 else ('win' if ztop>=3 else 'tie')))
    return pd.DataFrame(out)
f=rows2(C.xyz2_values(xyz_th),compnh)
print(len(f)); print(f.label.value_counts())
print(f.sort_values('zmin',ascending=False).to_string())

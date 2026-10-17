from lib import *
t=C.table(xyz,comp)
models=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','sd6_lr2','sd6_lr3','si1000_lr2','sd6_il']
cnt={'tie':0,'loss':0,'win':0}
rows=[]
for m in models:
  for rk in ['1','2','3','5','10','d']:
    x=xyz[(xyz.noise==m)&(xyz.rounds==rk)].iloc[0]
    cs=C.competitor_for(comp,m,rk)
    zs={}
    for c,r in cs.items():
        if c=='honeycomb': continue
        zs[c]=(x.pth-r.pth)/np.hypot(x.err,r.err)
    best=max(cs[c].pth for c in zs)
    zmin=min(zs.values()); 
    bestc=max(zs,key=lambda c:cs[c].pth)
    zbest=zs[bestc]
    lab='loss' if zmin<=-3 else ('win' if zbest>=3 else 'tie')
    cnt[lab]+=1
    rows.append((m,rk,round(x.pth*100,4),round(x.pth/best,3),round(zmin,2),lab))
for r in rows: print(r)
print(cnt)

import sys; sys.path.insert(0,'paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
th=pd.read_csv("docs/data/campaign/thresholds.csv"); th["rounds"]=th.rounds.astype(str)
xyz=C.xyz2_values(th); comp=C.code_values(th[th.decoder.str.contains(":")])
t=C.table(xyz,comp)
print(len(t[t.ratio.notna()]))
u=C.usable(th)
rows=[('si1000','1'),('si1000','2'),('si1000','5'),('si1000_lr2','1'),('si1000_lr2','2'),('si1000_lr2','3'),('si1000_lr2','5'),('si1000_lr2','10'),('h2_p_noxt','3')]
for n,r in rows:
    x=xyz[(xyz.noise==n)&(xyz.rounds==r)].iloc[0]
    sched,L=x.basis.split()[0],x.basis.split()[1][-1]
    name = n if (sched=='depth6' and L=='X') else f"{n}@{sched}-{L}"
    cands=[name] + ([n] if name==f"{n}@depth6-X" else [])
    m=u[(u.noise.isin(cands))&(u.decoder=='mwpm')&(u.rounds.astype(str)==r)]
    cf=C.competitor_for(comp,n,r)
    def zs(p,e):
        return {c:(p-v.pth)/np.hypot(e,v.err) for c,v in cf.items()}
    z1=zs(x.pth,x.err); mw=m.iloc[0]
    z2=zs(x.pth,mw.err)
    print(n,r,x.basis,x.decoder,f"{x.pth*100:.4f}({x.err*100:.4f})", 'mwpm same', name, f"{mw.pth*100:.4f}({mw.err*100:.4f})", 'ratio %.3f'%(x.pth/mw.pth),
      'zmin own %.2f'%min(z1.values()),'zmin mwpmerr %.2f'%min(z2.values()), 'table mwpm %.4f'%(x.mwpm*100), 'nominal ratio %.3f'%t[(t.noise==n)&(t.rounds==r)].ratio.iloc[0])

exec(open('/tmp/claude-108819/-pscratch-sd-s-sohan100/b9bb5492-172d-4149-9bf3-b4e9300d223c/scratchpad/fairlens/sym.py').read().split("t=pd.read_csv(R+'docs/data/campaign/competitors_table.csv')")[0])
import numpy as np
for rk in ['d','1','2','3','5','10']:
    xv=xyzval('em3',rk)
    cs={c:compval('em3',rk,c) for c in ['css','xzzx','xy']}
    top=max(cs,key=lambda c:cs[c][0])
    print(rk, round(100*xv[0],3), xv[4], xv[2]+xv[3], top, round(100*cs[top][0],3), round(xv[0]/cs[top][0],3), round((xv[0]-cs[top][0])/np.hypot(xv[1],cs[top][1]),1))
# si1000 r=1 sym stat-only
for base,rk in [('si1000','1'),('si1000_lr2','1'),('si1000_lr2','2'),('si1000_lr2','3'),('hyb_lr2','1')]:
    g=x[(x.base==base)&(x.rounds==rk)&x.decoder.isin(XD)&x.sched.isin(SC)]
    xv=xyzval(base,rk)
    r=g[(g.pth==xv[0])].iloc[0]
    print(base,rk,xv, 'stat',r.stat)

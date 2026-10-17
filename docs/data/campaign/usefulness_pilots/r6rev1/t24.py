from lab import *
xf=C.xyz2_values(xth)
for n,r in [('sd6_il','5'),('sd6_il','10'),('sd6_il','2'),('si1000','1'),('si1000','2'),('si1000','5'),('si1000','10'),('si1000','3'),('phen','d'),('phen','5'),('sd6_il','1')]:
    a=xf[(xf.noise==n)&(xf.rounds==r)].iloc[0]
    rs=C.competitor_for(comp,n,r)
    v=[rs[k] for k in ['css','xzzx','xy'] if k in rs]
    w=np.array([1/q.err**2 for q in v]); p=np.array([q.pth for q in v])
    pp=(w*p).sum()/w.sum(); pe=w.sum()**-.5
    chi=((p-pp)**2*w).sum()
    print(n,r,'pool %.4f(%.4f) chi2 %.2f z %.2f'%(pp*100,pe*100,chi,(a.pth-pp)/np.hypot(a.err,pe)))

import math
def brentq(g,a,b):
    for _ in range(200):
        m=(a+b)/2
        if g(a)*g(m)<=0: b=m
        else: a=m
    return (a+b)/2
eps=lambda pL,r:(1-(1-2*pL)**(1/r))/2
def rate(e,s,r): return eps(e/s,r)
# pairs: xyz2 runs list (err,shots,r), comp (err,shots,r)
P={'18/16':([(592,8176,9)],(579,8176,12)),
   '50/49':([(416,11248,15)],(468,16368,21)),
   '98/100':([(114,6896,21),(129,7152,21)],(518,49136,30))}
def pool(runs,delta):
    w=[1/(1/n+delta**2) for n,s,r in runs]
    lr=[math.log(rate(n,s,r)) for n,s,r in runs]
    # pooled log rate inverse variance
    return sum(a*b for a,b in zip(w,lr))/sum(w), 1/sum(w)
def chi2(runs):
    E=sum(n for n,s,r in runs); S=sum(s for n,s,r in runs)
    return sum((n-s*E/S)**2/(s*E/S) for n,s,r in runs)
out={}
for k,(xr,c) in P.items():
    Nx=sum(n for n,_,_ in xr); Nc=c[0]
    # Poisson pooled: pooled counts
    if len(xr)>1:
        ex=sum(n for n,_,_ in xr)/sum(s for _,s,_ in xr); rx=eps(ex,xr[0][2])
    else: rx=rate(*xr[0])
    rc=rate(*c); lnR=math.log(rx/rc)
    za=lnR/math.sqrt(1/Nx+1/Nc)
    fx=1.7 if len(xr)==1 else max(1,math.sqrt(chi2(xr)/(len(xr)-1)))
    zb=lnR/math.sqrt(fx**2/Nx+1.7**2/Nc)
    zc=lambda d: lnR/math.sqrt(pool(xr,d)[1]+1/Nc)
    ds=brentq(lambda d: zc(d)-3,0,3)
    out[k]=(lnR,Nx,Nc,fx,xr)
    print(k,'R %.3f +- %.3f'%(math.exp(lnR),math.exp(lnR)*math.sqrt(1/Nx+1/Nc)),'lnR %.3f za %.2f zb %.2f'%(lnR,za,zb),'zc',[round(zc(d),2) for d in (0.065,0.074,0.15,0.25,0.37)],'d* %.3f'%ds, 'chi2 %.2f'%chi2(xr) if len(xr)>1 else '')
# trend
a=out['98/100'];b=out['18/16']
L=a[0]-b[0]
va=1/a[1]+1/a[2]+1/b[1]+1/b[2]
vb=a[3]**2/a[1]+1.7**2/a[2]+1.7**2/b[1]+1.7**2/b[2]
def vc(d): return pool(a[4],d)[1]+1/a[2]+pool(b[4],d)[1]+1/b[2]
zc=lambda d:L/math.sqrt(vc(d))
print('trend',L,L/math.sqrt(va),L/math.sqrt(vb),[round(zc(d),2) for d in (0.065,0.074,0.37)],brentq(lambda d:zc(d)-3,0,3))
# BM-BM 50/49
x=(500,None);print()

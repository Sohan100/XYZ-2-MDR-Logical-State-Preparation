import numpy as np
from scipy import stats, optimize
def eps(n,N,r): pL=n/N; return (1-(1-2*pL)**(1/r))/2
def pool(runs,r):
    n=sum(a for a,b in runs); N=sum(b for a,b in runs); e=eps(n,N,r)
    # chi2 homogeneity
    p=n/N; chi=sum((a-b*p)**2/(b*p) for a,b in runs) if len(runs)>1 else 0
    return n,N,e,chi,len(runs)-1
def zz(x,c,rx,rc,fb=None,label=''):
    n1,N1,e1,chi,dof=pool(x,rx); n2,N2,e2,_,_=pool(c,rc)
    R=e1/e2; L=abs(np.log(R)); k=len(x)
    sa=np.sqrt(1/n1+1/n2)
    if dof>0:
        f=max(1,np.sqrt(chi/dof))
    else: f=1.7
    f2=1.7 if len(c)==1 else 1
    sb=np.sqrt(f**2/n1+1/n2); sb2=np.sqrt(f**2/n1+f2**2/n2)
    zb=L/sb; zb2=L/sb2
    if dof>0 and f>1:
        p=stats.t.sf(zb,dof); zbt=stats.norm.isf(p)
    else: zbt=zb
    zc=[L/np.sqrt(1/n1+1/n2+d**2/k) for d in (0.15,0.25,0.37)]
    try:
        ds=np.sqrt(k*((L/3)**2-1/n1-1/n2))
    except: ds=np.nan
    print(f"{label}: R={R:.3f}({R*sa:.3f}) n1={n1} n2={n2} chi={chi:.2f}/{dof} f={f:.2f} a={L/sa:.2f} b={zb:.2f}(t->{zbt:.2f}) b2(f2=1.7 comp)={zb2:.2f} c={zc[0]:.2f}/{zc[1]:.2f}/@.37 {zc[2]:.2f} d*={ds:.3f}")

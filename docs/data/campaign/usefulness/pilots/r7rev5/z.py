import math
from scipy import stats
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def pool(runs,r):
    e=sum(a for a,b in runs); s=sum(b for a,b in runs); p=e/s
    chi=sum((a-p*b)**2/(p*b) for a,b in runs) if len(runs)>1 else 0
    return eps(e,s,r),e,s,chi,len(runs)-1
def tgauss(t,dof):
    p=stats.t.sf(t,dof); return stats.norm.isf(p)
def z(name,A,rA,Bv,NB,fB=1.7,BM=True,kB=1):
    ea,na,sa,chi,dof=pool(A,rA); k=len(A)
    R=ea/Bv; l=abs(math.log(R))
    sa_=math.sqrt(1/na+1/NB)
    f=max(1,math.sqrt(chi/dof)) if dof>0 else 1.7
    sb=math.sqrt(f**2/na+fB**2/NB)
    zb=l/sb
    if dof>0 and f>1: zb=tgauss(zb,dof)
    out=f"{name}: R={R:.3f}({R*sa_:.3f}) a={l/sa_:.2f} b={zb:.2f}(f={f:.2f},chi={chi:.2f}/{dof})"
    if BM:
        c=[l/math.sqrt(sa_**2+d*d/k) for d in (0.15,0.25,0.37)]
        x=(l/3)**2-sa_**2
        ds=math.sqrt(k*x) if x>0 else 0
        out+=f" c={c[0]:.2f}/{c[1]:.2f}/{c[2]:.2f} d*={ds:.3f}"
    print(out)

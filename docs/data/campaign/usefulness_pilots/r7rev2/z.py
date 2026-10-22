import math
from scipy import stats, optimize
def eps(e,s,r): return 0.5*(1-(1-2*e/s)**(1/r))
def pool(runs,r):
    e=sum(a for a,b in runs); s=sum(b for a,b in runs); p=e/s
    chi=sum((a-p*b)**2/(p*b) for a,b in runs) if len(runs)>1 else None
    return eps(e,s,r),e,len(runs),chi
def f_of(k,chi):
    if k==1: return 1.7, None
    f=max(1,math.sqrt(chi/(k-1))); return f, (k-1 if f>1 else None)
def zc(A,rA,B,rB, cbm=True, compPoissonB=False, label=''):
    ea,na,ka,ca=pool(A,rA); eb,nb,kb,cb=pool(B,rB)
    R=ea/eb; L=abs(math.log(R))
    za=L/math.sqrt(1/na+1/nb)
    fa,dfa=f_of(ka,ca); fb,dfb=f_of(kb,cb)
    if compPoissonB: fb=1
    zb=L/math.sqrt(fa**2/na+fb**2/nb)
    zbt=zb
    if dfa: # t on dfa dof -> gaussian equiv
        p=stats.t.sf(zb,dfa); zbt=stats.norm.isf(p)
    out=dict(R=R,Rerr=R*math.sqrt(1/na+1/nb),za=za,zb=zb,zbt=zbt)
    if cbm:
        def var(d): return 1/sum(1/(1/a+d*d) for a,b in A) + 1/nb
        out['c15']=L/math.sqrt(var(.15)); out['c25']=L/math.sqrt(var(.25)); out['c37']=L/math.sqrt(var(.37))
        g=lambda d: L/math.sqrt(var(d))-3
        out['dstar']=optimize.brentq(g,0,5) if g(0)>0 else 0
    print(label, ' '.join(f"{k}={v:.3f}" for k,v in out.items()), f"na={na} nb={nb} chiA={ca}")
    return out

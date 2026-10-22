import math
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def pool(runs,r):
    e=sum(a for a,b in runs); s=sum(b for a,b in runs); p=e/s
    chi=sum((a-p*b)**2/(p*b) for a,b in runs) if len(runs)>1 else 0
    return eps(e,s,r),e,chi,len(runs)-1,len(runs)
def zz(name,A,rA,Bval,NB,fB=1.7,bm=True):
    ea,N1,chi,dof,k=pool(A,rA)
    R=ea/Bval; l=abs(math.log(R))
    sa=math.sqrt(1/N1+1/NB)
    f1=1.7 if k==1 else max(1,math.sqrt(chi/dof))
    sb=math.sqrt(f1**2/N1+fB**2/NB)
    zc=lambda d: l/math.sqrt(1/N1+1/NB+d*d/k)
    ds=math.sqrt(max(0,k*((l/3)**2-1/N1-1/NB)))
    print(f"{name}: R={R:.3f}({R*sa:.3f}) chi2={chi:.2f}/{dof} f1={f1:.2f} za={l/sa:.2f} zb={l/sb:.2f} zc={zc(0.15):.2f}/{zc(0.25):.2f} z37={zc(0.37):.2f} d*={ds:.3f}")

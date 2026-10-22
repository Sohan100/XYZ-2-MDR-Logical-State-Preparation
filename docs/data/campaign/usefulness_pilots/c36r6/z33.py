import math
from scipy import stats
def eps(err,shots,r):
    pl=err/shots; return 0.5*(1-(1-2*pl)**(1/r))
def pool(runs,r):
    e=sum(a for a,b in runs); s=sum(b for a,b in runs); p=e/s
    chi=sum((a-p*b)**2/(p*b) for a,b in runs) if len(runs)>1 else 0
    return eps(e,s,r),e,s,chi,len(runs)-1
def gauss_equiv(t,dof):
    p=stats.t.sf(t,dof); return stats.norm.isf(p)
def do(name,runs,r,Bval,NB):
    ea,na,sa,chi,dof=pool(runs,r); R=ea/Bval; L=abs(math.log(R)); k=len(runs)
    # propagation factor dln eps / dln pL
    pl=na/sa; x=1-2*pl; g=(2*pl*x**(1/r-1)/r)/(1-x**(1/r))  # d ln eps/d ln pL
    sa_=math.sqrt(1/na+1/NB); za=L/sa_
    sag=math.sqrt(g*g/na+1/NB)
    if dof>0:
        f=max(1,math.sqrt(chi/dof))
    else: f=1.7
    sb=math.sqrt(f*f/na+1/NB); tb=L/sb
    zb = gauss_equiv(tb,dof) if (dof>0 and f>1) else tb
    out=[]
    for d in (0.15,0.25):
        sc=math.sqrt(1/na+d*d/k+1/NB); out.append(L/sc)
    # delta at which (c) z=3
    lo,hi=0,1
    for _ in range(60):
        mid=(lo+hi)/2
        if L/math.sqrt(1/na+mid*mid/k+1/NB)>3: lo=mid
        else: hi=mid
    s166=math.sqrt(1.66**2/na+1/NB)
    lin_s=R*math.sqrt(1/na+1/NB)
    print(f"{name}: R={R:.3f} Poisson err {R*sa_:.3f} (g={g:.3f}) | (a) {za:.2f} | (b) f={f:.2f} t={tb:.2f} -> {zb:.2f} | (c) {out[0]:.2f}/{out[1]:.2f} delta3={lo:.3f} | uniform1.66 log {L/s166:.2f} | 1.52 log {L/math.sqrt(1.52**2/na+1/NB):.2f} | linear Pois {abs(1-R)/lin_s:.2f}")
do("m1",[(69,3187),(45,3201),(80,3401),(64,3269)],15,1.916e-3,34217)
do("A",[(174,3179)],15,5.116e-3,57973)
do("E",[(56,3379),(56,3308)],15,6.603e-4,11601)
do("f002",[(108,3237),(83,3238)],15,1.982e-3,32049)
do("f005Y",[(123,3195)],15,2.096e-3,33367)
do("eta20 3% Y",[(36,33978),(37,34130)],15,1.054e-4,6166)
do("eta10 Y",[(48,34284),(51,33493)],15,1.208e-4,7370)
do("eta100 Y",[(36,46743)],15,4.502e-5,2922)
do("r6d Y",[(52,13687),(47,13611)],30,3.163e-4,14848)
do("hwtr6 Y",[(218,21968)],15,1.089e-3,29621)

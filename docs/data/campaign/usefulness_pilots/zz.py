import math
from math import log, sqrt
from scipy import stats
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def pool(runs):
    E=sum(a for a,b in runs); S=sum(b for a,b in runs); p=E/S
    chi=sum((a-p*b)**2/(p*b) for a,b in runs) if len(runs)>1 else None
    return E,S,chi,len(runs)-1
def tconv(t,dof):
    p=stats.t.sf(t,dof); return stats.norm.isf(p)
def z(name,runs,r,Bval,NB):
    E,S,chi,dof=pool(runs); R=eps(E,S,r)/Bval; L=abs(log(R)); cv=1/NB; k=len(runs)
    a=L/sqrt(1/E+cv)
    if chi is None: f=1.7; tdof=None
    else: f=max(1,sqrt(chi/dof)); tdof=dof if f>1 else None
    bG=L/sqrt(f*f/E+cv); b=tconv(bG,tdof) if tdof else bG
    c15=L/sqrt(1/E+cv+0.15**2/k); c25=L/sqrt(1/E+cv+0.25**2/k)
    dmax=sqrt(max(0,(L/3)**2-1/E-cv)*k)
    s166=L/sqrt(1.66**2/E+cv); s152=L/sqrt(1.52**2/E+cv)
    lin=abs(1-R)/(R*sqrt(1/E+cv))
    print(f"{name:26s} R={R:.3f}({R*sqrt(1/E+cv):.3f}) N={E} chi={chi} f={f:.2f} a={a:.2f} b={b:.2f}(G {bG:.2f}) c15={c15:.2f} c25={c25:.2f} dmax={dmax:.3f} | uni1.66={s166:.2f} 1.52={s152:.2f} linP={lin:.2f}")
    return R,E,NB
z('eta20 3% X vs Z',[(40,33432),(22,33653)],15,6.775e-5,3985)
z('eta10 X vs Z',[(45,34371),(46,33496)],15,8.743e-5,5396)
# m1 better-basis lead with replicates individually
z('m1 Y vs Z (bm XZZX litrev)',[(52,3151),(43,3139),(61,3496)],15,1.566e-3,65)
import math
T2=50.
for tw,T1 in ((12.8,2050.),(8.8,2980.)):
    tau=0.1
    g=(1-math.exp(-tau**2/T2**2))/2; t1=(1-math.exp(-tau/T1))/4
    pTw=(1-math.exp(-tw**2/T2**2))/2
    print(tw, 'gauss',g,'t1',t1,'pT2 eq4',g-t1,'eta eq5',(g-t1)/t1, 'f strict g/pTw',g/pTw,'g/(1.05pTw)',g/(1.05*pTw),'(g+t1)/(1.05 pTw)',(g+t1)/(1.05*pTw), 'T2^2/(2T1)',T2**2/(2*T1))

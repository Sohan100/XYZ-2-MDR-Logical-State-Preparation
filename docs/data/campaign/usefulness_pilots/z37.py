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
m1=[(69,3187),(45,3201),(80,3401),(64,3269)]
z('m1 X',m1,15,1.916e-3,34217)
z('m1 Y vs Z',[(52,3151),(43,3139),(61,3496)],15,1.452e-3,25634)
z('A X',[(174,3179)],15,5.116e-3,57973)
z('A Y vs X?',[(143,3168)],15,5.116e-3,57973)
z('f002 X',[(108,3237),(83,3238)],15,1.982e-3,32049)
z('f002 Y vs X',[(67,3199),(74,3210)],15,1.982e-3,32049)
z('f005 Y',[(123,3195)],15,2.096e-3,33367)
z('f005 X',[(122,3205)],15,2.096e-3,33367)
z('f005 X vs Z',[(122,3205)],15,1.588e-3,25291)
z('E Y vs X',[(56,3379),(56,3308)],15,6.603e-4,11601)
z('E X vs Z',[(43,3383)],15,5.765e-4,9941)
z('eta10 Y',[(48,34284),(51,33493)],15,1.208e-4,7370)
z('eta20 3% Y',[(36,33978),(37,34130)],15,1.054e-4,6166)
z('eta100 Y',[(36,46743)],15,4.502e-5,2922)
z('eta100 X vs Z',[(29,46148)],15,1.740e-5,1152)
# shifts in XYZ2's own ratio, log form
def shift(n,R1,N1,NB1,f1,k1,R2,N2,NB2,f2,k2):
    d=abs(log(R2)-log(R1))
    P=sqrt(1/N1+1/NB1+1/N2+1/NB2)
    U=sqrt(1.66**2/N1+1/NB1+1.66**2/N2+1/NB2)
    B=sqrt(f1**2/N1+1/NB1+f2**2/N2+1/NB2)
    C15=sqrt(P**2+0.15**2/k1+0.15**2/k2); C25=sqrt(P**2+0.25**2/k1+0.25**2/k2)
    print(f"{n}: dlog={d:.3f} Pois {d/P:.2f} uni1.66 {d/U:.2f} own-f(Gauss) {d/B:.2f} c15 {d/C15:.2f} c25 {d/C25:.2f}")
shift('m1->f002',0.7004,258,34217,1.66,4,1.0207,191,32049,1.81,2)
shift('m1->f005Y',0.7004,258,34217,1.66,4,1.2710,123,33367,1.7,1)
# XYZ2 rate rises
print('X rise f002', 2.023e-3/1.342e-3-1, 'f005', 2.633e-3/1.342e-3-1, 'xzzx', 1.982/1.916-1, 2.096/1.916-1)
# XZZX at E vs XYZ2 at m1
R=1.342e-3/6.603e-4; print('m1 xyz2 / E xzzx', R, log(R)/sqrt(1/258+1/11601))
# device f
for tw in (12.756,12.8,8.8):
    T2=50.; dt=0.1
    pT=(1-math.exp(-tw**2/T2**2))/2
    der=(tw/T2**2)*math.exp(-tw**2/T2**2)*dt
    inc=(1-math.exp(-(tw+dt)**2/T2**2))/2-pT
    mark=(1-math.exp(-dt/T2))/2
    print(f"tw={tw} pT2={pT:.4f} lin f={dt/tw:.4f} gauss marg f={der/pT:.4f} (exact inc {inc/pT:.4f}) markov-layer/gaussPT={mark/pT:.4f} consistent markov f={mark/((1-math.exp(-tw/T2))/2):.4f}")
# timing
for s in (0.016984,0.022299,0.024813):
    print(s/15*1e3,'ms/round', s/15/12.8e-6, s/15/8.8e-6)

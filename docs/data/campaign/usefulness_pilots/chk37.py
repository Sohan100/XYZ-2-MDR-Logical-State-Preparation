from math import *
def eps(e,s,r): p=e/s; return 0.5*(1-(1-2*p)**(1/r))
def tsf(t,dof):
    if dof==1: return 0.5-atan(t)/pi
    if dof==3: x=t/sqrt(3); return 0.5-(x/(1+x*x)+atan(x))/pi
    raise
def tz(t,dof):
    p=tsf(t,dof); lo,hi=0,40
    for i in range(200):
        m=(lo+hi)/2
        if 0.5*erfc(m/sqrt(2))>p: lo=m
        else: hi=m
    return lo
def zz(name,A,rA,Bv,NB,single_s=1.7):
    E=sum(a for a,b in A); S=sum(b for a,b in A); ea=eps(E,S,rA); R=ea/Bv
    p=E/S; chi=sum((a-p*b)**2/(p*b) for a,b in A); k=len(A); dof=k-1
    s=max(1,sqrt(chi/dof)) if dof>0 else single_s
    L=abs(log(R)); cv=1/NB
    a=L/sqrt(1/E+cv); b=L/sqrt(s*s/E+cv)
    bt=tz(b,dof) if (dof>0 and s>1) else b
    c15=L/sqrt(1/E+cv+0.15**2/k); c25=L/sqrt(1/E+cv+0.25**2/k); c37=L/sqrt(1/E+cv+0.37**2/k)
    ds=sqrt(max(0,(L/3)**2-1/E-cv)*k)
    print(f"{name:20s} ea={ea:.4e} R={R:.4f}+-{R*sqrt(1/E+cv):.4f} chi2={chi:.2f}/{dof} s={s:.2f} a={a:.2f} b={b:.2f} bt={bt:.2f} c15={c15:.2f} c25={c25:.2f} c37={c37:.2f} d*={ds:.3f}")
    return R,E,NB
m1=zz("m1",[(64,3269),(69,3187),(45,3201),(80,3401)],15,1.916e-3,34217)
zz("A",[(174,3179)],15,5.116e-3,57973)
f2=zz("f0.02",[(108,3237),(83,3238)],15,1.982e-3,32049)
f5=zz("f0.05Y",[(123,3195)],15,2.096e-3,33367)
zz("f0.05X",[(122,3205)],15,2.096e-3,33367)
zz("E",[(56,3379),(56,3308)],15,6.603e-4,11601)
zz("Ebetter",[(43,3383)],15,5.765e-4,9941)
zz("cross m1 vs E",[(64,3269),(69,3187),(45,3201),(80,3401)],15,6.603e-4,11601)
zz("eta20 3% Y",[(36,33978),(37,34130)],15,1.054e-4,6166)
zz("eta10 Y",[(48,34284),(51,33493)],15,1.208e-4,7370)
zz("eta100 Y",[(36,46743)],15,4.502e-5,2922)
# shifts
for nm,(R2,E2,N2),k2 in (("f02",f2,2),("f05",f5,1)):
    L=log(R2/m1[0]); v=1/m1[1]+1/m1[2]+1/E2+1/N2
    print(nm,"shift",round(L,4),round(L/sqrt(v),2),[round(L/sqrt(v+d*d/4+d*d/k2),2) for d in (0.15,0.25)])
print("t3.44/3",tz(3.44,3),"t6.83/3",tz(6.83,3))
# timing
print(0.016984366/15*1e3,0.024813381/15*1e3, 0.016984366/15/13e-6, 0.024813381/15/9e-6)
# Eq14
tmin=1.5;T1R=1000;tres=3;pres=0.005;T2=50
tR=tmin*log(1+T1R/tmin); print("tauR*",tR)
for nm,pR,pT in (("xyz2",0.1063,0.1049),("xzzx",0.1032,0.1895)):
    t=tR
    for i in range(100): t=tR-tmin*log(1+pR/pT*(t+tres)*T1R/((1-2*pres)*T2**2))
    print(nm,t)
print("2window",6.678e-4/1.054e-4,6.678e-4/6.97e-4, 6.678/6.97*sqrt(1/218+1/60520))
print("Lx rises",2.023/1.342,2.633/1.342,1.982/1.916,2.096/1.916,254/52)

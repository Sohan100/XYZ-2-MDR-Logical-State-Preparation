from math import *
from statistics import NormalDist
class _T:
    def sf(self,t,d):
        if d==1: return 0.5-atan(t)/pi
        if d==3:
            x=t/sqrt(3); return 0.5-(atan(x)+x/(1+x*x))/pi
class _N:
    def isf(self,q): return NormalDist().inv_cdf(1-q)
class stats: t=_T(); norm=_N()
def eps(e,s,r=15): return 0.5*(1-(1-2*e/s)**(1/r))
def z(R,N,k,f,cv):
    L=abs(log(R)); return [L/sqrt(1/N+cv), L/sqrt(f*f/N+cv), L/sqrt(1/N+cv+.15**2/k), L/sqrt(1/N+cv+.25**2/k)]
# f=0.02
R=eps(191,3237+3238)/1.982e-3; print('f002',R, z(R,191,2,sqrt(3.28),1/32049))
zb=z(R,191,2,sqrt(3.28),1/32049)[1]; p=2*stats.t.sf(zb,1); print(' t1->z',stats.norm.isf(p/2))
R=2.664e-3/2.096e-3; print('f005Y',R, z(R,123,1,1.7,1/33367))
R=8.576e-4/5.765e-4; print('Ebetter',R,R*sqrt(1/43+1/9941), z(R,43,1,1.7,1/9941))
R=1.141; print('eta100 Y', log(R)/sqrt(1/36+1/2922))
# m1 t
p=2*stats.t.sf(3.44,3); print('m1 t3->z',stats.norm.isf(p/2))
# shifts
s0=1/258+1/34217
for nm,R,N,k,cv in [('f002',eps(191,6475)/1.982e-3,191,2,1/32049),('f005',2.664e-3/2.096e-3,123,1,1/33367)]:
    d=log(R)-log(0.700)
    for dl in (0,.15,.25):
        print(nm,dl, d, d/sqrt(s0+1/N+cv+dl**2/4+dl**2/k))
# bias diffs
r20=0.679; n20=73; c20=1/6166
for nm,R,N,cv in [('e10',0.807,99,1/7370),('e100',1.141,36,1/2922)]:
    print(nm,(log(R)-log(r20))/sqrt(1/n20+c20+1/N+cv))
# best tested
R=1.342e-3/6.603e-4; print('m1/E',R, log(R)/sqrt(1/258+1/11601))
print('E/E',1.134e-3/6.603e-4)
# L_X rises
print(2.023/1.342-1, 2.633/1.342-1, 1.982/1.916-1, 2.096/1.916-1)
# device f
T2=50
for t in (12.8,8.8):
    p=(1-exp(-t*t/T2**2))/2; m=(t/T2**2)*exp(-t*t/T2**2)*0.1
    print(t,p,m,m/p,0.1/t)
print((1-exp(-12.8/50))/2)
# Eq5 T1? unknown
print('chi2 pooled', (8.29+3.28+0.01), sqrt(11.58/5))

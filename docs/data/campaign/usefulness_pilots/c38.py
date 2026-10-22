import numpy as np
from scipy import stats
from scipy.optimize import brentq
def eps(e,n,r): return 0.5*(1-(1-2*e/n)**(1/r))
def chi2(runs):
    E=sum(e for e,n in runs);N=sum(n for e,n in runs);p=E/N
    return sum((e-n*p)**2/(n*p) for e,n in runs)
m1=[(69,3187),(45,3201),(80,3401),(64,3269)]
E=[(56,3379),(56,3308)]
C=[(108,3237),(83,3238)]
for name,runs in [('m1',m1),('E',E),('C',C),('m1Y',[(52,3151),(43,3139),(61,3496)])]:
  c=chi2(runs);k=len(runs);f=max(1,np.sqrt(c/(k-1)))
  e=sum(x for x,_ in runs);n=sum(y for _,y in runs)
  print(name,e,n,'%.4e'%eps(e,n,15),'chi2 %.3f'%c,'f %.3f'%f)
def z(R,N1,N2,f=1,delta=0,k=1):
  s=np.sqrt(f**2/N1+1/N2+delta**2/k); return abs(np.log(R))/s,s
x1=eps(258,13058,15); z1=1.916e-3; R=x1/z1; print('m1 R',R, R*np.sqrt(1/258+1/34217), R*np.sqrt(1.662**2/258+1/34217))
print('a',z(R,258,34217)); zb,_=z(R,258,34217,f=1.662);print('b',zb, 't->gauss', stats.norm.isf(stats.t.sf(zb,3)))
print('c15',z(R,258,34217,delta=.15,k=4),'c25',z(R,258,34217,delta=.25,k=4))
print('d*', brentq(lambda d: z(R,258,34217,delta=d,k=4)[0]-3,0,1))
x=eps(112,6687,15); zz=6.603e-4; R=x/zz; print('E R',x,zz,R,R*np.sqrt(1/112+1/11601))
print('a',z(R,112,11601),'c15',z(R,112,11601,delta=.15,k=2),'c25',z(R,112,11601,delta=.25,k=2),'f166',z(R,112,11601,f=1.66),'f152',z(R,112,11601,f=1.52))
print('d*', brentq(lambda d: z(R,112,11601,delta=d,k=2)[0]-3,0,1))
print('E as single-run k? delta per run, k=2 -> delta^2/2')
xb=eps(43,3383,15); zb=5.765e-4; Rb=xb/zb; print('better',xb,Rb,Rb*np.sqrt(1/43+1/9941),abs(np.log(Rb))/np.sqrt(1/43+1/9941),(Rb-1)/(Rb*np.sqrt(1/43+1/9941)))
x=eps(191,6475,15); zz=1.982e-3; R=x/zz; print('C',x,R,R*np.sqrt(1/191+1/32049), R*np.sqrt(1.81**2/191+1/32049), z(R,191,32049), z(R,191,32049,f=1.81))
print('Y C',eps(141,6409,15),'m1Y',eps(156,9786,15))
print('m1/E',x1/6.603e-4)

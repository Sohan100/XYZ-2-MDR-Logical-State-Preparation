import numpy as np
from scipy.optimize import brentq
def eps(pL,r): return (1-(1-2*pL)**(1/r))/2
# five-run XZZX pool
xs=[(7738,1972704),(7454,1912022),(506,131056),(497,131056),(4614,1198350)]
N=sum(e for e,s in xs); S=sum(s for e,s in xs)
ez=eps(N/S,21); print('pool',N,S,ez)
mu=[S and s*N/S for e,s in xs]; chi=sum((e-m)**2/m for (e,s),m in zip(xs,mu)); print('chi2',chi)
ys=[(55,25400),(58,23388),(56,22969),(43,22621),(53,21488),(47,22512),(57,22512),(33,16550)]
n=sum(e for e,s in ys); T=sum(s for e,s in ys); ey=eps(n/T,15); R=ey/ez
print('LY',n,T,ey,R,np.sqrt(1/n+1/N)*R)
mu=[s*n/T for e,s in ys]; c=sum((e-m)**2/m for (e,s),m in zip(ys,mu)); print('chi2 y',c,len(ys)-1)
za=abs(np.log(R))/np.sqrt(1/n+1/N); f=max(1,np.sqrt(c/7)); print('za',za,'f',f)
# delta*
def zc(d,ns,Nc,lnR,cd=0):
    v=1/sum(1/(1/k+d*d) for k in ns)+1/Nc+cd*cd; return abs(lnR)/np.sqrt(v)
lnR=np.log(R); print('dstar LY',brentq(lambda d: zc(d,[e for e,s in ys],N,lnR)-3,0,2))
# vs single
e1=eps(7738/1972704,21); R1=ey/e1; print('single',R1,abs(np.log(R1))/np.sqrt(1/n+1/7738), brentq(lambda d: zc(d,[e for e,s in ys],7738,np.log(R1))-3,0,2))
# competitor scatter
cs=[(1080,6291200)]+[(k,8388352) for k in (1268,1278,1369,1378)]
for sel in (cs, [cs[0],cs[1],cs[3],cs[4]]):
    N2=sum(e for e,s in sel);S2=sum(s for e,s in sel);mu=[s*N2/S2 for e,s in sel]
    c=sum((e-m)**2/m for (e,s),m in zip(sel,mu)); W=sum(m*(1-m/N2) for m in mu)
    from scipy.stats import chi2
    print('comp chi2',c,len(sel)-1,chi2.sf(c,len(sel)-1),'W',W,'delta',np.sqrt(max(0,c-len(sel)+1)/W))
# r=6d
ey6=eps((52+47)/(13687+13611),30); ez6=3.163e-4; R6=ey6/ez6; print('R6',R6)
print('z at .37',zc(.37,[52,47],14848,np.log(R6)),'with comp .044',zc(.37,[52,47],14848,np.log(R6),.044))
print('dstar',brentq(lambda d: zc(d,[52,47],14848,np.log(R6))-3,0,2))
# better/better
exx=eps((46+47)/(13542+13601),30); Rb=exx/2.033e-4; print('Rb',Rb,abs(np.log(Rb))/np.sqrt(1/93+1/9381),abs(np.log(Rb))/np.sqrt(1/93+2.89/9381),brentq(lambda d: zc(d,[46,47],9381,np.log(Rb))-3,0,2), zc(.37,[46,47],9381,np.log(Rb)))
# k=1
for lr,nn in ((np.log(R6),99),(np.log(exx/ez6),93)):
    print('k1',np.sqrt((lr/3)**2-1/nn-1/14848), abs(lr)/np.sqrt(1/nn+1/14848+.37**2))
# phen98/100 with comp delta: need values; biased10
# SI1000 35
for name,v,k in (('bm',1.170e-3,411),('cg',1.139e-3,1194),('mw',1.1698e-3,313454)):
    for rv,rk in ((6.347e-5,1514),(6.227e-5,1518),(6.278e-5,1514),(6.388e-5,1507)):
        r=v/rv; s=np.sqrt(1/k+1/rk); print(name,round(r,2),round(np.log(r)/s,1),round(np.log(r)/(1.7*s),1), round(np.log(r)/np.sqrt(1/k+1/rk+.137),2))
r=4.137e-4/7.248e-6; s=np.sqrt(1/1735+1/1501); print('sd6',r,np.log(r)/s,np.log(r)/s/1.7)

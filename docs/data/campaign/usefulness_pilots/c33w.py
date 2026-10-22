import numpy as np
from scipy.stats import chi2
def zc(d,ns,Nc,lnR,cd=0):
    v=1/sum(1/(1/k+d*d) for k in ns)+1/Nc+cd*cd; return abs(lnR)/np.sqrt(v)
print('phen',zc(.37,[114,129],518,np.log(2.36)),zc(.37,[114,129],518,np.log(2.36),.044))
b=[15,11,16,14,13,14,11,13,12,19,12]
print('b10',zc(.37,b,7868,np.log(3.35)),zc(.37,b,7868,np.log(3.35),.044))
sel=[(1080,6291200),(1278,8388352),(1369,8388352),(1378,8388352)]
N2=sum(e for e,s in sel);S2=sum(s for e,s in sel);mu=[s*N2/S2 for e,s in sel]
c=sum((e-m)**2/m for (e,s),m in zip(sel,mu)); W=sum(m*(1-m/N2) for m in mu)
print(c,chi2.sf(c,3),np.sqrt((c-3)/W))
sel=[(1080,6291200),(1273,8388352),(1369,8388352),(1378,8388352)]
N2=sum(e for e,s in sel);S2=sum(s for e,s in sel);mu=[s*N2/S2 for e,s in sel]
c=sum((e-m)**2/m for (e,s),m in zip(sel,mu)); W=sum(m*(1-m/N2) for m in mu)
print(c,chi2.sf(c,3),np.sqrt((c-3)/W))
# lambda 1e-3 slopes
def sl(a,ea,b,eb,f): return np.log(b/a)/np.log(f), np.hypot(ea/a,eb/b)/np.log(f)
x=sl(12.263,0.4376,31.361,3.5985,2); z=sl(13.557,0.5623,25.202,1.7584,2); print('b100 2->1',x,z,(x[0]-z[0])/np.hypot(x[1],z[1]))
x=sl(12.202,0.4922,41.536,4.768,2); z=sl(14.706,0.718,31.527,2.363,2); print('pz 2->1',x,z,(x[0]-z[0])/np.hypot(x[1],z[1]))
print('L 1e-3 diff b100',(31.361-25.202)/np.hypot(3.5985,1.7584),'pz',(41.536-31.527)/np.hypot(4.768,2.363))

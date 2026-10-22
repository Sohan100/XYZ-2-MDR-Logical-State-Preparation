import numpy as np
tmin,T1R,tres,pres,T2=1.5,1000.,3.,0.005,50.
ts=tmin*np.log(1+T1R/tmin)
def solve(R,T):
  t=ts
  for _ in range(200): t=ts-tmin*np.log(1+R/T*(t+tres)*T1R/((1-2*pres)*T2**2))
  return t
for k,(R,T) in {'XYZ2 TI':(16.5,4.2),'XZZX TI':(12.9,15.1),'SC TI':(14.5,3.94),'XYZ2 repo':(10.6,10.49),'XZZX repo':(10.3,18.95)}.items(): print(k, round(solve(R,T),2))

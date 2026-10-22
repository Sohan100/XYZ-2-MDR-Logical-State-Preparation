import numpy as np
tm,T1R,tres,pres,T2=1.5,1000.,3.,0.005,50.
pR=lambda t:1-(1-np.exp(-t/tm))*np.exp(-t/T1R)
pRR=lambda t:pres*(1-pR(t))+pR(t)*(1-pres)
pT2=lambda t:(1-np.exp(-(t+tres)**2/T2**2))/2
ts=tm*np.log(1+T1R/tm); print('tauR*',ts)
for t in [9.8,ts,5.8]:
  pt=pT2(t)*1.05
  print(t,'pR %.4f pRR %.4f pT2 %.4f pT %.4f'%(pR(t),pRR(t),pT2(t),pt))
  for n,(g,T,R) in {'XYZ2':(0.465,4.2,16.5),'XZZX':(0.37,15.1,12.9),'SC':(0.82,3.94,14.5)}.items():
    print('  ',n,'%.3f'%(0.1/g+100*pt/T+100*pRR(t)/R), ' pT2 only %.3f'%(0.1/g+100*pT2(t)/T+100*pRR(t)/R))
def qec(RT,TT):
  t=ts
  for i in range(200): t=ts-tm*np.log(1+ RT/TT*(t+tres)*T1R/((1-2*pres)*T2**2))
  return t
for n,(R,T) in {'XYZ2 TI':(16.5,4.2),'XZZX TI':(12.9,15.1),'SC TI':(14.5,3.94),'XYZ2 repo':(10.6,10.49),'XZZX repo':(10.3,18.95)}.items(): print(n,'%.2f'%qec(R,T))
# f
for t in [12.8,8.8]:
  inc=(t/T2**2)*np.exp(-t**2/T2**2)*0.1; p2=(1-np.exp(-t**2/T2**2))/2
  print(t,'coh f',inc/p2,inc/(1.05*p2),'lin',0.1/t, 'f=0.02 at ns', 0.02*p2/((t/T2**2)*np.exp(-t**2/T2**2))*1000, 0.02*t*1000)
  T1=(20+1)*T2**2/(2*t)  # eta=2 tau T1/T2^2 -1
  pt1=(1-np.exp(-0.1/T1))/4; pz=(1-np.exp(-0.01/T2**2))/2
  print('  T1',T1/1000,'pt1',pt1,'pz',pz,'f Z/pT2',pz/p2,'tot/pT',(pt1+pz)/(1.05*p2))

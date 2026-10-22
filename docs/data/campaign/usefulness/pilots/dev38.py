import numpy as np
tmin,T1R,tres,pres,T2=1.5,1000.,3.,0.005,50.
pR=lambda t:1-(1-np.exp(-t/tmin))*np.exp(-t/T1R)
pRR=lambda t:pres*(1-pR(t))+pR(t)*(1-pres)
pT2=lambda t:(1-np.exp(-(t+tres)**2/T2**2))/2
ts=tmin*np.log(1+T1R/tmin); print('tau*',ts)
planes={'XYZ2':(0.465,4.2,16.5),'XZZX':(0.37,15.1,12.9),'SC':(0.82,3.94,14.5)}
for t in (9.8,5.8):
  pt2=pT2(t); pt=1.05*pt2
  print(f"t={t} pR={pR(t)*100:.3f} pRR={pRR(t)*100:.3f} pT2={pt2*100:.3f} pT={pt*100:.3f}")
  for k,(g,T,R) in planes.items():
    print('  ',k, round(0.1/g+pt*100/T+pRR(t)*100/R,3))
  # f under per-phase convention, 100 ns idle
  tw=t+tres; T1=21*T2**2/(2*tw)  # eta=20 from Eq5
  tau=0.1
  p1=(1-np.exp(-tau/T1))/4*2  # pT1 = X+Y? Eq3
  p2g=(1-np.exp(-tau**2/T2**2))/2
  print('   T1 (eta=20)',T1,'pT1(100ns)',(1-np.exp(-tau/T1))/4,'gauss Z 100ns',p2g, 'fZ',p2g/pt2,'ftot',( (1-np.exp(-tau/T1))/4 + p2g)/pt)
  # coherent increment
  inc=(tw/T2**2)*np.exp(-tw**2/T2**2)*0.1
  print('   coherent f',inc/pt2,inc/pt,'linear',0.1/tw, 'f=0.02 at ns gauss',0.02/(inc/pt2)*100,'lin',0.02/(0.1/tw)*100)

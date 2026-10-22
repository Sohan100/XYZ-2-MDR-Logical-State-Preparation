import math
tm,T1R,tres,pres,T2=1.5,1000.,3.,0.005,50.
pR=lambda t:1-(1-math.exp(-t/tm))*math.exp(-t/T1R)
pRR=lambda t:pres*(1-pR(t))+pR(t)*(1-pres)
pT2=lambda t:(1-math.exp(-(t+tres)**2/T2**2))/2
print('tauR*',tm*math.log(1+T1R/tm))
for t in (9.8,9.756,5.8):
  print(t,'pR %.4f pRR %.4f pT2 %.4f pT %.4f'%(pR(t),pRR(t),pT2(t),pT2(t)*21/20))
tab={'xyz2':(0.465,4.2,16.5),'xzzx':(0.37,15.1,12.9),'sc':(0.82,3.94,14.5)}
for t in (9.8,5.8):
  for k,(g,T,R) in tab.items():
    pt=pT2(t)*21/20*100
    print(t,k,'%.3f'%(0.1/g+pt/T+pRR(t)*100/R), 'with pT2 only %.3f'%(0.1/g+pT2(t)*100/T+pRR(t)*100/R))
def qec(pth_R,pth_T):
  ts=tm*math.log(1+T1R/tm); t=ts
  for i in range(200): t=ts-tm*math.log(1+pth_R/pth_T*(t+tres)*T1R/((1-2*pres)*T2**2))
  return t
for k,(g,T,R) in tab.items(): print('eq14',k,qec(R,T))
print('repo xyz2',qec(10.6,10.49),'xzzx',qec(10.3,18.95))
# f
for t in (12.8,8.8):
  p2=pT2(t-tres); pT=p2*21/20; p1=pT-p2
  T1=-t/math.log(1-4*p1)
  a=(1-math.exp(-0.1/T1))/4; z=(1-math.exp(-0.01/T2**2))/2
  inc=(t/T2**2)*math.exp(-t**2/T2**2)*0.1
  print(t,'T1 %.3f ms'%(T1/1000),'p1 %.2e z %.2e'%(a,z),'fZ %.2e ftot %.2e'%(z/p2,(a+z)/pT),'coh %.4f %.4f'%(inc/p2,inc/pT),'lin %.4f'%(0.1/t))
  # layer for f=0.02
  print(' f=.02 at coh ns', 0.02*p2/((t/T2**2)*math.exp(-t**2/T2**2))*1000, 'lin', 0.02*t*1000)

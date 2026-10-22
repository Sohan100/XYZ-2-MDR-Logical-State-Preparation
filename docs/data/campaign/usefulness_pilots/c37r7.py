import math
from math import log, sqrt, exp
from scipy import stats
def tconv(t,dof): return stats.norm.isf(stats.t.sf(t,dof))
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
# cross ratio m1 xyz2 X vs E xzzx X
E1=258; NB=11601; k=4; f=1.662
R=1.342e-3/6.603e-4; L=log(R)
a=L/sqrt(1/E1+1/NB); bG=L/sqrt(f*f/E1+1/NB); b=tconv(bG,3)
c=lambda d: L/sqrt(1/E1+1/NB+d*d/k)
ds=sqrt(((L/3)**2-1/E1-1/NB)*k)
print('cross',R,'a',a,'bG',bG,'b(t3)',b,'c15',c(.15),'c25',c(.25),'c37',c(.37),'d*',ds)
# E: delta*
L=log(1.718); print('E c37', L/sqrt(1/112+1/11601+0.37**2/2), 'c065',L/sqrt(1/112+1/11601+0.065**2/2))
# eta20 3% delta*
# device f per-phase
T2=50.; dt=0.1
for tw,T1 in ((12.8,2051.),(8.8,2980.)):
    pT2w=(1-exp(-tw**2/T2**2))/2; pTw=pT2w*1.05
    q2=(1-exp(-dt**2/T2**2))/2; q1=dt/(4*T1)
    print(f"tw={tw} pT2w={pT2w:.4f} pTw={pTw:.4f} q2={q2:.2e} q1={q1:.2e} fZ={q2/pT2w:.2e} ftot={(q1+q2)/pTw:.2e} eta_layer={q2/q1:.2f}")
    # Gaussian coherent f per 100ns and layer time for f=0.02
    g=(tw/T2**2)*exp(-tw**2/T2**2)*dt/pT2w
    print('  gauss f',g,'ns for 0.02',100*0.02/g,'linear f',dt/tw,'ns for 0.02',0.02*tw*1000)
    x=tw**2/T2**2; print('  x',x,'factor',x*exp(-x)/(1-exp(-x)))
for s in (0.016984,0.024813):
    print(s/15*1e3, s/15/13e-6, s/15/9e-6)

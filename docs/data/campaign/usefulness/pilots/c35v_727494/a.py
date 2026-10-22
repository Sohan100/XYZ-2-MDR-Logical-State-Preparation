import math
from scipy.optimize import brentq
from scipy import stats
def z(R,n1s,N2,f1=None,f2=1.7,tdof=None):
    N1=sum(n1s); L=abs(math.log(R))
    a=L/math.sqrt(1/N1+1/N2)
    if f1 is None: f1=1.7 if len(n1s)==1 else 1.0
    b=L/math.sqrt(f1**2/N1+f2**2/N2)
    if tdof: b=stats.norm.isf(stats.t.sf(b,tdof))
    def c(dl): return L/math.sqrt(1/sum(1/(1/n+dl**2) for n in n1s)+1/N2)
    try: ds=brentq(lambda dl:c(dl)-3,0,5)
    except Exception: ds=None
    return round(a,2),round(b,2),round(c(.15),2),round(c(.25),2),ds and round(ds,3)
print('m1 X', z(0.700,[80,69,45,64],34217,f1=1.66,tdof=3))
print('m1 Y v Z', z(0.743,[61,52,43],25634))
def eps(pL,r): return (1-(1-2*pL)**(1/r))/2
e=[3.150e-4,9.715e-5]
e7=eps(29/13469,21); print('xyz2 d7 Y eps',e7)
s=(math.log(e[1])-math.log(e[0]))/80
interp=math.exp(math.log(e[0])+s*34); print('interp',interp,'R',e7/interp)
w=34/80; vI=(1-w)**2/4486+w**2/999
L=abs(math.log(e7/interp))
print('195 a', L/math.sqrt(1/29+vI), 'b', L/math.sqrt(1/29+2.89*vI))
print('195 c*', brentq(lambda dl: L/math.sqrt(1/sum(1/(1/n+dl**2) for n in [12,17])+vI)-3,0,2))
qe=161+(math.log(e7)-math.log(e[0]))/s; print('qubit eq',qe,195/qe,'sig_q',(1/math.sqrt(29))/abs(s))
print('vs d11', e7/e[1], e7/e[1]*math.sqrt(1/29+1/999))
print('lambda', 6.678e-4/e7, math.sqrt(1/218+1/29)*6.678e-4/e7)
print('r6d X v X k=1')

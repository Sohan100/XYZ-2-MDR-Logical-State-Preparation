import math
from scipy.optimize import brentq
def eps(err,shots,r):
    pL=err/shots; return (1-(1-2*pL)**(1/r))/2
runs=[(416,11248,15),(500,14791,15)]
E=sum(e for e,s,r in runs); S=sum(s for e,s,r in runs)
ex=eps(E,S,15)
mu=[E*s/S for e,s,r in runs]; chi=sum((e-m)**2/m for (e,s,r),m in zip(runs,mu)); f=max(1,math.sqrt(chi))
print('pool',ex,E,'chi2',chi,'f',f)
for name,(Nc,Sc,rc) in [('corr hi',(25446,940032,21)),('corr bq',(468,16368,21)),('bm',(198,8943,21))]:
    ec=eps(Nc,Sc,rc); R=ex/ec; l=math.log(R)
    za=l/math.sqrt(1/E+1/Nc); fc=1.7
    zb=l/math.sqrt(f**2/E+fc**2/Nc)
    vc=lambda d: 1/sum(1/(1/e+d*d) for e,s,r in runs)+1/Nc
    ds=brentq(lambda d: l/math.sqrt(vc(d))-3,0,5)
    print(name,'R=%.3f(%.3f)'%(R,R*math.sqrt(1/E+1/Nc)),'za %.2f zb %.2f'%(za,zb),'z(0.37) %.2f'%(l/math.sqrt(vc(0.37))),'delta* %.3f'%ds)
# single stat run vs bm
for e,s in [(500,14791)]:
    ex1=eps(e,s,15); ec=eps(198,8943,21); l=math.log(ex1/ec)
    print('single BM-BM', ex1/ec, l/math.sqrt(1/e+1/198), l/math.sqrt(1.7**2/e+1.7**2/198), math.sqrt((l/3)**2-1/e-1/198))

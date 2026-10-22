import math
from scipy.stats import t as T, norm
from scipy.optimize import brentq
def eps(err,shots,r):
  pL=err/shots; return (1-(1-2*pL)**(1/r))/2
def pool(runs,r):
  E=sum(e for e,s in runs); S=sum(s for e,s in runs)
  ep=eps(E,S,r)
  k=len(runs)
  if k>1:
    mu=[E*s/S for e,s in runs]; chi=sum((e-m)**2/m for (e,s),m in zip(runs,mu)); f=max(1,math.sqrt(chi/(k-1)))
  else: chi=None; f=1.7
  return ep,E,chi,f,k
def z(xruns,xr,cruns,cr,bm=True,cf=None,pois_err=True):
  ex,Ex,chix,fx,kx=pool(xruns,xr); ec,Ec,chic,fc,kc=pool(cruns,cr)
  if cf is not None: fc=cf
  R=ex/ec; l=abs(math.log(R))
  sa=math.sqrt(1/Ex+1/Ec); za=l/sa
  sb=math.sqrt(fx**2/Ex+fc**2/Ec); zb=l/sb
  # t conversion if f>1 estimated from replicates
  dofs=[]
  if kx>1 and fx>1: dofs.append(kx-1)
  if kc>1 and fc>1: dofs.append(kc-1)
  zbg=zb
  if dofs:
    p=T.sf(zb,min(dofs)); zbg=norm.isf(p)
  def zc(d):
    v=1/sum(1/(1/e+d*d) for e,s in xruns)+1/Ec
    return l/math.sqrt(v)
  zc15=zc(0.15); zc25=zc(0.25); zc37=zc(0.37)
  try: ds=brentq(lambda d: zc(d)-3,0,5) if zc(0)>3 else 0
  except Exception: ds=float('nan')
  return dict(R=round(R,4),Rerr=round(R*sa,4),za=round(za,2),zb=round(zb,2),zbg=round(zbg,2),zc065=round(zc(0.065),2),zc074=round(zc(0.074),2),zc15=round(zc15,2),zc25=round(zc25,2),zc37=round(zc37,2),dstar=round(ds,3),chix=chix,fx=fx,fc=fc)

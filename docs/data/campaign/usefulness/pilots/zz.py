import math
from scipy.optimize import brentq
from scipy.stats import t as tdist, norm
def eps(e,s,r): 
    pL=e/s; return (1-(1-2*pL)**(1/r))/2
def pool(runs):
    # runs: list of (err, shots); returns pooled rate and chi2
    E=sum(e for e,s in runs); S=sum(s for e,s in runs)
    mu=[E*s/S for e,s in runs]
    chi=sum((e-m)**2/m for (e,s),m in zip(runs,mu))
    return E,S,chi,len(runs)-1
def zc(lnR, ns, Nc, d):
    v=1/sum(1/(1/n+d*d) for n in ns)+1/Nc
    return abs(lnR)/math.sqrt(v)
def dstar(lnR, ns, Nc):
    if zc(lnR,ns,Nc,0)<3: return None
    f=lambda d: zc(lnR,ns,Nc,d)-3
    return brentq(f,0,10)
def tconv(z,dof):
    p=tdist.sf(z,dof); return norm.isf(p)
def report(R, ns, Nc, f1=1.0, f2=1.7, dof1=None):
    lnR=math.log(R); N1=sum(ns)
    za=abs(lnR)/math.sqrt(1/N1+1/Nc)
    zb=abs(lnR)/math.sqrt(f1**2/N1+f2**2/Nc)
    if f1>1 and dof1: zb_g=tconv(zb,dof1)
    else: zb_g=zb
    out=dict(R=R, err=R*math.sqrt(1/N1+1/Nc), za=za, zb=zb, zb_g=zb_g, c15=zc(lnR,ns,Nc,.15), c25=zc(lnR,ns,Nc,.25), c37=zc(lnR,ns,Nc,.37), c065=zc(lnR,ns,Nc,.065), c074=zc(lnR,ns,Nc,.074), dstar=dstar(lnR,ns,Nc))
    return {k:(round(v,4) if isinstance(v,float) else v) for k,v in out.items()}

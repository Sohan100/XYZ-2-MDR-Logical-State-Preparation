import math
from scipy.optimize import brentq
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def chi(runs):
    E=sum(e for e,s in runs); S=sum(s for e,s in runs); p=E/S
    c=sum((e-s*p)**2/(s*p) for e,s in runs)
    W=sum((s*p)*(1-s/S) for e,s in runs)
    return c,len(runs)-1,E,W
print("flagged sets: delta_E = sqrt((chi2-dof)/E), delta_W = sqrt((chi2-dof)/W)")
sets={
 'phen_b10 2% X':[(55,25400),(45,23371),(38,23016),(38,21488),(53,21488),(41,22512),(66,22569),(62,22000),(46,17450),(29,17450)],
 'd7 3% Y':[(32,7152),(26,7152),(21,7152),(33,6774),(10,6025)],
 'hw_tr 3% X':[(40,33432),(22,33653)],
 'm1 X':[(80,3401),(69,3187),(45,3201),(64,3269)],
}
for k,v in sets.items():
    c,d,E,W=chi(v)
    print(k, 'chi2 %.2f/%d E %d W %.1f dE %.3f dW %.3f'%(c,d,E,W,math.sqrt((c-d)/E),math.sqrt((c-d)/W)))
def zc(runs, Ncomp, lnR, delta):
    v=1/sum(1/(1/e+delta**2) for e,s in runs)+1/Ncomp
    return abs(lnR)/math.sqrt(v)
def dstar(runs,Ncomp,lnR):
    f=lambda d: zc(runs,Ncomp,lnR,d)-3
    if f(0)<0: return 0
    if f(5)>0: return float('inf')
    return brentq(f,0,5)
def report(name, runs, r, comp_eps, Ncomp):
    E=sum(e for e,s in runs); S=sum(s for e,s in runs)
    x=eps(E,S,r); R=x/comp_eps; lnR=math.log(R)
    sa=math.sqrt(1/E+1/Ncomp); za=abs(lnR)/sa
    if len(runs)>1:
        c,d,_,_=chi(runs); f=max(1,math.sqrt(c/d))
    else: f=1.7
    zb=abs(lnR)/math.sqrt(f*f/E+1/Ncomp)
    zb17=abs(lnR)/math.sqrt(f*f/E+2.89/Ncomp)
    ds=dstar(runs,Ncomp,lnR)
    print(f"{name}: R {R:.3f}({R*sa:.3f}) za {za:.2f} zb {zb:.2f} (f={f:.2f}; f1.7 on comp {zb17:.2f}) zc@0.37 {zc(runs,Ncomp,lnR,0.37):.2f} zc@0.374 {zc(runs,Ncomp,lnR,0.374):.2f} delta* {ds:.3f}")
report('r6d Y',[(52,13687),(47,13611)],30,3.163e-4,14848)
report('r6d X',[(46,13542),(47,13601)],30,3.163e-4,14848)
report('phen 98/100',[(114,6896),(129,7152)],21,3.550e-4,518)
report('biased10 Y',[(15,3119),(11,3202),(16,3194),(14,3444),(13,3415),(14,3328),(11,3328),(13,3584),(12,3550),(19,3750),(12,3750)],15,7.947e-5,7868)
report('hw_rr BM per-round',[(887,59048)],15,3.638e-4,14883)

import math
from scipy import stats, optimize
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def pt(name, runs, r, B, NB, fB=1.7, single_comp=True):
    e=sum(a for a,b in runs); s=sum(b for a,b in runs); ea=eps(e,s,r)
    p=e/s; k=len(runs)
    chi=sum((a-p*b)**2/(p*b) for a,b in runs) if k>1 else 0
    R=ea/B; L=abs(math.log(R))
    za=L/math.sqrt(1/e+1/NB)
    if k>1:
        f=max(1,math.sqrt(chi/(k-1)))
    else: f=1.7
    fb = fB if single_comp else 1.0
    zb=L/math.sqrt(f**2/e+fb**2/NB)
    if k>1 and f>1:
        p1=stats.t.sf(zb,k-1); zbg=stats.norm.isf(p1)
    else: zbg=zb
    def zc(d):
        v=1/sum(1/(1/a+d*d) for a,b in runs)+1/NB
        return L/math.sqrt(v)
    try: ds=optimize.brentq(lambda d: zc(d)-3, 0, 5)
    except Exception: ds=float('nan')
    print(f"{name}: R={R:.3f}+-{R*math.sqrt(1/e+1/NB):.3f} chi2={chi:.2f}/{k-1} f={f:.2f} (a)={za:.2f} (b)={zbg:.2f} (raw {zb:.2f}) (c)={zc(.15):.2f}/{zc(.25):.2f} z@.37={zc(.37):.2f} delta*={ds:.3f}")
pt("r6d Y", [(52,13687),(47,13611)],30,3.163e-4,14848)
pt("r6d X", [(46,13542),(47,13601)],30,3.163e-4,14848)
pt("6% Y", [(218,21968)],15,1.089e-3,29621)
pt("4% X r3d", [(62,22344)],15,3.050e-4,11782)
pt("3% Y", [(36,33978),(37,34130)],15,1.054e-4,6166)
pt("m1 X", [(69,3187),(45,3201),(80,3401),(64,3269)],15,1.916e-3,34217)
pt("A X", [(174,3179)],15,5.116e-3,57973)
pt("E Y", [(56,3379),(56,3308)],15,6.603e-4,11601)
pt("f02 X", [(108,3237),(83,3238)],15,1.982e-3,32049)
pt("f05 Y", [(123,3195)],15,2.096e-3,33367)
pt("eta100 X better", [(29,46148)],15,1.740e-5,1152)

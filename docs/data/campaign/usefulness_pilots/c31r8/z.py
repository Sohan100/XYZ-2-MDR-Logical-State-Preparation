from math import log,sqrt
from scipy import stats
from scipy.optimize import brentq
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def pool(runs):
    e=sum(a for a,b in runs); s=sum(b for a,b in runs); p=e/s
    chi=sum((a-p*b)**2/(p*b) for a,b in runs); return e,s,chi,len(runs)-1
def tconv(z,dof): return stats.norm.isf(stats.t.sf(z,dof))
def cvar(runs,r,delta):
    # inverse-variance pooled log variance of XYZ2 runs with per-run offset (C33)
    return 1/sum(1/(1/a+delta**2) for a,b in runs)
def zs(name,runs,r,B,cv,fB=1.0):
    e,s,chi,dof=pool(runs); ea=eps(e,s,r); R=ea/B; L=abs(log(R)); k=len(runs)
    sp=sqrt(1/e+cv)
    f=max(1,sqrt(chi/dof)) if dof>0 else 1.7
    zb=L/sqrt(f*f/e+fB*fB*cv)
    zbt=tconv(zb,dof) if (dof>0 and f>1) else zb
    # delta* simple form (C31) and C33 inverse-variance form
    d3=sqrt(max(0,(L/3)**2-1/e-cv)*k)
    g=lambda d: L/sqrt(cvar(runs,r,d)+cv)-3
    try: d3b=brentq(g,1e-6,5)
    except Exception: d3b=float('nan')
    z37=L/sqrt(cvar(runs,r,0.37)+cv)
    z15=L/sqrt(1/e+cv+0.15**2/k); z25=L/sqrt(1/e+cv+0.25**2/k)
    print(f"{name}: eps={ea:.4e} N={e}/{s} chi2={chi:.2f}/{dof} R={R:.4f}({R*sp:.3f}) a={L/sp:.2f} f={f:.2f} b={zb:.2f} bt={zbt:.2f} c15/25={z15:.2f}/{z25:.2f} d*={d3:.3f} d*(ivw)={d3b:.3f} z@0.37={z37:.2f}")
X2=[(66,22569),(62,22000),(55,25400),(45,23371),(38,23016),(38,21488),(53,21488),(41,22512),(46,17450),(29,17450)]
Y2=[(43,22621),(55,25400),(58,23388),(56,22969),(53,21488),(47,22512),(57,22512),(33,16550)]
B2=1.861e-4;N2=20809
zs("2% X all",X2,15,B2,1/N2)
zs("2% Y all",Y2,15,B2,1/N2)
zs("2% Y vs single XZZX run (C33)",Y2,15,1.875e-4,1/7738,fB=1.7)
zs("3% X",[(254,20416),(260,19762)],15,1.067e-3,1/25589,fB=1.7)
zs("2.5% X",[(119,20104),(117,21150)],15,4.897e-4,1/16015,fB=1.7)
zs("18/16 d4",[(408,12224)],9,5.441e-3,1/503,fB=1.7)
I=(5.441e-3)**(7/9)*(3.348e-3)**(2/9); cvI=(7/9)**2/503+(2/9)**2/490
zs("18/16 interp",[(408,12224)],9,I,cvI,fB=1.7)
zs("q0 L_Y",[(112,34800),(110,34800),(99,34800),(95,34800)],15,1.383e-4,1/759,fB=1.7)
zs("q0 L_X",[(100,34800),(88,34800),(72,34800),(92,34800)],15,1.383e-4,1/759,fB=1.7)
zs("q0 18/16",[(426,24560)],9,3.141e-3,1/596,fB=1.7)
# r=1 archived (eps=pL)
for nm,(e,s,eb,sb) in {"r1 archived X":(284,196352,1080,6291200),"r1 unarch X":(470,311040,1278,8388352),"r1 unarch Y":(753,327424,1278,8388352)}.items():
    a=e/s; b=eb/sb; R=a/b; sp=sqrt(1/e+1/eb); L=log(R)
    zb=L/sqrt(2.89/e+2.89/eb); d3=sqrt((L/3)**2-1/e-1/eb)
    print(nm, f"{a:.4e}({a/sqrt(e):.1e}) {b:.4e}({b/sqrt(eb):.1e}) R={R:.3f}({R*sp:.3f}) a={L/sp:.1f} b={zb:.1f} c15/25={L/sqrt(1/e+1/eb+.0225):.1f}/{L/sqrt(1/e+1/eb+.0625):.1f} d*={d3:.3f}")

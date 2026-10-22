import math
from scipy import stats, optimize
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def pool(runs,r):
    e=sum(a for a,b in runs); s=sum(b for a,b in runs); p=e/s
    chi=sum((a-p*b)**2/(p*b) for a,b in runs) if len(runs)>1 else 0
    return eps(e,s,r),e,chi,len(runs)-1
def z(name,A,rA,B,NB,fB=1.7,bm=True):
    ea,na,chi,dof=pool(A,rA); R=ea/B; L=abs(math.log(R))
    sa=math.sqrt(1/na+1/NB)
    f=max(1,math.sqrt(chi/dof)) if dof>0 else 1.7
    sb=math.sqrt(f*f/na+fB*fB/NB); zb=L/sb
    if dof>0 and f>1:
        p=2*stats.t.sf(zb,dof); zbs=f"{stats.norm.isf(p/2):.2f}(t{zb:.2f},f{f:.2f})"
    else: zbs=f"{zb:.2f}"
    def var_c(d): return 1/sum(1/(1/a+d*d) for a,b in A)+1/NB
    zc=[L/math.sqrt(var_c(d)) for d in (0.15,0.25,0.37)]
    try: ds=optimize.brentq(lambda d: L/math.sqrt(var_c(d))-3,0,5)
    except Exception: ds=None
    print(f"{name}: eps={ea:.4e} N={na} chi2={chi:.2f}/{dof} R={R:.3f}({R*sa:.3f}) a={L/sa:.2f} b={zbs} c={zc[0]:.2f}/{zc[1]:.2f}/{zc[2]:.2f} d*={ds}")
    return R,sa
z("6% Y",[(218,21968)],15,1.089e-3,29621)
z("6% Y vs XZZX bm",[(218,21968)],15,1.078e-3,238)
z("6% X vs Z",[(210,23755)],15,6.635e-4,18322)
z("4% X r3d",[(62,22344)],15,3.050e-4,11782)
z("4% Y vs Z",[(44,24063)],15,2.016e-4,7847)
z("3% Y",[(36,33978),(37,34130)],15,1.054e-4,6166)
z("3% X vs Z",[(40,33432),(22,33653)],15,6.775e-5,3985)
z("eta10 Y",[(48,34284),(51,33493)],15,1.208e-4,7370)
z("eta10 X vs Z",[(45,34371),(46,33496)],15,8.743e-5,5396)
z("eta100 Y",[(36,46743)],15,4.502e-5,2922)
z("eta100 X vs Z",[(29,46148)],15,1.740e-5,1152)
z("r6d Y",[(52,13687),(47,13611)],30,3.163e-4,14848)
z("r6d Y k=1 merged",[(99,27298)],30,3.163e-4,14848)
z("r6d X",[(46,13542),(47,13601)],30,3.163e-4,14848)
z("r6d X vs Z",[(46,13542),(47,13601)],30,2.033e-4,9381)
z("r6d X vs Z k=1",[(93,27143)],30,2.033e-4,9381)
I=math.exp(math.log(3.150e-4)+(195-161)/80*(math.log(9.715e-5)-math.log(3.150e-4))); print("interp",I)
# interp variance: weights
w11=(195-161)/80; w9=1-w11; NBeff=1/(w9**2/4486+w11**2/999); print("NBeff",NBeff)
z("195 Y interp",[(12,5410),(17,8059)],21,I,NBeff)
z("195 Y d11",[(12,5410),(17,8059)],21,9.715e-5,999)
z("m1 X",[(69,3187),(45,3201),(80,3401),(64,3269)],15,1.916e-3,34217)
z("m1 Y vs Z",[(52,3151),(43,3139),(61,3496)],15,1.452e-3,25634)
I2=math.exp(math.log(8.031e-4)+(195-161)/80*(math.log(3.458e-4)-math.log(8.031e-4)))
NB2=1/(w9**2/7709+w11**2/2025)
z("m1 195 X",[(11,1215),(12,1214)],21,I2,NB2)
z("f002",[(108,3237),(83,3238)],15,1.982e-3,32049)
z("E",[(56,3379),(56,3308)],15,6.603e-4,11601)
z("phen0 Y",[(112,34800),(110,34800),(99,34800),(95,34800)],15,1.383e-4,759)
# r=1
print("r=1")
for nm,(e,s),(e2,s2),bm in [("archived cg X",(1952,2096896),(1053,20971264),False),("depth6 bm X",(343,360192),(1116,20971264),True),("depth6 cg X",(1919,2096896),(1116,20971264),False),("both bm X",(293,327424),(1116,20971264),True),("both Y vs X",(428,343808),(1116,20971264),True),("second pair bm",(171,179968),(1146,20971264),True)]:
    pa=e/s; pb=e2/s2; R=pa/pb; rel=math.sqrt(1/e+1/e2)
    zc = math.log(R)/math.sqrt(1/e+1/e2+0.0625) if bm else float('nan')
    print(nm, f"{pa:.4e}({pa/math.sqrt(e):.2e}) {pb:.4e}({pb/math.sqrt(e2):.2e}) R={R:.2f}({R*rel:.2f}) a={math.log(R)/rel:.1f} b={math.log(R)/math.sqrt(2.89/e+2.89/e2):.1f} c25={zc:.2f}")
# qubit equivalence
lnA=math.log(3.150e-4); lnB=math.log(9.715e-5)
x=math.log(1.027e-4)
q=161+80*(x-lnA)/(lnB-lnA); print("q equiv",q)

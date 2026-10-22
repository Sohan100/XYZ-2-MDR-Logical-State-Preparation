import math
from scipy import stats
def eps(pL,r): return (1-(1-2*pL)**(1/r))/2
def chi2(runs):
    E=sum(n for n,s in runs); S=sum(s for n,s in runs)
    return sum((n-E*s/S)**2/(E*s/S) for n,s in runs)
def tconv(t,dof): return stats.norm.isf(stats.t.sf(t,dof))
def analyze(name,runs,r,ceps,cN,cf=1.0,useT=True):
    E=sum(n for n,s in runs); S=sum(s for n,s in runs); k=len(runs)
    R=eps(E/S,r)/ceps; l=abs(math.log(R))
    c2=chi2(runs) if k>1 else None
    f=max(1,math.sqrt(c2/(k-1))) if k>1 else 1.7
    za=l/math.sqrt(1/E+1/cN); zb=l/math.sqrt(f**2/E+cf**2/cN)
    zbt=tconv(zb,k-1) if (k>1 and f>1) else zb
    v=(l/3)**2-1/E-1/cN; ds=math.sqrt(k*v) if v>0 else float('nan')
    # C33 inverse-variance version
    def zc(d):
        var=1/sum(1/(1/n+d*d) for n,s in runs)+1/cN; return l/math.sqrt(var)
    lo,hi=0,5
    if zc(0)<3: ds2=float('nan')
    else:
        for _ in range(60):
            m=(lo+hi)/2
            if zc(m)>3: lo=m
            else: hi=m
        ds2=lo
    print(f"{name}: R {R:.4f} sigR {R*math.sqrt(1/E+1/cN):.3f} chi2 {c2} f {f:.3f} za {za:.2f} zb {zb:.2f} zbt {zbt:.2f} d* {ds:.3f} d*IV {ds2:.3f} z(0.37) {zc(0.37):.2f}")
LY=[(43,22621),(55,25400),(58,23388),(56,22969),(53,21488),(47,22512),(57,22512),(33,16550)]
analyze('2% LY pool',LY,15,1.861e-4,20809)
analyze('2% LY single',LY,15,1.875e-4,7738,1.7)
analyze('2% LY -b10Y',[x for i,x in enumerate(LY) if i!=1],15,1.861e-4,20809)
LX=[(66,22569),(62,22000),(55,25400),(45,23371),(38,23016),(38,21488),(53,21488),(41,22512),(46,17450),(29,17450)]
analyze('2% LX',LX,15,1.861e-4,20809)
analyze('2% LX -b10X',[x for i,x in enumerate(LX) if i!=2],15,1.861e-4,20809)
analyze('2% LX two stat',LX[:2],15,1.861e-4,20809)
analyze('2% LX other 8',LX[2:],15,1.861e-4,20809)
# q=0
qY=[(110,34800),(99,34800),(95,34800),(112,34800)]
qX=[(88,34800),(72,34800),(92,34800),(100,34800)]
analyze('q0 LY',qY,15,1.383e-4,759,1.7)
analyze('q0 LX',qX,15,1.383e-4,759,1.7)
# r=1
analyze('r1 arch',[(284,196352)],1,1.717e-4,1080,1.7)
analyze('r1 unarch',[(470,311040)],1,1.524e-4,1278,1.7)
print('LY unarch r1', (753/327424)/1.524e-4, (753/327424)/1.524e-4*math.sqrt(1/753+1/1278))
# 1.5%
X15=[(22,23917),(15,24557),(13,12941),(14,12798),(11,12718),(7,12898)]
analyze('1.5 X',X15,15,5.156e-5,2718,1.7)
analyze('1.5 X two',X15[:2],15,5.156e-5,2718,1.7)

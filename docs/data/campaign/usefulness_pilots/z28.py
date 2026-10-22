import math
from scipy import stats
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def go(name,runs,r,B,NB,f1=None):
    e=sum(a for a,b in runs); s=sum(b for a,b in runs); A=eps(e,s,r)
    k=len(runs); p=e/s
    chi=sum((a-p*b)**2/(p*b) for a,b in runs) if k>1 else 0
    R=A/B; L=abs(math.log(R))
    za=L/math.sqrt(1/e+1/NB)
    if k==1: f=1.7; zb=L/math.sqrt(f*f/e+1/NB); tb=None
    else:
        f=max(1,math.sqrt(chi/(k-1)))
        zb=L/math.sqrt(f*f/e+1/NB)
        if f>1:
            pv=stats.t.sf(zb,k-1); tb=zb; zb=stats.norm.isf(pv)
        else: tb=None
    zc=[L/math.sqrt(1/e+1/NB+d*d/k) for d in (0.15,0.25)]
    # delta for which zc=3
    dd=(L/3)**2-1/e-1/NB; dmax=math.sqrt(dd*k) if dd>0 else 0
    print(f"{name}: A={A:.4e} N={e} chi2={chi:.2f}/{k-1} R={R:.3f}({R*math.sqrt(1/e+1/NB):.3f}) za={za:.2f} zb={zb:.2f} t={tb} f={f:.2f} zc={zc[0]:.2f}/{zc[1]:.2f} dmax={dmax:.3f}")
go("r6d",[(52,13687),(47,13611)],30,3.163e-4,14848)
go("6%",[(218,21968)],15,1.089e-3,29621)
go("4%",[(62,22344)],15,3.050e-4,11782)
go("3%Y",[(36,33978),(37,34130)],15,1.054e-4,6166)
go("eta10",[(48,34284),(51,33493)],15,1.208e-4,7370)
go("eta100",[(36,46743)],15,4.502e-5,2922)
I=math.exp(math.log(3.150e-4)+(195-161)/80*(math.log(9.715e-5)-math.log(3.150e-4)))
go("195",[(12,5410),(17,8059)],21,I,4486)
go("m1",[(69,3187),(45,3201),(80,3401),(64,3269)],15,1.916e-3,34217)
I2=math.exp(math.log(8.031e-4)+(195-161)/80*(math.log(3.458e-4)-math.log(8.031e-4)))
print("I2",I2)
go("m1 195",[(11,1215),(12,1214)],21,I2,7709)
go("A",[(174,3179)],15,5.116e-3,57973)
go("f02",[(108,3237),(83,3238)],15,1.982e-3,32049)
go("f05",[(123,3195)],15,2.096e-3,33367)
go("f05X",[(122,3205)],15,2.096e-3,33367)
go("E",[(56,3379),(56,3308)],15,6.603e-4,11601)

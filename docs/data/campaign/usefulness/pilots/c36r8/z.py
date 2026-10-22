import math
from scipy import stats
def z(name,R,N1,N2,f1=1.7,f2=1.7,bm=False,k=1):
    L=abs(math.log(R)); a=L/math.sqrt(1/N1+1/N2)
    b1=L/math.sqrt(f1*f1/N1+1/N2); b2=L/math.sqrt(f1*f1/N1+f2*f2/N2)
    s=f"{name}: R={R:.3f} ({R*math.sqrt(1/N1+1/N2):.3f}) (a) {a:.2f} (b, f on XYZ2 only) {b1:.2f} (b, f both) {b2:.2f}"
    if bm:
        d2=(L/3)**2-(1/N1+1/N2); s+=f" delta*={math.sqrt(max(d2,0)*k):.3f} (c@0.37) {L/math.sqrt(1/N1+0.37**2/k+1/N2):.2f}"
    print(s)
z("hwtr r1",9.309e-4/5.021e-5,1952,1053)
z("6x3 cg",1.139e-3/6.388e-5,1194,1507)
z("6x3 mwpm",1.169842e-3/6.388e-5,313454,1507)
z("6x3 bm",1.170440e-3/6.388e-5,411,1507,bm=True)
z("LY both cg vs siCSSX",2.773e-3/1.794e-3,1453,1881)
z("LY both cg vs siCSSZ",2.773e-3/1.715e-3,1453,1798)
z("LY both cg vs ovcCSSX",2.773e-3/1.811675e-3,1453,391658)
z("LY both cg vs ovcCSSZ",2.773e-3/1.722671e-3,1453,313329)
z("6x3Z vs LYboth",4.899e-3/2.773e-3,2567,1453)
z("sq cg vs CSSZ corr",1.139e-3/1.715e-3,1194,1798)
z("sq mwpm vs CSSZ mwpm",1.169842e-3/1.719176e-3,313454,556163)
z("sq mwpm vs CSSZ corr ovc",1.169842e-3/1.722671e-3,313454,313329)
z("sq bm vs CSSZ corr",1.170440e-3/1.715e-3,411,1798,bm=True)
z("sq bm vs CSSZ corr ovc",1.170440e-3/1.722671e-3,411,313329,bm=True)
z("E",1.134e-3/6.603e-4,112,11601,f1=1.0,f2=1.7)
# device f
T2=50.
for tau in (12.8,8.8):
    pT=(1-math.exp(-tau**2/T2**2))/2; T1=21*T2**2/(2*tau); pT1=0.1/(4*T1); g=0.1**2/(2*T2**2)
    inc=(tau/T2**2)*math.exp(-tau**2/T2**2)*0.1
    print(tau,f"pT {pT:.5f} T1 {T1:.0f} pT1 {pT1:.3e} gauss {g:.1e} f_eq4 {g/pT:.2e} f_T1 {pT1/pT:.2e} f_sum {(pT1+g)/pT:.2e} lin {0.1/tau:.4f} inc {inc/pT:.4f} t02 gauss {0.1*0.02/(inc/pT):.3f} lin {0.02*tau:.3f}")

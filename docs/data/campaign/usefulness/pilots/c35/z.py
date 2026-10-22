import math
from scipy import stats
def eps(err,shots,r):
    pl=err/shots; return 0.5*(1-(1-2*pl)**(1/r))
def pool(runs,r):
    e=sum(a for a,b in runs); s=sum(b for a,b in runs)
    p=e/s; chi=sum((a-p*b)**2/(p*b) for a,b in runs) if len(runs)>1 else 0
    return eps(e,s,r), e, s, chi, len(runs)-1
def t2sig(t,dof):
    p=stats.t.sf(t,dof); return stats.norm.isf(p)
def rat(name,A,rA,Bval,NB,varB=None,flip=False):
    ea,na,sa,chi,dof=pool(A,rA); k=len(A)
    R=ea/Bval; lr=abs(math.log(R))
    vB = varB if varB is not None else 1/NB
    sa_=math.sqrt(1/na+vB); za=lr/sa_
    if dof>0:
        f=max(1,math.sqrt(chi/dof))
    else:
        f=1.7
    sb=math.sqrt(f*f/na+vB); zb=lr/sb
    if dof>0 and f>1: zb=t2sig(zb,dof)
    zc=[lr/math.sqrt(1/na+vB+d*d/k) for d in (0.15,0.25)]
    # max delta for z>=3
    need=(lr/3)**2-(1/na+vB)
    dmax=math.sqrt(need*k) if need>0 else 0
    su=math.sqrt(2.7556/na+vB); zu=lr/su
    lin=(1-R)/(R*su) if R<1 else (R-1)/(R*su)
    lab = 'PASS' if min(za,zb,*zc)>=3 else ('fails c25' if min(za,zb,zc[0])>=3 else '')
    print(f"{name:42s} R={R:.3f}({R*sa_:.3f}) uni1.66 ({R*su:.3f}) chi2={chi:.2f}/{dof} f={f:.2f} | za={za:.2f} zb={zb:.2f} zc={zc[0]:.2f}/{zc[1]:.2f} dmax={dmax:.3f} | zuni_log={zu:.2f} zuni_lin={lin:.2f} {lab}  [A={ea:.4e} N={na}]")
print("--- worse basis")
rat("6% Y vs X",[(218,21968)],15,1.089e-3,29621)
rat("4% X vs X",[(62,22344)],15,3.050e-4,11782)
rat("3% Y vs X",[(36,33978),(37,34130)],15,1.054e-4,6166)
a,b=3.150e-4,9.715e-5; w=34/80
I=math.exp(math.log(a)+w*(math.log(b)-math.log(a))); vI=(1-w)**2/4486+w**2/999
print("interp",I, math.sqrt(vI))
rat("195 Y vs interp (pool.py var)",[(12,5410),(17,8059)],21,I,4486)
rat("195 Y vs interp (interp var)",[(12,5410),(17,8059)],21,I,4486,varB=vI)
rat("195 X vs interp",[(7,5397),(19,8365)],21,I,4486,varB=vI)
rat("195 Y vs d11",[(12,5410),(17,8059)],21,9.715e-5,999)
rat("r6d Y vs X",[(52,13687),(47,13611)],30,3.163e-4,14848)
rat("r6d X vs X",[(46,13542),(47,13601)],30,3.163e-4,14848)
rat("m1 X4 vs X",[(69,3187),(45,3201),(80,3401),(64,3269)],15,1.916e-3,34217)
rat("m1 X3 (no m1_x_X_b) vs X",[(69,3187),(80,3401),(64,3269)],15,1.916e-3,34217)
a,b=8.031e-4,3.458e-4; Im=math.exp(math.log(a)+w*(math.log(b)-math.log(a))); vIm=(1-w)**2/7709+w**2/2025
rat("m1 195 X vs interp",[(11,1215),(12,1214)],21,Im,7709,varB=vIm)
rat("m1 195 Y vs interp Z?",[(7,1210),(8,1210)],21,Im,7709,varB=vIm)
rat("eta10 Y vs X",[(48,34284),(51,33493)],15,1.208e-4,7370)
rat("eta100 Y vs X",[(36,46743)],15,4.502e-5,2922)
print("--- better basis")
rat("6% X vs Z",[(210,23755)],15,6.635e-4,18322)
rat("4% Y vs Z",[(44,24063)],15,2.016e-4,7847)
rat("3% X vs Z",[(40,33432),(22,33653)],15,6.775e-5,3985)
rat("eta10 X vs Z",[(45,34371),(46,33496)],15,8.743e-5,5396)
rat("m1 Y vs Z",[(52,3151),(43,3139),(61,3496)],15,1.452e-3,25634)
rat("r6d X vs Z",[(46,13542),(47,13601)],30,2.033e-4,9381)
rat("r6d Y vs Z",[(52,13687),(47,13611)],30,2.033e-4,9381)
rat("eta100 X vs Z",[(29,46148)],15,1.740e-5,1152)
print("--- other")
rat("RunE Y vs X",[(56,3379),(56,3308)],15,6.603e-4,11601)
rat("phen q0 Y vs XZZX corr X",[(112,34800),(110,34800),(99,34800),(95,34800)],15,1.383e-4,759)
rat("f=0.02 X vs X",[(108,3237),(83,3238)],15,1.982e-3,32049)
rat("D0 alone X vs m1 X",[(64,3269)],15,1.916e-3,34217)
# XZZX BM vs corr
for nm,(e,n),(B,NB) in [("6% xzzx bm/corr",(1.078e-3,238),(1.089e-3,29621)),("m1 xzzx bm/corr",(2.074e-3,87),(1.916e-3,34217))]:
    R=e/B; print(nm, R, R*math.sqrt(1/n+1/NB))
# qubit equivalence
target=eps(29,13469,21)
q=161+80*math.log(3.150e-4/target)/math.log(3.150e-4/9.715e-5)
print("target",target,"qeq",q, 195/q)

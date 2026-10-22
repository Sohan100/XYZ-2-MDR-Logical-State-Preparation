from math import log, sqrt, exp
from scipy import stats
def eps(e,s,r): return 0.5*(1-(1-2*e/s)**(1/r))
def pool(runs,r):
    e=sum(a for a,b in runs); s=sum(b for a,b in runs); p=e/s
    chi=sum((a-p*b)**2/(p*b) for a,b in runs) if len(runs)>1 else None
    return eps(e,s,r), e, len(runs), chi
def fof(k,chi):
    if k==1: return 1.7, False
    f=max(1,sqrt(chi/(k-1))); return f, f>1
def tconv(z,dof):
    p=stats.t.sf(z,dof); return stats.norm.isf(p)
def show(name, A, rA, B=None, rB=None, Bfix=None):
    eA,NA,kA,chiA=pool(A,rA)
    fA,tA=fof(kA,chiA)
    if Bfix is None:
        eB,NB,kB,chiB=pool(B,rB); fB,tB=fof(kB,chiB); varB=1/NB; varBb=fB**2/NB; Bdesc=f"{eB:.4e}({NB},k={kB},chi={chiB})"
    else:
        eB,varB,varBb=Bfix; tB=False; kB=None; Bdesc=f"{eB:.4e} fixed"
    R=eA/eB; L=abs(log(R))
    sa=sqrt(1/NA+varB); za=L/sa
    sb=sqrt(fA**2/NA+varBb); zb=L/sb
    zbt = tconv(zb,kA-1) if tA else (tconv(zb,kB-1) if tB else zb)
    zc=[L/sqrt(1/NA+varB+d*d/kA) for d in (0.15,0.25)]
    dmax=sqrt(max(0,(L/3)**2-1/NA-varB)*kA)
    print(f"{name:34s} A={eA:.4e}({NA},k={kA},chi={None if chiA is None else round(chiA,2)}) B={Bdesc} R={R:.3f}({R*sa:.3f}) fA={fA:.2f} za={za:.2f} zb={zb:.2f} zbt={zbt:.2f} zc={zc[0]:.2f}/{zc[1]:.2f} dmax={dmax:.3f}")
    return R,sa
X2=[(66,22569),(62,22000),(55,25400),(45,23371),(38,23016),(38,21488),(53,21488),(41,22512),(46,17450),(29,17450)]
Y2=[(43,22621),(55,25400),(58,23388),(56,22969),(53,21488),(47,22512),(57,22512),(33,16550)]
Z2=[(7738,1972704),(7454,1912022),(506,131056),(497,131056),(4614,1198350)]
print(pool(Z2,21))
show("18/16 3% vs d4", [(408,12224)],9,[(503,8176)],12)
# interp: B = 7/9 d4corr + 2/9 d5bm  (log-linear? weights on log)
e4=5.441e-3; e5=3.348e-3; N4=503; N5=490
# check linear vs loglinear
lin=7/9*e4+2/9*e5; lg=exp(7/9*log(e4)+2/9*log(e5)); print('interp lin',lin,'log',lg)
I=lg
vB=(7/9)**2/N4+(2/9)**2/N5
show("18/16 interp", [(408,12224)],9,Bfix=(I,vB,2.89*vB))
show("3% 50/49 X", [(254,20416),(260,19762)],15,[(25589,1166789)],21)
show("3% 50/49 Y", [(238,20416),(193,19775)],15,[(25589,1166789)],21)
show("2.5% 50/49 X", [(119,20104),(117,21150)],15,[(16015,1572704)],21)
show("2% L_Y", Y2,15,Z2,21)
show("2% L_Y w/o ul b10 Y", [y for y in Y2 if y!=(55,25400)],15,Z2,21)
show("2% L_X", X2,15,Z2,21)
show("2% L_X w/o ul b10 X", [x for x in X2 if x!=(55,25400)],15,Z2,21)
show("2% L_X w/o stat", X2[2:],15,Z2,21)
show("2% L_X stat only", X2[:2],15,Z2,21)
show("q0 18/16", [(426,24560)],9,[(596,16368)],12)
show("q0 L_Y", [(112,34800),(110,34800),(99,34800),(95,34800)],15,[(759,262128)],21)
show("q0 L_X", [(100,34800),(88,34800),(72,34800),(92,34800)],15,[(759,262128)],21)
show("r1 X", [(470,311040)],1,[(1278,8388352)],1)
show("r1 Y vs X", [(753,327424)],1,[(1278,8388352)],1)

from math import log,sqrt
from scipy import stats, optimize
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def pool(runs):
    e=sum(a for a,b in runs); s=sum(b for a,b in runs); p=e/s
    chi=sum((a-p*b)**2/(p*b) for a,b in runs); return e,s,chi,len(runs)-1
def tconv(z,dof): return stats.norm.isf(stats.t.sf(z,dof))
def zs(name,runs,r,B,NB,fB=1.7):
    e,s,chi,dof=pool(runs); ea=eps(e,s,r); R=ea/B; L=abs(log(R)); k=len(runs)
    cv=1/NB; sp=sqrt(1/e+cv)
    f=max(1,sqrt(chi/dof)) if dof>0 else 1.7
    zb=L/sqrt(f*f/e+fB*fB*cv)
    zbt=tconv(zb,dof) if (dof>0 and f>1) else zb
    d3=sqrt(max(0,(L/3)**2-1/e-cv)*k)
    # inverse variance
    def zc(d): v=1/sum(1/(1/a+d*d) for a,b in runs)+cv; return L/sqrt(v)-3
    try: d3iv=optimize.brentq(zc,0,5)
    except Exception: d3iv=float('nan')
    print(f"{name}: eps={ea:.4e} N={e}/{s} chi2={chi:.2f}/{dof} R={R:.4f}({R*sp:.3f}) a={L/sp:.2f} f={f:.2f} b={zb:.2f} bt={zbt:.2f} d*={d3:.3f} d*iv={d3iv:.3f}")
X2=[(66,22569),(62,22000),(55,25400),(45,23371),(38,23016),(38,21488),(53,21488),(41,22512),(46,17450),(29,17450)]
Y2=[(43,22621),(55,25400),(58,23388),(56,22969),(53,21488),(47,22512),(57,22512),(33,16550)]
B2=1.861e-4;N2=20809
zs("2% X all",X2,15,B2,N2,1.0)
zs("2% X w/o stat",X2[2:],15,B2,N2,1.0)
zs("2% X stat only",X2[:2],15,B2,N2,1.0)
zs("2% X w/o b10_X",X2[:2]+X2[3:],15,B2,N2,1.0)
zs("2% Y all",Y2,15,B2,N2,1.0)
zs("2% Y single XZZX",Y2,15,1.875e-4,7738,1.7)
zs("2% Y w/o b10_Y",Y2[:1]+Y2[2:],15,B2,N2,1.0)
zs("3% X",[(254,20416),(260,19762)],15,1.067e-3,25589)
zs("2.5% X",[(119,20104),(117,21150)],15,4.897e-4,16015)
zs("18/16 d4",[(408,12224)],9,5.441e-3,503)
zs("q0 L_Y",[(112,34800),(110,34800),(99,34800),(95,34800)],15,1.383e-4,759)
zs("q0 L_X",[(100,34800),(88,34800),(72,34800),(92,34800)],15,1.383e-4,759)
zs("q0 18/16",[(426,24560)],9,3.141e-3,596)
X15=[(22,23917),(15,24557),(13,12941),(14,12798),(11,12718),(7,12898)]
zs("1.5% X",X15,15,5.156e-5,2718)
zs("1.5% X ab",X15[:2],15,5.156e-5,2718)
zs("1.5% Y",[(18,24006),(8,13084),(14,13051),(7,12889),(4,12673)],15,5.156e-5,2718)
I=(5.441e-3)**(7/9)*(3.348e-3)**(2/9)
cv=(7/9)**2/503+(2/9)**2/490; R=3.823e-3/I; Lr=abs(log(R))
print("interp",I,R,R*sqrt(1/408+cv),"a",Lr/sqrt(1/408+cv),"b",Lr/sqrt(2.89/408+2.89*cv),"d*",sqrt(max(0,(Lr/3)**2-1/408-cv)))
# r=1 archived
a=284/196352; b=1080/6291200; R=a/b; L=log(R); sp=sqrt(1/284+1/1080)
print("r1 arch",a,b,R,R*sp,L/sp,L/sqrt(2.89/284+2.89/1080),sqrt((L/3)**2-1/284-1/1080))
for nm,(e,s) in {"X":(470,311040),"Y":(753,327424)}.items():
    a=e/s; b=1278/8388352; R=a/b; sp=sqrt(1/e+1/1278); L=log(R)
    print("r1",nm,a,R,R*sp,L/sp,L/sqrt(2.89/e+2.89/1278),sqrt((L/3)**2-1/e-1/1278))
# XY
print("XY/XZZX", eps(157,30457,15)/1.861e-4)
# Lambda
print(8.632e-4/2.278e-4, 8.632e-4/2.278e-4*sqrt(1/514+1/188), 1.067e-3/2.490e-4, 1.067e-3/2.490e-4*sqrt(1/25589+1/486), 1.021e-3/2.490e-4,1.021e-3/2.490e-4*sqrt(1/688+1/486))
print("1.5 vs 2 distance", log(1.063/0.813)/sqrt(0.112**2+0.050**2))

import math
from scipy.optimize import brentq
from scipy import stats
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def zs(R,N1,N2,f1=1.0,f2=1.0,delta=0.0,k=1,dc=0.0):
    l=abs(math.log(R))
    a=l/math.sqrt(1/N1+1/N2)
    b=l/math.sqrt(f1**2/N1+f2**2/N2)
    return a,b
def zc(lnR, runs, Ncomp, delta, dc=0.0):
    # runs: list of error counts n_i of XYZ2 runs (pooled by inverse variance)
    w=sum(1/(1/n+delta**2) for n in runs)
    var=1/w+1/Ncomp+dc**2
    return abs(lnR)/math.sqrt(var)
def dstar(lnR,runs,Ncomp):
    f=lambda d: zc(lnR,runs,Ncomp,d)-3
    if f(0)<0: return None
    try: return brentq(f,0,5)
    except: return float('inf')
print("=== SI1000 35-qubit window vs rectangles")
rects={'css6x3X':(1514,23854848),'css3x6Z':(1518,24379136),'xzzx6x3X':(1514,24116992),'c36 css6x3X':(1507,23592704)}
xyz={'BM':(411,351150,1.170e-3),'corr_gauge':(1194,1048320,1.139e-3),'MWPM':(313454,267945550,1.1698e-3)}
for kx,(nx,sx,vx) in xyz.items():
    for kr,(nr,sr) in rects.items():
        R=(nx/sx)/(nr/sr); a,b=zs(R,nx,nr,f1=1.7 if kx!='MWPM' else 1.7,f2=1.7)
        out=f"{kx} vs {kr}: R={R:.2f} z(a)={a:.1f} z(b,f=1.7 both)={b:.1f}"
        if kx=='BM':
            out+=f" z(c,0.37)={zc(math.log(R),[nx],nr,0.37):.2f} z(b f1=1.7 only)={zs(R,nx,nr,1.7,1)[1]:.1f}"
        print(out)
print("SD6 35: xyz2 cg 1735/4194048 vs css6x3 1501/207093504")
R=(1735/4194048)/(1501/207093504); print(R, zs(R,1735,1501), zs(R,1735,1501,1.7,1.7))
print("=== r6d")
yY=eps(99,13687+13611,30); yX=eps(93,13542+13601,30)
zX=eps(14848,1132209,42); zZ=eps(9381,1107657,42)
print('eps',yY,yX,zX,zZ)
for nm,(e,n,c,nc,runs) in {'Y vs X':(yY,99,zX,14848,[52,47]),'X vs X':(yX,93,zX,14848,[46,47]),'X vs Z (better/better)':(yX,93,zZ,9381,[46,47]),'Y vs Z':(yY,99,zZ,9381,[52,47])}.items():
    R=e/c; a,b=zs(R,n,nc)
    print(nm, f"R={R:.3f}+-{R*math.sqrt(1/n+1/nc):.3f} a={a:.2f} b={b:.2f} dstar={dstar(math.log(R),runs,nc):.3f} z(c,0.37)={zc(math.log(R),runs,nc,0.37):.2f}")
    # k=1 shared offset
    l=abs(math.log(R)); v=(l/3)**2-1/n-1/nc
    print('   k=1 dstar', math.sqrt(v) if v>0 else None, ' z(c,0.37,k=1)=', l/math.sqrt(1/n+1/nc+0.37**2))
print("=== XZZX phen_b10 r=1 corr X replicates")
for runs in [[(1080,6291200),(1268,8388352),(1278,8388352),(1369,8388352),(1378,8388352)],[(1080,6291200),(1278,8388352),(1369,8388352),(1378,8388352)]]:
    E=sum(a for a,b in runs); S=sum(b for a,b in runs); p=E/S
    mu=[p*b for a,b in runs]
    chi=sum((a-m)**2/m for (a,b),m in zip(runs,mu)); dof=len(runs)-1
    W=sum(m*(1-m/E) for m in mu)
    d=math.sqrt(max(0,(chi-dof)/W))
    print(f"chi2={chi:.2f}/{dof} p={stats.chi2.sf(chi,dof):.4f} W={W:.0f} delta={d:.3f}")
R=yY/zX
print('r6d Y with comp delta 0.044:', zc(math.log(R),[52,47],14848,0.37,0.044), ' with 0.037', zc(math.log(R),[52,47],14848,0.37,0.037))
print("=== phen_b10 2% L_Y vs XZZX 5-run pool")
ys=[(43,22621),(55,25400),(58,23388),(56,22969),(53,21488),(47,22512),(57,22512),(33,16550)]
E=sum(a for a,b in ys); S=sum(b for a,b in ys); print(E,S)
ey=eps(E,S,15)
xz=[(7738,1972704),(7454,1912022),(506,131056),(497,131056),(4614,1198350)]
EX=sum(a for a,b in xz); SX=sum(b for a,b in xz); ex=eps(EX,SX,21)
p=EX/SX; chiX=sum((a-p*b)**2/(p*b) for a,b in xz)
print('XZZX pool',ex,EX,chiX)
R=ey/ex; a,b=zs(R,E,EX)
print(f"pool: R={R:.3f}+-{R*math.sqrt(1/E+1/EX):.3f} a={a:.2f} b={b:.2f} dstar={dstar(math.log(R),[a_ for a_,b_ in ys],EX):.3f}")
ex1=eps(7738,1972704,21); R=ey/ex1; a,b=zs(R,E,7738,1,1.7)
print(f"single: R={R:.3f} a={a:.2f} b={b:.2f} dstar={dstar(math.log(R),[a_ for a_,b_ in ys],7738):.3f}")

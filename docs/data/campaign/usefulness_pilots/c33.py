import math
from scipy import stats, optimize
def eps(E,S,r): pl=E/S; return 0.5*(1-(1-2*pl)**(1/r))
def chi(runs):
    E=sum(e for e,s in runs); S=sum(s for e,s in runs); p=E/S
    return sum((e-s*p)**2/(s*p) for e,s in runs), len(runs)-1
def ana(name, A, rA, Bval, NB, Bk=1, cBM=True, fsingle=1.7, fcomp=1.7, conv='eps'):
    E=sum(e for e,s in A); S=sum(s for e,s in A)
    a = eps(E,S,rA) if conv=='eps' else E/S
    R=a/Bval; L=abs(math.log(R))
    c,dof=chi(A)
    sa=math.sqrt(1/E+1/NB)
    za=L/sa
    # (b)
    if dof>0:
        f=max(1,math.sqrt(c/dof))
    else:
        f=fsingle
    fb=fcomp if Bk==1 else 1
    sb=math.sqrt(f**2/E+fb**2/NB)
    zb=L/sb
    tnote=''
    if dof>0 and f>1:
        t=zb; p=stats.t.sf(t,dof); zb_g=stats.norm.isf(p); tnote=f' t={t:.2f}/{dof} -> {zb_g:.2f}'
        zb=zb_g
    def zc(d):
        if not cBM: return za
        w=sum(1/(1/e+d*d) for e,s in A)
        return L/math.sqrt(1/w+1/NB)
    out=[zc(d) for d in (0.065,0.11,0.15,0.25)]
    # delta*
    if cBM and zc(0)>3:
        try: ds=optimize.brentq(lambda d: zc(d)-3, 0, 3)
        except Exception: ds=float('nan')
    else: ds=float('nan')
    print(f"{name:40s} R={R:.3f}({R*sa:.3f}) N={E} chi2={c:.2f}/{dof} (a){za:.2f} (b){zb:.2f}[f={f:.2f}]{tnote} (c).065 {out[0]:.2f} .11 {out[1]:.2f} .15 {out[2]:.2f} .25 {out[3]:.2f} d*={ds:.3f}")
print('--- leads')
ana('A 0.3% gates X', [(174,3179)],15,5.116e-3,57973)
ana('hw_tr 4% r3d X', [(62,22344)],15,3.050e-4,11782)
ana('hw_tr 6% Y', [(218,21968)],15,1.089e-3,29621)
ana('hw_tr 3% Y', [(36,33978),(37,34130)],15,1.054e-4,6166)
ana('hw_tr 3% X', [(40,33432),(22,33653)],15,1.054e-4,6166)
ana('r6d Y vs XZZX X', [(52,13687),(47,13611)],30,3.163e-4,14848)
ana('r6d X vs XZZX X', [(46,13542),(47,13601)],30,3.163e-4,14848)
lnI=math.log(3.150e-4)+(195-161)/80*(math.log(9.715e-5)-math.log(3.150e-4)); I=math.exp(lnI)
ana('195 Y vs interp', [(12,5410),(17,8059)],21,I,4486)
ana('m1 X 4 runs', [(69,3187),(45,3201),(80,3401),(64,3269)],15,1.916e-3,34217)
ana('m1 X w/o b', [(69,3187),(80,3401),(64,3269)],15,1.916e-3,34217)
X2=[(55,25400),(45,23371),(38,23016),(38,21488),(53,21488),(41,22512),(66,22569),(62,22000),(46,17450),(29,17450)]
Y2=[(55,25400),(58,23388),(56,22969),(43,22621),(53,21488),(47,22512),(57,22512),(33,16550)]
ana('phen_b10 2% X 10', X2,15,1.875e-4,7738)
ana('phen_b10 2% Y 8', Y2,15,1.875e-4,7738)
ana('phen_b10 3% 50/49 X', [(254,20416),(260,19762)],15,1.067e-3,25589)
ana('phen_b10 18/16 3% Y vs d4', [(408,12224)],9,5.441e-3,503)
print('--- losses')
ana('phen 18/16', [(592,8176)],9,6.323e-3,579)
ana('phen 50/49', [(416,11248)],15,1.400e-3,468)
ana('phen 98/100', [(114,6896),(129,7152)],21,3.550e-4,518)
ana('run E Y', [(56,3379),(56,3308)],15,6.603e-4,11601)
ana('phen q0 L_Y', [(112,34800),(110,34800),(99,34800),(95,34800)],15,1.383e-4,759)
ana('phen q0 L_X', [(100,34800),(88,34800),(72,34800),(92,34800)],15,1.383e-4,759)
ana('eta100 better X', [(29,46148)],15,1.740e-5,1152)
ana('hw_rr BM eps', [(887,59048)],15,3.638e-4,14883)
ana('hw_rr BM pL', [(887,59048)],15,7.585e-3,14883,conv='pL')
ana('hw_rr MWPM eps', [(70997,4470689)],15,3.647e-4,27593,cBM=False)
ana('hw_rr MWPM pL', [(70997,4470689)],15,7.603e-3,27593,cBM=False,conv='pL')
B10Y=[(15,3119),(11,3202),(16,3194),(14,3444),(13,3415),(14,3328),(11,3328),(13,3584),(12,3550),(19,3750),(12,3750)]
B10X=[(15,3280),(14,3232),(10,3168),(11,3436),(7,3328),(7,3328),(17,3584),(16,3550),(13,3750),(16,3750)]
# competitor XZZX X corr: ul 2437/1499079 r21, stat 2890/1712704, ovc 2541/1510350
NBc=2437+2890+2541; Sc=1499079+1712704+1510350
xz=eps(NBc,Sc,21); print('xzzx X corr pooled eps %.4e N=%d chi2=%.2f/%d'%(xz,NBc,*chi([(2437,1499079),(2890,1712704),(2541,1510350)])))
ana('biased10 Y 11 runs', B10Y,15,xz,NBc,Bk=3)
ana('biased10 X 10 runs', B10X,15,xz,NBc,Bk=3)

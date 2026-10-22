import numpy as np
from scipy import stats, optimize
def chi2(runs):  # runs: list of (err, shots)
    E=sum(e for e,s in runs); S=sum(s for e,s in runs)
    return sum((e-E*s/S)**2/(E*s/S) for e,s in runs), len(runs)-1
def zs(R, runs, N2, f2=1.7, comp_single=True, k_t=None):
    N1=sum(e for e,s in runs); L=abs(np.log(R))
    a=L/np.sqrt(1/N1+1/N2)
    if len(runs)==1: f1=1.7; dof=None
    else:
        c,dof=chi2(runs); f1=max(1,np.sqrt(c/dof)); 
    fb2 = f2 if comp_single else 1
    zb=L/np.sqrt(f1**2/N1+fb2**2/N2)
    if len(runs)>1 and f1>1:
        p=stats.t.sf(zb,dof); zb=stats.norm.isf(p)
    def zc(d): 
        S=1/sum(1/(1/e+d*d) for e,s in runs); return L/np.sqrt(S+1/N2)
    try: ds=optimize.brentq(lambda d: zc(d)-3,0,5)
    except: ds=None
    return round(a,2), round(zb,2), round(zc(.15),2), round(zc(.25),2), ds, round(zc(.37),2)
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def pool(runs,r): E=sum(e for e,s in runs); S=sum(s for e,s in runs); return eps(E,S,r)
P=[]
# 6% Y
x=pool([(218,21968)],15); print('6%',x/1.089e-3, zs(x/1.089e-3,[(218,21968)],29621))
x=pool([(62,22344)],15); print('4%',x/3.050e-4, zs(x/3.050e-4,[(62,22344)],11782))
r=[(36,33978),(37,34130)]; x=pool(r,15); print('3%Y',x/1.054e-4, zs(x/1.054e-4,r,6166), chi2(r))
r=[(52,13687),(47,13611)]; x=pool(r,30); print('6dY',x/3.163e-4, zs(x/3.163e-4,r,14848))
r=[(46,13542),(47,13601)]; x=pool(r,30); print('6dX vs X',x/3.163e-4, zs(x/3.163e-4,r,14848)); print('6dX vs Z',x/2.033e-4, zs(x/2.033e-4,r,9381))
r=[(56,3379),(56,3308)]; x=pool(r,15); print('E',x/6.603e-4, zs(x/6.603e-4,r,11601))
r=[(80,3401),(69,3187),(45,3201),(64,3269)]; x=pool(r,15); print('m1X',x/1.916e-3, zs(x/1.916e-3,r,34217), chi2(r))
r=[(61,3496),(52,3151),(43,3139)]; x=pool(r,15); print('m1Y vs Z',x/1.452e-3, zs(x/1.452e-3,r,25634), chi2(r))
r=[(44,24063)]; x=pool(r,15); print('4%YvsZ',x/2.016e-4, zs(x/2.016e-4,r,7847))
r=[(36,46743)]; x=pool(r,15); print('e100Y',x/4.502e-5, zs(x/4.502e-5,r,2922))
r=[(29,46148)]; x=pool(r,15); print('e100XvsZ',x/1.740e-5, zs(x/1.740e-5,r,1152))
r=[(48,34284),(51,33493)]; x=pool(r,15); print('e10Y',x/1.208e-4, zs(x/1.208e-4,r,7370))
# q=0 boundary
r=[(110,34800),(99,34800),(95,34800),(112,34800)]; x=pool(r,15); print('q0',x, x/1.383e-4, zs(x/1.383e-4,r,759), chi2(r))
# 195
r=[(12,5410),(17,8059)]; x=pool(r,21); print('195 Y',x, chi2(r))
q1,q2=161,241; e1,e2=3.150e-4,9.715e-5
# interp at 195
le=np.log(e1)+(np.log(e2)-np.log(e1))*(195-161)/80; xi=np.exp(le); print('interp',xi, x/xi, zs(x/xi,r,1/(1/4486*(1-34/80)**2+1/999*(34/80)**2)))
qeq=161+80*(np.log(x)-np.log(e1))/(np.log(e2)-np.log(e1)); print('qeq',qeq)
# r1
print('r1', 9.309e-4/5.021e-5, np.log(18.54)/np.sqrt(1/1952+1/1053), np.log(18.54)/np.sqrt(2.89*(1/1952+1/1053)))

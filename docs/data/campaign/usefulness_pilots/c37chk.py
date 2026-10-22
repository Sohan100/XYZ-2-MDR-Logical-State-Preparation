from math import log,sqrt,exp
from scipy import stats
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def pool(runs,r):
    e=sum(a for a,b in runs); s=sum(b for a,b in runs); p=e/s
    chi=sum((a-p*b)**2/(p*b) for a,b in runs) if len(runs)>1 else None
    return eps(e,s,r),e,chi,len(runs)
def Z(name,runs,r,B,NB):
    A,N,chi,k=pool(runs,r); R=A/B; L=abs(log(R)); cv=1/NB
    a=L/sqrt(1/N+cv)
    if chi is None: s=1.7; dof=None
    else: dof=k-1; s=max(1,sqrt(chi/dof))
    b=L/sqrt(s*s/N+cv)
    bg=b
    if s>1 and dof: bg=stats.norm.isf(stats.t.sf(b,dof))
    c=[L/sqrt(1/N+cv+d*d/k) for d in (0.15,0.25,0.37)]
    ds=sqrt(max(0,((L/3)**2-1/N-cv)*k))
    print(f"{name}: A={A:.4e} N={N} chi={chi} R={R:.3f}+-{R*sqrt(1/N+cv):.3f} a={a:.2f} b={b:.2f} (g {bg:.2f}) s={s:.2f} c={c[0]:.2f},{c[1]:.2f},{c[2]:.2f} d*={ds:.3f}")
    return A,N
m1=[(69,3187),(45,3201),(80,3401),(64,3269)]
Z('m1',m1,15,1.916e-3,34217)
Z('A',[(174,3179)],15,5.116e-3,57973)
Z('f02',[(108,3237),(83,3238)],15,1.982e-3,32049)
Z('f05Y',[(123,3195)],15,2.096e-3,33367)
Z('f05X',[(122,3205)],15,2.096e-3,33367)
Z('E',[(56,3379),(56,3308)],15,6.603e-4,11601)
Z('Ebetter',[(43,3383)],15,5.765e-4,9941)
Z('cross m1 vs E',m1,15,6.603e-4,11601)
Z('cross E vs E', [(56,3379),(56,3308)],15,6.603e-4,11601)
Z('hwtr3 Y',[(36,33978),(37,34130)],15,1.054e-4,6166)
Z('eta10 Y',[(48,34284),(51,33493)],15,1.208e-4,7370)
Z('eta10 X',[(45,34371),(46,33496)],15,1.208e-4,7370)
Z('eta100 Y',[(36,46743)],15,4.502e-5,2922)
# shifts
A0,N0=pool(m1,15)[:2]; 
for nm,runs,B,NB in (('f02',[(108,3237),(83,3238)],1.982e-3,32049),('f05Y',[(123,3195)],2.096e-3,33367)):
    A,N,chi,k=pool(runs,15)
    R0=A0/1.916e-3; R=A/B; d=log(R)-log(R0)
    v=1/N+1/NB+1/N0+1/34217
    out=[d/sqrt(v)]+[d/sqrt(v+dd*dd/4+dd*dd/k) for dd in (0.15,0.25)]
    print(nm,'shift',round(d,4),[round(x,2) for x in out])
# bias diffs
def lr(runs,B,NB):
    A,N,_,_=pool(runs,15); return log(A/B), 1/N+1/NB
l20,v20=lr([(36,33978),(37,34130)],1.054e-4,6166)
l10,v10=lr([(48,34284),(51,33493)],1.208e-4,7370)
l100,v100=lr([(36,46743)],4.502e-5,2922)
print('eta10vs20',(l10-l20)/sqrt(v10+v20),'eta100vs20',(l100-l20)/sqrt(v100+v20))
print('LX rises', pool([(108,3237),(83,3238)],15)[0]/A0-1, 2.633e-3/A0-1, 1.982/1.916-1, 2.096/1.916-1)
print('two-window',6.678e-4/1.054e-4, 6.678e-4/6.97e-4, 6.678e-4/6.97e-4*sqrt(1/218+1/60520))
print('latency',1.13e-3/13e-6,1.65e-3/9e-6, 0.0170/15,0.0248/15)

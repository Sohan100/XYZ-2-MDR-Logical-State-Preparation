import math, re, os
from scipy import stats, optimize
B='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/usefulness_pilots/'
def rd(f):
    t=open(B+f).read().split('\n')
    t=[l for l in t if 'shots=' in l and 'progress' not in l][-1]
    sh=int(re.search(r'shots=(\d+)',t).group(1)); e=int(re.search(r'err(?:ors)?=(\d+)',t).group(1)); r=int(re.search(r' r=(\d+)',t).group(1))
    return sh,e,r
def eps(pl,r): return (1-(1-2*pl)**(1/r))/2
def pool(fs):
    S=[rd(f) for f in fs]; sh=sum(s[0] for s in S); e=sum(s[1] for s in S); r=S[0][2]
    E=eps(e/sh,r)
    # chi2 homogeneity
    chi=0
    for s in S:
        mu=e*s[0]/sh; chi+=(s[1]-mu)**2/mu
    return dict(eps=E,N=e,k=len(S),n=[s[1] for s in S],mus=[e*s[0]/sh for s in S],chi=chi,dof=len(S)-1)
def tz(t,dof):
    p=stats.t.sf(t,dof); return stats.norm.isf(p)
def comp(name,x,c,delta=(0.15,0.25),fx=None,fc=None):
    R=x['eps']/c['eps']; L=abs(math.log(R))
    va=1/x['N']+1/c['N']; sa=math.sqrt(va)
    # (b)
    def f_of(p):
        if p['k']==1: return 1.7, False
        f=max(1,math.sqrt(p['chi']/p['dof'])); return f, f>1
    f1,e1=f_of(x); f2,e2=f_of(c)
    if fx: f1=fx
    vb=f1**2/x['N']+f2**2/c['N']; zb=L/math.sqrt(vb)
    if e1: zb=tz(zb,x['k']-1)
    # (b) with competitor Poisson (old)
    vb0=f1**2/x['N']+1/c['N']; zb0=L/math.sqrt(vb0)
    if e1: zb0=tz(zb0,x['k']-1)
    def vc(d):
        return 1/sum(1/(1/n+d*d) for n in x['n'])+1/c['N']
    zc=[L/math.sqrt(vc(d)) for d in delta]
    try: ds=optimize.brentq(lambda d: L/math.sqrt(vc(d))-3,0,5)
    except Exception: ds=float('nan')
    # k=1 variant
    vk1=lambda d: 1/x['N']+d*d+1/c['N']
    try: ds1=optimize.brentq(lambda d: L/math.sqrt(vk1(d))-3,0,5)
    except Exception: ds1=float('nan')
    print(f"{name}: R={R:.4f}({R*sa:.4f}) za={L/sa:.2f} zb={zb:.2f} (b,compPoisson {zb0:.2f}) zc={zc[0]:.2f}/{zc[1]:.2f} delta*={ds:.3f} k1delta*={ds1:.3f}  Nx={x['N']} chi={x['chi']:.2f}/{x['dof']} Nc={c['N']} epsx={x['eps']:.4e} epsc={c['eps']:.4e}")
    return R
P=pool
# 6%
comp('6% Y vs X', P(['litrev/hwtr_xyz2_Y.txt']), P(['litrev/hwtr_xzzx_X.txt']))
comp('6% X vs Z', P(['litrev/hwtr_xyz2_X.txt']), P(['litrev/hwtr_xzzx_Z.txt']))
comp('6% XZZX bm vs corr', P(['litrev/hwtr_xzzx_X_bm.txt']), P(['litrev/hwtr_xzzx_X.txt']))
comp('4% r3d X vs X', P(['litrev/hwtr4_xyz2_X.txt']), P(['litrev/hwtr4_xzzx_X.txt']))
comp('4% r3d Y vs Z', P(['litrev/hwtr4_xyz2_Y.txt']), P(['litrev/hwtr4_xzzx_Z.txt']))
comp('4% r3d Y vs X', P(['litrev/hwtr4_xyz2_Y.txt']), P(['litrev/hwtr4_xzzx_X.txt']))
comp('3% Y vs X', P(['cons/x3_Y_a.txt','cons/x3_Y_b.txt']), P(['cons/z3_X.txt']))
comp('3% X vs Z', P(['cons/x3_X_a.txt','cons/x3_X_b.txt']), P(['cons/z3_Z.txt']))
comp('3% X vs X', P(['cons/x3_X_a.txt','cons/x3_X_b.txt']), P(['cons/z3_X.txt']))
comp('r6d Y vs X', P(['cons/b_x_Y_a.txt','cons/b_x_Y_b.txt']), P(['cons/b_z_X.txt']))
comp('r6d X vs X', P(['cons/b_x_X_a.txt','cons/b_x_X_b.txt']), P(['cons/b_z_X.txt']))
comp('r6d X vs Z', P(['cons/b_x_X_a.txt','cons/b_x_X_b.txt']), P(['cons/b_z_Z.txt']))
comp('r6d Y vs Z', P(['cons/b_x_Y_a.txt','cons/b_x_Y_b.txt']), P(['cons/b_z_Z.txt']))
comp('eta10 Y vs X', P(['cons/e10_x_Y.txt','cons/e10b_x_Y.txt']), P(['cons/e10_z_X.txt']))
comp('eta10 X vs Z', P(['cons/e10_x_X.txt','cons/e10b_x_X.txt']), P(['cons/e10_z_Z.txt']))
comp('eta10 X vs X', P(['cons/e10_x_X.txt','cons/e10b_x_X.txt']), P(['cons/e10_z_X.txt']))
comp('eta100 Y vs X', P(['cons/e100_x_Y.txt']), P(['cons/e100_z_X.txt']))
comp('eta100 X vs Z', P(['cons/e100_x_X.txt']), P(['cons/e100_z_Z.txt']))
comp('m1 X vs X', P(['litrev/m1_xyz2_X.txt','cons/m1_x_X_a.txt','cons/m1_x_X_b.txt','hwrev_q7/D0_xyz2_X.txt']), P(['cons/m1_z_X.txt']))
comp('m1 Y vs Z', P(['litrev/m1_xyz2_Y.txt','cons/m1_x_Y_a.txt','cons/m1_x_Y_b.txt']), P(['cons/m1_z_Z.txt']))
comp('m1 Y vs X', P(['litrev/m1_xyz2_Y.txt','cons/m1_x_Y_a.txt','cons/m1_x_Y_b.txt']), P(['cons/m1_z_X.txt']))
comp('m1 xzzx bm vs corr', P(['litrev/m1_xzzx_X.txt']), P(['cons/m1_z_X.txt']))
comp('run E Y vs X', P(['hwrev_q7/E_xyz2_Y.txt','hwrev_q7/E2_xyz2_Y.txt']), P(['hwrev_q7/E_xzzx_X.txt']))
comp('C f0.02 X vs X', P(['hwrev_q7/C_xyz2_X.txt','hwrev_q7/D2_xyz2_X.txt']), P(['hwrev_q7/C_xzzx_X.txt']))
comp('XY X vs XYZ2 Y', P(['litrev/hwtr_xy_X.txt']), P(['litrev/hwtr_xyz2_Y.txt']))
comp('XY Z vs XYZ2 Y', P(['litrev/hwtr_xy_Z.txt']), P(['litrev/hwtr_xyz2_Y.txt']))
comp('XY X vs XZZX X', P(['litrev/hwtr_xy_X.txt']), P(['litrev/hwtr_xzzx_X.txt']))
comp('XY Z vs XZZX X', P(['litrev/hwtr_xy_Z.txt']), P(['litrev/hwtr_xzzx_X.txt']))
comp('Ybias XZZX bm vs XYZ2 4% X', P(['litlens/y_xzzx_X_bm.txt']), P(['litrev/hwtr4_xyz2_X.txt']))
comp('Ybias XZZX bm vs XZZX 4% X', P(['litlens/y_xzzx_X_bm.txt']), P(['litrev/hwtr4_xzzx_X.txt']))
comp('Ybias XZZX corr vs XZZX 4% X', P(['litlens/y_xzzx_X_corr.txt']), P(['litrev/hwtr4_xzzx_X.txt']))
comp('cg probe Y vs X', P(['cg_Y.txt']), P(['litrev/hwtr4_xzzx_X.txt']))
comp('cg probe X vs X', P(['cg_X.txt']), P(['litrev/hwtr4_xzzx_X.txt']))
# 195
y7=P(['litrev/hwtr_xyz2_Y_d7.txt','litrev/hwtr_xyz2_Y_d7b.txt']); x7=P(['litrev/hwtr_xyz2_X_d7.txt','litrev/hwtr_xyz2_X_d7b.txt'])
d9=P(['litrev/hwtr_xzzx_X_d9.txt']); d11=P(['litrev/hwtr_xzzx_X_d11.txt'])
print('195 Y', y7['eps'], y7['N'], y7['chi'], ' X', x7['eps'], x7['N'], x7['chi'])
q9,q11=161,241; l9,l11=math.log(d9['eps']),math.log(d11['eps'])
w=(195-q9)/(q11-q9); li=l9+w*(l11-l9); ei=math.exp(li); vi=(1-w)**2/d9['N']+w**2/d11['N']
print('interp eps at 195', ei, 'var', vi)
c={'eps':ei,'N':1/vi,'k':2,'chi':0,'dof':1}
R=y7['eps']/ei; L=abs(math.log(R)); va=1/y7['N']+vi
vb=1/y7['N']+2.89*((1-w)**2/d9['N']+w**2/d11['N'])
vc=lambda d: 1/sum(1/(1/n+d*d) for n in y7['n'])+vi
ds=optimize.brentq(lambda d: L/math.sqrt(vc(d))-3,0,5)
print(f"195 Y vs interp R={R:.4f}({R*math.sqrt(va):.4f}) za={L/math.sqrt(va):.2f} zb={L/math.sqrt(vb):.2f} zc={L/math.sqrt(vc(.15)):.2f}/{L/math.sqrt(vc(.25)):.2f} delta*={ds:.3f}")
comp('195 Y vs d11', y7, d11)
# qubit equivalence
le=math.log(y7['eps']); q=q9+(le-l9)/(l11-l9)*(q11-q9)
# error on q: propagate var of le, l9, l11
dl=l11-l9
dq_dle=(q11-q9)/dl; dq_dl9=(q11-q9)*(-(dl)+(le-l9))/dl**2; dq_dl11=-(q11-q9)*(le-l9)/dl**2
vq=dq_dle**2/y7['N']+dq_dl9**2/d9['N']+dq_dl11**2/d11['N']
print('qubit eq', q, math.sqrt(vq), 'ratio', 195/q, 195/q*math.sqrt(vq)/q, 'offset sig', (q-195)/math.sqrt(vq))
print('Lambda', P(['litrev/hwtr_xyz2_Y.txt'])['eps']/y7['eps'], 'xzzx', P(['litrev/hwtr_xzzx_X.txt'])['eps']/math.exp(l9+ (195-161)/80*0) )

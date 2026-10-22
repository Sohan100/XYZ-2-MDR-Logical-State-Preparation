import math
from scipy.optimize import brentq
from scipy import stats
def eps(pl,r): return (1-(1-2*pl)**(1/r))/2
def pool(runs,r):
    n=sum(e for e,s in runs); s=sum(s for e,s in runs); return eps(n/s,r), n, runs
def chi2(runs):
    n=sum(e for e,s in runs); S=sum(s for e,s in runs)
    return sum((e-n*s/S)**2/(n*s/S) for e,s in runs), len(runs)-1
def z(x, runsx, ey, ny, ny_single=True, label=''):
    R=x/ey; L=abs(math.log(R)); N1=sum(e for e,s in runsx)
    sa=math.sqrt(1/N1+1/ny)
    if len(runsx)>1:
        c,dof=chi2(runsx); f=max(1,math.sqrt(c/dof))
    else: f=1.7; dof=None; c=None
    fb2=2.89 if ny_single else 1
    sb=math.sqrt(f**2/N1+fb2/ny); zb=L/sb
    if len(runsx)>1 and f>1:
        p=2*stats.t.sf(zb,dof); zb_n=stats.norm.isf(p/2)
    else: zb_n=zb
    def sc(d): return math.sqrt(1/sum(1/(1/e+d*d) for e,s in runsx)+1/ny)
    zc=[L/sc(d) for d in (0.15,0.25)]
    try: ds=brentq(lambda d:L/sc(d)-3,0,3)
    except: ds=None
    print(f"{label:22s} R={R:.4f}({R*sa:.4f}) a={L/sa:.2f} b={zb:.2f}->{zb_n:.2f} f={f:.2f} c={zc[0]:.2f}/{zc[1]:.2f} d*={ds}")
    return R
# 6% d5
y6=[(218,21968)]; ey=eps(218/21968,15)
z(ey,y6,1.089e-3,29621,label='6% Y vs X')
z(eps(210/23755,15),[(210,23755)],6.635e-4,18322,label='6% X vs Z')
z(ey,y6,1.078e-3,238,label='xzzx bm? ')
print('XZZX bm/corr 6%',1.078e-3/1.089e-3, math.sqrt(1/238+1/29621)*1.078/1.089)
# 4%
z(eps(62/22344,15),[(62,22344)],3.050e-4,11782,label='4% X vs X')
z(eps(44/24063,15),[(44,24063)],2.016e-4,7847,label='4% Y vs Z')
z(eps(44/24063,15),[(44,24063)],3.050e-4,11782,label='4% Y vs X')
# 3%
y3=[(36,33978),(37,34130)]; e=eps(73/68108,15)
z(e,y3,1.054e-4,6166,label='3% Y vs X')
x3=[(40,33432),(22,33653)]; z(eps(62/67085,15),x3,6.775e-5,3985,label='3% X vs Z'); print('x3 chi2',chi2(x3))
# r6d
by=[(52,13687),(47,13611)]; bx=[(46,13542),(47,13601)]
ey6=eps(99/27298,30); ex6=eps(93/27143,30); print('r6d eps',ey6,ex6, chi2(by),chi2(bx))
z(ey6,by,3.163e-4,14848,label='r6d Y vs X'); z(ex6,bx,3.163e-4,14848,label='r6d X vs X'); z(ex6,bx,2.033e-4,9381,label='r6d X vs Z')
for nm,ee,N in (('Y',ey6,99),('X',ex6,93)):
    L=abs(math.log(ee/3.163e-4)); print('shared d*',nm, math.sqrt((L/3)**2-1/N-1/14848))
print('r6d Y alone at 3d 0.40?', eps(44/24063,15)/3.050e-4, 'r6d', ey6/3.163e-4)
# eta10
e10=[(48,34284),(51,33493)]; e=eps(99/67777,15); print('e10 Y eps',e)
z(e,e10,1.208e-4,7370,label='eta10 Y vs X')
e10x=[(45,34371),(46,33496)]; z(eps(91/67867,15),e10x,8.743e-5,5396,label='eta10 X vs Z')
z(eps(36/46743,15),[(36,46743)],4.502e-5,2922,label='eta100 Y vs X')
z(eps(29/46148,15),[(29,46148)],1.740e-5,1152,label='eta100 X vs Z')
# m1
m1x=[(80,3401),(69,3187),(45,3201),(64,3269)]; print('m1x chi2',chi2(m1x))
z(eps(258/13058,15),m1x,1.916e-3,34217,label='m1 X vs X')
m1y=[(61,3496),(52,3151),(43,3139)]; print('m1y chi2',chi2(m1y))
z(eps(156/9786,15),m1y,1.452e-3,25634,label='m1 Y vs Z')
print('m1 xzzx bm/corr',2.074e-3/1.916e-3, 2.074/1.916*math.sqrt(1/87+1/34217))
# run E
E=[(56,3379),(56,3308)]; z(eps(112/6687,15),E,6.603e-4,11601,label='run E'); print('E chi2',chi2(E))
# gate idle
C=[(108,3237),(83,3238)]; z(eps(191/6475,15),C,1.982e-3,32049,label='fgate .02')
# boundary
q0=[(110,34800),(99,34800),(95,34800),(112,34800)]; eq0=eps(416/139200,15); print('q0',eq0,chi2(q0))
z(eq0,q0,1.383e-4,759,label='q0 Y')
# 195
y7=[(12,5410),(17,8059)]; e7=eps(29/13469,21); print('d7 eps',e7, chi2(y7))
l9,l11=math.log(3.150e-4),math.log(9.715e-5)
li=l9+(l11-l9)*(195-161)/80; ei=math.exp(li)
# uncertainty of interp
s9,s11=1/math.sqrt(4486),1/math.sqrt(999); w=(195-161)/80
si=math.sqrt(((1-w)*s9)**2+(w*s11)**2)
print('interp',ei,si)
z(e7,y7,ei,1/si**2,ny_single=False,label='195 vs interp')
z(e7,y7,9.715e-5,999,label='195 vs d11')
q=161+80*(math.log(e7)-l9)/(l11-l9)
# error
dl=l11-l9
g=lambda a,b,c: 161+80*(a-b)/(c-b)
sq=math.sqrt((80/dl*(1/math.sqrt(29)))**2+(80*(math.log(e7)-l11)/dl**2*s9)**2+(80*(math.log(e7)-l9)/dl**2*s11)**2)
print('qeq',q,sq,195/q,195/q*sq/q,'offset',q-195,(q-195)/sq)
print('Lambda',ey/e7, ey/e7*math.sqrt(1/218+1/29), 1.089e-3/1.911e-4)
print('d11 qubit fraction', 241/195)
# XY
for v in (3.505e-3,3.692e-3): print('XY',v/ey, v/ey*math.sqrt(1/218+1/992), v/1.089e-3)
x4=eps(62/22344,15); print('ybias', 9.029e-4/x4, 9.029e-4/x4*math.sqrt(1/306+1/62), 9.029e-4/3.050e-4, 9.029e-4/3.050e-4*math.sqrt(1/306+1/11782), 2.581e-3/3.050e-4)
# bias diffs
def lnsig(e,N1,ey,N2): return math.log(e/ey), 1/N1+1/N2
a=lnsig(eps(73/68108,15),73,1.054e-4,6166); b=lnsig(eps(99/67777,15),99,1.208e-4,7370); c=lnsig(eps(36/46743,15),36,4.502e-5,2922)
print('bias diff',(b[0]-a[0])/math.sqrt(a[1]+b[1]),(c[0]-a[0])/math.sqrt(a[1]+c[1]))
print('q0 vs hwtr eta10 Y', eq0/eps(99/67777,15), 1.383e-4/1.208e-4)
# r1
r=9.309e-4/5.021e-5; s=math.sqrt(1/1952+1/1053); print('r1',r,r*s,math.log(r)/s, math.log(r)/math.sqrt(2.89/1952+2.89/1053))
for e,n,s2 in ((343,360192,None),(293,327424,None),(171,179968,None)):
    pass
xz=1116/20971264; xz2=1146/20971264
for nm,e,n,c,N in (('d6',343,360192,xz,1116),('both',293,327424,xz,1116),('d6b',171,179968,xz2,1146),('Y',428,343808,xz,1116)):
    R=(e/n)/c; s=math.sqrt(1/e+1/N); sc=math.sqrt(1/e+0.37**2+1/N); print(nm,R,R*s,math.log(R)/sc, e/n)
print('cg', 6.927e-4/3.050e-4, 6.927e-4/3.050e-4*math.sqrt(1/10873+1/11782), 5.833e-4/3.050e-4)
print('L_Y over XZZX Z?', (428/343808)/(760/20971264))

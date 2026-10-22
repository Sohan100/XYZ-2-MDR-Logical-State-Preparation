import math
from scipy.optimize import brentq
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def zc(lnR, runs, Ncomp, delta, dc=0.0):
    w=sum(1/(1/n+delta**2) for n in runs)
    return abs(lnR)/math.sqrt(1/w+1/Ncomp+dc**2)
# phen 98/100: bq/phen.log d=7 Y 114/6896, 129/7152 r=21 vs XZZX d=10 X 3.550e-4 (518)
e=eps(243,6896+7152,21); R=e/3.550e-4; l=math.log(R)
print('phen98', R, zc(l,[114,129],518,0.37), zc(l,[114,129],518,0.37,0.044), brentq(lambda d: zc(l,[114,129],518,d,0.044)-3,0,2))
# biased10
print('b10 at .37 w comp .044', zc(math.log(3.35),[15,11,16,14,13,14,11,13,12,19,12],7868,0.37,0.044))
# hw_rr corr_gauge vs corr with 0.044 on both
for R,n1,n2 in [(2.098,33044,14883),(2.089,70997,27593)]:
    print('hw_rr', math.log(R)/math.sqrt(1/n1+1/n2+2*0.044**2))
# SI1000 window BM vs rect with comp offset
print('win', zc(math.log(18.44),[411],1514,0.37,0.044))
# delta* list check (from claim): which hold at 0.065 / 0.074 / 0.15
ds={'hw_tr 6% r3d':0.148,'hw_tr 195 L_Y':0.126,'hw_tr 3% L_Y':0.075,'phen_b10 2% L_Y':0.132,'phen_b10 3% L_X':0.078,'phen_b10 18/16 vs d4':0.097,
'run E':0.217,'phen 50/49':0.189,'phen_b10 q0 L_Y':0.213,'phen 18/16':0.085,'hw_rr BM':0.225,'r6d better/better':0.227}
for d in (0.065,0.074,0.15,0.284):
    print(d,[k for k,v in ds.items() if v>=d])

import math
exec(open('v.py').read().split("P='/pscratch")[0])
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
# 195 L_Y vs interp
I=math.exp(math.log(3.150e-4)+(195-161)/80*(math.log(9.715e-5)-math.log(3.150e-4)))
eA=29; RA=eps(29,5410+8059,21)/I; L=abs(math.log(RA)); 
from scipy import optimize
for nm,A,R,eB in [("195 LY",[(12,5410),(17,8059)],RA,4486)]:
    eA=sum(a for a,b in A); sa=math.sqrt(1/eA+1/eB)
    zc=lambda d: abs(math.log(R))/math.sqrt(1/sum(1/(1/a+d*d) for a,b in A)+1/eB)
    print(nm,R,R*sa,abs(math.log(R))/sa, abs(math.log(R))/math.sqrt(1/eA+2.89/eB), zc(.15),zc(.25),optimize.brentq(lambda d:zc(d)-3,0,3))
# biased10 xzzx chi2
runs=[(2437,1499079),(2890,1712704),(2541,1510350)]
E=sum(a for a,b in runs);S=sum(b for a,b in runs);p=E/S
print("b10 xzzx chi2",sum((a-p*b)**2/(p*b) for a,b in runs), eps(E,S,21))
# 18/16 interp
Ie=math.exp(math.log(5.441e-3)+(18-16)/(25-16)*(math.log(3.348e-3)-math.log(5.441e-3)))
print("interp",Ie)
def u166(nm,R,eA,eB): print(nm, abs(math.log(R))/math.sqrt(1.66**2/eA+1/eB))
u166("runE",eps(112,3379+3308,15)/6.603e-4,112,11601)
u166("hw4",eps(62,22344,15)/3.050e-4,62,11782)
u166("b10 3% LX",eps(514,20416+19762,15)/1.067e-3,514,25589)
u166("1816 interp",eps(408,12224,9)/Ie,408,503)
u166("1816 interp 490",eps(408,12224,9)/Ie,408,490)
u166("m1",eps(258,3187+3201+3401+3269,15)/1.916e-3,258,34217)
u166("hw6",eps(218,21968,15)/1.089e-3,218,29621)
u166("1816 d4",eps(408,12224,9)/5.441e-3,408,503)
u166("r6d",eps(99,13687+13611,30)/3.163e-4,99,14848)
print("runE f1.52", abs(math.log(eps(112,3379+3308,15)/6.603e-4))/math.sqrt(1.52**2/112+1/11601))
R=(70997/4470689)/(27593/3629009); print("mwpm d.25",math.log(R)/math.sqrt(1/70997+1/27593+.0625))
R=(33044/2076184)/(14883/1962124); print("cg d.25",math.log(R)/math.sqrt(1/33044+1/14883+.0625))
print("eps rr", eps(887,59048,15), eps(14883,1962124,21))

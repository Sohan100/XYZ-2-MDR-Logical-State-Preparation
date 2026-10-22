import math
def eps(e,s,r):
    pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
# opening r=3d
A=eps(62,22344,15); B=3.050e-4
R=A/B; s=math.sqrt(1/62+1/11782)
print("r3d R",R,"sigR",R*s,"lin z",(1-R)/(R*s),"log z",abs(math.log(R))/s)
s2=math.sqrt(1.66**2/62+1/11782); print("scaled1.66 lin",(1-R)/(R*s2),"log",abs(math.log(R))/s2)
# r6d Y
eY=eps(99,27298,30); BX=3.163e-4
R6=eY/BX; s=math.sqrt(1/99+1/14848)
print("r6d R",R6,R6*s,"lin",(1-R6)/(R6*s),"log",abs(math.log(R6))/s)
for f in [1.66,1.7]:
  s2=math.sqrt(f**2/99+1/14848); print(f,"log",abs(math.log(R6))/s2)
# run offset delta over 2 runs: sigma_ln add delta/sqrt(2)? 
for d in [0.15,0.25,0.43]:
  s3=math.sqrt(1/99+1/14848+d**2/2); print("delta",d,abs(math.log(R6))/s3)
# chi2 replicate
e=[(52,13687),(47,13611)]; p=99/27298; print("chi2",sum((a-p*b)**2/(p*b) for a,b in e))
# item 3
LX3=eps(62,22344,15); sLX3=LX3/math.sqrt(62)
LX6=eps(93,27143,30); sLX6=LX6/math.sqrt(93)
LY3=eps(44,24063,15); sLY3=LY3/math.sqrt(44)
LY6=eY; sLY6=eY/math.sqrt(99)
print("LX",LX3,sLX3,LX6,sLX6,"drop z",(LX3-LX6)/math.hypot(sLX3,sLX6))
print("LY",LY3,sLY3,LY6,sLY6)
# eps = b + B/r ; r=15,30: B=(e15-e30)*30, b=2e30-e15
def fit(e15,s15,e30,s30):
  b=2*e30-e15; sb=math.hypot(2*s30,s15)
  Bv=30*(e15-e30); sB=30*math.hypot(s15,s30)
  share=(Bv/15)/e15
  # share = 2(1 - e30/e15)
  rr=e30/e15; srr=rr*math.hypot(s15/e15,s30/e30); sshare=2*srr
  return b,sb,Bv,sB,share,sshare
fx=fit(LX3,sLX3,LX6,sLX6); fy=fit(LY3,sLY3,LY6,sLY6)
print("X",fx); print("Y",fy)
print("share X above 0.5:",(fx[4]-0.5)/fx[5], "contrast",(fx[4]-fy[4])/math.hypot(fx[5],fy[5]))
print("LY/XZZX X r3d",LY3/3.050e-4,"r6d",LY6/3.163e-4)
# XZZX rise
a=3.050e-4; sa=a/math.sqrt(11782); b=3.163e-4; sb=b/math.sqrt(14848); print("xzzx rise",(b-a)/math.hypot(sa,sb), sa, sb)

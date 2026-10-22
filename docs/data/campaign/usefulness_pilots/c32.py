import math
from scipy.stats import chi2 as C
def eps(e,n,r): p=e/n; return (1-(1-2*p)**(1/r))/2
def eps2(e,n,r): p=e/n; return 1-(1-p)**(1/r)
Y=[(15,3119),(11,3202),(16,3194),(14,3444),(13,3415),(14,3328),(11,3328),(13,3584),(12,3550),(19,3750),(12,3750)]
X=[(15,3280),(14,3232),(10,3168),(11,3436),(7,3328),(7,3328),(17,3584),(16,3550),(13,3750),(16,3750)]
def pool(L,r=15):
    e=sum(a for a,b in L); n=sum(b for a,b in L)
    ep=eps(e,n,r); 
    # chi2 homogeneity
    pp=e/n; c=sum((a-b*pp)**2/(b*pp*(1-pp)) for a,b in L)
    return e,n,ep,ep/math.sqrt(e),c,len(L)-1,C.sf(c,len(L)-1)
for nm,L in [('Y',Y),('X',X)]:
    print(nm,len(L),pool(L))
Z=[(2541,1510350),(2890,1712704),(2437,1499079),(400,245000)]
zs=[eps(a,b,21) for a,b in Z]; print('xzzx',zs,[eps2(a,b,21) for a,b in Z])
e=sum(a for a,b in Z); n=sum(b for a,b in Z); pp=e/n
c=sum((a-b*pp)**2/(b*pp*(1-pp)) for a,b in Z); print('chi2',c,C.sf(c,3))
eY,nY=150,37664; eX,nX=126,34406
for nm,(ee,nn) in [('Y',(eY,nY)),('X',(eX,nX))]:
  ey=eps(ee,nn,15)
  for (a,b),zz in zip(Z,zs):
    R=ey/zz; s=math.sqrt(1/ee+1/a)
    sb=math.sqrt(1/ee+1.7**2/a)
    sc15=math.sqrt(1/ee+1/a+0.15**2/11); sc25=math.sqrt(1/ee+1/a+0.25**2/11)
    print(nm,round(R,3),round(s,4),'za',round(math.log(R)/s,1),'zb17',round(math.log(R)/sb,1),'zc',round(math.log(R)/sc15,1),round(math.log(R)/sc25,1),'pertick',round(R*6/8,2))
# corr_gauge
for nm,L in [('X',[(1216,163584),(437,55000)]),('Y',[(1252,163584),(430,55000)])]:
  for a,b in L: ee=eps(a,b,15); print(nm,a,b,ee,ee/math.sqrt(a),eps2(a,b,15))
  e1,n1=L[0];e2,n2=L[1]; p1=e1/n1;p2=e2/n2; print(' sig diff', (p2-p1)/math.sqrt(p1*(1-p1)/n1+p2*(1-p2)/n2))
  e=e1+e2;n=n1+n2; ep=eps(e,n,15); print(' pooled',e,n,ep,ep/math.sqrt(e))
  for (a,b),zz in zip(Z,zs): R=ep/zz; print('  R',round(R,3),'z',round(math.log(R)/math.sqrt(1/e+1/a),1))

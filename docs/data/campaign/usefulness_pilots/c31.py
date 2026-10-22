import math
from scipy import stats
def z(R,N1,N2,f1=1,f2=1,k=1,delta=0):
    l=abs(math.log(R)); return l/math.sqrt(1/N1+1/N2), l/math.sqrt(f1**2/N1+f2**2/N2)
def dstar(R,N1,N2,k):
    v=(math.log(R)/3)**2-1/N1-1/N2
    return math.sqrt(k*v) if v>0 else float('nan')
def tconv(t,dof):
    p=stats.t.sf(t,dof); return stats.norm.isf(p)
# 18/16
R=3.823e-3/5.441e-3; print('18/16',R,z(R,408,503,1.7,1.7),dstar(R,408,503,1))
# interp
import numpy as np
lin=7/9*math.log(5.441e-3)+2/9*math.log(3.348e-3); I=math.exp(lin); R=3.823e-3/I
var2=(7/9)**2/503+(2/9)**2/490
print('interp',I,R, math.log(1/R)/math.sqrt(1/408+var2), math.log(1/R)/math.sqrt(1.7**2/408+1.7**2*var2), math.sqrt(((math.log(R)/3)**2-1/408-var2)) if (math.log(R)/3)**2-1/408-var2>0 else 'neg')
# 3%
R=8.632e-4/1.067e-3; print('3%',R,z(R,514,25589,1,1.7),dstar(R,514,25589,2))
# 2.5
eps=3.834e-4; R=eps/4.897e-4; print('2.5',R,z(R,236,16015,1,1.7),dstar(R,236,16015,2))
# 2% LY
R=None

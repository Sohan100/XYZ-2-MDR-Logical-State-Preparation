import math
from scipy.stats import chi2 as C
def st(runs):
    E=sum(a for a,b in runs); S=sum(b for a,b in runs); p=E/S
    c=sum((a-p*b)**2/(p*b) for a,b in runs); dof=len(runs)-1
    W=sum((b/S*E)*(1-b/S) for a,b in runs)
    return round(c,2),dof,round(C.sf(c,dof),3),round(W,1),round(math.sqrt(max(0,c-dof)/W),3)
x2=[(55,25400),(45,23371),(38,23016),(38,21488),(53,21488),(41,22512),(66,22569),(62,22000),(46,17450),(29,17450)]
x2b=[(55,25400),(45,23371),(38,23016),(132,65488),(66,22569),(62,22000),(75,34900)]
y7=[(32,7152),(26,7152),(21,7152),(33,6774),(10,6025)]
y7b=[(32,7152),(47,14304),(33,6774),(10,6025)]
m1=[(80,3401),(69,3187),(45,3201),(64,3269)]
m1b=[(80,3401),(114,6388),(64,3269)]
for n,r in [('x2',x2),('x2 batch',x2b),('y7',y7),('y7 batch',y7b),('m1',m1),('m1 batch(a+b)',m1b)]: print(n,st(r))

import math, sys
def z(R,N1,N2,f1=1,f2=1,k=1):
    l=abs(math.log(R))
    a=l/math.sqrt(1/N1+1/N2)
    b=l/math.sqrt(f1**2/N1+f2**2/N2)
    c=[l/math.sqrt(1/N1+1/N2+d*d/k) for d in (0.15,0.25,0.37)]
    ds=math.sqrt(max(0,k*((l/3)**2-1/N1-1/N2)))
    return round(a,2),round(b,2),[round(x,2) for x in c],round(ds,3)

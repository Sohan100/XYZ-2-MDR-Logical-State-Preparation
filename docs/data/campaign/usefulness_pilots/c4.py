import math
from scipy.stats import chi2
s2=math.sqrt(2)
LX={1:(11.0031,0.38694),2:(4.03799,0.20002),3:(2.33794,0.089547)}
LY={1:(15.79507,0.63555),2:(5.35841,0.13096),3:(2.83458,0.061298)}
LC={1:(13.84486,1.23368),2:(6.557622,0.139887),3:(3.981811,0.053102)}
for name,L in (('X',LX),('Y',LY)):
  for p in (1,2,3):
    a,ea=L[p]; c,ec=LC[p]
    f=math.log(a)/(s2*math.log(c))
    df=f*math.hypot((ea/a)/math.log(a),(ec/c)/math.log(c))
    print(name,p,'ratio %.3f frac %.3f(%.3f)'%(a/c**s2,f,df))
up=chi2.ppf(1-0.00135,6)/2; lo=chi2.ppf(0.00135,600)/2
print(up,lo, 693.4336*(lo/300)/(up/2), 693.4336/(up/2))
xyz={1:5.877e-4,2:2.745e-3,3:7.151e-3}
c3={1:7.265e-4,2:2.745e-3,3:4.623e-3}
c5={1:3.921e-5,2:3.35e-4,3:1.084e-3}
for p in (1,2,3):
  fn=(35-17)/32; fd=(math.sqrt(18)-3)/2
  i1=math.exp(math.log(c3[p])+fn*(math.log(c5[p])-math.log(c3[p])))
  i2=math.exp(math.log(c3[p])+fd*(math.log(c5[p])-math.log(c3[p])))
  print(p, xyz[p]/c3[p], i1, xyz[p]/i1, xyz[p]/i2)

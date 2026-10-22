import numpy as np
from scipy.stats import chi2, norm
p3=norm.sf(3)
def lo(n): return chi2.ppf(p3,2*n)/2
def up(n): return chi2.ppf(1-p3,2*(n+1))/2
print('lo32',lo(32),'up1',up(1),'lo301',lo(301),'up3',up(3))
print(82.2*(lo(32)/32)*(1/up(1)), 849.9*(lo(301)/301)*(3/up(3)))
s2=np.sqrt(2)
def parity(Y2,X2,Y1,X1):
    g2=np.log(Y2)-s2*np.log(X2); g1=np.log(Y1)-s2*np.log(X1)
    dg=g1-g2  # per factor 2 decrease
    n=-g1/dg
    return 1e-3/2**n, g1/(g1-0)
# steps d7->9
b_X1=8.140688e-08/2.757061e-09; b_Y1=1.298914e-07/1.667390e-09; p_Y1=9.842850e-08/2.404815e-09
x_b1=3.921476e-07/1.684138e-08; x_p1=2.368604e-07/7.947685e-09
b_X2=4.730813e-06/3.782521e-07; b_Y2=5.175477e-06/4.410517e-07; p_Y2=4.288514e-06/4.102281e-07
x_b2=6.546407e-06/5.012065e-07; x_p2=3.684174e-06/2.653559e-07
print('steps',b_X1,b_Y1,p_Y1,x_b1,x_p1,b_X2,b_Y2,p_Y2,x_b2,x_p2)
print('frac1',np.log(b_X1)/(s2*np.log(x_b1)),np.log(b_Y1)/(s2*np.log(x_b1)),np.log(p_Y1)/(s2*np.log(x_p1)))
print('frac2',np.log(b_X2)/(s2*np.log(x_b2)),np.log(b_Y2)/(s2*np.log(x_b2)),np.log(p_Y2)/(s2*np.log(x_p2)))
print('parity b X',parity(b_X2,x_b2,b_X1,x_b1)[0])
print('parity b Y',parity(b_Y2,x_b2,b_Y1,x_b1)[0])
print('parity p Y',parity(p_Y2,x_p2,p_Y1,x_p1)[0])
# upper ends
b_X1u=8.140688e-08/2.045080e-09; p_Y1u=9.842850e-08/1.728902e-09
print('68% up', b_X1u, p_Y1u, parity(b_X2,x_b2,b_X1u,x_b1)[0], parity(p_Y2,x_p2,p_Y1u,x_p1)[0])
# lambda.csv
print('lam b',parity(12.263311,13.556778,31.361373,25.201840)[0],'lam p',parity(12.202421,14.705592,41.535794,31.526740)[0])
# with 3e-3->2e-3 slopes extrapolated from 2e-3
def par2(Y3,X3,Y2,X2):
    g3=np.log(Y3)-s2*np.log(X3); g2=np.log(Y2)-s2*np.log(X2)
    slope=(g2-g3)/np.log(1.5)
    return 2e-3*np.exp(-(-g2)/slope)
print('3to2', par2(5.453917,7.841633,12.263311,13.556778), par2(5.673773,9.116592,12.202421,14.705592))

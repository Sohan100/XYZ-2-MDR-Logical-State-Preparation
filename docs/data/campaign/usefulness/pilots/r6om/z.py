import math
from math import log, sqrt
# name: R, N1 (xyz2 errors), k runs, own-replicate chi2/dof, comp rel var
pts = {
 'hw_tr 6% 99/97 Y': (0.613, 218, 1, None, 1/29621),
 'hw_tr 4% r3d X': (0.608, 62, 1, None, 1/11782),
 'hw_tr 3% Y': (7.15e-5/1.054e-4, 73, 2, (0.01,1), 1/6166),
 'hw_tr 195 Y': (0.538, 29, 2, (0.02,1), 0.0),
 'hw_tr r6d Y': (0.384, 99, 2, (0.23,1), 1/14848),
 'hw_tr eta10 Y': (0.807, 99, 2, (0.17,1), 1/7370),
 'm1 X': (0.700, 258, 4, (8.29,3), 1/34217),
 'A X': (0.752, 174, 1, None, 1/57973),
 'E Y': (1.718, 112, 2, (0.01,1), 1/11601),
 'eta100 better X/Z': (2.409, 29, 1, None, 1/1152),
 'phen_b10 3% 50/49 X': (0.809, 514, 2, (0.40,1), 1/25589),
 'phen_b10 18/16 vs d4': (0.70, 408, 1, None, 1/503),
 'phen_b10 18/16 interp': (3.823e-3/4.884e-3, 408, 1, None, (7/9)**2/503+(2/9)**2/490),
 'phen_b10 2% L_Y 50/49': (0.81, 402, 8, (4.16,7), 1/20809),
 'phen_b10 2.5% 50/49 X': (3.834e-4/4.897e-4, 236, 2, None, 1/16015),
 'phen q0 L_X': (1.22, 352, 4, (4.73,3), 1/759),
 'phen q0 L_Y': (1.445, 416, 4, (1.98,3), 1/759),
 'phen 18/16': (1.36, 592, 1, None, 1/579),
 'phen 50/49': (1.83, 416, 1, None, 1/468),
 'phen 98/100': (2.36, 243, 2, (0.46,1), 1/518),
 'hw_rr BM Y': (1.98, 887, 1, None, 1/14883),
}
def zs(R,N1,k,rep,cv):
    L=abs(log(R))
    pois=L/sqrt(1/N1+cv)
    lin_p=abs(1-R)/(R*sqrt(1/N1+cv))
    f166=L/sqrt(1.66**2/N1+cv)
    lin166=abs(1-R)/(R*sqrt(1.66**2/N1+cv))
    if rep: f=max(1,sqrt(rep[0]/rep[1]))
    else: f=1.7
    own=L/sqrt(f**2/N1+cv)
    c15=L/sqrt(1/N1+cv+0.15**2/k); c25=L/sqrt(1/N1+cv+0.25**2/k); c065=L/sqrt(1/N1+cv+0.065**2/k)
    # max delta for 3 sigma
    d3=sqrt(max(0,(L/3)**2-1/N1-cv)*k)
    return pois,lin_p,f166,lin166,f,own,c065,c15,c25,d3
print(f"{'point':28s} R     logP  linP  log1.66 lin1.66 f_b  (b)   c.065 c.15  c.25  dmax3")
for n,v in pts.items():
    r=zs(*v)
    print(f"{n:28s} {v[0]:.3f} {r[0]:5.2f} {r[1]:5.2f} {r[2]:6.2f} {r[3]:6.2f} {r[4]:4.2f} {r[5]:5.2f} {r[6]:5.2f} {r[7]:5.2f} {r[8]:5.2f} {r[9]:5.3f}")
# Stouffer of hw_tr eta20 edge
names=['hw_tr 6% 99/97 Y','hw_tr 4% r3d X','hw_tr 3% Y','hw_tr r6d Y','hw_tr 195 Y']
for lab,idx in (('pois',0),('1.66',2),('c.25',8),('c.15',7)):
    zz=[zs(*pts[n])[idx] for n in names]
    print('Stouffer',lab, round(sum(zz)/sqrt(len(zz)),2), 'w/o r6d', round(sum(zz[:3]+zz[4:])/sqrt(4),2))

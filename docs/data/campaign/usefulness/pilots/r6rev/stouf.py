import math
from scipy import stats
# (label, ratio, N_xyz2, k runs, N_xzzx, extra var)
pts=[('6%',0.613,218,1,29621,0),('4% r3d',0.608,62,1,11782,0),('3% Y',0.679,73,2,6166,0),('r6d Y',0.384,99,2,14848,0),('195 Y',0.538,29,2,1e9,0.016**2+1/4486*0+0)]
for d in (0.0,0.065,0.11,0.15,0.25):
    zs=[]
    for lab,R,N,k,M,ev in pts:
        s=math.sqrt(1/N+d*d/k+1/M+ev)
        zs.append(-math.log(R)/s)
    S=sum(zs)/math.sqrt(len(zs)); S4=(sum(zs)-zs[3])/2
    print(d,[round(z,2) for z in zs],'stouffer5',round(S,2),'without r6d',round(S4,2))
# (b) uniform f=1.66 on single runs (own f=1 for homogeneous pairs) + Student t not needed
for f in (1.66,):
    zs=[-math.log(R)/math.sqrt(f*f/N+1/M) for lab,R,N,k,M,ev in pts]
    print('b f=1.66 all',[round(z,2) for z in zs], round(sum(zs)/math.sqrt(5),2))

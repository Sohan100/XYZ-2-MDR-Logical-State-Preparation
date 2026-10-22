from math import *
from scipy.stats import t as T, norm
def eps(err,shots,r): pL=err/shots; return (1-(1-2*pL)**(1/r))/2
def z(R,v): return abs(log(R))/sqrt(v)
def dstar(R,v,k):  # delta^2/k added
    a=(log(R)/3)**2 - v
    return sqrt(max(a,0)*k)
# opening
x3=eps(62,22344,15); zx=3.050e-4; Nzx=11782
R=x3/zx; v=1/62+1/Nzx
print('r3d',R,R*sqrt(v),z(R,v),z(R,2.89*v),[z(R,v+d*d) for d in(0.15,0.25)],dstar(R,v,1))
y6=eps(99,27298,30); x6=eps(93,27143,30); zX=eps(14848,1132209,42); zZ=eps(9381,1107657,42)
print('eps',y6,x6,zX,zZ)
for nm,e,n in(('Y',y6,99),('X',x6,93)):
    R=e/zX; v=1/n+1/14848
    print('r6d',nm,R,R*sqrt(v),z(R,v),z(R,1/n+2.89/14848),[z(R,v+d*d/2) for d in(0.15,0.25,0.37)],dstar(R,v,2),'shared',dstar(R,v,1))
R=x6/zZ; v=1/93+1/9381
print('better',R,R*sqrt(v),z(R,v),z(R,1/93+2.89/9381),dstar(R,v,2))
# item3
y3=eps(44,24063,15)
print('LY vs Z',y3/2.016e-4, y3/2.016e-4*sqrt(1/44+1/7847), y6/zZ, y6/zZ*sqrt(1/99+1/9381), y3/zx, y6/zX)
print('XZZX rise', log(3.163/3.050)/sqrt(1/11782+1/14848))
# item4
c=1.383e-4; Nc=759
def it4(nm,e,n,f=1,k=None):
    R=e/c; v=1/n+1/Nc
    out=[nm,round(R,4),round(R*sqrt(v),4),round(z(R,v),2),round(z(R,f*f/n+2.89/Nc),2)]
    if k: out+= [round(z(R,v+d*d/k),2) for d in (0.15,0.25)]+[round(dstar(R,v,k),3)]
    print(out)
it4('BM Y',1.998e-4,416,1,4); it4('BM X',1.690e-4,352,sqrt(4.73/3),4)
zb=z(1.690e-4/c,1.575/352+2.89/759); print('t3 conv', norm.isf(T.sf(zb,3)))
for nm,e,n in(('cgY',2.635e-4,516),('cgX',2.235e-4,438),('bpfY',1.907e-4,73),('bpfX',1.541e-4,59),('bpcY',2.158e-4,33),('bpcX',9.327e-5,15),('cfeY',1.671e-4,7),('cfeX',9.537e-5,4)):
    it4(nm,e,n,1.7)
print('xzzx bm', 1.236/1.383, z(1.236/1.383,1/53+1/759))
print('q0/hwtr', 1.998e-4/9.751e-5, 1.998e-4/9.751e-5*sqrt(1/416+1/99), 1.690e-4/8.950e-5, 1.690/0.8950*sqrt(1/352+1/91), 1.383/1.208, 1.383/1.208*sqrt(1/759+1/7370), 9.751/12.08, 9.751/12.08*sqrt(1/99+1/7370))
print('pool', (112+110+99+95)/(4*34800), eps(416,4*34800,15), eps(352,4*34800,15), eps(99,67777,15), eps(91,67867,15))
# item7
for nm,a,na,b,nb in(('hw',9.309e-4,1952,5.021e-5,1053),('ph',1.446e-3,284,1.717e-4,1080)):
    R=a/b; v=1/na+1/nb
    print(nm,R,R*sqrt(v),z(R,v),z(R,2.89*v),z(R,v+0.0625),z(R,v+0.37**2),dstar(R,v,1))

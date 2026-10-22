from math import *
def eps(n,s,r): return n/s/r
def z(R,var): return abs(log(R))/sqrt(var)
def deltastar(R,ns,Nc,shared=False):
    lo,hi=0,5
    for _ in range(100):
        d=(lo+hi)/2
        if shared: v=1/sum(ns)+d*d+1/Nc
        else: v=1/sum(1/(1/n+d*d) for n in ns)+1/Nc
        if abs(log(R))/sqrt(v)>3: lo=d
        else: hi=d
    return lo
# r=6d
eY=(52+47)/(13687+13611)/30; eX=(46+47)/(13542+13601)/30
ezX=3.163e-4; ezZ=9381/1107657/42
print('ezZ',ezZ, 'ezX', 14848/1132209/42)
for nm,e,ns,Nc,ec in [('Y vs X',eY,[52,47],14848,ezX),('X vs X',eX,[46,47],14848,ezX),('X vs Z',eX,[46,47],9381,ezZ)]:
    R=e/ec; N=sum(ns)
    print(nm,R,R*sqrt(1/N+1/Nc),z(R,1/N+1/Nc),z(R,1/N+2.89/Nc),deltastar(R,ns,Nc),deltastar(R,ns,Nc,True))
# chi2 X pair
# r=3d
R=1.855e-4/3.050e-4; print('3d',R,z(R,1/62+1/11782),z(R,2.89/62+2.89/11782))
# item 3
eX15=62/22344/15; print('eX15',eX15,'eX30',eX)
b=2*eX-eX15; B=30*(eX15-eX); print('b',b,'B',B,'share',B/15/eX15)
sX15=eX15/sqrt(62); sX30=eX/sqrt(93)
sb=sqrt(4*sX30**2+sX15**2); sB=30*sqrt(sX15**2+sX30**2)
print('sb',sb,'sB',sB)
# share = B/(15 eX15) = 2(1 - eX/eX15)
r=eX/eX15; sr=r*sqrt(1/62+1/93); print('share',2*(1-r),2*sr,(2*(1-r)-0.5)/(2*sr))
eY15=44/24063/15; rY=eY/eY15; print('Yshare',2*(1-rY),2*rY*sqrt(1/44+1/99), 'eY15',eY15)
print('LX ratio',1/r, (1/r)*sqrt(1/62+1/93), log(1/r)/sqrt(1/62+1/93))
print('xzzx rise z', log(3.163/3.050)/sqrt(1/11782+1/14848))
print('LY/XZZX X 3d', eY15/3.050e-4, ' 6d', eY/3.163e-4)
# item 4
Y0=416/(4*34800)/15; X0=352/(4*34800)/15
print('Y0 X0',Y0,X0)
ezc=1.383e-4
for nm,e,n,f in [('BM Y',Y0,416,1),('BM X',X0,352,1.255)]:
    R=e/ezc; print(nm,R,R*sqrt(1/n+1/759),z(R,1/n+1/759),z(R,f*f/n+2.89/759),deltastar(R,[112,110,99,95] if 'Y' in nm else [100,88,72,92],759))
for nm,e,n in [('cg Y',2.635e-4,516),('cg X',2.235e-4,438),('bpf Y',1.907e-4,73),('bpf X',1.541e-4,59),('bpc Y',2.158e-4,33),('bpc X',9.327e-5,15),('cfe Y',1.671e-4,7),('cfe X',9.537e-5,4)]:
    R=e/ezc; print(nm,R,R*sqrt(1/n+1/759),z(R,1/n+1/759),z(R,2.89/n+2.89/759))
eY10=99/67777/15; eX10=91/67867/15
print('hwtr10',eY10,eX10, eY10/1.208e-4, z(eY10/1.208e-4,1/99+1/7370), (eY10/1.208e-4)*sqrt(1/99+1/7370))
print('rises',Y0/eY10,Y0/eY10*sqrt(1/416+1/99),X0/eX10,X0/eX10*sqrt(1/352+1/91),ezc/1.208e-4,ezc/1.208e-4*sqrt(1/759+1/7370))
print('xzzx bm', 1.236e-4/ezc, z(1.236e-4/ezc,1/53+1/759))
# item 7
for nm,a,na,b_,nb,bm in [('hw',9.309e-4,1952,5.021e-5,1053,False),('phen',1.446e-3,284,1.717e-4,1080,True)]:
    R=a/b_; print(nm,R,R*sqrt(1/na+1/nb),z(R,1/na+1/nb),z(R,2.89/na+2.89/nb), deltastar(R,[na],nb) if bm else None)

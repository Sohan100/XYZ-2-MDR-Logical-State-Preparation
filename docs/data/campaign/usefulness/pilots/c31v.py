import numpy as np
from scipy import stats, optimize
def eps(pl,r): return (1-(1-2*pl)**(1/r))/2
def pool(runs,r):
    # runs (err,shots)
    E=sum(e for e,_ in runs); S=sum(s for _,s in runs)
    pl=E/S
    exp=[pl*s for _,s in runs]
    chi=sum((e-m)**2/m for (e,_),m in zip(runs,exp)) if len(runs)>1 else 0
    return eps(pl,r),E,chi,len(runs)
def zs(R,N1,N2,f1,f2,k,dofk=None):
    L=abs(np.log(R))
    za=L/np.sqrt(1/N1+1/N2)
    zb=L/np.sqrt(f1**2/N1+f2**2/N2)
    if dofk:
        p=stats.t.sf(zb,dofk); zbt=stats.norm.isf(p)
    else: zbt=zb
    d2=k*((L/3)**2-1/N1-1/N2)
    ds=np.sqrt(d2) if d2>0 else float('nan')
    return R, R*np.sqrt(1/N1+1/N2), za, zb, zbt, ds
def dstar_iv(runs,r,N2,R):
    L=abs(np.log(R))
    def g(d): 
        v=1/sum(1/(1/e+d*d) for e,_ in runs)+1/N2
        return L/np.sqrt(v)-3
    try: return optimize.brentq(g,1e-6,5)
    except: return float('nan')
r=15
# 2% L_Y
Y=[(43,22621),(55,25400),(58,23388),(56,22969),(53,21488),(47,22512),(57,22512),(33,16550)]
ey,Ny,cy,k=pool(Y,r); print('2%Y',Ny,sum(s for _,s in Y),cy)
xz=[(7738,1972704),(7454,1912022),(506,131056),(497,131056),(4614,1198350)]
ez,Nz,cz,_=pool(xz,21); print('xzzx pool',ez,Nz,cz)
print('2%Y',zs(ey/ez,Ny,Nz,np.sqrt(max(1,cy/7)),1,8), dstar_iv(Y,r,Nz,ey/ez))
e1=eps(7738/1972704,21)
print('2%Y vs single',zs(ey/e1,Ny,7738,max(1,np.sqrt(cy/7)),1.7,8),dstar_iv(Y,r,7738,ey/e1))
Y2=[y for y in Y if y!=(55,25400)]
ey2,Ny2,cy2,k2=pool(Y2,r); print('2%Y no b10',zs(ey2/ez,Ny2,Nz,max(1,np.sqrt(cy2/6)),1,7))
X=[(66,22569),(62,22000),(55,25400),(45,23371),(38,23016),(38,21488),(53,21488),(41,22512),(46,17450),(29,17450)]
ex,Nx,cx,k=pool(X,r); f=np.sqrt(cx/9); print('2%X',Nx,cx,f, zs(ex/ez,Nx,Nz,f,1,10,9), dstar_iv(X,r,Nz,ex/ez))
X2=[x for x in X if x!=(55,25400)]
ex2,Nx2,cx2,k2=pool(X2,r); f2=np.sqrt(cx2/8); print('2%X no b10',cx2,f2,zs(ex2/ez,Nx2,Nz,f2,1,9,8))
# 3%
X3=[(254,20416),(260,19762)]; e3,N3,c3,_=pool(X3,r); print('3%',c3)
# XZZX r=21 for d=7
ez3=1.067e-3
print('3%',zs(e3/ez3,N3,25589,1,1.7,2),dstar_iv(X3,r,25589,e3/ez3))
# 2.5
X25=[(119,20104),(117,21150)]; e25,N25,c25,_=pool(X25,r); print('2.5',c25,e25)
print('2.5',zs(e25/4.897e-4,N25,16015,1,1.7,2),dstar_iv(X25,r,16015,e25/4.897e-4))
# 18/16
R=3.823e-3/5.441e-3; print('18/16 d4',zs(R,408,503,1.7,1.7,1))
# interp
ei=np.exp(7/9*np.log(5.441e-3)+2/9*np.log(3.348e-3)); print('interp',ei)
L=np.log(3.823e-3/ei); va=1/408+(7/9)**2/503+(2/9)**2/490
print('interp R',3.823e-3/ei, abs(L)/np.sqrt(va), abs(L)/np.sqrt(1.7**2/408+1.7**2*((7/9)**2/503+(2/9)**2/490)), np.sqrt((L/3)**2-va) if (L/3)**2>va else 'neg', 'simple', np.sqrt((L/3)**2-1/408-1/503) )
# q=0
Yq=[(110,34800),(99,34800),(95,34800),(112,34800)]; eq,Nq,cq,_=pool(Yq,15); print('q0 Y',eq,cq,zs(eq/1.383e-4,Nq,759,max(1,np.sqrt(cq/3)),1.7,4), dstar_iv(Yq,15,759,eq/1.383e-4))
Xq=[(88,34800),(72,34800),(92,34800),(100,34800)]; eqx,Nqx,cqx,_=pool(Xq,15); f=np.sqrt(cqx/3); print('q0 X',eqx,cqx,f,zs(eqx/1.383e-4,Nqx,759,f,1.7,4,3))
print('q0 18/16',zs(1.958e-3/3.141e-3,426,596,1.7,1.7,1))
# r=1
print('r1 arch',zs(1.446e-3/1.717e-4,284,1080,1.7,1.7,1), 284/196352, 1080/6291200)
print('r1 unarch',zs(1.511e-3/1.524e-4,470,1278,1.7,1.7,1), (2.300e-3/1.524e-4), 2.300e-3/1.524e-4*np.sqrt(1/753+1/1278))
X15=[(22,23917),(15,24557),(13,12941),(14,12798),(11,12718),(7,12898)]
e,N,c,_=pool(X15,15); print('1.5X',e,N,c,e/5.156e-5)
e2,N2,_,_=pool(X15[:2],15); print('two',e2/5.156e-5, e2/5.156e-5*np.sqrt(1/37+1/2718))
Y15=[(18,24006),(8,13084),(14,13051),(7,12889),(4,12673)]
e,N,c,_=pool(Y15,15); print('1.5Y',e,N,e/5.156e-5,e/5.156e-5*np.sqrt(1/N+1/2718))

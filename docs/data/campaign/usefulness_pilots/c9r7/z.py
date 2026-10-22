import math
from scipy import stats, optimize
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def zc(lnR, ns, Nc, d):
    v=1/sum(1/(1/n+d*d) for n in ns)+1/Nc
    return abs(lnR)/math.sqrt(v)
def dstar(lnR, ns, Nc):
    if zc(lnR,ns,Nc,0)<3: return None
    try: return optimize.brentq(lambda d: zc(lnR,ns,Nc,d)-3, 0, 5)
    except: return '>5'
def tconv(t,dof):
    p=stats.t.sf(t,dof); return stats.norm.isf(p)
# opening r=3d
R=1.855e-4/3.050e-4; l=math.log(R)
print('r3d',R, l/ -math.sqrt(1/62+1/11782), -l/math.sqrt(2.89/62+2.89/11782), zc(l,[62],11782,.15), zc(l,[62],11782,.25), dstar(l,[62],11782))
# r6d
eY=eps(99,13687+13611,30); eX=eps(93,13542+13601,30)
for name,e,ns,ec,Nc in [('6dY',eY,[52,47],3.163e-4,14848),('6dX',eX,[46,47],3.163e-4,14848),('6dXvsZ',eX,[46,47],2.033e-4,9381),('6dYvsZ',eY,[52,47],2.033e-4,9381)]:
    R=e/ec; l=math.log(R); N=sum(ns)
    print(name, e, R, R*math.sqrt(1/N+1/Nc), 'a',-l/math.sqrt(1/N+1/Nc), 'b',-l/math.sqrt(1/N+2.89/Nc), 'c15',zc(l,ns,Nc,.15),'c25',zc(l,ns,Nc,.25),'c37',zc(l,ns,Nc,.37),'d*',dstar(l,ns,Nc),
      'k1 d*', math.sqrt((l/3)**2-1/N-1/Nc) if (l/3)**2>1/N+1/Nc else None, 'k1 z37', -l/math.sqrt(1/N+1/Nc+.37**2))
# chi2 replicate for r6d Y pool
def chi(runs):
    e=sum(a for a,b in runs); s=sum(b for a,b in runs); p=e/s
    return sum((a-p*b)**2/(p*b) for a,b in runs)
print('chi Y',chi([(52,13687),(47,13611)]),'chi X',chi([(46,13542),(47,13601)]))
# item 4 q=0
xc=1.383e-4; Nc=759
Yruns=[112,110,99,95]; Xruns=[100,88,72,92]
eYq=sum(Yruns)/(4*34800); eXq=sum(Xruns)/(4*34800)
print('q0 eps', eYq, eXq, 'note: r=15 conv')
print('---- item 4')
eYq=eps(416,4*34800,15); eXq=eps(352,4*34800,15)
Nc=759; ec=1.383e-4
chiY=chi([(112,34800),(110,34800),(99,34800),(95,34800)]); chiX=chi([(100,34800),(88,34800),(72,34800),(92,34800)])
fY=max(1,math.sqrt(chiY/3)); fX=max(1,math.sqrt(chiX/3))
print('chi',chiY,chiX,'f',fY,fX)
for name,e,N,f,ns in [('BM Y',eYq,416,fY,[112,110,99,95]),('BM X',eXq,352,fX,[100,88,72,92])]:
    R=e/ec; l=math.log(R); a=l/math.sqrt(1/N+1/Nc); b=l/math.sqrt(f*f/N+2.89/Nc)
    print(name,e,R,R*math.sqrt(1/N+1/Nc),'a',a,'b',b,'b as t3',tconv(b,3) if f>1 else b,'c15',zc(l,ns,Nc,.15),'c25',zc(l,ns,Nc,.25),'d*',dstar(l,ns,Nc))
for name,e,N in [('cg Y',2.635e-4,516),('cg X',2.235e-4,438),('bpf Y',1.907e-4,73),('bpf X',1.541e-4,59),('bpc Y',2.158e-4,33),('bpc X',9.327e-5,15),('cfe Y',1.671e-4,7),('cfe X',9.537e-5,4)]:
    R=e/ec; l=math.log(R); print(name,R,R*math.sqrt(1/N+1/Nc),'a',abs(l)/math.sqrt(1/N+1/Nc),'b',abs(l)/math.sqrt(2.89/N+2.89/Nc))
# hw_tr eta10 comparisons
eY10=eps(99,34284+33493,15); eX10=eps(91,34371+33496,15)
print('hwtr e10 Y',eY10,'X',eX10,'Y/XZZX',eY10/1.208e-4, (eY10/1.208e-4)*math.sqrt(1/99+1/7370))
print('q0/hwtr Y',eYq/eY10, eYq/eY10*math.sqrt(1/416+1/99),'X',eXq/eX10,eXq/eX10*math.sqrt(1/352+1/91),'XZZX',ec/1.208e-4,ec/1.208e-4*math.sqrt(1/759+1/7370))
print('XZZX bm vs corr',1.236e-4/ec, math.log(ec/1.236e-4)/math.sqrt(1/53+1/759))
print('---- item 7')
for name,e1,n1,e2,n2,bm in [('hwtr',9.309e-4,1952,5.021e-5,1053,False),('phen',1.446e-3,284,1.717e-4,1080,True)]:
    R=e1/e2; l=math.log(R); print(name,R,R*math.sqrt(1/n1+1/n2),'a',l/math.sqrt(1/n1+1/n2),'b',l/math.sqrt(2.89/n1+2.89/n2),'bx',l/math.sqrt(2.89/n1+1/n2), 'c25',zc(l,[n1],n2,.25) if bm else None,'c37',zc(l,[n1],n2,.37) if bm else None,'d*',dstar(l,[n1],n2) if bm else None)
print('hwtr exact', 1952/2096896, 1053/20971264, (1952/2096896)/(1053/20971264))
print('phen exact', 284/196352, 1080/6291200, (284/196352)/(1080/6291200))
# item 3
eX15=1.855e-4; nX15=62; eX30=eX; nX30=93
b=2*eX30-eX15; sb=math.sqrt((2*eX30)**2/nX30+eX15**2/nX15)
B=30*(eX15-eX30); sB=30*math.sqrt(eX15**2/nX15+eX30**2/nX30)
print('item3 b',b,sb,'B',B,sB,'share',B/15/eX15)

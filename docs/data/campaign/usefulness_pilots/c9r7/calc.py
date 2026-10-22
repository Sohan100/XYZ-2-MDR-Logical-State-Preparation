import math
from scipy.optimize import brentq
from scipy import stats
def eps(e,s,r): pl=e/s; return 0.5*(1-(1-2*pl)**(1/r))
def pool(runs):
    e=sum(a for a,b in runs); s=sum(b for a,b in runs); p=e/s
    chi=sum((a-p*b)**2/(p*b) for a,b in runs) if len(runs)>1 else 0
    return e,s,chi,len(runs)-1
def zs(name,R,nA,nB,fA,fB,runsA=None,kdof=None,bm=True):
    L=abs(math.log(R))
    za=L/math.sqrt(1/nA+1/nB)
    zb=L/math.sqrt(fA**2/nA+fB**2/nB)
    zbt=None
    if kdof and fA>1:
        p=stats.t.sf(zb,kdof); zbt=stats.norm.isf(p)
    ds=None
    if bm and runsA:
        def zc(d):
            v=1/sum(1/(1/n+d*d) for n in runsA)+1/nB
            return L/math.sqrt(v)-3
        if zc(0)>0:
            try: ds=brentq(zc,0,5)
            except Exception: ds=float('inf')
        else: ds=0
    print(f"{name}: R={R:.4f}({R*math.sqrt(1/nA+1/nB):.4f}) za={za:.2f} zb={zb:.2f}" + (f" zb_t={zbt:.2f}" if zbt else "") + (f" delta*={ds:.3f}" if ds is not None else ""))
    return ds
# Opening r=3d
eX=eps(62,22344,15); print('eps X r3d',eX)
zs('r3d X vs XZZX X', eX/3.050e-4, 62,11782,1.7,1.7,[62])
# r=6d
eY6=eps(99,27298,30); eX6=eps(93,27143,30)
print('eY6',eY6,'eX6',eX6, pool([(52,13687),(47,13611)]), pool([(46,13542),(47,13601)]))
zs('r6d Y vs X', eY6/3.163e-4, 99,14848,1,1.7,[52,47])
zs('r6d X vs X', eX6/3.163e-4, 93,14848,1,1.7,[46,47])
zs('r6d X vs Z (better/better)', eX6/2.033e-4, 93,9381,1,1.7,[46,47])
zs('r6d Y vs Z', eY6/2.033e-4, 99,9381,1,1.7,[52,47])
# k=1 shared offset delta*: single offset on pooled
for nm,R,n in [('Y',eY6/3.163e-4,99),('X',eX6/3.163e-4,93),('Xbb',eX6/2.033e-4,93)]:
    L=abs(math.log(R)); nb=14848 if nm!='Xbb' else 9381
    d2=(L/3)**2-1/n-1/nb; print('k=1 delta*',nm, math.sqrt(d2) if d2>0 else 0)
    # z at 0.37 indep
for nm,R,runs in [('Y',eY6/3.163e-4,[52,47]),('X',eX6/3.163e-4,[46,47])]:
    L=abs(math.log(R)); v=1/sum(1/(1/n+0.37**2) for n in runs)+1/14848; print('z at 0.37',nm,L/math.sqrt(v))
# item 3
eX3=eps(62,22344,15); eY3=eps(44,24063,15)
print('L_X r3d',eX3,eX3/math.sqrt(62),'r6d',eX6,eX6/math.sqrt(93),'ratio',eX3/eX6, eX3/eX6*math.sqrt(1/62+1/93), 'lnz',math.log(eX3/eX6)/math.sqrt(1/62+1/93))
print('L_Y r3d',eY3,eY3/math.sqrt(44),'r6d',eY6,eY6/math.sqrt(99))
b=2*eX6-eX3; sb=math.sqrt((2*eX6/math.sqrt(93))**2+(eX3/math.sqrt(62))**2)
B=30*(eX3-eX6); sB=30*math.sqrt((eX6/math.sqrt(93))**2+(eX3/math.sqrt(62))**2)
sh=(B/15)/eX3
# share error: share = 2(1-eX6/eX3) -> d share = 2*eX6/eX3*sqrt(1/62+1/93)
ssh=2*eX6/eX3*math.sqrt(1/62+1/93)
print('b',b,sb,'B',B,sB,'share',sh,ssh,'(sh-0.5)/ssh',(sh-0.5)/ssh)
shY=2*(1-eY6/eY3); sshY=2*eY6/eY3*math.sqrt(1/44+1/99); print('Y share',shY,sshY,'contrast',(sh-shY)/math.hypot(ssh,sshY))
print('LY/XZZX X r3d',eY3/3.050e-4,'r6d',eY6/3.163e-4)
print('XZZX rise', 3.163/3.050, math.log(3.163e-4/3.050e-4)/math.sqrt(1/11782+1/14848))
# item 4
Ypool=pool([(112,34800),(110,34800),(99,34800),(95,34800)]); Xpool=pool([(100,34800),(88,34800),(72,34800),(92,34800)])
print(Ypool,Xpool)
eYq=eps(416,139200,15); eXq=eps(352,139200,15); print('eYq',eYq,'eXq',eXq)
eYh=eps(99,67777,15); eXh=eps(91,67867,15); print('hw eta10 Y',eYh,'X',eXh)
print('rise Y',eYq/eYh, eYq/eYh*math.sqrt(1/416+1/99),'X',eXq/eXh,eXq/eXh*math.sqrt(1/352+1/91),'XZZX',1.383e-4/1.208e-4,1.383/1.208*math.sqrt(1/759+1/7370))
zs('hw_tr eta10 Y vs XZZX X',eYh/1.208e-4,99,7370,1,1.7,[48,51])
zs('q0 BM Y',eYq/1.383e-4,416,759,1,1.7,[112,110,99,95])
fX=max(1,math.sqrt(Xpool[2]/3)); print('fX',fX)
zs('q0 BM X',eXq/1.383e-4,352,759,fX,1.7,[100,88,72,92],kdof=3)
for nm,e,n in [('cg Y',2.635e-4,516),('cg X',2.235e-4,438),('bpf Y',1.907e-4,73),('bpf X',1.541e-4,59),('bpc Y',2.158e-4,33),('bpc X',9.327e-5,15),('cfe Y',1.671e-4,7),('cfe X',9.537e-5,4)]:
    zs(nm,e/1.383e-4,n,759,1.7,1.7,bm=False)
print('XZZX bm/corr',1.236e-4/1.383e-4, math.log(1.383/1.236)/math.sqrt(1/53+1/759))
# item 7
zs('r1 hw_tr', 9.309e-4/5.021e-5, 1952,1053,1.7,1.7,bm=False)
ds=zs('r1 phen', 1.446e-3/1.717e-4, 284,1080,1.7,1.7,[284])
L=math.log(1.446e-3/1.717e-4); print('phen z at .37',L/math.sqrt(1/284+1/1080+0.37**2), 'at .25', L/math.sqrt(1/284+1/1080+0.25**2))

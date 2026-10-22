import math
runs={'Ybm':(887,59048),'Xbm':(574,57771),'Ycg':(33044,2076184),'Ymw':(70997,4470689),'Xmw':(48956,4718132),
'zXc':(14883,1962124),'zXm':(27593,3629009),'zZm':(26868,3544469),'cX':(27267,3576266),'cZ':(27170,3609792)}
p={k:e/n for k,(e,n) in runs.items()}
def cmp(a,b,bm):
    e1,n1=runs[a]; e2,n2=runs[b]; R=p[a]/p[b]; L=math.log(R)
    sa=math.sqrt(1/e1+1/e2); sb=math.sqrt(2.89/e1+2.89/e2); sb1=math.sqrt(2.89/e1+1/e2)
    binrel=math.sqrt((1-p[a])/e1+(1-p[b])/e2)
    sR=R*binrel; zlin=(R-1)/sR
    s1=math.sqrt(p[a]*(1-p[a])/n1); s2=math.sqrt(p[b]*(1-p[b])/n2); zd=(p[a]-p[b])/math.hypot(s1,s2)
    out=f"{a}/{b}: R={R:.4f}({sR:.4f}) lin z={zlin:.2f} diff z={zd:.2f} | (a) {L/sa:.1f} (b both) {L/sb:.1f} (b xyz only) {L/sb1:.1f}"
    if bm:
        zc=[L/math.sqrt(1/e1+1/e2+d*d) for d in (0.15,0.25)]
        dmax=math.sqrt((L/3)**2-(1/e1+1/e2))
        out+=f" (c) {zc[0]:.2f}/{zc[1]:.2f} dmax={dmax:.3f}"
    else:
        zc=[L/math.sqrt(1/e1+1/e2+d*d) for d in (0.15,0.25)]
        out+=f" [if (c) applied anyway {zc[0]:.2f}/{zc[1]:.2f}]"
    print(out)
for k in p: print(k, f"{p[k]:.5e} +- {math.sqrt(p[k]*(1-p[k])/runs[k][1]):.2e}")
cmp('Ybm','zXc',1); cmp('Xbm','zXc',1); cmp('Ymw','zXm',0); cmp('Xmw','zXm',0); cmp('Ycg','zXc',0)
cmp('cX','zXm',0); cmp('cZ','zXm',0); cmp('Ycg','Ymw',0); cmp('Ybm','Ymw',0)

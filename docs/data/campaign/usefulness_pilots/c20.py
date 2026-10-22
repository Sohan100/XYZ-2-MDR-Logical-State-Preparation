from math import *
def p(e,n): q=e/n; return q, sqrt(q*(1-q)/n)
def R(a,b,bm=False,f=1.7):
    (e1,n1),(e2,n2)=a,b
    p1,s1=p(e1,n1);p2,s2=p(e2,n2);r=p1/p2;lr=log(r)
    sr=r*sqrt((s1/p1)**2+(s2/p2)**2)
    sa=sqrt(1/e1+1/e2); sb=sqrt(f*f/e1+f*f/e2)
    out=f"R={r:.4f}({sr:.4f}) lnR={lr:.4f} sa={sa:.4f} za={lr/sa:.2f} sb={sb:.4f} zb={lr/sb:.2f}"
    if bm:
        zs=[lr/sqrt(sa**2+d*d) for d in (0.065,0.074,0.15,0.25,0.37)]
        ds=sqrt((lr/3)**2-sa**2)
        out+=" zc="+",".join(f"{z:.2f}" for z in zs)+f" d*={ds:.3f}"
        out+=f" zb_only1={lr/sqrt(f*f/e1+1/e2):.2f}"
    else:
        zs=[lr/sqrt(sa**2+d*d) for d in (0.15,0.25,0.37)]; out+=" zc_if="+",".join(f"{z:.2f}" for z in zs)
    print(out)
Ym=(70997,4470689);Xz=(27593,3629009);Yc=(33044,2076184);Xc=(14883,1962124);Yb=(887,59048);Xb=(574,57771);Xm=(48956,4718132)
for x in [Ym,Xz,Yc,Xc,Yb,Xb,Xm,(27267,3576266),(27170,3609792),(96690,7040000)]:
    print(x, "%.5e(%.2e)"%p(*x))
print("Ym/Xz");R(Ym,Xz)
print("Yc/Xc");R(Yc,Xc)
print("Yb/Xc");R(Yb,Xc,True)
print("Xm/Xz");R(Xm,Xz)
print("Xb/Xc");R(Xb,Xc,True)
print("cssX/Xz");R((27267,3576266),Xz)
print("cssZ/Xz");R((27170,3609792),Xz)
print("Yb/Yc");R(Yb,Yc)
print("Yc vs Ym");R(Yc,Ym)
d5=(96690,7040000)
print("Ym/d5");R(Ym,d5)
print("Xm/d5");R(Xm,d5)
# combined sigma for Yc vs Ym in binomial
p1,s1=p(*Yc);p2,s2=p(*Ym);print("diff sig",(p1-p2)/hypot(s1,s2))
# summary z
print((0.10634-0.10504)/hypot(.0046,.005626),(0.10634-0.10319)/hypot(.0046,.004371))
print((0.10596-0.09782)/hypot(.002761,.004142),(0.10596-0.09846)/hypot(.002761,.002613), 0.10596/0.09782,0.10596/0.09846)
print(log(16.5/14.5),hypot(.12,.08),log(16.5/14.5)/hypot(.12,.08),16.5/14.5)
print(0.4/0.10634, 4*25,3*25,3*36, 4*25-1,2*49-1)

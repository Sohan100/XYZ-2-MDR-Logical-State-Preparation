from math import log,sqrt
def z(a,Na,b,Nb,f=1,d=0): return abs(log(a/b))/sqrt(f*f/Na+1/Nb+d*d)
def ds(a,Na,b,Nb,f=1):
    L=abs(log(a/b)); v=(L/3)**2-(1/Na+1/Nb)
    return sqrt(v) if v>0 else 0
def full(n,a,Na,b,Nb):
    print(n, 'R=%.3f'%(a/b), 'a=%.2f b=%.2f c37=%.2f c25=%.2f d*=%.3f'%(z(a,Na,b,Nb),z(a,Na,b,Nb,1.7),z(a,Na,b,Nb,1,.37),z(a,Na,b,Nb,1,.25),ds(a,Na,b,Nb)))
cg3=(1.139e-3,1194); mw3=(1.201e-3,1259); bm3=(1.1704e-3,411); mw3o=(1.1698e-3,313454)
r63X=(6.347e-5,1514); r36Z=(6.227e-5,1518); x63=(6.278e-5,1514); r63Xb=(6.388e-5,1507)
for nx,x in [('cg3',cg3),('mw3',mw3),('mw3o',mw3o)]:
  for ny,y in [('63X',r63X),('36Z',r36Z),('x63',x63)]:
    print(nx,ny,'R=%.2f z=%.1f'%(x[0]/y[0],z(*x,*y)))
for ny,y in [('63X',r63X),('36Z',r36Z),('x63',x63),('63Xb',r63Xb)]: full('bm3 vs '+ny,*bm3,*y)
bm5=(1.009e-4,38); cg5=(1.184e-4,1242); D7X=(4.725e-5,1189); D7Z=(4.399e-5,1107); r86=(1.430e-5,1327)
full('bm5 vs D7X',*bm5,*D7X); full('bm5 vs D7Z',*bm5,*D7Z); full('bm5 vs 86',*bm5,*r86)
print('cg5 vs 86 R=%.2f z=%.1f'%(cg5[0]/r86[0],z(*cg5,*r86)))
print('cg5 vs D7', cg5[0]/D7X[0], z(*cg5,*D7X), cg5[0]/D7Z[0], z(*cg5,*D7Z))
cg7=(1.125e-5,354); D9=(6.474e-6,353)
print('d7', cg7[0]/D9[0], z(*cg7,*D9), 1.184/1.009, 1.125e-5/(1.184/1.009)/6.474e-6)
D5=[(3.085e-4,1294),(3.255e-4,1365),(3.2221e-4,74201),(3.1967e-4,73480)]
print('D5', [ (x[0]/y[0], z(*x,*y)) for x in [cg3,mw3] for y in D5])
D4=[('si cssXc',1.794e-3,1881),('si cssZc',1.715e-3,1798),('si cssXm',1.784e-3,1870),('si cssZm',1.723e-3,1806),('o cssZm',1.7192e-3,556163),('o cssZc',1.7227e-3,313329),('o cssXc',1.8117e-3,391658),('o cssXm',1.8115e-3,694246)]
XZ4=[('xzX',1.887e-3,1978),('xzZ',1.784e-3,1870)]
for nx,x in [('cg3',cg3),('mw3',mw3),('mw3o',mw3o),('bm3',bm3)]:
  print(nx, ['%s %.3f %.1f'%(n,a/x[0],z(a,N,*x)) for n,a,N in D4+XZ4])
bY=(2.773e-3,1453); mY=(2.809e-3,1472); d6Y=(3.4498e-3,985268)
for nx,x in [('bothYcg',bY),('bothYmw',mY),('d6Ymw',d6Y)]:
  print(nx, ['%s %.3f %.1f'%(n,x[0]/a,z(a,N,*x)) for n,a,N in D4])
print('6x3Z', (5.145-4.899)/sqrt(.099**2+.096**2), 5.145e-3/2.773e-3)
# sd6
s=[('cg',4.137e-4,1735),('mw',4.127e-4,1731)]; c=[('cX',3.951e-4,1657),('cZ',3.853e-4,1616),('xX',3.970e-4,1665)]
for n,a,N in s: print(n,['%s %.3f %.2f'%(m,a/b,z(a,N,b,M)) for m,b,M in c], a/7.248e-6, z(a,N,7.248e-6,1501))
print('sd6 99', 2.082e-5/2.563e-6, z(2.082e-5,1048,2.563e-6,258), 2.082e-5/2.720e-6, z(2.082e-5,1048,2.720e-6,251))
# ion
for a in [8.331e-5,8.742e-5]: print('hel', a/2.980e-5, a/9.162e-6, a/9.573e-6, a/2.344e-6)
for a in [9.314e-4,9.558e-4]: print('h2', a/1.983e-4, a/9.525e-5, a/9.596e-5, a/3.606e-5)
print('mem', 8.636/5.700, 8.636/5.700*sqrt((1.1/8.636)**2+(.086/5.7)**2), 5.700/2.980, 5.7/2.98*sqrt((.086/5.7)**2+(.094/2.98)**2), z(5.700e-5,4363,2.980e-5,1000), 5.7e-5/9.573e-6, 5.7e-5/9.162e-6)
# qubit interp
import numpy as np
def qeq(pl, pts):
    # pts list of (n,p) sorted
    for (n1,p1),(n2,p2) in zip(pts,pts[1:]):
        if p2<=pl<=p1:
            t=(log(pl)-log(p1))/(log(p2)-log(p1)); return n1+t*(n2-n1)
    return None
cssX=[(31,1.794e-3),(49,3.255e-4),(71,2.019e-4),(97,4.725e-5),(127,2.443e-5)]
cssZ=[(31,1.715e-3),(49,3.085e-4),(71,1.971e-4),(97,4.399e-5),(127,2.384e-5),(161,6.474e-6)]
for nq,vals in [(35,[1.139e-3,1.201e-3]),(99,[1.184e-4]),(195,[1.125e-5])]:
  for v in vals:
    for nm,pts in [('X',cssX),('Z',cssZ)]:
      q=qeq(v,pts); print(nq,v,nm, q and nq/q)

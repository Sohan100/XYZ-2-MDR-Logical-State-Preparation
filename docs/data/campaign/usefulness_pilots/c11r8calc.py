from math import log, sqrt
def brentq(g,a,b):
    for _ in range(200):
        m=(a+b)/2
        if g(a)*g(m)<=0: b=m
        else: a=m
    return (a+b)/2
def z(p1,n1,p2,n2,f=1.0,d=0.0):
    return abs(log(p1/p2))/sqrt(f*f/n1+1/n2+d*d)
def dstar(p1,n1,p2,n2,f=1.0):
    g=lambda d: z(p1,n1,p2,n2,f,d)-3
    if g(0)<0: return None
    return brentq(g,0,10)
BM3=(1.170e-3,411); BM5=(1.009e-4,38)
r6x3=(6.347e-5,1514); r3x6=(6.227e-5,1518); x6x3=(6.278e-5,1514); r8x6=(1.430e-5,1327)
D7X=(4.725e-5,1189); D7Z=(4.399e-5,1107)
for name,a,b in [('BM3 v 6x3',BM3,r6x3),('BM3 v 3x6',BM3,r3x6),('BM3 v x6x3',BM3,x6x3),('BM5 v 8x6',BM5,r8x6),('BM5 v D7X',BM5,D7X),('BM5 v D7Z',BM5,D7Z)]:
    print(name, round(a[0]/b[0],3), [round(z(*a,*b,f,d),2) for f,d in [(1,0),(1.7,0),(1,0.25),(1,0.37),(1.7,0.37)]], 'dstar', round(dstar(*a,*b),3))
# corr_gauge, MWPM d3 vs rects
for xa in [(1.139e-3,1194),(1.201e-3,1259),(1.1698e-3,313454)]:
    for b in [r6x3,r3x6,x6x3]:
        print('d3',xa[0],b[0],round(xa[0]/b[0],2),round(z(*xa,*b),1))
# 6x3 Z both runs
print('6x3Z diff', (5.145e-3-4.899e-3)/sqrt(9.9e-5**2+9.6e-5**2))
# L_Y both vs CSS D4
for y in [(2.773e-3,1453),(2.809e-3,1472)]:
    for c in [(1.794e-3,1881),(1.715e-3,1798),(1.784e-3,1870),(1.723e-3,1806),(1.8117e-3,391658),(1.7192e-3,556163),(1.7227e-3,313329),(1.8115e-3,694246)]:
        print('LY',y[0],c[0],round(y[0]/c[0],3),round(z(*y,*c),1))
# SD6
for a in [(4.137e-4,1735),(4.127e-4,1731)]:
    for c in [(3.951e-4,1657),(3.853e-4,1616),(3.970e-4,1665),(7.248e-6,1501)]:
        print('sd6',a[0],c[0],round(a[0]/c[0],3),round(z(*a,*c),2))
print('cg5 v 8x6', 1.184e-4/1.430e-5, z(1.184e-4,1242,*r8x6))
print('memscale', 8.636e-5/5.700e-5, 5.700e-5/2.980e-5, z(5.700e-5,4363,2.980e-5,1000), 5.7e-5/9.162e-6, 5.7e-5/9.573e-6)
print('helios', 8.331/2.980, 8.742/2.980, 8.331/0.9573, 8.742/0.9162, 8.331/0.2344, 8.742/0.2344)
print('h2', 9.314/1.983, 9.558/1.983, 9.314/0.9596, 9.558/0.9525, 9.314/0.3606, 9.558/0.3606)

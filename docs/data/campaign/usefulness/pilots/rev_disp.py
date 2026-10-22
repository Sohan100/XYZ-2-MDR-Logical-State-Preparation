import math, statistics
def chi2sf(x,k):
    # survival function via regularized gamma (series), fine for small k
    from math import exp, lgamma, log
    a=k/2; xx=x/2
    # lower regularized gamma P(a,xx)
    s=0; term=1/a; n=0; tot=term
    while term>1e-15*tot and n<1000:
        n+=1; term*=xx/(a+n); tot+=term
    P=tot*exp(-xx+a*log(xx)-lgamma(a))
    return 1-P
def het(name, runs):
    E=sum(e for e,s in runs); S=sum(s for e,s in runs); p=E/S
    c=sum((e-s*p)**2/(s*p) for e,s in runs)
    print(name, 'pooled %d/%d'%(E,S), 'chi2=%.2f dof=%d p=%.4f'%(c,len(runs)-1, chi2sf(c,len(runs)-1)))
het('phen_b10 p=2% XYZ2 X bm', [(55,25400),(45,23371),(38,23016),(66,22569),(62,22000)])
het('phen_b10 p=2% XYZ2 Y bm', [(55,25400),(58,23388),(56,22969),(43,22621)])
het('m1 99 X', [(69,3187),(45,3201),(80,3401)])
het('m1 99 Y', [(52,3151),(43,3139),(61,3496)])
het('hwtr p3 X', [(40,33432),(22,33653)])
het('hwtr p3 Y', [(36,33978),(37,34130)])
het('hwtr eta10 X', [(45,34371),(46,33496)])
het('hwtr eta10 Y', [(48,34284),(51,33493)])
het('hwtr r30 X', [(46,13542),(47,13601)])
het('hwtr r30 Y', [(52,13687),(47,13611)])
het('phen0 eta10 Y', [(112,34800),(110,34800),(99,34800),(95,34800)])
het('phen0 eta10 X', [(100,34800),(88,34800),(72,34800),(92,34800)])
het('phenb10 p3 d7 X', [(30,7152),(26,7152),(37,6640),(38,6694),(28,5842),(29,6000)])
het('hwtr p6 d7 X', [(7,5397),(19,8365)])
het('hwtr p6 d7 Y', [(12,5410),(17,8059)])
het('biased10 X', [(15,3280),(14,3232),(10,3168),(7,3328),(7,3328),(17,3584)])
het('biased10 Y', [(15,3119),(11,3202),(16,3194),(14,3328),(11,3328),(13,3584)])
def eps(E,S,r): pl=E/S; return 0.5*(1-(1-2*pl)**(1/r))
y=eps(99,67777,15); x=eps(91,67867,15); z=1.208e-4
print('eta10 XYZ2 Y %.3e X %.3e ratio %.3f +- %.3f z=%.2f'%(y,x,y/z,y/z/math.sqrt(99),(1-y/z)/(y/z/math.sqrt(99))))
y=eps(73,68108,15); x=eps(62,67085,15); z=1.054e-4
print('p3 Y %.3e X %.3e ratio %.3f +- %.3f'%(y,x,y/z,y/z/math.sqrt(73)))
x=eps(62,22344,15); y=eps(44,24063,15); z=3.05e-4
print('p4 X %.3e Y %.3e ratio %.3f +- %.3f'%(x,y,x/z,x/z/math.sqrt(62)))
y=eps(218,21968,15); z=1.089e-3
print('p6 ratio %.3f +- %.3f'%(y/z,y/z/math.sqrt(218)))
y=eps(36,46743,15); x=eps(29,46148,15); z=4.502e-5
print('eta100 Y %.3e ratio %.3f +- %.3f'%(y,y/z,y/z/math.sqrt(36)))
x=eps(194,9789,15); y=eps(156,9786,15); z=1.916e-3
print('m1 X %.3e Y %.3e ratio %.3f +- %.3f'%(x,y,x/z,x/z/math.sqrt(194)))
rates=[69/3187,45/3201,80/3401]
m=statistics.mean(rates); sd=statistics.stdev(rates)/math.sqrt(3)
print('m1 X scatter rel err %.3f'%(sd/m))
x7=eps(23,2429,21); y7=eps(15,2420,21)
a,b=8.03e-4,3.46e-4; f=(195-161)/(241-161); zi=math.exp(math.log(a)+f*(math.log(b)-math.log(a)))
print('m1 195 X %.3e Y %.3e xzzx interp %.3e ratio %.3f +- %.3f'%(x7,y7,zi,x7/zi,x7/zi/math.sqrt(23)))
print('Lambda xyz2 m1 %.2f xzzx %.2f %.2f'%(x/x7, 1.916e-3/8.03e-4, 8.03e-4/3.46e-4))
y=eps(29,13469,21); x=eps(26,13762,21)
a,b=3.15e-4,9.715e-5; zi=math.exp(math.log(a)+f*(math.log(b)-math.log(a)))
print('p6 195 Y %.3e X %.3e interp %.3e ratio %.3f +- %.3f'%(y,x,zi,y/zi,y/zi/math.sqrt(29)))
print('p6 Lambda xyz2 %.2f xzzx %.2f %.2f'%(eps(218,21968,15)/y, 1.089e-3/3.15e-4, 3.15e-4/9.715e-5))
y=eps(416,139200,15); x=eps(352,139200,15)
print('phen0 Y %.3e X %.3e ratio vs 1.383e-4 %.3f +- %.3f'%(y,x,y/1.383e-4,y/1.383e-4*math.sqrt(1/416+1/759)))
y=eps(99,27298,30); x=eps(93,27143,30)
print('r6d Y %.3e X %.3e ratio %.3f'%(y,x,y/3.163e-4))
# phen_b10 sizes
print('phenb10 18/16 p3: %.3f'%(3.823e-3/5.441e-3))
xp=eps(514,40178,15); yp=eps(431,40191,15)
print('phenb10 50/49 p3: X %.3e Y %.3e ratio %.3f +- %.3f'%(xp,yp,xp/1.067e-3,xp/1.067e-3/math.sqrt(514)))
x2=eps(266,116356,15); y2=eps(212,94378,15)
print('phenb10 50/49 p2: X %.3e Y %.3e ratio %.3f +- %.3f'%(x2,y2,x2/1.875e-4,x2/1.875e-4/math.sqrt(266)))
x7=eps(188,39480,21); y7=eps(122,34255,21)
print('phenb10 98/100 p3: X %.3e Y %.3e vs d10 2.49e-4 ratio %.3f +- %.3f'%(x7,y7,x7/2.49e-4,x7/2.49e-4*math.sqrt(1/188+1/486)))
print('---')
het('phen_b10 p=2% XYZ2 X bm all7', [(55,25400),(45,23371),(38,23016),(66,22569),(62,22000),(46,17450),(29,17450)])
het('phen_b10 p=2% XYZ2 Y bm all5', [(55,25400),(58,23388),(56,22969),(43,22621),(33,16550)])
E=341;S=151706; x=eps(E,S,15); print('p2 X pooled eps %.3e ratio %.3f +- %.3f'%(x,x/1.875e-4,x/1.875e-4/math.sqrt(E)))
E=245;S=110928; y=eps(E,S,15); print('p2 Y pooled eps %.3e'%y)
x=eps(29,22900,21); y=eps(51,34100,21); print('p25 d7 X %.3e Y %.3e'%(x,y))

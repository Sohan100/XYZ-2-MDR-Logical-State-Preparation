import csv, math
L={}
for r in csv.DictReader(open('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/eq/lambda.csv')):
    if r['mode']=='chosen' and r['model'] in ('biased100','purez'):
        L[(r['model'],r['code'],float(r['p']))]=(float(r['Lambda']),float(r['Lambda_err']),r['Lambda_change'])
s2=math.sqrt(2)
for m in ('biased100','purez'):
    sl={}
    for c in ('xyz2','xzzx'):
        a,ea,_=L[(m,c,0.003)]; b,eb,_=L[(m,c,0.002)]
        s=math.log(b/a)/math.log(1.5); e=math.hypot(ea/a,eb/b)/math.log(1.5)
        sl[c]=(s,e); print(m,c,'Lambda 3e-3 %.3f(%.3f) 2e-3 %.3f(%.3f) slope %.3f +- %.3f'%(a,ea,b,eb,s,e))
    d=sl['xyz2'][0]-sl['xzzx'][0]; ed=math.hypot(sl['xyz2'][1],sl['xzzx'][1])
    print('  diff %.3f +- %.3f z %.2f'%(d,ed,d/ed))
    d2=sl['xyz2'][0]-s2*sl['xzzx'][0]; ed2=math.hypot(sl['xyz2'][1],s2*sl['xzzx'][1])
    print('  xyz2 - sqrt2*xzzx %.3f +- %.3f z %.2f; ratio slope_x/(sqrt2 slope_z) %.3f'%(d2,ed2,d2/ed2, sl['xyz2'][0]/(s2*sl['xzzx'][0])))
    # crossover from 3e-3->2e-3 slopes, starting at 2e-3
    N=math.log(L[(m,'xyz2',0.002)][0]); D=s2*math.log(L[(m,'xzzx',0.002)][0])
    print('  frac 2e-3 %.3f'%(N/D))
    if d2>0:
        x=(D-N)/d2; print('  parity from 3e-3->2e-3 slopes at p=%.2g'%(0.002*math.exp(-x)))
    # from 2e-3->1e-3 lambda.csv
    N1=math.log(L[(m,'xyz2',0.001)][0]); D1=s2*math.log(L[(m,'xzzx',0.001)][0])
    sn=(N1-N)/math.log(2); sd=(D1-D)/math.log(2)
    print('  frac 1e-3 %.3f; slopes 2e-3->1e-3 xyz2 %.3f xzzx %.3f (sqrt2*xzzx %.3f)'%(N1/D1,sn,sd/s2,sd))
    if sn>sd: x=(D1-N1)/(sn-sd); print('  parity from lambda.csv 2e-3->1e-3: p=%.2g'%(0.001*math.exp(-x)))
# d=7->9 steps
steps={('biased100','X'):(12.51,29.5,13.06,23.3,44.0),('biased100','Y'):(11.73,77.9,13.06,23.3,125),('purez','Y'):(10.46,40.9,13.88,29.8,62.9)}
for k,(a2,a1,b2,b1,up) in steps.items():
    for lab,a1v in (('central',a1),('upper68',up)):
        N=math.log(a2);N1=math.log(a1v);D=s2*math.log(b2);D1=s2*math.log(b1)
        sn=(N1-N)/math.log(2); sd=(D1-D)/math.log(2)
        f1=N1/D1
        if f1>=1: print(k,lab,'frac at 1e-3 %.3f >=1, parity between 2e-3 and 1e-3 at p=%.2g'%(f1,0.002*math.exp(-(D-N)/(sn-sd))))
        elif sn>sd: print(k,lab,'frac %.3f parity p=%.2g'%(f1,0.001*math.exp(-(D1-N1)/(sn-sd))))
        else: print(k,lab,'no parity')

import csv,sys,math
D='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
T=list(csv.DictReader(open(D+'thresholds.csv')))
idx={}
for r in T:
    idx.setdefault((r['noise'],r['decoder'],r['rounds']),[]).append(r)
def find(noise,dec,rounds,val):
    best=None
    for r in idx.get((noise,dec,rounds),[]):
        if abs(float(r['pth'])-val)<1e-9: return r
        if best is None or abs(float(r['pth'])-val)<abs(float(best['pth'])-val): best=r
    return best
def xyzkey(noise,lab):
    dec,b=[s.strip() for s in lab.split(',',1)]
    b=b.replace(' only','')
    sched,L=b.split(' L_')
    if sched=='depth6' and L=='X':
        n=noise if (noise,dec) and (noise+'@depth6-X') not in {k[0] for k in idx} else noise+'@depth6-X'
    else: n=f'{noise}@{sched}-{L}'
    return n,dec
C=list(csv.DictReader(open(D+'competitors_table.csv')))
for r in C:
    if not r['ratio']: continue
    noise,R=r['noise'],r['rounds']
    n,dec=xyzkey(noise,r['xyz2_dec'])
    x=find(n,dec,R,float(r['xyz2']))
    bo=r['best_other']; cl=r[bo+'_dec']
    cd,cb=[s.strip() for s in cl.split(',')[:2]]
    parts=[p.strip() for p in cl.split(',')]
    cn=parts[2][3:] if len(parts)>2 and parts[2].startswith('as ') else noise
    c=find(cn,f'{bo}-{cb}:{cd}',R,float(r[bo]))
    if x is None or c is None:
        print(noise,R,'missing',n,dec,bo,cl); continue
    dx,dc=float(x['pth'])-float(c['pth']),0
    et=math.hypot(float(x['err']),float(c['err'])); es=math.hypot(float(x['stat'] or 0),float(c['stat'] or 0))
    print(f"{noise:14s} {R:3s} xyz {100*float(x['pth']):.3f}({100*float(x['err']):.3f}) {x['method']} nu{float(x['nu'] or 0):.2f} | {bo} {100*float(c['pth']):.3f}({100*float(c['err']):.3f}) {c['method']} | ratio {float(r['ratio']):.3f} z_tot {dx/et:+.1f} z_stat {dx/es:+.1f}")

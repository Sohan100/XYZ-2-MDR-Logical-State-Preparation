import csv,math
D='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
R=list(csv.DictReader(open(D+'thresholds.csv')))
T=list(csv.DictReader(open(D+'competitors_table.csv')))
def find(model,rounds,pth):
    best=None
    for x in R:
        if x['rounds']!=rounds: continue
        if x['noise'].split('@')[0]!=model: continue
        if not x['pth']: continue
        if abs(float(x['pth'])-pth)<1e-12: return x
    return None
for t in T:
    if not t['ratio']: continue
    m,r=t['noise'],t['rounds']
    xy=float(t['xyz2']); bo=t['best_other']; bv=float(t[bo])
    a=find(m,r,xy); b=find(m,r,bv)
    if a is None or b is None: print(m,r,'missing'); continue
    ea,eb=float(a['err']),float(b['err']); sa,sb=float(a['stat']),float(b['stat'])
    zt=(xy-bv)/math.hypot(ea,eb); zs=(xy-bv)/math.hypot(sa,sb)
    print('%-14s %-3s %.3f %s/%s %-10s nu%.2f chi%.1f | %s %.3f %s %s | ratio %.3f zt %.1f zs %.1f'%(m,r,xy*100,a['noise'],a['decoder'],a['method'],float(a['nu'] or -1),float(a['chi2'] or -1),bo,bv*100,b['decoder'],b['method'],xy/bv,zt,zs))

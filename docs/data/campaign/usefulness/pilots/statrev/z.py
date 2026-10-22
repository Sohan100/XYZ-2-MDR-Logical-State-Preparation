import csv, math, collections
D='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
T=list(csv.DictReader(open(D+'thresholds.csv')))
C=list(csv.DictReader(open(D+'competitors_table.csv')))
def find(noise_base, rounds, val):
    best=None
    for t in T:
        if t['noise'].split('@')[0]!=noise_base or t['rounds']!=rounds: continue
        try: p=float(t['pth'])
        except: continue
        if abs(p-val)<1e-9*max(1,val) or abs(p-val)<1e-12:
            best=t; break
    return best
zs=[]
for r in C:
    if not r['ratio']: continue
    n=r['noise']; rr=r['rounds']; R='d' if rr=='d' else rr
    rounds = rr
    x=float(r['xyz2']); bo=r['best_other']; o=float(r[bo])
    tx=find(n, rr, x); to=find(n, rr, o)
    if tx is None or to is None:
        print('missing',n,rr, tx is None, to is None); continue
    ex=float(tx['err']); eo=float(to['err'])
    z=(x-o)/math.hypot(ex,eo)
    zs.append((n,rr,z))
    print(f"{n:14s} r={rr:3s} xyz2={100*x:.3f}({100*ex:.3f}) {tx['noise']}:{tx['decoder']} {tx['method']} nu={float(tx['nu']):.2f} | {bo}={100*o:.3f}({100*eo:.3f}) {to['decoder']} {to['method']} | ratio={x/o:.3f} z={z:+.1f}")

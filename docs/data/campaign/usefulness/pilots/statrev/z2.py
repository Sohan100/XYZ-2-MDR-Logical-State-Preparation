import csv, math
D='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
T=list(csv.DictReader(open(D+'thresholds.csv')))
C=list(csv.DictReader(open(D+'competitors_table.csv')))
base={'sd6_il':'sd6','sd6_lr2':'sd6','sd6_lr3':'sd6','si1000_lr2':'si1000','hyb':None,'hyb_lr2':None}
def find(nb, rounds, val, comp=False):
    for t in T:
        if t['rounds']!=rounds: continue
        nn=t['noise'].split('@')[0]
        if comp and isinstance(nb,(list,tuple)):
            if nn not in nb: continue
        elif nn!=nb: continue
        try: p=float(t['pth'])
        except: continue
        if abs(p-val)<1e-12: return t
for r in C:
    if not r['ratio'] or r['noise'] not in base: continue
    n=r['noise']; rr=r['rounds']
    x=float(r['xyz2']); bo=r['best_other']; o=float(r[bo])
    tx=find(n, rr, x)
    to=find(['sd6','si1000','em3','sd6_il','hyb'], rr, o, comp=True)
    # second best surface code
    others={k:float(r[k]) for k in ('css','xzzx','xy') if r[k]}
    sbest=max(others,key=others.get)
    ts=find(['sd6','si1000','em3'], rr, others[sbest], comp=True)
    s=lambda t: float(t['err']) if t else float('nan')
    z=(x-o)/math.hypot(s(tx),s(to)) if tx and to else float('nan')
    zs=(x-others[sbest])/math.hypot(s(tx),s(ts)) if tx and ts else float('nan')
    print(f"{n:11s} r={rr:3s} xyz2={100*x:.3f}({100*s(tx):.3f}) {tx['noise'] if tx else ''}:{tx['decoder'] if tx else ''} | {bo}={100*o:.3f}({100*s(to):.3f}) {to['noise'] if to else ''}:{to['decoder'] if to else ''} z={z:+.1f} | best surf {sbest} {100*others[sbest]:.3f}({100*s(ts):.3f}) ratio={x/others[sbest]:.3f} z={zs:+.1f}")

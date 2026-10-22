import csv,math
B='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
rows=list(csv.DictReader(open(B+'thresholds.csv')))
tab=list(csv.DictReader(open(B+'competitors_table.csv')))
def find(noise,rd,val,codeprefix=None):
    best=None
    for r in rows:
        if r['rounds']!=rd: continue
        if not (r['noise']==noise or r['noise'].startswith(noise+'@')): continue
        if codeprefix and not r['decoder'].startswith(codeprefix): continue
        if not codeprefix and ':' in r['decoder']: continue
        try: v=float(r['pth'])
        except: continue
        if abs(v-val)<1e-12: return r
    return None
out=[]
for t in tab:
    if not t['ratio']: continue
    n,rd=t['noise'],t['rounds']
    x=float(t['xyz2']); rx=find(n,rd,x)
    w=t['best_other']; o=float(t[w]); 
    ro=find(n,rd,o,w+'-') 
    if ro is None:
        # competitor may come from base noise (e.g. sd6 for lr variants)
        base={'sd6_lr2':'sd6','sd6_lr3':'sd6','sd6_il':'sd6','hyb':'sd6','hyb_lr2':'sd6','si1000_lr2':'si1000'}.get(n)
        if w=='honeycomb' and n in('hyb','hyb_lr2'): base='em3'
        if base: ro=find(base,rd,o,w+'-')
    ex=float(rx['err']) if rx and rx['err'] else float('nan')
    eo=float(ro['err']) if ro and ro['err'] else float('nan')
    z=(x-o)/math.sqrt(ex**2+eo**2) if ex==ex and eo==eo else float('nan')
    print(f"{n:14s} {rd:2s} xyz2 {x*100:.3f}({ex*100:.3f}) [{rx['noise'] if rx else '?'}:{rx['decoder'] if rx else '?'} {rx['method'] if rx else ''} nu={float(rx['nu']) if rx and rx['nu'] else 0:.2f}] vs {w} {o*100:.3f}({eo*100:.3f}) [{ro['decoder'] if ro else '?'} {ro['method'] if ro else ''}] ratio {float(t['ratio']):.3f} z {z:+.1f}")

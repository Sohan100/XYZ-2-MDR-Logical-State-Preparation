import csv, sys
rows=list(csv.DictReader(open('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')))
def f(x):
    try: return round(float(x)*100,3)
    except: return x
def show(prefix, rounds, decs=None, exact=False):
    for r in rows:
        ok = (r['noise']==prefix) if exact else r['noise'].startswith(prefix)
        if ok and r['rounds']==rounds and (decs is None or any(r['decoder'].startswith(d) or r['decoder']==d for d in decs)):
            print(r['noise'], r['decoder'], r['method'], f(r['pth']), f(r['err']), 'nu',r['nu'][:5], 'chi2',r['chi2'][:5], 'drift',r['drift'][:5], r['flag'], 'd',r['dmin'],r['dmax'])
if __name__=='__main__':
    prefix, rounds = sys.argv[1], sys.argv[2]
    decs = sys.argv[3].split(',') if len(sys.argv)>3 and sys.argv[3] else None
    exact = len(sys.argv)>4
    show(prefix, rounds, decs, exact)

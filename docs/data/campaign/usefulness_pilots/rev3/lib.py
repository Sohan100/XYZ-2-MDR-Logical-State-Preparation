import csv,math
B='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
rows=list(csv.DictReader(open(B+'thresholds.csv')))
def F(x):
    try: return float(x)
    except: return float('nan')
def sel(noise,rd,decpat=None,method='fss',code=None):
    out=[]
    for r in rows:
        if r['noise']!=noise or r['rounds']!=rd or (method and r['method']!=method): continue
        d=r['decoder']
        if code is None and ':' in d: continue
        if code is not None and not d.startswith(code+':'): continue
        if decpat and d.split(':')[-1] not in decpat: continue
        out.append((F(r['pth'])*100,F(r['err'])*100,d,F(r['nu']),F(r['chi2']),F(r['drift']),r['flag']))
    return out
def best(noise,rd,decpat=None,code=None):
    s=sel(noise,rd,decpat,code=code)
    return max(s) if s else None

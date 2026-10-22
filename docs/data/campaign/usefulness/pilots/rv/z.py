import csv,math
T=list(csv.DictReader(open('docs/data/campaign/thresholds.csv')))
C=list(csv.DictReader(open('docs/data/campaign/competitors.csv')))
allr=T+C
def find(noise,rounds,val):
    if val in ('',None): return None
    v=float(val)
    base=noise
    for x in allr:
        if x['noise'].split('@')[0]==base and x['rounds']==rounds and x['pth'] and abs(float(x['pth'])-v)<1e-12:
            return x
    return None
for r in csv.DictReader(open('docs/data/campaign/competitors_table.csv')):
    if r['rounds'] not in ('1','2','3','5','10','d'): continue
    xr=find(r['noise'],r['rounds'],r['xyz2'])
    if not xr or not r['best_other']: continue
    bo=r['best_other']; br=find(r['noise'],r['rounds'],r[bo])
    if not br: 
        # try base model for 'as sd6'
        dec=r[bo+'_dec']
        if 'as ' in dec:
            base=dec.split('as ')[1]
            br=find(base,r['rounds'],r[bo])
    if not br: print(r['noise'],r['rounds'],'no br'); continue
    a=float(xr['pth']);ea=float(xr['err']);b=float(br['pth']);eb=float(br['err'])
    z=(a-b)/math.hypot(ea,eb)
    print('%-14s %3s xyz2 %.4f(%.4f) [%s %s %s] vs %-9s %.4f(%.4f) [%s %s] ratio %.3f z %+.2f'%(r['noise'],r['rounds'],a*100,ea*100,xr['noise'],xr['decoder'],xr['method'],bo,b*100,eb*100,br['decoder'],br['method'],a/b,z))

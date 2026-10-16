import csv,math,sys
T=list(csv.DictReader(open('docs/data/campaign/thresholds.csv')))
C=list(csv.DictReader(open('docs/data/campaign/competitors.csv')))
allr=T+C
def find(noise,rounds,val):
    v=float(val)
    for x in allr:
        if x['noise'].split('@')[0]==noise and x['rounds']==rounds and x['pth'] and abs(float(x['pth'])-v)<1e-12:
            return x
other=sys.argv[1]
models=sys.argv[2].split(',')
for r in csv.DictReader(open('docs/data/campaign/competitors_table.csv')):
    if r['noise'] not in models or r['rounds'] not in ('1','2','3','5','10','d'): continue
    if not r['xyz2'] or not r[other]: print(r['noise'],r['rounds'],'missing'); continue
    xr=find(r['noise'],r['rounds'],r['xyz2']); br=find(r['noise'],r['rounds'],r[other])
    if br is None and 'as ' in r[other+'_dec']:
        br=find(r[other+'_dec'].split('as ')[1],r['rounds'],r[other])
    a=float(xr['pth']);ea=float(xr['err']);b=float(br['pth']);eb=float(br['err'])
    print('%-10s %3s xyz2 %.4f(%.4f) vs %s %.4f(%.4f) [%s %s nu %s] ratio %.3f z %+.2f'%(r['noise'],r['rounds'],a*100,ea*100,other,b*100,eb*100,br['decoder'],br['method'],br['nu'][:5],a/b,(a-b)/math.hypot(ea,eb)))

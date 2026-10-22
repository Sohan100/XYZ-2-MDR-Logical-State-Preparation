import csv,sys,re
rows=list(csv.DictReader(open('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')))
noise_re=re.compile(sys.argv[1]); rounds=sys.argv[2].split(','); dec_re=re.compile(sys.argv[3] if len(sys.argv)>3 else '.')
meth=sys.argv[4] if len(sys.argv)>4 else None
def f(x,s=100,n=3):
    try: return ('%.'+str(n)+'f')%(float(x)*s)
    except: return x
for r in rows:
    if noise_re.fullmatch(r['noise']) and r['rounds'] in rounds and dec_re.search(r['decoder']) and (meth is None or r['method']==meth):
        print(r['noise'],r['rounds'],r['decoder'],r['method'],'pth',f(r['pth']),'(',f(r['err']),')','nu',f(r['nu'],1,2),'chi2',f(r['chi2'],1,2),'dr',f(r['drift'],1,2),'cl',f(r['cross_last']),'d',r['dmin'],r['dmax'],'flag',r['flag'])

import csv,sys
rows=list(csv.DictReader(open('thresholds.csv')))
def fl(x,s=100):
    try: return round(float(x)*s,3)
    except: return ''
def show(noise_prefix, rounds, decs=None):
    for r in rows:
        n=r['noise']
        if (n==noise_prefix or n.startswith(noise_prefix+'@')) and r['rounds']==rounds and (decs is None or any(r['decoder'].startswith(d) or r['decoder']==d for d in decs)):
            print(n, r['decoder'], r['method'], fl(r['pth']), fl(r['err']), 'nu',r['nu'][:5],'chi2',r['chi2'][:5],'n',r['npts'],'d',r['dmin'],r['dmax'],'x1',fl(r['cross_first']),'xl',fl(r['cross_last']),'drift',r['drift'][:5],r['flag'])
args=sys.argv[1:]
show(args[0],args[1],args[2:] or None)

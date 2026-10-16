import csv,sys
base='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
T=list(csv.DictReader(open(base+'thresholds.csv')))
C=list(csv.DictReader(open(base+'competitors.csv')))
def fmt(x):
    try: return '%.3f'%(float(x)*100)
    except: return '-'
def show(rows):
    for x in rows:
        print(x['noise'],x['decoder'],'r=',x['rounds'],x['method'],'pth',fmt(x['pth']),'err',fmt(x['err']),'stat',fmt(x['stat']),'syst',fmt(x['syst']),'nu %.2f'%float(x['nu'] or 'nan'),'chi2',x['chi2'][:5],'n',x['npts'],'d',x['dmin'],x['dmax'],'drift',x['drift'][:5],'flag',x['flag'])
noise=sys.argv[1]; r=sys.argv[2]; which=sys.argv[3] if len(sys.argv)>3 else 'x'
if which in ('both','x'):
    show([x for x in T if (x['noise']==noise or x['noise'].startswith(noise+'@')) and x['rounds']==r])
if which in ('both','c'):
    show([x for x in C if x['noise']==noise and x['rounds']==r])

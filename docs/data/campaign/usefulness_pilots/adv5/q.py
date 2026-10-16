import csv,sys
R=list(csv.DictReader(open('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')))
def show(model,rounds,code=None,method=None,dec=None):
    for x in R:
        n=x['noise']
        base=n.split('@')[0] if '@' in n else n
        # code identification
        if base.split(':')[0]!=model and not n.startswith(model+'@') and not n.startswith(model+':'): 
            if not (n.split('@')[0]==model): continue
        if x['rounds']!=rounds: continue
        if code and code not in n: continue
        if method and x['method']!=method: continue
        if dec and x['decoder']!=dec: continue
        print(n,x['decoder'],x['method'],'%.4f(%.4f) st %.4f sy %.4f nu %.2f chi2 %.2f d%s-%s drift %s %s'%(float(x['pth'])*100 if x['pth'] else -1,float(x['err'])*100 if x['err'] else -1,float(x['stat'])*100 if x['stat'] else -1,float(x['syst'])*100 if x['syst'] else -1,float(x['nu']) if x['nu'] else -1,float(x['chi2']) if x['chi2'] else -1,x['dmin'],x['dmax'],x['drift'][:5],x['flag']))
if __name__=='__main__':
    a=sys.argv[1:]
    show(a[0],a[1],*(a[2:] if len(a)>2 else []))

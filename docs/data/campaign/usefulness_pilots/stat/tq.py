import csv,sys
rows=list(csv.DictReader(open('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')))
noise=sys.argv[1]; rnd=sys.argv[2]; pat=sys.argv[3] if len(sys.argv)>3 else ''
for r in rows:
    base=r['noise'].split('@')[0]
    if base==noise and r['rounds']==rnd and pat in (r['noise']+'|'+r['decoder']):
        f=lambda k: float(r[k]) if r[k] not in ('',None) else float('nan')
        print(f"{r['noise']:22s} {r['decoder']:18s} {r['method']:4s} pth={100*f('pth'):.3f}({100*f('err'):.3f}) st={100*f('stat'):.3f} sy={100*f('syst'):.3f} nu={f('nu'):.2f} chi2={f('chi2'):.2f} n={r['npts']} d={r['dmin']}-{r['dmax']} cf={100*f('cross_first'):.3f} cl={100*f('cross_last'):.3f}({100*f('cross_last_err'):.3f}) drift={f('drift'):.2f} {r['flag']}")

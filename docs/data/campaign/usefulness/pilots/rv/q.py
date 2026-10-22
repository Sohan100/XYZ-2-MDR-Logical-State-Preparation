import csv,sys,re
T='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv'
rows=list(csv.DictReader(open(T)))
pat_n,pat_d,pat_r=sys.argv[1],sys.argv[2],sys.argv[3]
for r in rows:
  if re.fullmatch(pat_n,r['noise']) and re.fullmatch(pat_d,r['decoder']) and re.fullmatch(pat_r,r['rounds']):
    f=lambda k:float(r[k]) if r[k] else float('nan')
    print(f"{r['noise']:22s} {r['decoder']:20s} r={r['rounds']:3s} {r['method']:5s} pth={100*f('pth'):.4f}({100*f('err'):.4f}) nu={f('nu'):.2f} chi2={f('chi2'):.2f} n={r['npts']} d={r['dmin']}-{r['dmax']} cf={100*f('cross_first'):.3f} cl={100*f('cross_last'):.3f} drift={f('drift'):.2f} flag={r['flag']}")

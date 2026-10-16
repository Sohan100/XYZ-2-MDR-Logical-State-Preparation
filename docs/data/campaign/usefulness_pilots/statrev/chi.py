import csv, statistics
D='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
T=[t for t in csv.DictReader(open(D+'thresholds.csv')) if t['method']=='fss' and t['chi2']]
xy=[float(t['chi2']) for t in T if ':' not in t['decoder'] and t['noise'].split('@')[0] not in ('helios_p','h2_p')]
co=[float(t['chi2']) for t in T if ':' in t['decoder'] and t['noise'] not in ('helios_p','h2_p')]
q=lambda v: [round(statistics.quantiles(v,n=20)[i],2) for i in (9,17,18)]
print('xyz2 fss chi2 median/90/95', q(xy), len(xy))
print('comp fss chi2 median/90/95', q(co), len(co))
# correlation stat err vs chi2 for xyz2
import math
pairs=[(float(t['chi2']), float(t['stat'])/float(t['pth'])) for t in T if ':' not in t['decoder'] and float(t['pth'])>0]
lo=[s for c,s in pairs if c<1.5]; hi=[s for c,s in pairs if c>4]
print('rel stat err median chi2<1.5', statistics.median(lo), 'chi2>4', statistics.median(hi), len(lo), len(hi))

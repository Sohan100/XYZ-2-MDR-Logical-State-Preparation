import csv, math, itertools, collections
D='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
T=[t for t in csv.DictReader(open(D+'thresholds.csv')) if t['noise']=='sd6' and ':' in t['decoder'] and t['method']=='fss']
idx={(t['decoder'],t['rounds']):t for t in T}
rounds=sorted(set(t['rounds'] for t in T))
decs=['bm','corr','mwpm','bposd','tesseract']
for mapping in [ {'css':'X','xzzx':'X','xy':'X'}, {'css':'X','xzzx':'Z','xy':'X'},{'css':'X','xzzx':'X','xy':'Z'},{'css':'X','xzzx':'Z','xy':'Z'}]:
  for other in ['X','Z']:
    pass
# compute all pairwise pulls for same decoder and round, both basis choices
res=collections.defaultdict(list)
for r in rounds:
  for dec in decs:
    for (c1,b1),(c2,b2) in itertools.combinations([(c,b) for c in ('css','xzzx','xy') for b in 'XZ'],2):
      if c1==c2: continue
      a=idx.get((f'{c1}-{b1}:{dec}',r)); b=idx.get((f'{c2}-{b2}:{dec}',r))
      if not a or not b: continue
      d=float(a['pth'])-float(b['pth'])
      s1=math.hypot(float(a['stat']),float(b['stat'])); s2=math.hypot(float(a['err']),float(b['err']))
      res[(c1,b1,c2,b2)].append((d/s1,d/s2))
for k,v in res.items():
  rs1=math.sqrt(sum(x[0]**2 for x in v)/len(v)); rs2=math.sqrt(sum(x[1]**2 for x in v)/len(v))
  print(k,len(v),f"rms stat {rs1:.2f} rms tot {rs2:.2f}")

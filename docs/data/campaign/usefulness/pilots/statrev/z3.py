import csv, math
D='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/'
T=list(csv.DictReader(open(D+'thresholds.csv')))
def get(noise, dec, r):
    for t in T:
        if t['noise']==noise and t['decoder']==dec and t['rounds']==r and t['method']=='fss': return t
cases=[('si1000@both-X','bm','2','si1000','css-Z:bm'),('si1000@both-X','bp_corr','5','si1000','css-X:bm'),('si1000@both-X','bm','10','si1000','xy-Z:bm'),
('h2_p_noxt@both-X','bp_corr','3','h2_p_noxt','xzzx-Z:bm'),('sd6_il@depth6-Y','bp_corr','2','sd6','css-X:bm'),('sd6_il@both-X','bp_full','5','sd6','xzzx-X:bm'),('sd6_il@depth6-Y','seq_soft','10','sd6','css-Z:corr'),
('phen@depth6-X','bp_corr','d','phen','css-Z:bm'),('phen_b10@both-X','bp_full','5','phen_b10','xzzx-Z:bm'),('si1000@depth6-Y','tnml','1','si1000','css-X:bm'),('si1000@both-X','mwpm','1','si1000','css-X:bm'),('si1000@both-X','mwpm','1','si1000','xzzx-Z:bm')]
for a,da,r,b,db in cases:
    x=get(a,da,r); o=get(b,db,r)
    if not x or not o: print('miss',a,da,r,b,db); continue
    d=float(x['pth'])-float(o['pth'])
    zt=d/math.hypot(float(x['err']),float(o['err'])); zs=d/math.hypot(float(x['stat']),float(o['stat']))
    print(f"{a}:{da} r={r} {100*float(x['pth']):.3f} vs {b}:{db} {100*float(o['pth']):.3f}  z_tot={zt:+.2f} z_stat={zs:+.2f} drift_x={float(x['drift']):.2f} drift_o={float(o['drift']):.2f}")

import sys
sys.path.insert(0,'/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/paper/analysis')
import competitor_report as C, pandas as pd, numpy as np
th=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')
th['rounds']=th.rounds.astype(str)
u=C.usable(th)
def xyz_dec(n,r,decs,errcol='err'):
  g=u[(~u.decoder.str.contains(':'))&(u.rounds==r)]
  g=g[g.noise.map(lambda s:C.xyz2_label(s)[0])==n]
  g=g[g.decoder.isin(decs)]
  best=None
  for s,gs in g.groupby(g.noise.map(lambda s:C.xyz2_label(s)[1])):
    bl={}
    for L in 'XY':
      gl=gs[gs.noise.map(lambda s:C.xyz2_label(s)[2])==L]
      if len(gl): bl[L]=gl.sort_values('pth').iloc[-1]
    if len(bl)==2:
      w=min(bl.values(),key=lambda r:r.pth)
      if best is None or w.pth>best.pth: best=w
  return best
def css_dec(n,r,decs):
  g=u[(u.decoder.str.startswith('css-'))&(u.rounds==r)&(u.noise==n)]
  g=g[g.decoder.map(lambda s:s.split(':')[1]).isin(decs)]
  bb={}
  for b in 'XZ':
    gb=g[g.decoder.str.startswith(f'css-{b}:')]
    if len(gb): bb[b]=gb.sort_values('pth').iloc[-1]
  return min(bb.values(),key=lambda r:r.pth) if len(bb)==2 else None
def show(tag,x,c,col='err'):
  z=(x.pth-c.pth)/np.hypot(x[col],c[col])
  print(tag,f"{x.noise}:{x.decoder} {x.pth*100:.4f}({x[col]*1e6:.0f}) vs {c.decoder} {c.pth*100:.4f}({c[col]*1e6:.0f}) ratio {x.pth/c.pth:.3f} z {z:.2f}")
alld=sorted(set(u.decoder))
xd=[d for d in alld if ':' not in d]
cd=sorted(set(d.split(':')[1] for d in alld if d.startswith('css-')))
print(xd,cd)
for n in ['biased10','biased100','purez']:
  for r in ['d','1','2','3','5','10']:
    x=xyz_dec(n,r,xd); c=css_dec(n,r,cd)
    show(f'{n} r={r} best stat',x,c,'stat')
    x=xyz_dec(n,r,['mwpm']); c=css_dec(n,r,['mwpm'])
    if x is not None and c is not None: show(f'{n} r={r} mwpm',x,c)
  for decs in [(['corr_gauge'],['corr']),(['bm'],['bm'])]:
    x=xyz_dec(n,'d',decs[0]); c=css_dec(n,'d',decs[1]); show(f'{n} d {decs}',x,c)
  x=xyz_dec(n,'d',['corr_gauge']); c=css_dec(n,'d',cd); show(f'{n} d cg vs cssbest',x,c)
print('---nonBP')
nb=[d for d in xd if d not in ('bp_corr','bp_full','bm')]
for n in ['biased10','biased100','purez']:
  x=xyz_dec(n,'d',nb); c=css_dec(n,'d',cd); show(f'{n} d nonBP vs cssbest',x,c)
  for d in nb:
    x=xyz_dec(n,'d',[d])
    if x is not None: print('   ',d,f"{x.pth*100:.4f}")

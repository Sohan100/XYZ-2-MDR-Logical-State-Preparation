import sys,math
sys.path.insert(0,'paper/analysis')
import pandas as pd
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv')
u=C.usable(th)
def best(noise,code,basis,rk,decs=None):
  g=u[(u.noise==noise)&(u.rounds.astype(str)==rk)&(u.decoder.str.startswith(f'{code}-{basis}:'))]
  if decs: g=g[g.decoder.str.split(':').str[1].isin(decs)]
  if not len(g): return None
  return g.sort_values('pth').iloc[-1]
def xbest(noise,rk,decs=None,scheds=('both','depth6')):
  # xyz2: per schedule weaker of best per basis, then best schedule
  out=[]
  for s in scheds:
    bb={}
    for L in 'XY':
      nm=f'{noise}@{s}-{L}' if not (s=='depth6' and L=='X') else noise
      g=u[(u.noise==nm)&(u.rounds.astype(str)==rk)]
      if s=='depth6' and L=='X': g=g[~g.decoder.str.contains(':')]
      if decs: g=g[g.decoder.isin(decs)]
      if len(g): bb[L]=g.sort_values('pth').iloc[-1]
    if len(bb)==2: out.append(min(bb.values(),key=lambda r:r.pth))
  return max(out,key=lambda r:r.pth) if out else None
for m in ('biased10','biased100','purez'):
  for decs,cd in ((['corr_gauge'],['corr']),(['bm'],['bm']),(['mwpm'],['mwpm']),(None,None)):
    x=xbest(m,'d',decs)
    cs=[best(m,'css',b,'d',cd) for b in 'XZ']
    c=min(cs,key=lambda r:r.pth)
    print(m,decs,'%s %.4f(%.4f)'%(x.decoder,x.pth*100,x.err*100),'%s %.4f(%.4f)'%(c.decoder,c.pth*100,c.err*100),'ratio %.3f z %.2f'%(x.pth/c.pth,(x.pth-c.pth)/math.hypot(x.err,c.err)))
print('---C5')
for m in ('biased10','biased100','purez'):
  x=xbest(m,'d')
  zs=[best(m,'xzzx',b,'d') for b in 'XZ']; zz=min(zs,key=lambda r:r.pth)
  print(m,x.pth*100,x.err*100,zz.decoder,zz.pth*100,zz.err*100,(x.pth-zz.pth)/math.hypot(x.err,zz.err))

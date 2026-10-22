import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
xz={}
for r in rows:
    if r['code']=='xzzx' and r['basis']=='X' and r['dec']=='corr' and r['rkey']=='3d':
        xz[(r['model'],r['p'],r['n_qubits'])]=r
pairs={'35':'17','99':'97','195':'161','323':'337','483':'449'}
bm={}
for r in rows:
    if r['code']=='xyz2' and r['dec']=='bm' and r['rkey']=='3d' and r['model'] in('biased100','purez'):
        bm.setdefault((r['model'],r['p'],r['n_qubits']),[]).append(r)
for k in sorted(bm):
    ent=[(r['basis'],int(r['errors']),r['eps'],r['eps_lo'],r['eps_hi'],r['upper']) for r in bm[k]]
    m,p,n=k; c=xz.get((m,p,pairs.get(n,'')))
    ce=float(c['eps']) if c else None
    out=[]
    for b,e,eps,lo,hi,up in ent:
        if eps and e>0 and ce: out.append(f"{b}:{float(eps):.3e}({e}) R={float(eps)/ce:.2f} [{float(lo)/float(c['eps_hi']):.2f},{float(hi)/float(c['eps_lo']):.2f}]")
        else: out.append(f"{b}:{eps}({e}) up={up}")
    print(m,p,n,'xzzx',(c['eps'],c['errors']) if c else None,' | '.join(out))

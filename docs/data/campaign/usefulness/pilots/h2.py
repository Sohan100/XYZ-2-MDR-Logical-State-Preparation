import re, glob, collections, math, json
from scipy import stats
recs=[]
pat=re.compile(r'(\S+)=(\S+)')
for f in glob.glob('cons/*.txt')+glob.glob('litrev/*.txt')+glob.glob('ul/*.txt')+glob.glob('bq/*.log')+glob.glob('stat/*.log'):
    for line in open(f):
        line=line.strip()
        if not line or line.startswith('partial') or 'progress' in line: continue
        kv=dict(pat.findall(line))
        toks=[t for t in line.split() if '=' not in t]
        if 'shots' not in kv: continue
        err=kv.get('errors',kv.get('err'))
        if err is None: continue
        sh=int(kv['shots']); e=int(err)
        key=tuple(toks[:6])+tuple(sorted((k,v) for k,v in kv.items() if k in ('d','p','pT','pG','pR','r','eta','q','nq','qubits')))
        recs.append((key,sh,e,f))
for line in open('ovc/resb10.jsonl'):
    j=json.loads(line); recs.append(((j['label'],j['dec'],str(j['d']),str(j['p']),str(j['r'])),j['shots'],j['errors'],'ovc/resb10'))
g=collections.defaultdict(list)
for k,s,e,f in recs: g[k].append((s,e,f))
tot=0;dof=0
for k,v in g.items():
    if len(v)<2: continue
    S=sum(s for s,_,_ in v); E=sum(e for _,e,_ in v)
    if E==0: continue
    rate=E/S
    chi=sum((e-s*rate)**2/(s*rate) for s,e,_ in v)
    p=stats.chi2.sf(chi,len(v)-1)
    tot+=chi; dof+=len(v)-1
    print(f"{k} n={len(v)} E={E} chi2={chi:.2f}/{len(v)-1} p={p:.3f} ", [ (e,s) for s,e,_ in v])

print('total',tot,dof,stats.chi2.sf(tot,dof))
def mom(filt):
    T=0;D=0;W=0;n=0
    for k,v in g.items():
        if len(v)<2 or not filt(k): continue
        S=sum(s for s,_,_ in v); E=sum(e for _,e,_ in v)
        if E==0: continue
        rate=E/S
        T+=sum((e-s*rate)**2/(s*rate) for s,e,_ in v); D+=len(v)-1; n+=1
        W+=sum(s*rate*(1-s*rate/E) for s,_,_ in v)
    print(n,T,D,W,math.sqrt(max(0,T-D)/W))
mom(lambda k: True)
mom(lambda k: ('xyz2' in k or any('xyz2' in str(x) for x in k)) and any(x=='bm' or 'bm' in str(x) for x in k))

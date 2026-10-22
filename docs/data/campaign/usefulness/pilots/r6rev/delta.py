import re, glob, collections, math, json, sys
from scipy import stats
exec(open('statrev/homog.py').read().split("for line in open('ovc/resb10.jsonl')")[0])
try:
  for line in open('ovc/resb10.jsonl'):
    j=json.loads(line); recs.append(((j['label'],j['dec'],str(j['d']),str(j['p']),str(j['r'])),j['shots'],j['errors'],'ovc/resb10'))
except FileNotFoundError: pass
g=collections.defaultdict(list)
for k,s,e,f in recs: g[k].append((s,e,f))
tot=dof=W=0
for k,v in g.items():
    if len(v)<2: continue
    S=sum(s for s,_,_ in v); E=sum(e for _,e,_ in v)
    if E==0: continue
    rate=E/S
    chi=sum((e-s*rate)**2/(s*rate) for s,e,_ in v)
    w=sum(s*rate*(1-s*rate/E) for s,_,_ in v)
    tot+=chi; dof+=len(v)-1; W+=w
print(tot,dof,W, 'delta_mom=',math.sqrt(max(0,(tot-dof)/W)))
# upper 95%: chi2 at which p=0.05 lower tail
for q in (0.84,0.95):
    # find delta s.t. expected chi2 = observed at lower quantile... simple: chi2 upper bound on noncentral: use tot + z*sqrt(2dof)
    pass
print('delta if excess= tot-dof+1.64*sqrt(2*dof):', math.sqrt((tot-dof+1.645*math.sqrt(2*dof))/W))

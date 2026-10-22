import re, ast, math
tot=0; dof=0; W=0; Wx=0; totx=0; dofx=0
for line in open('homog.out'):
    if not line.startswith('('): continue
    m=re.search(r'chi2=([\d.]+)/(\d+)', line)
    chi=float(m.group(1)); df=int(m.group(2))
    lst=ast.literal_eval(line[line.rindex('['):].strip())
    E=sum(e for e,s in lst); S=sum(s for e,s in lst)
    w=sum((s*E/S)*(1-(s/S)) for e,s in lst)
    tot+=chi; dof+=df; W+=w
    key=line[:line.index(' n=')]
    isx = ('xyz2' in key) and ("'bm'" in key or ' bm' in key)
    if isx: totx+=chi; dofx+=df; Wx+=w
print('all', tot, dof, W, 'delta', math.sqrt(max(0,(tot-dof)/W)))
print('xyz2 bm', totx, dofx, Wx, 'delta', math.sqrt(max(0,(totx-dofx)/Wx)) if totx>dofx else 0)
# rough upper: chi2 95% for dof
from math import sqrt

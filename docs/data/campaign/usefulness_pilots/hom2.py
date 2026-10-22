import runpy, io, contextlib, math, sys
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    g=runpy.run_path('statrev/homog.py')
print(buf.getvalue().splitlines()[-1])
G=g['g']
def mm(filt):
    tot=0;dof=0;den=0;n=0
    for k,v in G.items():
        if len(v)<2 or not filt(k,v): continue
        S=sum(s for s,_,_ in v); E=sum(e for _,e,_ in v)
        if E==0: continue
        rate=E/S
        tot+=sum((e-s*rate)**2/(s*rate) for s,e,_ in v); dof+=len(v)-1
        den+=sum(s*rate*(1-s*rate/E) for s,e,_ in v); n+=1
    d=math.sqrt(max(0,(tot-dof)/den))
    print(f"sets={n} chi2={tot:.2f}/{dof} den={den:.0f} delta={d:.3f}")
mm(lambda k,v: True)
mm(lambda k,v: 'xyz2' in ' '.join(map(str,k)) and 'bm' in ' '.join(map(str,k)))
# rough upper: chi2 at 95% upper
from scipy import stats

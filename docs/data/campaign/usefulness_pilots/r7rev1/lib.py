import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
comp_csv=pd.read_csv(R+'/docs/data/campaign/competitors.csv'); comp_csv['rounds']=comp_csv.rounds.astype(str)
xyz=C.xyz2_values(th)
compA=C.code_values(th[th.decoder.str.contains(":")])
compB=C.code_values(comp_csv[comp_csv.decoder.str.contains(":")])
tab=pd.read_csv(R+'/docs/data/campaign/competitors_table.csv'); tab['rounds']=tab.rounds.astype(str)
def row(noise,rk,src='th'):
    comp = compA if src=='th' else compB
    x=xyz[(xyz.noise==noise)&(xyz.rounds==rk)]
    cs=C.competitor_for(comp,noise,rk)
    return x, cs
def zrow(noise,rk,src='th',codes=('css','xzzx','xy','honeycomb')):
    x,cs=row(noise,rk,src)
    if not len(x): return None
    x=x.iloc[0]
    out={}
    for c,r in cs.items():
        if c not in codes: continue
        if pd.isna(r.get('pth')): continue
        z=(x.pth-r.pth)/np.hypot(x.err,r.err)
        out[c]=(r.pth*100,r.err*100,r.decoder,r.basis,z, x.pth/r.pth)
    return (x.pth*100,x.err*100,x.decoder,x.basis), out
def statrow(noise,rk,codes=('css','xzzx','xy')):
    # stat-only z using stat column; need to find rows
    x,cs=row(noise,rk,'th')
    if not len(x): return None
    x=x.iloc[0]
    # locate xyz2 row stat
    sched,L=x.basis.split()[0], x.basis.split()[1][-1]
    nm=noise if (sched=='depth6' and L=='X') else f'{noise}@{sched}-{L}'
    xr=th[(th.noise==nm)&(th.rounds==rk)&(th.decoder==x.decoder)&(th.method=='fss')]
    xs=xr.stat.iloc[0]
    out={}
    for c,r in cs.items():
        if c not in codes: continue
        src=r.noise
        cr=th[(th.noise==src)&(th.rounds==rk)&(th.decoder==f'{c}-{r.basis.split(",")[0]}:{r.decoder}')&(th.method=='fss')]
        cs_=cr.stat.iloc[0]
        out[c]=(x.pth-r.pth)/np.hypot(xs,cs_)
    return out
xyz_th=th[~th.decoder.str.contains(':')]
_lab=xyz_th.noise.map(C.xyz2_label)
_sched=_lab.map(lambda x:x[1])
sym_th=xyz_th[(_sched=='both')&xyz_th.decoder.isin(['mwpm','corr_gauge','bm'])]
xsym=C.xyz2_values(sym_th)
def zsym(noise,rk,codes=('css','xzzx','xy')):
    x=xsym[(xsym.noise==noise)&(xsym.rounds==rk)]
    if not len(x): return None
    x=x.iloc[0]; cs=C.competitor_for(compA,noise,rk)
    out={}
    for c,r in cs.items():
        if c not in codes: continue
        out[c]=(round(r.pth*100,3),round(float((x.pth-r.pth)/np.hypot(x.err,r.err)),2),round(x.pth/r.pth,3))
    return (round(x.pth*100,3),round(x.err*100,3),x.decoder,x.basis),out

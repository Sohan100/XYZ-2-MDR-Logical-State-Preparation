import sys, numpy as np, pandas as pd
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/paper/analysis')
import competitor_report as C
th=pd.read_csv(R+'/docs/data/campaign/thresholds.csv')
xyz=C.xyz2_values(th); comp=C.code_values(th[th.decoder.str.contains(':')])
def row(n,rk):
    x=xyz[(xyz.noise==n)&(xyz.rounds==rk)].iloc[0]
    cs=C.competitor_for(comp,n,rk)
    out={'xyz2':(x.pth,x.err,x.decoder,x.basis)}
    for k,v in cs.items():
        if pd.notna(v.get('pth',np.nan)): out[k]=(v.pth,v.err,v.decoder,v.basis)
    return out
if __name__=='__main__':
    for n in sys.argv[1].split(','):
        for rk in ['1','2','3','5','10','d']:
            try: o=row(n,rk)
            except Exception as e: continue
            x=o['xyz2']
            s=f"{n} r={rk} xyz2 {100*x[0]:.3f}({100*x[1]:.3f}) {x[2]},{x[3]} |"
            for k,v in o.items():
                if k=='xyz2': continue
                s+=f" {k} {100*v[0]:.3f}({100*v[1]:.3f}) {v[2]} {v[3]} z={(x[0]-v[0])/np.hypot(x[1],v[1]):.2f} rat={x[0]/v[0]:.3f};"
            print(s)

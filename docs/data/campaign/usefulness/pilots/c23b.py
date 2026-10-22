import sys; sys.path.insert(0,'paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
th=pd.read_csv("docs/data/campaign/thresholds.csv"); th["rounds"]=th.rounds.astype(str)
xyz=C.xyz2_values(th)
c=th[th.decoder.str.contains(":")]
t0=C.table(xyz,C.code_values(c))
pr=c[(c.method=='pair')&c.decoder.str.contains('bposd|tesseract')&(c.flag.fillna('')!='drift')&(c.dmin==9)&(c.dmax==11)].copy()
pr['method']='fss'
c2=pd.concat([c,pr])
comp2=C.code_values(c2)
t1=C.table(xyz,comp2)
m=t0.merge(t1,on=['noise','rounds'],suffixes=('','_n'))
ch=m[(m.ratio.notna())&((m.ratio-m.ratio_n).abs()>1e-9)]
print(ch[['noise','rounds','xyz2','best_other','ratio','best_other_n','ratio_n']].to_string())
comp0=C.code_values(c)
for r in ['1','5','10','d']:
    x=xyz[(xyz.noise=='h2_p')&(xyz.rounds==r)].iloc[0]
    for lab,cp in [('old',comp0),('new',comp2)]:
        cf=C.competitor_for(cp,'h2_p',r)
        print(r,lab,x.decoder,f"{x.pth*100:.4f}({x.err*100:.4f})",{k:f"{v.basis}:{v.decoder} {v.pth*100:.4f}({v.err*100:.4f}) z={(x.pth-v.pth)/np.hypot(x.err,v.err):.2f}" for k,v in cf.items()})

from lab import *
pc=cth[(cth.method=='pair')&(cth.dmin==9)&(cth.dmax==11)&(cth.flag.fillna('')!='drift')&cth.decoder.str.contains(':(?:bposd|tesseract)$')&cth.pth.notna()].copy(); pc['method']='fss'
comp2=C.code_values(pd.concat([cth,pc]))
xf=C.xyz2_values(xth)
for m in ['biased10','biased100','purez']:
    for r in ['1','2','3','5','10','d']:
        a=xf[(xf.noise==m)&(xf.rounds==r)].iloc[0]
        c1=comp[(comp.noise==m)&(comp.rounds==r)&(comp.code=='css')].iloc[0]; c2=comp2[(comp2.noise==m)&(comp2.rounds==r)&(comp2.code=='css')].iloc[0]
        print(m,r,'css %.4f -> %.4f(%.4f) %s  z %.2f'%(c1.pth*100,c2.pth*100,c2.err*100,c2.decoder,(a.pth-c2.pth)/np.hypot(a.err,c2.err)), 'raised' if c2.pth>c1.pth+1e-12 else '')

exec(open('/tmp/claude-108819/-pscratch-sd-s-sohan100/b9bb5492-172d-4149-9bf3-b4e9300d223c/scratchpad/r10v/lib.py').read())
for xd,cd,lab in [(['bm'],['bm'],'bm-bm'),(None,None,'best'),(['mwpm'],['mwpm'],'mwpm'),(['corr_gauge'],['corr'],'cg-corr')]:
    t=sub(th,xd,cd)
    X=C.xyz2_values(t); comp=C.code_values(t[t.decoder.str.contains(":")])
    for noise in ['biased10','biased100','purez']:
        for rk in ['1','2','3','5','10','d']:
            x=X[(X.noise==noise)&(X.rounds==rk)]
            c=comp[(comp.noise==noise)&(comp.rounds==rk)&(comp.code=='css')]
            if len(x)==0 or len(c)==0 or pd.isna(c.pth.iloc[0]): continue
            xp,xe=x.pth.iloc[0],x.err.iloc[0]; cp,ce=c.pth.iloc[0],c.err.iloc[0]
            print(lab,noise,rk,f"{xp*100:.4f}({xe*100:.4f}) {x.decoder.iloc[0]} {x.basis.iloc[0]} vs css {cp*100:.4f}({ce*100:.4f}) {c.decoder.iloc[0]} {c.basis.iloc[0]} R={xp/cp:.3f} z={(xp-cp)/np.hypot(xe,ce):.2f}")

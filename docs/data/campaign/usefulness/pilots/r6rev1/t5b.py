from lab import *
pd.set_option('display.width',250)
def adm(df):
    pc=df[(df.method=='pair')&(df.dmax>=11)&(df.flag.fillna('')!='drift')&df.pth.notna()].copy(); pc['method']='fss'
    return pd.concat([df,pc])
comp2=C.code_values(adm(cth)); 
xf1=C.xyz2_values(xth); xf2=C.xyz2_values(adm(xth))
for lab_,xv,cv in [('comp only',xf1,comp2),('both',xf2,comp2)]:
    print(lab_)
    for m in ['biased10','biased100','purez']:
        for r in ['1','2','3','5','10','d']:
            a=xv[(xv.noise==m)&(xv.rounds==r)].iloc[0]; z=cv[(cv.noise==m)&(cv.rounds==r)&(cv.code=='xzzx')].iloc[0]
            print(' ',m,r,'%.3f(%.0f) %s %s vs %.3f(%.0f) %s ratio %.3f z %.2f'%(a.pth*100,a.err*1e5,a.decoder,a.basis,z.pth*100,z.err*1e5,z.decoder,a.pth/z.pth,(a.pth-z.pth)/np.hypot(a.err,z.err)))

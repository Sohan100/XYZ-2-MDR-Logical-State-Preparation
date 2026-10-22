import sys, numpy as np, pandas as pd
sys.path.insert(0,'paper/analysis')
import competitor_report as C
th=pd.read_csv('docs/data/campaign/thresholds.csv'); th['rounds']=th.rounds.astype(str)
U=C.usable(th)
r1=U[U.rounds=='1']
cp=r1[r1.decoder.str.contains(':')].copy()
cp['code']=cp.decoder.str.split('-').str[0]; cp['basis']=cp.decoder.str.split('-').str[1].str.split(':').str[0]
models=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','em3','hyb','hyb_lr2','sd6_lr2','sd6_lr3','si1000_lr2','phen','phen_b10','sd6_il','helios_p','h2_p']
res={}
for m in models:
    x=r1[(r1.noise.isin([m,m+"@depth6-X"]))&(~r1.decoder.str.contains(":"))]
    if not len(x): print(m,'no depth6-X'); continue
    b=x.sort_values('pth').iloc[-1]
    srcs=[m]+C.COMPETITOR_HARDWARE.get(m,[])
    c=cp[cp.noise.isin(srcs)]
    bases={}
    for (code,B),g in c.groupby(['code','basis']):
        bases[(code,B)]=g.sort_values('pth').iloc[-1]
    # merge across srcs: best per code-basis already max over srcs
    zs={f"{k[0]}-{k[1]}":((b.pth-v.pth)/np.hypot(b.err,v.err),(b.pth-v.pth)/np.hypot(b.stat,v.stat),v.pth*100,v.noise) for k,v in bases.items()}
    nonh={k:v for k,v in zs.items() if not k.startswith('honeycomb')}
    zmin=min(v[0] for v in zs.values()); 
    lab='loss' if zmin<=-3 else ('win' if all(v[0]>=3 for v in zs.values()) else 'tie')
    print(m,b.decoder,round(b.pth*100,3),round(b.err*100,3),'nu',round(b.nu,2),lab,'zmin',round(zmin,2))
    for k,v in sorted(zs.items(),key=lambda kv:kv[1][0]):
        print('    ',k,round(v[2],3),v[3],'z',round(v[0],2),'stat',round(v[1],2))

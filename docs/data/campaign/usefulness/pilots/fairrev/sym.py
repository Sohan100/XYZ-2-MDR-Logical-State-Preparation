import sys
sys.path.insert(0,'paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
th=pd.read_csv('docs/data/campaign/thresholds.csv')
th['rounds']=th.rounds.astype(str)
comp_th=th[th.decoder.str.contains(':')]
xyz_th=th[~th.decoder.str.contains(':')]
comp=C.code_values(comp_th)
full=C.table(C.xyz2_values(xyz_th), comp)
# symmetric: both schedule, decoders mwpm/corr_gauge/bm
lab=xyz_th.noise.map(C.xyz2_label)
sched=lab.map(lambda x:x[1])
sym_th=xyz_th[(sched=='both')&xyz_th.decoder.isin(['mwpm','corr_gauge','bm'])]
sym=C.table(C.xyz2_values(sym_th), comp)
# also: both schedule, all decoders
both_th=xyz_th[(sched=='both')]
bt=C.table(C.xyz2_values(both_th), comp)
# matched decoders but any schedule
md_th=xyz_th[xyz_th.decoder.isin(['mwpm','corr_gauge','bm'])]
md=C.table(C.xyz2_values(md_th), comp)
m=full[['noise','rounds','xyz2','xyz2_dec','best_other','ratio']].merge(sym[['noise','rounds','xyz2','xyz2_dec','ratio']],on=['noise','rounds'],suffixes=('','_sym'))
m=m.merge(bt[['noise','rounds','ratio']].rename(columns={'ratio':'ratio_both'}),on=['noise','rounds']).merge(md[['noise','rounds','ratio']].rename(columns={'ratio':'ratio_md'}),on=['noise','rounds'])
m=m[m.ratio.notna()]
m['drop']=1-m.xyz2_sym/m.xyz2
print(m.to_string())
print('----- z values')
xs=C.xyz2_values(sym_th); xf=C.xyz2_values(xyz_th)
for (n,r) in [('si1000','1'),('si1000_lr2','1'),('si1000_lr2','2'),('si1000_lr2','3'),('hyb_lr2','1'),('phen','1'),('phen_b10','d'),('phen_b10','10'),('em3','d'),('phen','2'),('si1000','2')]:
    rs=C.competitor_for(comp,n,r)
    top=max(rs.values(),key=lambda q:q.pth)
    for lab,xx in (('full',xf),('sym',xs)):
        a=xx[(xx.noise==n)&(xx.rounds==r)]
        if len(a)==0: continue
        a=a.iloc[0]
        z=(a.pth-top.pth)/np.hypot(a.err,top.err)
        print(n,r,lab,round(a.pth*100,3),round(a.err*100,3),a.decoder,a.basis,'vs',top.code,round(top.pth*100,3),round(top.err*100,3),'ratio',round(a.pth/top.pth,3),'z',round(z,2))

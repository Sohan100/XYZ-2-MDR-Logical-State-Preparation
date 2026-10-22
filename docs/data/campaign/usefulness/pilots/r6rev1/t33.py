from lab import *
cc=pd.read_csv(R+'/docs/data/campaign/competitors.csv'); cc['rounds']=cc.rounds.astype(str)
comp_c=C.code_values(cc)
xf=C.xyz2_values(xth)
L=labels(xf,comp_c)
print(L.label.value_counts().to_dict())
for n,r in [('phen_b10','d'),('phen','1'),('phen_b10','10'),('phen','d'),('si1000','10'),('h2_p','5'),('sd6_il','5'),('si1000_lr2','5')]:
    a=L[(L.noise==n)&(L.rounds==r)].iloc[0]; print(n,r,round(a.ztop,2),round(a.zmin,2),a.argmin)

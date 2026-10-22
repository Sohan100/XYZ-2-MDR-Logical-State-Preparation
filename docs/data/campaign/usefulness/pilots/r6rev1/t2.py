import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',400)
Lf=pd.read_csv('Lf.csv',dtype={'rounds':str}); Ls=pd.read_csv('Ls.csv',dtype={'rounds':str})
M=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','sd6_lr2','sd6_lr3','si1000_lr2','sd6_il']
R=['1','2','3','5','10','d']
a=Lf[Lf.noise.isin(M)&Lf.rounds.isin(R)]
print(len(a), a.label.value_counts().to_dict(), a.ratio.max())
print(a[a.label=='tie'].to_string())
print(a[(a.label=='loss')].sort_values('zmin',ascending=False).head(8).to_string())
print(a.sort_values('zmin').head(3).to_string())
b=Ls[Ls.noise.isin(M)&Ls.rounds.isin(R)]
m=a.merge(b,on=['noise','rounds'],suffixes=('','_s'))
print(m[m.label!=m.label_s][['noise','rounds','label','zmin','label_s','zmin_s','argmin_s','ratio_s']].to_string())

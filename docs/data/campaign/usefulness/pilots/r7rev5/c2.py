import pandas as pd
T=pd.read_csv('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/competitors_table.csv')
m=['sd6','si1000','biased10','biased100','purez','helios_p_noxt','h2_p_noxt','sd6_lr2','sd6_lr3','si1000_lr2','sd6_il']
x=T[T.noise.isin(m)&T.rounds.astype(str).isin(['1','2','3','5','10','d'])&T.ratio.notna()]
print(len(x), x.ratio.max(), x.sort_values('ratio').tail(3)[['noise','rounds','ratio']].values.tolist())
ion=T[T.noise.isin(['helios_p','h2_p','helios_p_noxt','h2_p_noxt'])&T.ratio.notna()]
print(len(ion), ion.ratio.min(), sorted(ion.ratio.round(3))[-3:])
print(T[T.ratio>1][['noise','rounds','ratio','winner']].values.tolist())

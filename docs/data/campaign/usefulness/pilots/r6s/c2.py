import os, sys
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lab.py')).read())
L = labels(xf); Ls = labels(xs); Lst = labels(xf, statonly=True)
L['label_stat'] = Lst.label.values; L['zmin_stat'] = Lst.zmin.values
print(L.label.value_counts().to_dict(), Ls.label.value_counts().to_dict(), len(L), len(Ls))
C2 = ['sd6', 'si1000', 'biased10', 'biased100', 'purez', 'helios_p_noxt', 'h2_p_noxt', 'sd6_lr2', 'sd6_lr3', 'si1000_lr2', 'sd6_il']
R = ['1', '2', '3', '5', '10', 'd']
sub = L[L.noise.isin(C2) & L.rounds.isin(R)]
print(len(sub), sub.label.value_counts().to_dict())
print(sub[['noise', 'rounds', 'ratio', 'zmin', 'label', 'zmin_stat', 'label_stat', 'dec', 'basis', 'zs']].sort_values('zmin').to_string())
print('---- sym C2')
subs = Ls[Ls.noise.isin(C2) & Ls.rounds.isin(R)]
print(subs[['noise', 'rounds', 'ratio', 'zmin', 'label', 'dec', 'basis', 'zs']].sort_values('zmin').tail(20).to_string())
print('---- sym ties all')
print(Ls[Ls.label != 'loss'][['noise', 'rounds', 'ratio', 'zmin', 'ztop', 'label']].to_string())
print('---- table ties all')
print(L[L.label != 'loss'][['noise', 'rounds', 'ratio', 'zmin', 'ztop', 'label', 'zmin_stat']].to_string())

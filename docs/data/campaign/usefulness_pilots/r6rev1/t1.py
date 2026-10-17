from lab import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',400)
xf=C.xyz2_values(xth); xs=C.xyz2_values(sym_th)
Lf=labels(xf,comp); Ls=labels(xs,comp)
print(Lf.label.value_counts(), len(Lf)); print(Ls.label.value_counts(), len(Ls))
Lf.to_csv('Lf.csv',index=False); Ls.to_csv('Ls.csv',index=False)

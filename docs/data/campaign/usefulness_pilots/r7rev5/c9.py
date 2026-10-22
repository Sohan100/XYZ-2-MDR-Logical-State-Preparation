from th import *
t=TH[TH.rounds.astype(str)=='1']
x=t[t.noise.str.startswith('phen_b10@')|(t.noise=='phen_b10')]
x=x[~x.decoder.str.contains(':')]
print(x.pth.min(),x.pth.max(), len(x))
print(x.sort_values('pth')[['noise','decoder','method','pth','err','flag','dmax']].head(4).to_string())
print(x.sort_values('pth')[['noise','decoder','method','pth','err','flag','dmax']].tail(4).to_string())
u=C.usable(x); print('usable',len(u),u.pth.min(),u.pth.max())
z=t[(t.noise=='phen_b10')&(t.decoder.str.startswith('xzzx'))]
print(z[['decoder','method','pth','err','flag','dmax']].sort_values('pth').to_string())

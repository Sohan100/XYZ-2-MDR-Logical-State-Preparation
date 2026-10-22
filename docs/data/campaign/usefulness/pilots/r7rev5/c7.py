from th import *
u=C.usable(TH)
for m in ['biased10','biased100','purez']:
    x=u[u.noise.str.startswith(m+'@')|(u.noise==m)]
    x=x[~x.decoder.str.contains(':')]
    x=x[x.rounds.astype(str)=='d']
    # restrict to bm
    xb=x[x.decoder=='bm']
    rows=[]
    for n,g in xb.groupby('noise'):
        rows.append((n,g.pth.max(),g.err.iloc[g.pth.values.argmax()]))
    print(m,rows)
    c=u[(u.noise==m)&(u.decoder.str.startswith('css-'))&(u.rounds.astype(str)=='d')]
    print(c[['decoder','pth','err']].to_string())

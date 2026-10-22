import os, sys
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lab.py')).read())
L = labels(xf)
cc = pd.read_csv('docs/data/campaign/competitors.csv'); cc['rounds'] = cc.rounds.astype(str)
compc = C.code_values(cc)
def pool3(compv, n, r):
    rs = C.competitor_for(compv, n, r)
    vals = [(v.pth, v.err) for k, v in rs.items() if k in ('css', 'xzzx', 'xy')]
    w = np.array([1 / e**2 for _, e in vals]); p = np.array([v for v, _ in vals])
    P = (w * p).sum() / w.sum(); E = 1 / np.sqrt(w.sum())
    chi2 = (w * (p - P)**2).sum()
    return P, E, chi2, max(p)
for n, r in [('sd6_il', '5'), ('sd6_il', '2'), ('sd6_il', '10'), ('si1000', '1'), ('si1000', '2'), ('si1000', '5'), ('si1000', '10'), ('si1000', '3'), ('phen', 'd'), ('phen', '5'), ('phen', '10'), ('si1000_lr2', '5'), ('h2_p_noxt', '3'), ('h2_p', '5')]:
    x = L[(L.noise == n) & (L.rounds == r)].iloc[0]
    for nm, cv in (('th', comp), ('cc', compc)):
        P, E, chi2, mx = pool3(cv, n, r)
        print(n, r, nm, 'pool', round(P * 100, 4), round(E * 100, 4), 'chi2', round(chi2, 2), 'max/pool', round(mx / P, 4), 'z', round((x.xyz - P) / np.hypot(x.err, E), 2), 'zmin', round(x.zmin, 2))

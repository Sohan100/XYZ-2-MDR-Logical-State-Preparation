import os, sys
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lab.py')).read())
L = labels(xf)
C2 = ['sd6', 'si1000', 'biased10', 'biased100', 'purez', 'helios_p_noxt', 'h2_p_noxt', 'sd6_lr2', 'sd6_lr3', 'si1000_lr2', 'sd6_il']
# pair-crossing variant
pc = comp_th[(comp_th.method == 'pair') & (comp_th.dmin == 9) & (comp_th.dmax == 11) & (comp_th.flag.fillna('') != 'drift') & comp_th.decoder.str.contains('bposd|tesseract')].copy()
pc['method'] = 'fss'
comp2 = C.code_values(pd.concat([comp_th, pc]))
L2 = labels(xf, compv=comp2)
m = L.merge(L2, on=['noise', 'rounds'], suffixes=('', '_pc'))
ch = m[(abs(m.ratio - m.ratio_pc) > 1e-9)]
print(ch[['noise', 'rounds', 'ratio', 'ratio_pc', 'zmin', 'zmin_pc', 'label', 'label_pc', 'top_pc', 'zs_pc']].to_string())
# trend
T = pd.read_csv('docs/data/campaign/competitors_table.csv'); T['rounds'] = T.rounds.astype(str)
def ratio_err(n, r):
    row = L[(L.noise == n) & (L.rounds == r)].iloc[0]
    rs = C.competitor_for(comp, n, r); top = rs[row.top]
    rel = np.hypot(row.err / row.xyz, top.err / top.pth)
    # stat
    e1 = ST.get(xkey(type('o', (), dict(noise=n, basis=row.basis, decoder=row.dec, rounds=r))()), np.nan)
    e2 = ST.get((top.noise, f"{row.top}-{top.basis.split(',')[0]}:{top.decoder}", r), np.nan)
    rels = np.hypot(e1 / row.xyz, e2 / top.pth)
    return row.ratio, row.ratio * rel, row.ratio * rels
for n in ['sd6', 'si1000', 'biased100', 'purez', 'helios_p_noxt', 'h2_p_noxt', 'biased10']:
    v = {r: ratio_err(n, r) for r in ['1', '2', '3', 'd']}
    def z(a, b, k):
        return (v[a][0] - v[b][0]) / np.hypot(v[a][k], v[b][k])
    print(n, {r: (round(x[0], 3), round(x[1], 3)) for r, x in v.items()}, '1vd', round(z('1', 'd', 1), 2), round(z('1', 'd', 2), 2), '1v2', round(z('1', '2', 1), 2), '3v1', round(z('1', '3', 1), 2), round(z('1', '3', 2), 2), '3vd', round(z('d', '3', 1), 2), round(z('d', '3', 2), 2))
# nu medians
u = C.usable(th)
u = u.copy()
isc = u.decoder.str.contains(':')
u['base'] = [n if c else C.xyz2_label(n)[0] for n, c in zip(u.noise, isc)]
u = u[u.base.isin(C2)]
for r in ['1', '2', '3', '5', '10', 'd']:
    print(r, round(u[(~isc[u.index]) & (u.rounds == r)].nu.median(), 3), round(u[(isc[u.index]) & (u.rounds == r)].nu.median(), 3))
oth = u[(~isc[u.index]) & ~u.rounds.isin(['1', '2', '3', '5', '10', 'd'])]
print('xyz-only', oth.groupby('rounds').nu.median().round(3).to_dict())
# competitor nu at r=d for C2 models: competitors use base noise only, sd6_lr2 etc have no competitor rows

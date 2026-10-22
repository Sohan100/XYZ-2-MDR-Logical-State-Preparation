import sys
sys.path.insert(0, 'paper/analysis')
import pandas as pd, numpy as np
import competitor_report as C
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
th = pd.read_csv('docs/data/campaign/thresholds.csv')
th['rounds'] = th.rounds.astype(str)
comp_th = th[th.decoder.str.contains(':')]
xyz_th = th[~th.decoder.str.contains(':')]
comp = C.code_values(comp_th)
lab = xyz_th.noise.map(C.xyz2_label)
sched = lab.map(lambda x: x[1])
sym_th = xyz_th[(sched == 'both') & xyz_th.decoder.isin(['mwpm', 'corr_gauge', 'bm'])]
ST = {(r.noise, r.decoder, r.rounds): r.stat for r in th.itertuples()}


def xkey(r):
    b = r.basis.replace(' only', '')
    s, L = b.split(' L_')
    nm = r.noise if (s == 'depth6' and L == 'X') else f"{r.noise}@{s}-{L}"
    return (nm, r.decoder, r.rounds)


def labels(xv, statonly=False, compv=None):
    compv = comp if compv is None else compv
    out = []
    for r in xv.itertuples():
        if pd.isna(r.pth):
            continue
        rs = C.competitor_for(compv, r.noise, r.rounds)
        if not rs:
            continue
        zs = {}
        for code, v in rs.items():
            e1, e2 = r.err, v.err
            if statonly:
                e1 = ST.get(xkey(r), np.nan)
                src = v.noise
                e2 = ST.get((src, f"{code}-{v.basis.split(',')[0]}:{v.decoder}", r.rounds), np.nan)
            zs[code] = (r.pth - v.pth) / np.hypot(e1, e2)
        top = max(rs, key=lambda k: rs[k].pth)
        zmin = min(zs.values()); ztop = zs[top]
        lb = 'loss' if zmin <= -3 else ('win' if ztop >= 3 else 'tie')
        out.append(dict(noise=r.noise, rounds=r.rounds, xyz=r.pth, err=r.err, dec=r.decoder, basis=r.basis,
                        top=top, ratio=r.pth / rs[top].pth, zmin=zmin, ztop=ztop, label=lb,
                        zs={k: round(v, 2) for k, v in zs.items()}))
    return pd.DataFrame(out)


xf = C.xyz2_values(xyz_th)
xs = C.xyz2_values(sym_th)

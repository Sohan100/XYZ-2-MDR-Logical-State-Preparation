import numpy as np
# (name, R, N_xyz2, N_comp, extra_ln_var)
pts = [
    ('hwtr6', 0.6132, 218, 29621, 0),
    ('hwtr4', 0.608, 62, 11782, 0),
    ('hwtr3', 0.679, 73, 1e9, 0),
    ('r6d', 0.384, 99, 14848, 0),
    ('195', 0.538, 29, 1e9, 0.016**2),
    ('m1', 0.700, 258, 34217, 0),
    ('A', 0.752, 174, 57973, 0),
    ('f002', 1.021, 191, 32049, 0),
    ('f005', 1.271, 123, 33367, 0),
    ('E', 1.718, 112, 11601, 0),
    ('eta10', 0.807, 99, 7370, 0),
    ('pb10_3pct', 0.809, 514, 25589, 0),
    ('pb10_25', 0.783, 236, 16015, 0),
    ('pb10_1816_d4', 0.7026, 408, 503, 0),
    ('pb10_1816_interp', 0.7828, 408, 1e12, (7/9)**2/503 + (2/9)**2/490),
    ('pb10_2LY', 0.8124, 402, 20809, 0),
    ('phen1816', 1.3627, 592, 579, 0),
    ('phen5049', 1.825, 416, 468, 0),
    ('si_bm99', 2.29, 38, 1100, 0),
]
print('%-18s %6s | %6s %6s %6s | %6s %6s %6s' % ('pt', 'R', 'linP', 'logP', 'scP', 'linS', 'logS', 'scS'))
for nm, R, n1, n2, ev in pts:
    out = []
    for f in (1.0, 1.66):
        s = np.sqrt(f**2 / n1 + 1 / n2 + ev)
        lin = abs(1 - R) / (R * s)
        lg = abs(np.log(R)) / s
        # score: XYZ^2 count under H0 rate equal comp: expected n1/R, var f^2*n1/R (+comp & extra in relative terms)
        e0 = n1 / R
        sc = abs(n1 - e0) / np.sqrt(f**2 * e0 + e0**2 * (1 / n2 + ev))
        out += [lin, lg, sc]
    print('%-18s %6.3f | %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f' % ((nm, R) + tuple(out)))
# E with pooled 1.52
for f in (1.52,):
    s = np.sqrt(f**2 / 112 + 1 / 11601)
    print('E 1.52', abs(1 - 1.718) / (1.718 * s), np.log(1.718) / s)
# phen_b10 2% LX with own f=1.531
s = np.sqrt(1.531**2 / 473 + 1 / 20809); print('2%LX f1.53 lin', (1 - 0.782) / (0.782 * s), 'log', -np.log(0.782) / s)

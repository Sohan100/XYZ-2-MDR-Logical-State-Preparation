"""Crossing points of logical error curves of two distances, with fitted curves for the plots.

Each distance is fitted with a weighted least-squares line of ln p_L against p over
a window around the crossing. The crossing of two distances is where the lines
meet, and its uncertainty comes from a parametric bootstrap of the binomial
counts. The same fitted lines are drawn in the figures, so the vertical line of
a threshold always passes through the intersection that the reader sees.
"""
import numpy as np

RNG = np.random.default_rng(11)


def _fit(p, k, n, deg=1, center=None):
    """Weighted polynomial fit of ln(k/n) against p - center. Returns coefficients (highest first)."""
    y = np.log(np.clip(k / n, 1e-12, None))
    # binomial error of ln p_L is sqrt((1-q)/(n q))
    q = np.clip(k / n, 1e-9, 1 - 1e-9)
    w = 1.0 / np.sqrt(np.clip((1 - q) / (n * q), 1e-12, None))
    x = p - (center if center is not None else 0.0)
    return np.polyfit(x, y, deg, w=w)


def curve(g, d, window, deg=1, center=None):
    """Fitted ln p_L(p) of distance d from the rows of g inside window."""
    q = g[(g.d == d) & (g.value >= window[0]) & (g.value <= window[1]) & (g.errors > 0)]
    if len(q) < deg + 2:
        return None
    c = _fit(q.value.values, q.errors.values.astype(float), q.shots.values.astype(float), deg, center)
    return lambda x: np.exp(np.polyval(c, np.asarray(x) - (center or 0.0)))


def _root(ca, cb, lo, hi, center):
    diff = np.polysub(ca, cb)
    r = np.roots(diff)
    r = r[np.isreal(r)].real + center
    r = r[(r >= lo - 0.25 * (hi - lo)) & (r <= hi + 0.25 * (hi - lo))]
    if len(r) == 0:
        return np.nan
    mid = 0.5 * (lo + hi)
    return float(r[np.argmin(np.abs(r - mid))])


def crossing(g, d1, d2, window, deg=1, boot=400):
    """Crossing of the fitted curves of d1 and d2 inside window, with bootstrap error."""
    center = 0.5 * (window[0] + window[1])
    qa = g[(g.d == d1) & (g.value >= window[0]) & (g.value <= window[1]) & (g.errors > 0)]
    qb = g[(g.d == d2) & (g.value >= window[0]) & (g.value <= window[1]) & (g.errors > 0)]
    if len(qa) < deg + 2 or len(qb) < deg + 2:
        return np.nan, np.nan
    pa, ka, na = qa.value.values, qa.errors.values.astype(float), qa.shots.values.astype(float)
    pb, kb, nb = qb.value.values, qb.errors.values.astype(float), qb.shots.values.astype(float)
    x0 = _root(_fit(pa, ka, na, deg, center), _fit(pb, kb, nb, deg, center), *window, center)
    xs = []
    for _ in range(boot):
        ka2 = RNG.binomial(na.astype(int), ka / na).astype(float)
        kb2 = RNG.binomial(nb.astype(int), kb / nb).astype(float)
        if (ka2 == 0).any() or (kb2 == 0).any():
            continue
        xs.append(_root(_fit(pa, ka2, na, deg, center), _fit(pb, kb2, nb, deg, center), *window, center))
    xs = np.array([x for x in xs if np.isfinite(x)])
    err = float(np.std(xs)) if len(xs) > 20 else np.nan
    return x0, err


def window_around(g, d1, d2, guess, rel=0.15, min_points=4):
    """Window [guess(1-rel), guess(1+rel)], widened until both distances have min_points points."""
    for r in (rel, 0.2, 0.25, 0.3, 0.4):
        w = (guess * (1 - r), guess * (1 + r))
        na = ((g.d == d1) & (g.value >= w[0]) & (g.value <= w[1])).sum()
        nb = ((g.d == d2) & (g.value >= w[0]) & (g.value <= w[1])).sum()
        if na >= min_points and nb >= min_points:
            return w
    return (guess * 0.6, guess * 1.4)


def refine(g, d1, d2, guess, rel=0.15, deg=1, boot=400, iters=3):
    """Iterate window and crossing until the crossing is the centre of its own window."""
    x = guess
    for _ in range(iters):
        w = window_around(g, d1, d2, x, rel)
        x_new, err = crossing(g, d1, d2, w, deg, boot=0)
        if not np.isfinite(x_new):
            break
        if abs(x_new - x) < 1e-3 * x:
            x = x_new
            break
        x = x_new
    w = window_around(g, d1, d2, x, rel)
    x, err = crossing(g, d1, d2, w, deg, boot)
    return x, err, w

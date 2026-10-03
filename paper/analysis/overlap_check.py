"""Report text that overlaps other text, legends, lines or markers in a matplotlib figure.

Import this module before drawing; every Figure.savefig call then prints the
overlaps it finds for that figure.
"""
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

PAD = 0.6  # points of clearance required around text


def _boxes(fig, r):
    out = []
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                out.append(("text", ax, t.get_text(), t.get_window_extent(r)))
        if ax.title.get_text().strip():
            out.append(("title", ax, ax.title.get_text(), ax.title.get_window_extent(r)))
        for side in ("left", "right"):
            tt = getattr(ax, f"_{side}_title")
            if tt.get_text().strip():
                out.append(("title", ax, tt.get_text(), tt.get_window_extent(r)))
        for lg in ax.findobj(Legend):
            out.append(("legend", ax, "legend", lg.get_window_extent(r)))
        axbb = ax.get_window_extent(r)
        for axis, which in ((ax.xaxis, 0), (ax.yaxis, 1)):
            for tk in axis._update_ticks():
                for t in (tk.label1, tk.label2):
                    if not (t.get_visible() and t.get_text().strip()):
                        continue
                    bb = t.get_window_extent(r)
                    c = (bb.x0 + bb.x1) / 2 if which == 0 else (bb.y0 + bb.y1) / 2
                    lo, hi = (axbb.x0, axbb.x1) if which == 0 else (axbb.y0, axbb.y1)
                    if lo - 1 <= c <= hi + 1:
                        out.append(("tick", ax, t.get_text(), bb))
        for lab in (ax.xaxis.label, ax.yaxis.label):
            if lab.get_text().strip():
                out.append(("axlabel", ax, lab.get_text(), lab.get_window_extent(r)))
    return out


def _grow(bb, pts, dpi):
    g = pts * dpi / 72.0
    return matplotlib.transforms.Bbox([[bb.x0 - g, bb.y0 - g], [bb.x1 + g, bb.y1 + g]])


def _seg_hits(bb, P):
    """True if any segment of the polyline P (N x 2, display coords) enters bb."""
    if len(P) == 0:
        return False
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) == 0:
        return False
    inside = (P[:, 0] > bb.x0) & (P[:, 0] < bb.x1) & (P[:, 1] > bb.y0) & (P[:, 1] < bb.y1)
    if inside.any():
        return True
    for a, b in zip(P[:-1], P[1:]):
        for t in np.linspace(0, 1, 25):
            x, y = a + t * (b - a)
            if bb.x0 < x < bb.x1 and bb.y0 < y < bb.y1:
                return True
    return False


def _clip_to_axes(ax, P):
    bb = ax.get_window_extent()
    keep = (P[:, 0] >= bb.x0 - 1) & (P[:, 0] <= bb.x1 + 1) & (P[:, 1] >= bb.y0 - 1) & (P[:, 1] <= bb.y1 + 1)
    return P[keep] if keep.any() else P[:0]


def check(fig, name="figure"):
    fig.draw_without_rendering()
    r = fig._get_renderer()
    dpi = fig.dpi
    boxes = _boxes(fig, r)
    issues = []
    # text against text
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ki, ai, si, bi = boxes[i]
            kj, aj, sj, bj = boxes[j]
            if ki == "tick" and kj == "tick":
                continue
            if ki == "legend" and kj == "legend" and ai is not aj:
                continue
            if bi.overlaps(bj):
                ov = matplotlib.transforms.Bbox.intersection(bi, bj)
                if ov is not None and ov.width > 0.5 and ov.height > 0.5:
                    issues.append(f"{ki} '{si}' overlaps {kj} '{sj}'")
    # text and legends against lines and markers
    for ax in fig.axes:
        legs = list(ax.findobj(Legend))
        own = set()
        for lg in legs:
            own.update(id(h) for h in lg.findobj())
        artists = [l for l in ax.lines if l.get_visible() and id(l) not in own]
        cols = [c for c in ax.collections if isinstance(c, LineCollection) and c.get_visible()]
        for kind, a2, s, bb in boxes:
            if a2 is not ax or kind in ("tick", "axlabel"):
                continue
            g = _grow(bb, PAD, dpi)
            for ln in artists:
                xy = ln.get_xydata()
                if len(xy) == 0:
                    continue
                P = ln.get_transform().transform(xy)
                P = _clip_to_axes(ax, P)
                hit = False
                if ln.get_linestyle() not in ("None", "", " ", "none") and len(P) > 1:
                    hit = _seg_hits(g, P)
                if not hit and ln.get_marker() not in (None, "None", "", " ", "none"):
                    ms = ln.get_markersize() * dpi / 72.0 / 2
                    for x, y in P:
                        mb = matplotlib.transforms.Bbox([[x - ms, y - ms], [x + ms, y + ms]])
                        if mb.overlaps(g):
                            hit = True
                            break
                if hit:
                    lab = ln.get_label()
                    issues.append(f"{kind} '{s}' touches line '{lab}' (color {matplotlib.colors.to_hex(ln.get_color())})")
            for c in cols:
                for seg in c.get_segments():
                    P = c.get_transform().transform(np.asarray(seg))
                    P = _clip_to_axes(ax, P)
                    if len(P) > 1 and _seg_hits(g, P):
                        issues.append(f"{kind} '{s}' touches error bar")
                        break
    # remove duplicates, keep order
    seen, out = set(), []
    for s in issues:
        if s not in seen:
            seen.add(s)
            out.append(s)
    tag = "OK" if not out else f"{len(out)} overlaps"
    print(f"[overlap] {name}: {tag}")
    for s in out:
        print(f"    {s}")
    return out


_orig = Figure.savefig


def _savefig(self, fname, *a, **k):
    try:
        check(self, str(fname).rsplit("/", 1)[-1])
    except Exception as e:  # never block saving
        print(f"[overlap] check failed for {fname}: {e}")
    return _orig(self, fname, *a, **k)


Figure.savefig = _savefig


def _hits_lines(ax, g, dpi, skip=()):
    legs = list(ax.findobj(Legend))
    own = set()
    for lg in legs:
        own.update(id(h) for h in lg.findobj())
    for ln in ax.lines:
        if not ln.get_visible() or id(ln) in own or ln in skip:
            continue
        xy = ln.get_xydata()
        if len(xy) == 0:
            continue
        P = _clip_to_axes(ax, ln.get_transform().transform(xy))
        if ln.get_linestyle() not in ("None", "", " ", "none") and len(P) > 1 and _seg_hits(g, P):
            return True
        if ln.get_marker() not in (None, "None", "", " ", "none"):
            ms = ln.get_markersize() * dpi / 72.0 / 2
            for x, y in P:
                if matplotlib.transforms.Bbox([[x - ms, y - ms], [x + ms, y + ms]]).overlaps(g):
                    return True
    for c in ax.collections:
        if isinstance(c, LineCollection) and c.get_visible():
            for seg in c.get_segments():
                P = _clip_to_axes(ax, c.get_transform().transform(np.asarray(seg)))
                if len(P) > 1 and _seg_hits(g, P):
                    return True
    return False


def is_clear(ax, t, pad=PAD):
    """True if text t is inside its axes and clear of lines, markers, legends and other text."""
    fig = ax.figure
    fig.draw_without_rendering()
    r = fig._get_renderer()
    bb = t.get_window_extent(r)
    axbb = ax.get_window_extent(r)
    if bb.x0 < axbb.x0 or bb.x1 > axbb.x1 or bb.y0 < axbb.y0 or bb.y1 > axbb.y1:
        return False
    g = _grow(bb, pad, fig.dpi)
    if _hits_lines(ax, g, fig.dpi):
        return False
    for kind, a2, s, b2 in _boxes(fig, r):
        if kind == "text" and s == t.get_text() and a2 is ax:
            continue
        if b2.overlaps(g):
            return False
    return True


def place(ax, s, candidates, **kw):
    """Add text s at the first clear candidate (x, y, ha, va) in axes coordinates."""
    for x, y, ha, va in candidates:
        t = ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, **kw)
        if is_clear(ax, t):
            return t
        t.remove()
    x, y, ha, va = candidates[0]
    print(f"[overlap] no clear spot for '{s}'")
    return ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, **kw)


CORNERS = [(0.96, 0.05, "right", "bottom"), (0.04, 0.05, "left", "bottom"),
           (0.04, 0.95, "left", "top"), (0.96, 0.95, "right", "top"),
           (0.5, 0.05, "center", "bottom"), (0.5, 0.95, "center", "top")]

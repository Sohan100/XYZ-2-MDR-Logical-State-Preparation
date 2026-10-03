"""CPU time per shot of each decoder on one core.

Usage: python paper/analysis/timing.py

Decodes the same samples at d = 5, r = 5 and SD6 noise with p = 4e-3 and
writes paper/build/timing.csv. Copy it to docs/data/thresholds/timing.csv to
use it in fig_decoders. Belief-matching and sequential BP need `ldpc`, and
Tesseract needs `tesseract-decoder`.
"""
import sys
import time

import pandas as pd

from common import ROOT, out_dir

sys.path.insert(0, str(ROOT / "src"))
from mdr.ft import CircuitNoise, FTMDRCircuit  # noqa: E402
from mdr.ft.two_level_decoder import TwoLevelDecoder  # noqa: E402

CONFIGS = [
    ("mwpm", dict(mode="mwpm"), 4000),
    ("corr_links", dict(mode="corr_split", lower="links"), 4000),
    ("corr_gauge", dict(mode="corr_split", lower="gauge"), 4000),
    ("seq_match", dict(mode="seq_match"), 1000),
    ("seq_soft", dict(mode="seq_soft"), 600),
    ("bm", dict(mode="bp_full", bp_method="product_sum", bp_iters=10), 400),
    ("tesseract", dict(mode="tesseract"), 100),
]

if __name__ == "__main__":
    ft = FTMDRCircuit(5, 5, CircuitNoise.uniform(4e-3), final="frame", detectors="combined")
    dets, _ = ft.build().compile_detector_sampler(seed=11).sample(4000, separate_observables=True)
    rows = []
    for name, kw, n in CONFIGS:
        try:
            dec = TwoLevelDecoder(ft, **kw)
        except ImportError as err:
            print(f"{name:12s} skipped ({err})")
            continue
        dec.decode_batch(dets[:5])
        t = time.process_time()
        dec.decode_batch(dets[:n])
        us = (time.process_time() - t) / n * 1e6
        rows.append((name, us, n))
        print(f"{name:12s} {us:10.1f} us/shot ({n} shots)", flush=True)
    path = out_dir() / "timing.csv"
    pd.DataFrame(rows, columns=["decoder", "us_per_shot", "shots"]).to_csv(path, index=False)
    print("wrote", path)

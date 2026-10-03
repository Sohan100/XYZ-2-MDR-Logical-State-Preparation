"""Paths and data loading shared by the analysis scripts of the paper.

By default the scripts read `docs/data` of this repository and write to
`paper/build`. Set XYZ2_DATA or XYZ2_PAPER_OUT to change either location.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("XYZ2_DATA", ROOT / "docs" / "data"))
THR = DATA / "thresholds"
OUT = Path(os.environ.get("XYZ2_PAPER_OUT", ROOT / "paper" / "build"))

KEYS = ["noise", "value", "d", "rounds", "final", "decoder"]
NOISE_ORDER = ["sd6", "si1000", "biased10", "biased100", "purez", "em3"]


def read(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 80 else pd.DataFrame()


def load_sweeps(extra=()):
    """Every simulated point, one row per point.

    Shots and errors of repeated runs of the same point are summed. `extra`
    lists further CSV files written by scripts/run_decoder_threshold_sweep.py,
    which are pooled with the published data. The environment variable
    XYZ2_EXTRA_SWEEPS adds more files, separated by the path separator.
    """
    extra = list(extra) + [p for p in os.environ.get("XYZ2_EXTRA_SWEEPS", "").split(os.pathsep) if p]
    parts = [read(THR / "sweeps_pooled.csv")] + [read(p) for p in extra]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if "seconds" not in df:
        df["seconds"] = 0.0
    df = df.groupby(KEYS, as_index=False).agg(shots=("shots", "sum"), errors=("errors", "sum"),
                                              seconds=("seconds", "sum"))
    df["p_L"] = df.errors / df.shots
    df["stderr"] = np.sqrt(np.maximum(df.p_L * (1 - df.p_L), 1e-15) / df.shots)
    return df


def load_fits():
    """Threshold fits as {(noise, decoder): (threshold, uncertainty)}."""
    t = read(THR / "fits.csv")
    return {} if t.empty else {(r.noise, r.decoder): (r.pth, r.unc) for r in t.itertuples()}


def out_dir(sub=""):
    path = OUT / sub if sub else OUT
    path.mkdir(parents=True, exist_ok=True)
    return path

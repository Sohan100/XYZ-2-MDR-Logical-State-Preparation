"""Tests of the threshold campaign driver (scripts/campaign.py)."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("campaign", ROOT / "scripts" / "campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_grid_brackets_expected_threshold():
    for noise in ("sd6", "biased100", "em3", "helios_p_noxt"):
        for dec in ("mwpm", "corr_gauge", "bm", "cfe0"):
            for r in ("1", "d"):
                g = campaign.grid(noise, dec, r)
                c = campaign.center(noise, dec, r)
                assert min(g) < 0.7 * c and max(g) > 1.4 * c
                assert g == sorted(g)
    xt = campaign.grid("helios_p", "mwpm", "d")
    assert len(xt) == 14 and xt[0] < 1e-4 < xt[-1]


def test_tasks_respect_decoder_limits():
    tasks = campaign.make_tasks(["sd6"], ["tesseract", "cfe", "cfe_tn"], ["1", "2", "5", "d"], campaign.DISTANCES)
    tess = {t["d"] for t in tasks if t["decoder"] == "tesseract"}
    assert max(tess) == 9
    for r in ("1", "5", "d"):
        assert max(t["d"] for t in tasks if t["decoder"] == "cfe" and t["rounds"] == r) == 21
    tn = {r: {t["d"] for t in tasks if t["decoder"] == "cfe_tn" and t["rounds"] == r} for r in ("1", "2", "5", "d")}
    assert max(tn["1"]) == 21 and max(tn["2"]) == 9 and tn["5"] == {3} and tn["d"] == {3}
    # the fast CFE needs up to 9.5 GB at d = 21, r = 21 (docs/data/campaign/memory_probe.csv)
    assert all(t["mem"] < 16 for t in tasks if t["decoder"] == "cfe")
    # replicas share the time budget and the error target of their point
    big = [t for t in tasks if t["decoder"] == "cfe" and t["rounds"] == "d" and t["d"] == 21]
    assert max(t["budget"] for t in tasks) <= 4 * 3600 + 1
    reps = {}
    for t in big:
        reps.setdefault(t["id"].rsplit("|", 1)[0], []).append(t)
    assert any(len(v) > 1 for v in reps.values())
    assert len({t["id"] for t in tasks}) == len(tasks)
    # a second stage leaves out the tasks of the first
    first = {t["id"] for t in tasks[:100]}
    again = campaign.make_tasks(["sd6"], ["tesseract", "cfe", "cfe_tn"], ["1", "2", "5", "d"], campaign.DISTANCES,
                                exclude=first)
    assert len(again) == len(tasks) - 100


def test_tn_dmax_limits_cfe_tn():
    tasks = campaign.make_tasks(["sd6"], ["cfe", "cfe_tn"], ["1", "2", "3", "d"], campaign.DISTANCES,
                                tn_dmax={"1": 21, "2": 5})
    tn = {r: {t["d"] for t in tasks if t["decoder"] == "cfe_tn" and t["rounds"] == r} for r in ("1", "2", "3", "d")}
    assert max(tn["1"]) == 21 and max(tn["2"]) == 5 and not tn["3"] and not tn["d"]
    assert max(t["d"] for t in tasks if t["decoder"] == "cfe" and t["rounds"] == "d") == 21


def test_full_matrix_centres_and_caps():
    # every number of rounds from 1 to 21 and r = d, every decoder of the registry
    assert campaign.ROUNDS[0] == "1" and campaign.ROUNDS[-2] == "21" and campaign.ROUNDS[-1] == "d"
    from run_decoder_threshold_sweep import DECODERS as REG
    assert set(campaign.DECODERS) <= set(REG)
    # the wide grid gives the slow decoders 14 points too
    assert len(campaign.grid("sd6", "cfe", "7", wide=True)) == 14 and len(campaign.grid("sd6", "cfe", "7")) == 10
    # measured centres: kept, interpolated in log r, towards the r = d value at large r, proxies, formula
    meas = {("sd6", "mwpm", "6"): 0.007, ("sd6", "mwpm", "8"): 0.0064, ("sd6", "mwpm", "d"): 0.0043,
            ("sd6", "bm", "1"): 0.018}
    assert campaign.center_from(meas, "sd6", "mwpm", "6") == 0.007
    assert 0.0064 < campaign.center_from(meas, "sd6", "mwpm", "7") < 0.007
    assert 0.0043 < campaign.center_from(meas, "sd6", "mwpm", "12") < 0.0064
    assert campaign.center_from(meas, "sd6", "mwpm", "21") == 0.0043 == campaign.center_from(meas, "sd6", "mwpm", "d")
    assert campaign.center_from(meas, "sd6", "bp_full", "1") == 0.018          # proxy: belief-matching
    assert campaign.center_from(meas, "si1000", "mwpm", "7") == campaign.center("si1000", "mwpm", "7")
    # Tesseract to d = 21 when asked; the tensor-network decoders only where the network can be contracted
    tasks = campaign.make_tasks(["sd6"], ["tesseract", "tnml"], ["1", "21"], campaign.DISTANCES,
                                dmax={"tesseract": 21}, wide=True, centers=meas)
    assert max(t["d"] for t in tasks if t["decoder"] == "tesseract") == 21
    assert {t["rounds"] for t in tasks if t["decoder"] == "tnml"} == {"1"}


def test_memory_covers_measured_peaks():
    # every peak measured on Perlmutter, with the batch memory of the fast decoders scaled to the
    # BATCH_BITS cap of the worker, is below the estimate that the runner reserves, with a margin of 1.3
    with open(ROOT / "docs" / "data" / "campaign" / "memory_probe.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) > 100
    for r in rows:
        d, peak, setup = int(r["d"]), float(r["peak_gb"]), float(r["setup_gb"])
        bits = int(r["batch_shots"]) * int(r["detectors"])
        if r["decoder"] in campaign.FAST and bits > campaign.BATCH_BITS:
            peak = setup + (peak - setup) * campaign.BATCH_BITS / bits
        assert campaign.memory(r["decoder"], d, r["rounds"]) >= 1.3 * peak, r


def test_progress_and_finished(tmp_path):
    t = dict(id="a", target=10, max_shots=100, budget=50.0)
    f = tmp_path / "points_0.csv"
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(campaign.RAW)
        w.writerow(["sd6", 0.004, 3, 3, "frame", "mwpm", 40, 4, 0.1, 0.05, 20.0, "a", ""])
        w.writerow(["sd6", 0.004, 3, 3, "frame", "mwpm", 30, 3, 0.1, 0.05, 20.0, "a", ""])
        fh.write("sd6,0.004,3,3,frame,mwpm,1")  # a line still being written by another job
    prog = campaign.progress([str(f)])
    assert prog["a"][:3] == [70, 7, 40.0]
    assert not campaign.finished(t, prog["a"])
    prog["a"][2] = 50.0
    assert campaign.finished(t, prog["a"])
    # a worker that stopped at its budget wrote "done", even if the rounded seconds fall short
    with open(f, "a", newline="") as fh:
        fh.write("\n")
        csv.writer(fh).writerow(["sd6", 0.004, 3, 3, "frame", "mwpm", 5, 0, 0.0, 0.0, 9.9, "a", "done"])
    prog = campaign.progress([str(f)])
    assert prog["a"][2] < 50.0 and prog["a"][3] == "done" and campaign.finished(t, prog["a"])


def test_run_and_merge(tmp_path):
    tasks = [t for t in campaign.make_tasks(["sd6"], ["mwpm"], ["1"], [3]) if t["value"] > 0.02][:2]
    for t in tasks:
        t["budget"] = 3.0
        t["target"] = 20
    out = tmp_path / "points_0.csv"
    campaign.run(tasks, str(out), workers=2, mem_gb=4.0, log=lambda m: None)
    rows = list(csv.DictReader(open(out)))
    assert {r["task"] for r in rows} == {t["id"] for t in tasks}
    assert all(not r["note"].startswith("failed") for r in rows)
    # a second run finds the tasks finished
    campaign.run(tasks, str(out), workers=2, mem_gb=4.0, log=lambda m: None)
    assert len(list(csv.DictReader(open(out)))) == len(rows)
    merged = tmp_path / "points.csv"
    n = campaign.merge([str(out)], str(merged))
    assert n == 2
    m = list(csv.DictReader(open(merged)))
    assert set(m[0]) == set(campaign.COLS)
    assert sum(int(r["shots"]) for r in m) == sum(int(r["shots"]) for r in rows)


def test_task_file_roundtrip(tmp_path):
    tasks = campaign.make_tasks(["em3"], ["mwpm"], ["d"], [3, 5])
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    back = [json.loads(line) for line in p.read_text().splitlines()]
    assert back == tasks


@pytest.mark.parametrize("decoder", ["cfe", "cfe0", "cfe_tn"])
def test_cfe_variants_decode(decoder):
    import numpy as np
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    pytest.importorskip("ldpc")
    from mdr.ft import FTMDRCircuit
    from mdr.ft.two_level_decoder import TwoLevelDecoder
    from run_decoder_threshold_sweep import DECODERS, NOISE

    rounds = 1 if decoder == "cfe_tn" else 3
    ft = FTMDRCircuit(3, rounds, NOISE["sd6"](3e-3), final="frame", detectors="combined")
    dec = TwoLevelDecoder(ft, **DECODERS[decoder])
    dets, obs = dec.circuit.compile_detector_sampler(seed=1).sample(200, separate_observables=True)
    fails = int(np.sum(np.any(dec.decode_batch(dets) != obs, axis=1)))
    assert fails < 20

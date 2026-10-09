"""Shared helpers for the fairness audit (circuits, census, Monte Carlo)."""
from __future__ import annotations

import collections
import math
import time

import numpy as np
import stim

from mdr.ft.circuit_noise import CircuitNoise
from mdr.ft.competitor_circuits import competitor_circuit
from mdr.ft.competitor_decoders import competitor_decoder

NOISE_OPS = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR",
             "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "E", "ELSE_CORRELATED_ERROR"}


def stim_generated(d: int, rounds: int, p: float, basis: str) -> stim.Circuit:
    """Stim's reference rotated surface-code memory with the four SD6-like knobs at p."""
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}", distance=d, rounds=rounds,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


def ablate(c: stim.Circuit, drop_reset_idle: bool, drop_gate_idle: bool) -> stim.Circuit:
    """
    Audit-only ablation of a competitor gates circuit (the repo is not changed):
    drop the DEPOLARIZE1 of the ancilla-reset step (as if reset and measurement
    shared one MR step) and/or the DEPOLARIZE1 on qubits idling in a gate layer.
    """
    out = stim.Circuit()
    seg = []

    def flush():
        names = {i.name for i in seg}
        kind = "gate" if names & {"CX", "CY", "CZ"} else "reset" if "RX" in names and "MX" not in names else "other"
        for i in seg:
            if i.name == "DEPOLARIZE1" and ((kind == "reset" and drop_reset_idle)
                                           or (kind == "gate" and drop_gate_idle)):
                continue
            out.append(i)
        seg.clear()

    for inst in c.flattened():
        seg.append(inst)
        if inst.name == "TICK":
            flush()
    flush()
    return out


def build(kind: str, d: int, rounds: int, p: float, basis: str) -> stim.Circuit:
    """
    kind: 'stim' (stim.Circuit.generated CSS), a code of competitor_circuit under
    sd6, or 'css_mr' / 'css_mr_noidle' (audit ablations of the CSS circuit, see `ablate`).
    """
    if kind == "stim":
        return stim_generated(d, rounds, p, basis)
    if kind in ("css_mr", "css_mr_noidle"):
        c = competitor_circuit("css", d, rounds, CircuitNoise.uniform(p), basis=basis)
        return ablate(c, True, kind == "css_mr_noidle")
    return competitor_circuit(kind, d, rounds, CircuitNoise.uniform(p), basis=basis)


def data_qubits(c: stim.Circuit) -> set:
    """Qubits measured exactly once (the data qubits of a memory with >= 2 rounds)."""
    cnt = collections.Counter()
    for inst in c.flattened():
        if inst.name in ("M", "MX", "MY", "MR", "MRX", "MRY"):
            for t in inst.targets_copy():
                cnt[t.value] += 1
    return {q for q, k in cnt.items() if k == 1}


def census(c: stim.Circuit) -> dict:
    """
    Noise census: for every (channel, role) the number of sites (qubits for
    1-qubit channels, pairs for 2-qubit channels) and the summed total error
    probability (sum over sites of the channel's total probability).
    role is 'data', 'anc' or 'pair' (2-qubit channels).
    """
    data = data_qubits(c)
    out = collections.defaultdict(lambda: [0, 0.0])
    prev = None
    for inst in c.flattened():
        name = inst.name
        args = inst.gate_args_copy()
        tg = [t.value for t in inst.targets_copy() if not t.is_combiner]
        # measurement flips folded into the measurement (M(p) etc.)
        if name in ("M", "MX", "MY", "MR", "MRX", "MRY", "MPP") and args and args[0] > 0:
            for q in tg:
                key = (name + "(flip)", "data" if q in data else "anc")
                out[key][0] += 1
                out[key][1] += args[0]
        if name not in NOISE_OPS:
            prev = name
            continue
        ptot = sum(args) if name.startswith("PAULI_CHANNEL") else args[0]
        if name in ("DEPOLARIZE2", "PAULI_CHANNEL_2"):
            for i in range(0, len(tg), 2):
                key = (name, "pair")
                out[key][0] += 1
                out[key][1] += ptot
        elif name in ("E", "ELSE_CORRELATED_ERROR"):
            key = (name, "corr")
            out[key][0] += 1
            out[key][1] += ptot
        else:
            # label X_ERROR/Z_ERROR as reset- or measurement-flip by context
            label = name
            if name in ("X_ERROR", "Z_ERROR") and prev in ("R", "RX", "RY", "MR", "MRX"):
                label = name + "@reset"
            for q in tg:
                key = (label, "data" if q in data else "anc")
                out[key][0] += 1
                out[key][1] += ptot
        if name not in ("X_ERROR", "Z_ERROR"):
            prev = name
    return {k: tuple(v) for k, v in sorted(out.items())}


def ops_census(c: stim.Circuit) -> dict:
    cnt = collections.Counter()
    for inst in c.flattened():
        if inst.name in ("H", "CX", "CZ", "CY", "R", "RX", "M", "MX", "MR", "TICK"):
            n = len(inst.targets_copy())
            cnt[inst.name] += 1 if inst.name == "TICK" else (n // 2 if inst.name.startswith("C") else n)
    return dict(cnt)


def wilson(k: int, n: int, z: float = 1.0):
    """Wilson score interval (1 sigma by default)."""
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_task(task: dict) -> dict:
    """Monte-Carlo one (kind, d, rounds, p, basis, decoder) point until max_errors or max_shots."""
    c = build(task["kind"], task["d"], task["rounds"], task["p"], task["basis"])
    dec = competitor_decoder(c, task["decoder"])
    sampler = c.compile_detector_sampler(seed=task.get("seed"))
    shots = errors = 0
    t0 = time.time()
    batch = task.get("batch", 2000)
    while errors < task["max_errors"] and shots < task["max_shots"] \
            and time.time() - t0 < task.get("max_seconds", 600):
        dets, obs = sampler.sample(batch, separate_observables=True)
        pred = dec.decode_batch(dets)
        errors += int(np.any(pred != obs, axis=1).sum())
        shots += batch
    lo, hi = wilson(errors, shots)
    return {**{k: task[k] for k in ("kind", "d", "rounds", "p", "basis", "decoder")},
            "shots": shots, "errors": errors, "ler": errors / shots, "lo": lo, "hi": hi,
            "seconds": round(time.time() - t0, 1)}

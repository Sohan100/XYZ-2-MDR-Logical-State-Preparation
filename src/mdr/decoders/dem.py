"""
Stim detector-error-model parsing for MDR state-preparation decoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Iterable

import numpy as np
import stim


@dataclass(frozen=True)
class MechanismOutcome:
    """
    One mutually exclusive outcome of a detector fault mechanism.
    """

    probability: float
    detector_ids: tuple[int, ...]
    logical_ids: tuple[int, ...]


@dataclass(frozen=True)
class MechanismFactor:
    """
    A categorical detector fault mechanism.
    """

    outcomes: tuple[MechanismOutcome, ...]


@dataclass(frozen=True)
class DetectorFaultModel:
    """
    Decoder-facing detector fault model.
    """

    num_detectors: int
    num_observables: int
    mechanisms: tuple[MechanismFactor, ...]
    source: str = "stim_dem"


def _unique_sorted(values: Iterable[int]) -> tuple[int, ...]:
    """
    Return deterministic sorted unique ids.
    """
    return tuple(sorted(set(int(v) for v in values)))


def _targets_to_ids(
    targets: Iterable[stim.DemTarget],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Split Stim DEM targets into detector and logical ids.
    """
    detectors: list[int] = []
    logicals: list[int] = []
    for target in targets:
        if target.is_separator():
            continue
        if target.is_relative_detector_id():
            detectors.append(target.val)
        elif target.is_logical_observable_id():
            logicals.append(target.val)
    return _unique_sorted(detectors), _unique_sorted(logicals)


def detector_fault_model_from_stim_dem(
    dem: stim.DetectorErrorModel,
) -> DetectorFaultModel:
    """
    Convert a Stim detector error model into categorical fault factors.

    The current threshold workflow uses ordinary Stim ``error(p)`` DEM
    instructions. Separator targets are interpreted as one correlated error
    event whose detector/logical support is the union of the separated target
    groups, matching the underlying DEM event rather than treating separator
    groups as independent faults.
    """
    flat = dem.flattened()
    mechanisms: list[MechanismFactor] = []
    for inst in flat:
        if inst.type in {"detector", "logical_observable", "shift_detectors"}:
            continue
        if inst.type != "error":
            raise ValueError(f"Unsupported DEM instruction: {inst.type}")

        args = inst.args_copy()
        if len(args) != 1:
            raise ValueError(f"Expected one probability for {inst!s}")
        probability = float(args[0])
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"Invalid DEM probability: {probability}")

        group_targets = [
            target
            for group in inst.target_groups()
            for target in group
            if not target.is_separator()
        ]
        detector_ids, logical_ids = _targets_to_ids(group_targets)

        outcomes: list[MechanismOutcome] = []
        if probability < 1.0:
            outcomes.append(
                MechanismOutcome(
                    probability=1.0 - probability,
                    detector_ids=(),
                    logical_ids=(),
                )
            )
        if probability > 0.0:
            outcomes.append(
                MechanismOutcome(
                    probability=probability,
                    detector_ids=detector_ids,
                    logical_ids=logical_ids,
                )
            )
        if outcomes:
            mechanisms.append(MechanismFactor(outcomes=tuple(outcomes)))

    return DetectorFaultModel(
        num_detectors=flat.num_detectors,
        num_observables=flat.num_observables,
        mechanisms=tuple(mechanisms),
    )


def detector_fault_model_from_circuit(
    circuit: stim.Circuit,
) -> DetectorFaultModel:
    """
    Build a detector fault model from a detector-annotated Stim circuit.
    """
    dem = circuit.detector_error_model(
        decompose_errors=False,
        approximate_disjoint_errors=True,
    )
    return detector_fault_model_from_stim_dem(dem)


def brute_force_log_z(
    model: DetectorFaultModel,
    observed_detectors: np.ndarray,
) -> np.ndarray:
    """
    Exact logical-class posterior for tiny detector fault models.
    """
    observed = np.asarray(observed_detectors, dtype=np.uint8)
    if observed.shape != (model.num_detectors,):
        raise ValueError(
            "observed_detectors shape must be "
            f"({model.num_detectors},), got {observed.shape}"
        )
    logical_classes = 1 << max(model.num_observables, 1)
    log_z = np.full(logical_classes, -np.inf, dtype=float)

    choices = [range(len(mechanism.outcomes)) for mechanism in model.mechanisms]
    for assignment in product(*choices):
        detectors = np.zeros(model.num_detectors, dtype=np.uint8)
        logical_mask = 0
        logp = 0.0
        for mechanism, outcome_index in zip(model.mechanisms, assignment):
            outcome = mechanism.outcomes[outcome_index]
            if outcome.probability <= 0.0:
                logp = -np.inf
                break
            logp += math.log(outcome.probability)
            for detector_id in outcome.detector_ids:
                detectors[detector_id] ^= 1
            for logical_id in outcome.logical_ids:
                logical_mask ^= 1 << logical_id
        if np.isneginf(logp):
            continue
        if np.array_equal(detectors, observed):
            log_z[logical_mask] = np.logaddexp(log_z[logical_mask], logp)
    return log_z

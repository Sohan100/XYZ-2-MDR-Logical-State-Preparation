"""
Regression tests for the MDR MPS/MLD decoder.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from mdr.decoders.dem import (
    DetectorFaultModel,
    MechanismFactor,
    MechanismOutcome,
    brute_force_log_z,
    detector_fault_model_from_circuit,
)
from mdr.decoders.mps_mld import MpsMldDecoder


def _binary_factor(
    probability: float,
    detectors: tuple[int, ...] = (),
    logicals: tuple[int, ...] = (),
) -> MechanismFactor:
    """
    Build a simple binary detector fault factor.
    """
    return MechanismFactor(
        outcomes=(
            MechanismOutcome(probability=1.0 - probability, detector_ids=(), logical_ids=()),
            MechanismOutcome(
                probability=probability,
                detector_ids=detectors,
                logical_ids=logicals,
            ),
        )
    )


def test_zero_detector_record_prefers_no_logical_flip() -> None:
    """
    A clean temporal detector record should prefer the no-logical-flip class.
    """
    model = DetectorFaultModel(
        num_detectors=1,
        num_observables=1,
        mechanisms=(
            _binary_factor(0.1, detectors=(0,), logicals=(0,)),
            _binary_factor(0.01, logicals=(0,)),
        ),
    )
    decoder = MpsMldDecoder(model=model, num_checks=1, rounds=2)
    syndrome_rounds = np.zeros((1, 2, 1), dtype=np.uint8)

    result = decoder.decode_batch(
        syndrome_rounds=syndrome_rounds,
        observable_labels=["Logical X"],
    )

    assert result.decoder_known.tolist() == [True]
    assert result.observable_flips["Logical X"].tolist() == [0]
    assert result.log_likelihood_gap[0] > 0


def test_exact_matches_bruteforce_small_dem() -> None:
    """
    Untruncated frontier contraction should match brute force on small DEMs.
    """
    model = DetectorFaultModel(
        num_detectors=2,
        num_observables=1,
        mechanisms=(
            _binary_factor(0.11, detectors=(0,)),
            _binary_factor(0.07, detectors=(1,), logicals=(0,)),
            _binary_factor(0.05, detectors=(0, 1)),
            _binary_factor(0.03, logicals=(0,)),
        ),
    )
    decoder = MpsMldDecoder(model=model, num_checks=2, rounds=2)
    observed = np.array([1, 0], dtype=np.uint8)

    contracted = decoder._contract_one(observed).log_z
    brute = brute_force_log_z(model, observed)

    assert contracted[:2] == pytest.approx(brute[:2])


def test_truncation_is_deterministic() -> None:
    """
    Fixed chi truncation should produce repeatable flips and diagnostics.
    """
    model = DetectorFaultModel(
        num_detectors=2,
        num_observables=1,
        mechanisms=(
            _binary_factor(0.2, detectors=(0,)),
            _binary_factor(0.2, detectors=(1,)),
            _binary_factor(0.2, detectors=(0, 1), logicals=(0,)),
            _binary_factor(0.2, logicals=(0,)),
        ),
    )
    decoder = MpsMldDecoder(
        model=model,
        num_checks=2,
        rounds=2,
        max_bond_dimension=2,
    )
    syndrome_rounds = np.array([[[0, 0], [1, 0]]], dtype=np.uint8)

    first = decoder.decode_batch(
        syndrome_rounds=syndrome_rounds,
        observable_labels=["Logical X"],
    )
    second = decoder.decode_batch(
        syndrome_rounds=syndrome_rounds,
        observable_labels=["Logical X"],
    )

    assert first.observable_flips["Logical X"].tolist() == second.observable_flips[
        "Logical X"
    ].tolist()
    assert first.log_likelihood_gap.tolist() == second.log_likelihood_gap.tolist()
    assert first.truncation_mass_lost.tolist() == second.truncation_mass_lost.tolist()


def test_detector_model_accepts_pauli_channel_2() -> None:
    """
    Circuit-level two-qubit Pauli channels should be DEM-extractable.
    """
    circuit = stim.Circuit(
        """
        R 0 1
        PAULI_CHANNEL_2(0.01,0,0,0.01,0,0,0,0,0,0,0,0,0,0,0) 0 1
        M 0 1
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )

    model = detector_fault_model_from_circuit(circuit)

    assert model.num_detectors == 1
    assert model.num_observables == 1
    assert model.mechanisms

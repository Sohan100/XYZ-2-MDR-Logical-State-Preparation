"""
Frontier-MPS-style maximum-likelihood decoder for detector fault models.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import numpy as np

from .base import DecodeResult
from .dem import DetectorFaultModel, MechanismFactor


def _logsumexp(values: list[float]) -> float:
    """
    Stable log-sum-exp for a short list.
    """
    finite = [value for value in values if not math.isinf(value)]
    if not finite:
        return -math.inf
    max_value = max(finite)
    return max_value + math.log(sum(math.exp(v - max_value) for v in finite))


@dataclass(frozen=True)
class _ContractionResult:
    log_z: np.ndarray
    truncation_mass_lost: float


class MpsMldDecoder:
    """
    Approximate degenerate MLD decoder using a bounded frontier contraction.

    With ``max_bond_dimension=None`` the contraction is exact for the supplied
    detector fault model. With a finite bond dimension, low-probability
    frontier states are deterministically truncated after each mechanism. This
    gives the threshold workflow a tunable chi-like convergence knob without
    introducing the global raw-syndrome lookup table that fails at larger
    distances.
    """

    def __init__(
        self,
        *,
        model: DetectorFaultModel,
        num_checks: int,
        rounds: int,
        max_bond_dimension: int | None = None,
    ) -> None:
        if num_checks < 0:
            raise ValueError("num_checks must be nonnegative.")
        if rounds < 0:
            raise ValueError("rounds must be nonnegative.")
        if max_bond_dimension is not None and max_bond_dimension < 1:
            raise ValueError("max_bond_dimension must be positive or None.")
        self.model = model
        self.num_checks = int(num_checks)
        self.rounds = int(rounds)
        self.max_bond_dimension = max_bond_dimension
        self._last_touch = self._compute_last_touch()
        self._factor_detector_ids = [
            self._factor_detector_support(factor)
            for factor in self.model.mechanisms
        ]
        self._cache: dict[bytes, _ContractionResult] = {}

    def _compute_last_touch(self) -> dict[int, int]:
        """
        Return the last mechanism index touching each detector.
        """
        last_touch: dict[int, int] = {}
        for index, mechanism in enumerate(self.model.mechanisms):
            for outcome in mechanism.outcomes:
                for detector_id in outcome.detector_ids:
                    last_touch[detector_id] = index
        return last_touch

    @staticmethod
    def _factor_detector_support(
        mechanism: MechanismFactor,
    ) -> tuple[int, ...]:
        """
        Return every detector touched by a mechanism.
        """
        ids = {
            detector_id
            for outcome in mechanism.outcomes
            for detector_id in outcome.detector_ids
        }
        return tuple(sorted(ids))

    def _observed_temporal_detectors(
        self,
        syndrome_rounds: np.ndarray,
    ) -> np.ndarray:
        """
        Convert raw syndrome rounds into temporal detector bits.
        """
        syndromes = np.asarray(syndrome_rounds, dtype=np.uint8)
        expected_shape = (syndromes.shape[0], self.rounds, self.num_checks)
        if syndromes.ndim != 3 or syndromes.shape != expected_shape:
            raise ValueError(
                "syndrome_rounds must have shape "
                f"(shots, {self.rounds}, {self.num_checks}); "
                f"got {syndromes.shape}"
            )
        if self.rounds <= 1:
            detectors = np.zeros((syndromes.shape[0], 0), dtype=np.uint8)
        else:
            detectors = np.bitwise_xor(
                syndromes[:, 1:, :],
                syndromes[:, :-1, :],
            ).reshape(syndromes.shape[0], -1)
        if detectors.shape[1] != self.model.num_detectors:
            raise ValueError(
                "Temporal detector count does not match DEM: "
                f"{detectors.shape[1]} != {self.model.num_detectors}"
            )
        return detectors

    @staticmethod
    def _extend_bits(
        old_ids: list[int],
        new_ids: list[int],
        bits: tuple[int, ...],
    ) -> list[int]:
        """
        Move an active-detector bit tuple onto a larger active id list.
        """
        bit_by_id = dict(zip(old_ids, bits))
        return [int(bit_by_id.get(detector_id, 0)) for detector_id in new_ids]

    def _truncate_states(
        self,
        states: Dict[tuple[tuple[int, ...], int], float],
    ) -> tuple[Dict[tuple[tuple[int, ...], int], float], float]:
        """
        Apply deterministic chi truncation and return dropped mass fraction.
        """
        if (
            self.max_bond_dimension is None
            or len(states) <= self.max_bond_dimension
        ):
            return states, 0.0

        items = sorted(
            states.items(),
            key=lambda item: (-item[1], item[0]),
        )
        kept = dict(items[: self.max_bond_dimension])
        dropped_logs = [value for _, value in items[self.max_bond_dimension :]]
        all_logs = [value for _, value in items]
        dropped = _logsumexp(dropped_logs)
        total = _logsumexp(all_logs)
        if math.isinf(dropped) or math.isinf(total):
            return kept, 0.0
        return kept, float(math.exp(dropped - total))

    def _contract_one(
        self,
        observed_detectors: np.ndarray,
    ) -> _ContractionResult:
        """
        Contract the detector fault model for one observed detector record.
        """
        observed = np.asarray(observed_detectors, dtype=np.uint8)
        if observed.shape != (self.model.num_detectors,):
            raise ValueError(
                "observed_detectors shape must be "
                f"({self.model.num_detectors},), got {observed.shape}"
            )

        cache_key = observed.tobytes()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        for detector_id, bit in enumerate(observed):
            if bit and detector_id not in self._last_touch:
                logical_classes = 1 << max(self.model.num_observables, 1)
                result = _ContractionResult(
                    log_z=np.full(logical_classes, -np.inf, dtype=float),
                    truncation_mass_lost=0.0,
                )
                self._cache[cache_key] = result
                return result

        active_ids: list[int] = []
        states: Dict[tuple[tuple[int, ...], int], float] = {((), 0): 0.0}
        truncation_mass_lost = 0.0

        for index, mechanism in enumerate(self.model.mechanisms):
            expanded_ids = sorted(
                set(active_ids).union(self._factor_detector_ids[index])
            )
            expanded_index = {
                detector_id: pos
                for pos, detector_id in enumerate(expanded_ids)
            }
            next_ids = [
                detector_id
                for detector_id in expanded_ids
                if self._last_touch.get(detector_id, -1) > index
            ]
            next_index = {
                detector_id: pos for pos, detector_id in enumerate(next_ids)
            }
            next_states: Dict[tuple[tuple[int, ...], int], float] = {}

            for (bits, logical_mask), log_prob in states.items():
                expanded_bits_base = self._extend_bits(
                    active_ids,
                    expanded_ids,
                    bits,
                )
                for outcome in mechanism.outcomes:
                    if outcome.probability <= 0.0:
                        continue
                    expanded_bits = list(expanded_bits_base)
                    for detector_id in outcome.detector_ids:
                        expanded_bits[expanded_index[detector_id]] ^= 1

                    compatible = True
                    for detector_id in expanded_ids:
                        if self._last_touch.get(detector_id, -1) == index:
                            observed_bit = int(observed[detector_id])
                            if expanded_bits[expanded_index[detector_id]] != (
                                observed_bit
                            ):
                                compatible = False
                                break
                    if not compatible:
                        continue

                    next_bits = [0] * len(next_ids)
                    for detector_id in next_ids:
                        next_bits[next_index[detector_id]] = expanded_bits[
                            expanded_index[detector_id]
                        ]
                    next_logical_mask = int(logical_mask)
                    for logical_id in outcome.logical_ids:
                        next_logical_mask ^= 1 << logical_id
                    key = (tuple(next_bits), next_logical_mask)
                    value = log_prob + math.log(outcome.probability)
                    previous = next_states.get(key, -math.inf)
                    next_states[key] = float(np.logaddexp(previous, value))

            states, dropped_fraction = self._truncate_states(next_states)
            truncation_mass_lost += dropped_fraction
            active_ids = next_ids
            if not states:
                break

        logical_classes = 1 << max(self.model.num_observables, 1)
        log_z = np.full(logical_classes, -np.inf, dtype=float)
        for (bits, logical_mask), log_prob in states.items():
            if bits:
                continue
            log_z[logical_mask] = np.logaddexp(log_z[logical_mask], log_prob)

        result = _ContractionResult(
            log_z=log_z,
            truncation_mass_lost=truncation_mass_lost,
        )
        self._cache[cache_key] = result
        return result

    def decode_batch(
        self,
        *,
        syndrome_rounds: np.ndarray,
        observable_labels: list[str],
    ) -> DecodeResult:
        """
        Decode a batch of raw syndrome histories.
        """
        detectors = self._observed_temporal_detectors(syndrome_rounds)
        shots = detectors.shape[0]
        flips = np.zeros(shots, dtype=np.uint8)
        known = np.ones(shots, dtype=bool)
        gaps = np.zeros(shots, dtype=float)
        masses = np.zeros(shots, dtype=float)

        for shot in range(shots):
            contracted = self._contract_one(detectors[shot])
            log_z = contracted.log_z
            masses[shot] = contracted.truncation_mass_lost
            if len(log_z) < 2 or np.all(np.isneginf(log_z[:2])):
                known[shot] = False
                gaps[shot] = 0.0
                flips[shot] = 0
                continue
            z0 = float(log_z[0])
            z1 = float(log_z[1])
            flips[shot] = 1 if z1 > z0 else 0
            best = max(z0, z1)
            second = min(z0, z1)
            gaps[shot] = (
                math.inf
                if math.isinf(second) and not math.isinf(best)
                else best - second
            )

        return DecodeResult(
            frame_x=None,
            frame_z=None,
            observable_flips={
                label: flips.copy() for label in observable_labels
            },
            decoder_known=known,
            log_likelihood_gap=gaps,
            truncation_mass_lost=masses,
        )

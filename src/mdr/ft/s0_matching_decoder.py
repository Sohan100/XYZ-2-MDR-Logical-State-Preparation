"""

s0_matching_decoder.py
----------------------------------------------------------------------------
Spacetime minimum-weight matching decoder for the FT MDR circuits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import stim


@dataclass(frozen=True)
class LogicalErrorEstimate:
    """
    Monte-Carlo estimate of the logical error rate of one circuit.
    """

    shots: int
    errors: int
    seconds: float

    @property
    def rate(self) -> float:
        return self.errors / max(self.shots, 1)

    @property
    def stderr(self) -> float:
        p = self.rate
        return float(np.sqrt(max(p * (1.0 - p), 1e-30) / max(self.shots, 1)))


class S0MatchingDecoder:
    """
    Decode a detector-annotated FT MDR circuit with PyMatching.

    With `detectors="s0"` every circuit fault flips at most two
    frame-deterministic detectors after Stim's decomposition, so the model
    is a matching graph. This replaces the per-round lookup-table toggles:
    the decoder sees the whole space-time history at once and returns
    whether the Logical X outcome must be flipped (the logical part of the
    recovery). `kind="bposd"` and `kind="tesseract"` decode the full
    hypergraph model instead (optional packages `stimbposd`,
    `tesseract-decoder`).

    Attributes
    ----------
    circuit : stim.Circuit Circuit being decoded. kind : str Decoder backend.
    """

    def __init__(self, circuit: stim.Circuit, kind: str = "pymatching") -> None:
        self.circuit = circuit
        self.kind = kind
        if kind == "pymatching":
            import pymatching

            dem = circuit.detector_error_model(decompose_errors=True)
            self._matching = pymatching.Matching.from_detector_error_model(dem)
            self._decode = self._matching.decode_batch
        elif kind == "bposd":
            from stimbposd import BPOSD

            dem = circuit.detector_error_model(decompose_errors=False)
            self._decode = BPOSD(dem, max_bp_iters=30, osd_order=7).decode_batch
        elif kind == "tesseract":
            import tesseract_decoder.tesseract as tesseract

            dem = circuit.detector_error_model(decompose_errors=False)
            cfg = tesseract.TesseractConfig(dem=dem, det_beam=20,
                                            beam_climbing=True)
            self._decode = cfg.compile_decoder().decode_batch
        else:
            raise ValueError("kind must be 'pymatching', 'bposd' or 'tesseract'.")

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """
        Return predicted observable flips for a batch of detection events.
        """
        return np.asarray(self._decode(detection_events), dtype=bool)

    def decode_measurements(self, measurements: np.ndarray) -> np.ndarray:
        """
        Decode raw measurement records, for example from hardware shots.

        `measurements` has shape `(shots, circuit.num_measurements)` in the
        order of the circuit (the order of the classical register when the
        circuit is exported with `scripts/export_ft_mdr_qasm.py`). Returns a
        boolean vector that is True where the decoded Logical X outcome is
        -1, i.e. where the prepared plus state failed.
        """
        meas = np.asarray(measurements, dtype=bool)
        converter = self.circuit.compile_m2d_converter()
        dets, obs = converter.convert(measurements=meas,
                                      separate_observables=True)
        pred = self.decode_batch(dets)
        return np.any(pred != obs, axis=1)

    def estimate(
        self,
        max_shots: int = 1_000_000,
        max_errors: int = 1000,
        batch: int = 20_000,
        seed: Optional[int] = None,
        time_limit: float = 600.0,
    ) -> LogicalErrorEstimate:
        """
        Sample the circuit and count shots whose decoded Logical X is wrong.
        """
        sampler = self.circuit.compile_detector_sampler(seed=seed)
        shots = errors = 0
        start = time.time()
        while (shots < max_shots and errors < max_errors
               and time.time() - start < time_limit):
            size = min(batch, max_shots - shots)
            dets, obs = sampler.sample(size, separate_observables=True)
            pred = self.decode_batch(dets)
            errors += int(np.sum(np.any(pred != obs, axis=1)))
            shots += size
        return LogicalErrorEstimate(shots, errors, time.time() - start)

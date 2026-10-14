"""Minimum-weight perfect matching (MWPM) baseline via PyMatching.

Used to put the neural decoder's logical error rate in context against the
standard surface-code decoder on the same circuit and shots.
"""
from __future__ import annotations

import numpy as np

from .data import CodeTask


def mwpm_logical_error_rate(task: CodeTask, shots: int = 100_000,
                            seed: int | None = 0) -> float:
    """Logical error rate of MWPM on fresh samples from ``task``."""
    import pymatching

    dem = task.circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    dets, obs = task.sample_raw(shots, seed=seed)
    pred = matching.decode_batch(dets)                      # (shots, n_logical)
    wrong = np.any(pred != obs, axis=1)
    return float(wrong.mean())

"""Surface-code training data from Stim.

Builds a rotated-surface-code memory experiment and samples:

  * detection events, reshaped to ``(shots, rounds, n_stab)`` so each QEC round
    is one time step and each stabilizer location is one token, and
  * the logical-observable flips (the training labels).

Detectors are placed onto a canonical per-round stabilizer grid using their
Stim coordinates ``(x, y, t)``: the set of distinct ``(x, y)`` positions defines
the stabilizer tokens, and ``t`` indexes the round. Rounds that contain only a
subset of the detectors (e.g. the first/last layer of a memory experiment) are
simply zero-filled at the absent locations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim


@dataclass
class CodeTask:
    """A surface-code decoding task plus the layout needed to tensorise it."""

    circuit: stim.Circuit
    distance: int
    rounds: int
    basis: str
    p: float
    n_stab: int
    stab_coords: np.ndarray        # (n_stab, 2)
    n_round_layers: int            # number of distinct detector rounds
    _round_of: np.ndarray          # (num_detectors,) round-layer index per detector
    _stab_of: np.ndarray           # (num_detectors,) stabilizer index per detector

    @property
    def n_logical(self) -> int:
        return self.circuit.num_observables

    @property
    def num_detectors(self) -> int:
        return self.circuit.num_detectors

    # -- sampling ----------------------------------------------------------
    def sample_raw(self, shots: int, seed: int | None = None):
        """Return ``(dets, obs)`` as uint8 arrays of shape
        ``(shots, num_detectors)`` and ``(shots, n_logical)``."""
        sampler = self.circuit.compile_detector_sampler(seed=seed)
        dets, obs = sampler.sample(shots, separate_observables=True)
        return dets.astype(np.uint8), obs.astype(np.uint8)

    def to_layers(self, dets_raw: np.ndarray) -> np.ndarray:
        """Reshape raw detector vectors ``(shots, num_detectors)`` to the
        model layout ``(shots, n_round_layers, n_stab)`` (float32)."""
        shots = dets_raw.shape[0]
        out = np.zeros((shots, self.n_round_layers, self.n_stab), dtype=np.float32)
        out[:, self._round_of, self._stab_of] = dets_raw
        return out

    def sample(self, shots: int, seed: int | None = None):
        """Convenience: ``(layers, obs, dets_raw)`` for training/eval."""
        dets_raw, obs = self.sample_raw(shots, seed=seed)
        return self.to_layers(dets_raw), obs.astype(np.float32), dets_raw


def make_task(distance: int = 3, rounds: int = 3, p: float = 1e-3,
              basis: str = "Z") -> CodeTask:
    """Construct a rotated-surface-code memory task with a uniform depolarising
    noise model of strength ``p``."""
    code = f"surface_code:rotated_memory_{basis.lower()}"
    circuit = stim.Circuit.generated(
        code,
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )

    coords = circuit.get_detector_coordinates()        # idx -> [x, y, t]
    idx = sorted(coords)
    xy = [(coords[i][0], coords[i][1]) for i in idx]
    ts = [coords[i][2] for i in idx]

    uniq_xy = sorted(set(xy))
    stab_index = {p_: k for k, p_ in enumerate(uniq_xy)}
    uniq_t = sorted(set(ts))
    round_index = {tv: k for k, tv in enumerate(uniq_t)}

    round_of = np.array([round_index[coords[i][2]] for i in idx], dtype=np.int64)
    stab_of = np.array([stab_index[(coords[i][0], coords[i][1])] for i in idx],
                       dtype=np.int64)

    return CodeTask(
        circuit=circuit,
        distance=distance,
        rounds=rounds,
        basis=basis,
        p=p,
        n_stab=len(uniq_xy),
        stab_coords=np.array(uniq_xy, dtype=np.float32),
        n_round_layers=len(uniq_t),
        _round_of=round_of,
        _stab_of=stab_of,
    )

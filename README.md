# AlphaQubit (open reference implementation)

A compact, readable **PyTorch reimplementation** of the recurrent-transformer
neural decoder for surface codes described in

> J. Bausch, A. W. Senior, F. J. H. Heras, T. Edlich, A. Davies, M. Newman,
> C. Jones, K. Satzinger, M. Y. Niu, S. Blackwell, et al.
> *"Learning high-accuracy error decoding for quantum processors."*
> Nature **636**, 834–840 (2024).

It trains on [Stim](https://github.com/quantumlib/Stim)-simulated surface-code
data and is benchmarked against the standard MWPM decoder
([PyMatching](https://github.com/oscarhiggott/PyMatching)) on the same shots.

> **Independent project.** This is a from-the-paper reimplementation for
> research and teaching. It is not affiliated with, endorsed by, or derived
> from Google DeepMind's code or weights, and it does not aim to reproduce the
> paper's accuracy numbers (see *Scope* below).

## What it does

Given the per-round detection events from a surface-code experiment, the model
predicts whether a logical error occurred. It keeps the three load-bearing
ideas of the paper:

- **one representation vector per stabilizer** (a token per detector location);
- a **syndrome transformer** that mixes stabilizer tokens each QEC round, with
  an **attention bias from the 2D code geometry** (the spatial-structure prior);
- a **recurrence** carrying a per-stabilizer state across rounds, so a model
  trained at one round count decodes **any** number of rounds at inference.

A readout head pools the final state into per-logical-observable logits.

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .           # or: pip install -r requirements.txt
```

Needs `torch`, `stim`, `pymatching`, `numpy`. CPU is fine for small codes;
a GPU helps for `d >= 5` or large models (`--device cuda`).

## Quickstart

Train a small model on a distance-3 code and compare to MWPM (a couple of
minutes on CPU):

```bash
python examples/quickstart.py
```

or via the CLIs:

```bash
alphaqubit-train --distance 3 --rounds 3 --p 0.05 --steps 600 \
                 --d-model 64 --n-blocks 2 --out alphaqubit_d3.pt
alphaqubit-eval alphaqubit_d3.pt --shots 100000
```

Library use:

```python
from alphaqubit import make_task, AlphaQubit, AlphaQubitConfig, model_logical_error_rate

task  = make_task(distance=3, rounds=3, p=0.05)
cfg   = AlphaQubitConfig(n_stab=task.n_stab, n_logical=task.n_logical, d_model=64)
model = AlphaQubit(cfg, stab_coords=task.stab_coords)
# ... train (see alphaqubit/train.py) ...
print(model_logical_error_rate(model, task, shots=100_000))
```

## Architecture

```
detection events (B, rounds, n_stab)
        │
        ▼  per-round loop (recurrence)
  inject syndrome   : state + Linear(detector[, soft])
  syndrome transformer × n_blocks : attention(+spatial bias) → MLP
  GRU update        : state ← GRUCell(transformer_out, state)
        │  (after the last round)
        ▼
  LayerNorm → mean-pool over stabilizers → MLP head → logical logits
```

The spatial bias is an MLP mapping each pair of stabilizers' `(dx, dy)` offset
to a per-head additive attention score, so the attention is aware of the code
layout without hard-coding it.

## Scope and simplifications

This is a faithful reimplementation of the **architecture's ideas**, not a
reproduction of the paper's system or results. Deliberate differences:

- **No analog/soft readout or leakage inputs by default.** The input layer has
  an optional per-detector soft channel (`AlphaQubitConfig(soft_input=True)`),
  but the data generator here emits hard detection events only.
- **Learned pairwise spatial attention bias** instead of the paper's
  dilated-convolution stack over the 2D stabilizer grid.
- **GRU recurrence** for the across-round state update.
- **Simulated-only training by default.** Fine-tuning on recorded experimental
  shots works by feeding fixed `(detector, observable)` arrays into the same
  training loop (see `alphaqubit/train.py`).
- Matching the paper's accuracy needs large-scale pretraining, analog inputs,
  and a fine-tuning stage on real data; none of that is provided here.

## Results (smoke test)

Distance-3 rotated memory, `p = 0.006` (below the d=3 circuit-level threshold),
tiny model (`d_model=64`, `n_blocks=2`, ~0.1 M params), 800 training steps on a
laptop CPU, evaluated on 100k fresh shots. Reproduce with
`examples/quickstart.py`; numbers vary a little with seed.

| decoder     | logical error rate |
|-------------|--------------------|
| AlphaQubit  | ~3.1%              |
| MWPM        | ~2.4%              |

The reference model lands in the same ballpark as MWPM with a tiny network and
a few minutes of training — a correctness check, **not** a benchmark. Closing
and beating that gap (as the paper does) needs a larger model, far more
training, analog/soft inputs, and real-data fine-tuning.

## Tests

```bash
pip install -e ".[test]"
pytest -q
```

## Citing

If you use this code, please cite the original paper (above). For the
reimplementation itself, cite this repository.

## License

Apache-2.0. See [LICENSE](LICENSE).

"""Quickstart: train a small AlphaQubit on a distance-3 surface code and
compare it to MWPM. Runs on CPU in a couple of minutes.

    python examples/quickstart.py
"""
from alphaqubit.train import train
from alphaqubit.evaluate import model_logical_error_rate
from alphaqubit.baseline import mwpm_logical_error_rate

# p = 0.006 is below the d=3 circuit-level threshold, so there is real signal to
# decode (MWPM ~2.4% LER). It is still high enough that logical failures are
# frequent and quick to measure. Drop p toward ~1e-3 (and raise eval_shots) for
# the low-error regime.
model, task, best = train(
    distance=3, rounds=3, p=0.006,
    d_model=64, n_blocks=2, n_heads=4,
    steps=800, batch=1024, lr=1e-3, eval_every=200, eval_shots=40_000,
    out="alphaqubit_d3.pt",
)

ler = model_logical_error_rate(model, task, shots=100_000)
mwpm = mwpm_logical_error_rate(task, shots=100_000)
print(f"\nAlphaQubit LER = {ler:.4%}")
print(f"MWPM       LER = {mwpm:.4%}")

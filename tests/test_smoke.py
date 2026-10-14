"""Fast smoke tests: shapes, a forward pass, data tensorisation, and that a
few training steps drive the loss down. No GPU, runs in seconds."""
import numpy as np
import torch

from alphaqubit.data import make_task
from alphaqubit.model import AlphaQubit, AlphaQubitConfig


def test_task_shapes():
    task = make_task(distance=3, rounds=3, p=0.05)
    layers, obs, raw = task.sample(32, seed=0)
    assert layers.shape == (32, task.n_round_layers, task.n_stab)
    assert obs.shape == (32, task.n_logical)
    assert raw.shape == (32, task.num_detectors)
    # layers must be a lossless re-packing of the raw detector vector
    assert int(layers.sum()) == int(raw.sum())


def test_forward_shape():
    task = make_task(distance=3, rounds=2, p=0.05)
    cfg = AlphaQubitConfig(n_stab=task.n_stab, n_logical=task.n_logical,
                           d_model=32, n_blocks=1, n_heads=2)
    model = AlphaQubit(cfg, stab_coords=task.stab_coords)
    layers, _, _ = task.sample(8, seed=1)
    out = model(torch.from_numpy(layers))
    assert out.shape == (8, task.n_logical)


def test_handles_variable_rounds():
    # the recurrence should accept a round count unseen at construction time
    task = make_task(distance=3, rounds=5, p=0.05)
    cfg = AlphaQubitConfig(n_stab=task.n_stab, n_logical=task.n_logical,
                           d_model=32, n_blocks=1, n_heads=2)
    model = AlphaQubit(cfg, stab_coords=task.stab_coords)
    x = torch.zeros(4, 9, task.n_stab)      # 9 arbitrary time steps
    assert model(x).shape == (4, task.n_logical)


def test_training_reduces_loss():
    # p below the d=3 circuit-level threshold, so there is real signal to learn.
    torch.manual_seed(0)
    task = make_task(distance=3, rounds=3, p=0.008)
    cfg = AlphaQubitConfig(n_stab=task.n_stab, n_logical=task.n_logical,
                           d_model=48, n_blocks=2, n_heads=4)
    model = AlphaQubit(cfg, stab_coords=task.stab_coords)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def avg_loss():
        layers, obs, _ = task.sample(2048, seed=999)
        with torch.no_grad():
            return loss_fn(model(torch.from_numpy(layers)),
                           torch.from_numpy(obs)).item()

    before = avg_loss()
    for step in range(150):
        layers, obs, _ = task.sample(512, seed=step)
        loss = loss_fn(model(torch.from_numpy(layers)), torch.from_numpy(obs))
        opt.zero_grad(); loss.backward(); opt.step()
    after = avg_loss()
    assert after < before - 0.05, f"loss did not drop: {before:.3f} -> {after:.3f}"

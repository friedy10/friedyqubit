"""Evaluation: neural-decoder logical error rate, and a CLI to compare it with
the MWPM baseline on the same task."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .data import CodeTask, make_task
from .model import AlphaQubit, AlphaQubitConfig


@torch.no_grad()
def model_logical_error_rate(model: AlphaQubit, task: CodeTask, shots: int = 100_000,
                             device: str = "cpu", batch_size: int = 4096,
                             seed: int | None = 0) -> float:
    """Logical error rate of the neural decoder on fresh samples from ``task``."""
    model.eval()
    layers, obs, _ = task.sample(shots, seed=seed)
    wrong = 0
    for i in range(0, shots, batch_size):
        x = torch.from_numpy(layers[i:i + batch_size]).to(device)
        logits = model(x)
        pred = (logits > 0).to(torch.uint8).cpu().numpy()        # sigmoid>0.5
        y = obs[i:i + batch_size].astype(np.uint8)
        wrong += int(np.any(pred != y, axis=1).sum())
    return wrong / shots


def _load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    task = make_task(**ckpt["task"])
    model = AlphaQubit(AlphaQubitConfig(**ckpt["config"]), stab_coords=task.stab_coords)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model, task


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate an AlphaQubit checkpoint.")
    ap.add_argument("checkpoint")
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-mwpm", action="store_true", help="skip the MWPM baseline")
    args = ap.parse_args(argv)

    model, task = _load_checkpoint(args.checkpoint, args.device)
    ler = model_logical_error_rate(model, task, args.shots, args.device)
    print(f"task: d={task.distance} rounds={task.rounds} basis={task.basis} "
          f"p={task.p}  (n_stab={task.n_stab}, layers={task.n_round_layers})")
    print(f"AlphaQubit  LER = {ler:.4%}  ({model.num_parameters():,} params)")
    if not args.no_mwpm:
        from .baseline import mwpm_logical_error_rate
        mwpm = mwpm_logical_error_rate(task, args.shots)
        print(f"MWPM        LER = {mwpm:.4%}")


if __name__ == "__main__":
    main()

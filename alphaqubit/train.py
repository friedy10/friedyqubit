"""Train an AlphaQubit decoder on Stim-simulated surface-code data.

Data is sampled fresh from Stim every step, so training sees an effectively
infinite stream of shots (the "pretraining on simulated data" regime of the
paper). Fine-tuning on recorded experimental shots can reuse the same loop by
feeding a fixed array of (detector, observable) pairs instead.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from .data import make_task
from .evaluate import model_logical_error_rate
from .model import AlphaQubit, AlphaQubitConfig


def train(distance=3, rounds=3, p=1e-3, basis="Z",
          d_model=128, n_heads=4, n_blocks=3, d_ff=256, dropout=0.0,
          steps=2000, batch=1024, lr=1e-3, weight_decay=1e-4,
          eval_every=200, eval_shots=50_000, device="cpu", seed=0,
          out="alphaqubit.pt", verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    task = make_task(distance=distance, rounds=rounds, p=p, basis=basis)
    cfg = AlphaQubitConfig(n_stab=task.n_stab, n_logical=task.n_logical,
                           d_model=d_model, n_heads=n_heads, n_blocks=n_blocks,
                           d_ff=d_ff, dropout=dropout)
    model = AlphaQubit(cfg, stab_coords=task.stab_coords).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    if verbose:
        print(f"task: d={distance} rounds={rounds} basis={basis} p={p} "
              f"| n_stab={task.n_stab} layers={task.n_round_layers} "
              f"n_logical={task.n_logical}")
        print(f"model: {model.num_parameters():,} params | device={device}")

    best = None
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        layers, obs, _ = task.sample(batch, seed=seed + step)
        x = torch.from_numpy(layers).to(device)
        y = torch.from_numpy(obs).to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if verbose and (step % max(1, eval_every // 4) == 0):
            print(f"  step {step:5d}  loss {loss.item():.4f}  "
                  f"({step / (time.time() - t0):.1f} it/s)")
        if step % eval_every == 0 or step == steps:
            ler = model_logical_error_rate(model, task, eval_shots, device,
                                           seed=10_000 + step)
            if verbose:
                print(f"  step {step:5d}  eval LER {ler:.4%}")
            if best is None or ler < best:
                best = ler
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": vars(cfg),
                    "task": {"distance": distance, "rounds": rounds,
                             "p": p, "basis": basis},
                    "eval_ler": ler, "step": step,
                }, out)
    if verbose:
        print(f"done. best eval LER {best:.4%}  -> {out}")
    return model, task, best


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train an AlphaQubit decoder.")
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--p", type=float, default=1e-3)
    ap.add_argument("--basis", default="Z", choices=["Z", "X"])
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-blocks", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-shots", type=int, default=50_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="alphaqubit.pt")
    args = ap.parse_args(argv)
    train(distance=args.distance, rounds=args.rounds, p=args.p, basis=args.basis,
          d_model=args.d_model, n_heads=args.n_heads, n_blocks=args.n_blocks,
          steps=args.steps, batch=args.batch, lr=args.lr,
          eval_every=args.eval_every, eval_shots=args.eval_shots,
          device=args.device, seed=args.seed, out=args.out)


if __name__ == "__main__":
    main()

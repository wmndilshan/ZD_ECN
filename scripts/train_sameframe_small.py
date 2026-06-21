#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from evaluate_zdecn_dqs import INPUT_SIZE, OUTPUT_SIZE, VERTEX_COUNT


def build_model(hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(INPUT_SIZE, hidden),
        nn.GELU(approximate="tanh"),
        nn.Linear(hidden, hidden),
        nn.GELU(approximate="tanh"),
        nn.Linear(hidden, OUTPUT_SIZE),
    )


def pve_mm(predicted: torch.Tensor, target: torch.Tensor, batch_size: int) -> float:
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, predicted.shape[0], batch_size):
            diff = predicted[start:start + batch_size] - target[start:start + batch_size]
            diff = diff.reshape(-1, VERTEX_COUNT, 3)
            values.append(torch.linalg.vector_norm(diff, dim=2).mean(dim=1).cpu())
    return float(torch.cat(values).mean().item())


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> tuple[float, float]:
    model.eval()
    losses: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    loss_fn = nn.MSELoss(reduction="mean")
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size]
            yb = y[start:start + batch_size]
            pred = model(xb)
            losses.append(loss_fn(pred, yb).detach().cpu())
            preds.append(pred.cpu())
    pred_all = torch.cat(preds)
    return float(torch.stack(losses).mean().item()), pve_mm(pred_all, y.cpu(), batch_size)


def load_history(path: Path) -> list[dict[str, float | int | bool]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    summary: dict[str, object],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "summary": summary,
            "architecture": "144->64->64->20670, GELU(tanh), same-frame pose-correction target",
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset_sameframe_pose_3k.npz"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper/sameframe_small_train"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--small-run-target-pve", type=float, default=2.5534801483154297)
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.dataset, allow_pickle=False)
    x_np = data["theta_rot6d"].astype(np.float32)
    y_np = data["target_delta_mm"].astype(np.float32)
    train_indices = data["train_indices"].astype(np.int64)
    val_indices = data["val_indices"].astype(np.int64)

    x_train = torch.from_numpy(x_np[train_indices])
    y_train = torch.from_numpy(y_np[train_indices])
    x_val = torch.from_numpy(x_np[val_indices])
    y_val = torch.from_numpy(y_np[val_indices])

    baseline_val_pve = pve_mm(torch.zeros_like(y_val), y_val, args.eval_batch_size)
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    loss_fn = nn.MSELoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "training_history.jsonl"
    history: list[dict[str, float | int | bool]] = load_history(history_path)
    start_epoch = 1

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "summary" in checkpoint and "epochs" in checkpoint["summary"]:
            start_epoch = int(checkpoint["summary"]["epochs"]) + 1
        elif history:
            start_epoch = int(history[-1]["epoch"]) + 1

    crossing_epoch = next((int(row["epoch"]) for row in history if row["val_pve_mm"] < baseline_val_pve), None)
    small_run_crossing_epoch = next((int(row["epoch"]) for row in history if row["val_pve_mm"] < args.small_run_target_pve), None)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        order = torch.randperm(x_train.shape[0])
        train_loss_sum = 0.0
        train_batches = 0
        for start in range(0, order.numel(), args.batch_size):
            batch_indices = order[start:start + args.batch_size]
            pred = model(x_train[batch_indices])
            loss = loss_fn(pred, y_train[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            train_batches += 1

        val_loss, val_pve = evaluate(model, x_val, y_val, args.eval_batch_size)
        crossed = val_pve < baseline_val_pve
        if crossed and crossing_epoch is None:
            crossing_epoch = epoch
        crossed_small_run = val_pve < args.small_run_target_pve
        if crossed_small_run and small_run_crossing_epoch is None:
            small_run_crossing_epoch = epoch
        row = {
            "epoch": epoch,
            "train_mse": train_loss_sum / train_batches,
            "val_mse": val_loss,
            "val_pve_mm": val_pve,
            "baseline_val_pve_mm": baseline_val_pve,
            "small_run_target_pve_mm": args.small_run_target_pve,
            "below_lbs_baseline": crossed,
            "below_small_run_target": crossed_small_run,
        }
        history.append(row)
        with history_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

        summary = {
            "dataset": str(args.dataset),
            "epochs": epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "train_samples": int(train_indices.size),
            "val_samples": int(val_indices.size),
            "baseline_val_pve_mm": baseline_val_pve,
            "small_run_target_pve_mm": args.small_run_target_pve,
            "lbs_crossing_epoch": crossing_epoch,
            "small_run_target_crossing_epoch": small_run_crossing_epoch,
            "final_val_pve_mm": val_pve,
            "healthy_directionality": crossing_epoch is not None and crossing_epoch <= 20,
        }
        if args.checkpoint_every > 0 and (epoch % args.checkpoint_every == 0 or epoch == args.epochs):
            save_checkpoint(args.output_dir / f"checkpoint_epoch_{epoch:04d}.pt", model, optimizer, summary)
            save_checkpoint(args.output_dir / "checkpoint_last.pt", model, optimizer, summary)
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    summary = {
        "dataset": str(args.dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "train_samples": int(train_indices.size),
        "val_samples": int(val_indices.size),
        "baseline_val_pve_mm": baseline_val_pve,
        "small_run_target_pve_mm": args.small_run_target_pve,
        "lbs_crossing_epoch": crossing_epoch,
        "small_run_target_crossing_epoch": small_run_crossing_epoch,
        "final_val_pve_mm": history[-1]["val_pve_mm"],
        "healthy_directionality": crossing_epoch is not None and crossing_epoch <= 20,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_checkpoint(args.output_dir / "checkpoint_last.pt", model, optimizer, summary)
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()

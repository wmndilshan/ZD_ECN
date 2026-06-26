#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        rows.append(json.loads(line))
    return rows


def checkpoint_epochs(output_dir: Path) -> list[int]:
    epochs = []
    for path in output_dir.glob("checkpoint_epoch_*.pt"):
        match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def trend(current: float, previous: float | None) -> str:
    if previous is None:
        return "first checkpoint"
    delta = current - previous
    if delta < -0.03:
        return "improving"
    if delta > 0.10:
        return "concerning"
    return "stable"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only monitor for full ZD-ECN retraining progress.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper/full_retrain_sameframe"))
    parser.add_argument("--lbs-baseline", type=float, default=5.41)
    parser.add_argument("--small-run-target", type=float, default=2.30)
    args = parser.parse_args()

    rows = load_rows(args.output_dir / "training_history.jsonl")
    checkpoints = checkpoint_epochs(args.output_dir)
    if not rows:
        print("No training rows yet.")
        return

    by_epoch = {int(row["epoch"]): row for row in rows}
    previous_pve: float | None = None
    for epoch in checkpoints:
        row = by_epoch.get(epoch)
        if row is None:
            continue
        current_pve = float(row["val_pve_mm"])
        print(f"[Epoch {epoch}] val PVE = {current_pve:.3f}mm  [trend: {trend(current_pve, previous_pve)}]")
        previous_pve = current_pve

    lbs_cross = next((int(row["epoch"]) for row in rows if float(row["val_pve_mm"]) < args.lbs_baseline), None)
    small_cross = next((int(row["epoch"]) for row in rows if float(row["val_pve_mm"]) < args.small_run_target), None)
    latest = rows[-1]
    print(
        json.dumps(
            {
                "latest_epoch": int(latest["epoch"]),
                "latest_val_pve_mm": float(latest["val_pve_mm"]),
                "checkpoint_epochs": checkpoints,
                "lbs_baseline_crossing_epoch": lbs_cross,
                "small_run_range_crossing_epoch": small_cross,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

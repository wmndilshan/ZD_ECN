# ZD-ECN Training Dataset

`scripts/build_training_dataset.py` is the canonical target-generation script
for new ZD-ECN retraining runs.

## Canonical Decisions

- Raw pose source: `surreal/zd_ecn_subset/smpl_data.npz`.
- Subject exclusion: subjects `18` and `19` are excluded.
- Beta handling: zero-beta only. The production ZD-ECN input is pose-only
  `rot6d_144`, so beta-dependent targets would be shape variation the model
  cannot observe.
- Target formula:
  `target_mm = (skin_A(theta)(v_shaped + pose_bs(theta)) - skin_A(theta)(v_shaped)) * 1000`.
- Coordinate frame: posed frame; both sides use the same posed joint transforms
  `A(theta)`.
- Units: target deltas are saved in millimeters.
- Split: deterministic train/validation split, default `80/20`, seed `42`.
- Sampling: natural, unweighted random sampling without replacement from real
  CMU MoCap pose sequences. Synthetic stratification is intentionally not used.

## Small Validation Dataset

Before a full retrain, generate a small 3,000-pose dataset from the existing
evaluation poses:

```bash
.venv/bin/python scripts/build_training_dataset.py \
  --from-existing-eval-set data/dataset_eval_3k.npz \
  --samples 3000 \
  --output data/dataset_sameframe_pose_3k.npz
```

The LBS baseline for this target is zero correction, so the baseline PVE is the
mean magnitude of `target_delta_mm` on the validation split.

## Full Retraining Dataset

Only after the small validation run shows healthy directionality, generate the
full 30,000-pose dataset:

```bash
.venv/bin/python scripts/build_training_dataset.py \
  --samples 30000 \
  --output data/dataset_sameframe_pose_30k.npz
```

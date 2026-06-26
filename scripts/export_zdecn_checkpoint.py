#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_zdecn_dqs import INPUT_SIZE, OUTPUT_SIZE
from train_sameframe_small import build_model


LAYERS = [
    ("0", "weights_l0.bin", "biases_l0.bin", (64, 144), (64,)),
    ("2", "weights_l1.bin", "biases_l1.bin", (64, 64), (64,)),
    ("4", "weights_l2.bin", "biases_l2.bin", (20670, 64), (20670,)),
]


def load_checkpoint_model(checkpoint_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def layer_arrays(model: torch.nn.Module) -> list[tuple[np.ndarray, np.ndarray]]:
    modules = dict(model.named_modules())
    arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for module_name, _weight_file, _bias_file, weight_shape, bias_shape in LAYERS:
        layer = modules[module_name]
        weight = layer.weight.detach().cpu().numpy().astype(np.float32)
        bias = layer.bias.detach().cpu().numpy().astype(np.float32)
        if weight.shape != weight_shape:
            raise ValueError(f"Layer {module_name} weight shape {weight.shape} != {weight_shape}")
        if bias.shape != bias_shape:
            raise ValueError(f"Layer {module_name} bias shape {bias.shape} != {bias_shape}")
        arrays.append((weight, bias))
    return arrays


def write_weight_outputs(
    arrays: list[tuple[np.ndarray, np.ndarray]],
    output_fp16: Path,
    shader_asset_dir: Path | None,
    output_fp32: Path | None = None,
) -> dict[str, object]:
    output_fp16.parent.mkdir(parents=True, exist_ok=True)
    if output_fp32 is not None:
        output_fp32.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[np.ndarray] = []
    fp32_chunks: list[np.ndarray] = []
    manifest_layers: list[dict[str, object]] = []

    if shader_asset_dir is not None:
        shader_asset_dir.mkdir(parents=True, exist_ok=True)

    for index, ((weight, bias), (_module_name, weight_file, bias_file, weight_shape, bias_shape)) in enumerate(zip(arrays, LAYERS, strict=True)):
        weight_fp16 = np.ascontiguousarray(weight.astype(np.float16))
        bias_fp16 = np.ascontiguousarray(bias.astype(np.float16))
        weight_fp32 = np.ascontiguousarray(weight.astype(np.float32))
        bias_fp32 = np.ascontiguousarray(bias.astype(np.float32))
        chunks.extend([weight_fp16.reshape(-1), bias_fp16.reshape(-1)])
        fp32_chunks.extend([weight_fp32.reshape(-1), bias_fp32.reshape(-1)])

        if shader_asset_dir is not None:
            weight_fp16.tofile(shader_asset_dir / weight_file)
            bias_fp16.tofile(shader_asset_dir / bias_file)

        manifest_layers.append(
            {
                "layer": index,
                "fan_in": int(weight_shape[1]),
                "fan_out": int(weight_shape[0]),
                "weights_file": weight_file,
                "biases_file": bias_file,
                "weights_bytes": int(weight_fp16.nbytes),
                "biases_bytes": int(bias_fp16.nbytes),
            }
        )

    np.concatenate(chunks).astype(np.float16).tofile(output_fp16)
    if output_fp32 is not None:
        np.concatenate(fp32_chunks).astype(np.float32).tofile(output_fp32)
    return {
        "combined_fp16": str(output_fp16),
        "combined_fp32": str(output_fp32) if output_fp32 is not None else None,
        "packing": "W0,b0,W1,b1,W2,b2; fp16 row-major weights [out,in], then fp16 biases [out]",
        "layers": manifest_layers,
    }


def write_fixture_outputs(model: torch.nn.Module, fixture_dir: Path, batch_size: int) -> dict[str, object]:
    inputs_path = fixture_dir / "inputs_rot6d_f32.bin"
    inputs = np.fromfile(inputs_path, dtype=np.float32).reshape(-1, INPUT_SIZE)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            x = torch.from_numpy(inputs[start:start + batch_size])
            outputs.append(model(x).cpu().numpy().astype(np.float32))
    output = np.concatenate(outputs)
    if output.shape != (inputs.shape[0], OUTPUT_SIZE):
        raise ValueError(f"Unexpected fixture output shape: {output.shape}")
    output_path = fixture_dir / "outputs_ref_f32.bin"
    np.ascontiguousarray(output).tofile(output_path)
    return {
        "fixture_dir": str(fixture_dir),
        "inputs": str(inputs_path),
        "outputs": str(output_path),
        "samples": int(inputs.shape[0]),
        "output_shape": [int(inputs.shape[0]), OUTPUT_SIZE],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a ZD-ECN PyTorch checkpoint to GLSL fp16 SSBO assets.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-bin", type=Path)
    parser.add_argument("--output-fp16", type=Path)
    parser.add_argument("--output-fp32", type=Path)
    parser.add_argument("--shader-asset-dir", type=Path)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--fixture-batch-size", type=int, default=16)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    output_fp16 = args.output_fp16 or args.output_bin
    if output_fp16 is None:
        parser.error("one of --output-fp16 or --output-bin is required")

    model = load_checkpoint_model(args.checkpoint)
    arrays = layer_arrays(model)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "architecture": "144->64->64->20670, GELU(tanh)",
        "weight_export": write_weight_outputs(arrays, output_fp16, args.shader_asset_dir, args.output_fp32),
    }
    if args.fixture_dir is not None:
        metadata["fixture_export"] = write_fixture_outputs(model, args.fixture_dir, args.fixture_batch_size)
    if args.shader_asset_dir is not None:
        manifest_path = args.shader_asset_dir / "production_asset_manifest.json"
        manifest = {
            "source": str(output_fp16),
            "packing": metadata["weight_export"]["packing"],
            "layers": metadata["weight_export"]["layers"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        metadata["shader_manifest"] = str(manifest_path)

    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

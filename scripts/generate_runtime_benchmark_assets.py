#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

INPUT_SIZE = 144
HIDDEN_SIZE = 64
OUTPUT_SIZE = 20670
SEED = 0x5EED1234


def rodrigues(axis_angle):
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = np.divide(axis_angle, theta, out=np.zeros_like(axis_angle), where=theta > 1e-8)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = np.zeros_like(x)
    k = np.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], axis=-1).reshape(axis.shape[:-1] + (3, 3))
    eye = np.broadcast_to(np.eye(3, dtype=np.float32), k.shape)
    theta_m = theta[..., None]
    return eye + np.sin(theta_m) * k + (1.0 - np.cos(theta_m)) * (k @ k)


def first_pose_rot6d(smpl_npz):
    data = np.load(smpl_npz)
    pose_key = next(key for key in sorted(data.files) if key.startswith('pose_'))
    poses = np.asarray(data[pose_key], dtype=np.float32)
    pose = poses[0] if poses.ndim == 2 else poses.reshape(-1, 72)[0]
    joints = pose[:72].reshape(24, 3)
    rotations = rodrigues(joints)
    rot6d = rotations[:, :, :2].transpose(0, 2, 1).reshape(-1).astype(np.float32)
    if rot6d.shape[0] != INPUT_SIZE:
        raise ValueError(f'Expected {INPUT_SIZE} rot6d values, got {rot6d.shape[0]}')
    return pose_key, rot6d


def gelu_nodes(name, input_name, output_name):
    pow_out = f'{name}_pow3'
    beta_mul = f'{name}_beta_mul'
    inner = f'{name}_inner'
    alpha_mul = f'{name}_alpha_mul'
    tanh = f'{name}_tanh'
    one_plus = f'{name}_one_plus'
    half_x = f'{name}_half_x'
    return [
        helper.make_node('Pow', [input_name, f'{name}_three'], [pow_out], name=f'{name}_Pow3'),
        helper.make_node('Mul', [pow_out, f'{name}_beta'], [beta_mul], name=f'{name}_BetaMul'),
        helper.make_node('Add', [input_name, beta_mul], [inner], name=f'{name}_Inner'),
        helper.make_node('Mul', [inner, f'{name}_alpha'], [alpha_mul], name=f'{name}_AlphaMul'),
        helper.make_node('Tanh', [alpha_mul], [tanh], name=f'{name}_Tanh'),
        helper.make_node('Add', [tanh, f'{name}_one'], [one_plus], name=f'{name}_OnePlus'),
        helper.make_node('Mul', [input_name, f'{name}_half'], [half_x], name=f'{name}_HalfX'),
        helper.make_node('Mul', [half_x, one_plus], [output_name], name=f'{name}_Out'),
    ]


def scalar_initializer(name, value):
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def make_model(weights, biases, output_path):
    initializers = []
    nodes = []
    previous = 'input'
    shapes = [(INPUT_SIZE, HIDDEN_SIZE), (HIDDEN_SIZE, HIDDEN_SIZE), (HIDDEN_SIZE, OUTPUT_SIZE)]
    for idx, (fan_in, fan_out) in enumerate(shapes):
        w = weights[idx].reshape(fan_out, fan_in).T.astype(np.float32)
        b = biases[idx].astype(np.float32)
        w_name = f'W{idx}'
        b_name = f'B{idx}'
        matmul = f'L{idx}_matmul'
        linear = 'output' if idx == 2 else f'L{idx}_linear'
        initializers.append(numpy_helper.from_array(w, name=w_name))
        initializers.append(numpy_helper.from_array(b, name=b_name))
        nodes.append(helper.make_node('MatMul', [previous, w_name], [matmul], name=f'L{idx}_MatMul'))
        nodes.append(helper.make_node('Add', [matmul, b_name], [linear], name=f'L{idx}_Add'))
        if idx < 2:
            gelu_out = f'L{idx}_gelu'
            prefix = f'L{idx}_gelu_const'
            initializers.extend([
                scalar_initializer(f'{prefix}_three', 3.0),
                scalar_initializer(f'{prefix}_beta', 0.044715),
                scalar_initializer(f'{prefix}_alpha', 0.7978845608),
                scalar_initializer(f'{prefix}_one', 1.0),
                scalar_initializer(f'{prefix}_half', 0.5),
            ])
            nodes.extend(gelu_nodes(prefix, linear, gelu_out))
            previous = gelu_out
        else:
            previous = linear

    graph = helper.make_graph(
        nodes,
        'zd_ecn_mlp_144_64_64_20670',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, INPUT_SIZE])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, OUTPUT_SIZE])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpl-data', default='surreal/zd_ecn_subset/smpl_data.npz')
    parser.add_argument('--shader-assets', default='android_compute_benchmark/app/src/main/assets/shaders')
    parser.add_argument('--onnx-assets', default='onnx_runtime_benchmark/app/src/main/assets')
    args = parser.parse_args()

    shader_assets = Path(args.shader_assets)
    onnx_assets = Path(args.onnx_assets)
    shader_assets.mkdir(parents=True, exist_ok=True)
    onnx_assets.mkdir(parents=True, exist_ok=True)

    pose_key, rot6d = first_pose_rot6d(args.smpl_data)
    (shader_assets / 'input_rot6d.bin').write_bytes(rot6d.tobytes())
    (onnx_assets / 'input_rot6d.bin').write_bytes(rot6d.tobytes())

    rng = np.random.default_rng(SEED)
    layer_shapes = [(INPUT_SIZE, HIDDEN_SIZE), (HIDDEN_SIZE, HIDDEN_SIZE), (HIDDEN_SIZE, OUTPUT_SIZE)]
    weights = []
    biases = []
    for idx, (fan_in, fan_out) in enumerate(layer_shapes):
        scale = np.sqrt(6.0 / (fan_in + fan_out)).astype(np.float32)
        weight = rng.uniform(-scale, scale, size=(fan_out, fan_in)).astype(np.float32)
        bias = rng.uniform(-0.05, 0.05, size=(fan_out,)).astype(np.float32)
        weights.append(weight)
        biases.append(bias)
        (shader_assets / f'weights_l{idx}.bin').write_bytes(weight.astype(np.float16).tobytes())
        (shader_assets / f'biases_l{idx}.bin').write_bytes(bias.tobytes())

    make_model(weights, biases, onnx_assets / 'zd_ecn_mlp.onnx')
    metadata = (
        f'source_pose_key={pose_key}\n'
        f'architecture=144->64->64->20670\n'
        f'input_representation=rot6d from first SURREAL pose\n'
        f'seed={SEED}\n'
        f'shader_weights=fp16\n'
        f'onnx_weights=fp32\n'
    )
    (shader_assets / 'benchmark_asset_metadata.txt').write_text(metadata)
    (onnx_assets / 'benchmark_asset_metadata.txt').write_text(metadata)
    print(metadata)


if __name__ == '__main__':
    main()

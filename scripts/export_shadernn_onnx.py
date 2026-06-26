#!/usr/bin/env python3
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from pathlib import Path

INPUT_SIZE = 144
HIDDEN_SIZE = 64
OUTPUT_SIZE = 20670
SEED = 0x5EED1234

def make_model(weights, biases, output_path):
    initializers = []
    nodes = []
    
    # Input is 4D [1, 144, 1, 1]
    nodes.append(helper.make_node(
        'Flatten',
        ['input'],
        ['input_flat'],
        name='L0_Flatten',
        axis=1
    ))
    
    previous = 'input_flat'
    shapes = [(INPUT_SIZE, HIDDEN_SIZE), (HIDDEN_SIZE, HIDDEN_SIZE), (HIDDEN_SIZE, OUTPUT_SIZE)]
    
    for idx, (fan_in, fan_out) in enumerate(shapes):
        w = weights[idx].astype(np.float32)  # shape (fan_out, fan_in)
        b = biases[idx].astype(np.float32)  # shape (fan_out,)
        w_name = f'W{idx}'
        b_name = f'B{idx}'
        gemm_out = 'output' if idx == 2 else f'L{idx}_gemm'
        
        initializers.append(numpy_helper.from_array(w, name=w_name))
        initializers.append(numpy_helper.from_array(b, name=b_name))
        
        nodes.append(helper.make_node(
            'Gemm',
            [previous, w_name, b_name],
            [gemm_out],
            name=f'L{idx}_Gemm',
            alpha=1.0,
            beta=1.0,
            transB=1
        ))
        
        if idx < 2:
            relu_out = f'L{idx}_relu'
            nodes.append(helper.make_node(
                'Relu',
                [gemm_out],
                [relu_out],
                name=f'L{idx}_Relu'
            ))
            previous = relu_out

    graph = helper.make_graph(
        nodes,
        'zd_ecn_snn_mlp_144_64_64_20670',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, INPUT_SIZE, 1, 1])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, OUTPUT_SIZE])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 6
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"ONNX model saved to {output_path}")

def main():
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

    output_dir = Path('models')
    output_dir.mkdir(parents=True, exist_ok=True)
    make_model(weights, biases, output_dir / 'zdecn_snn.onnx')

if __name__ == '__main__':
    main()

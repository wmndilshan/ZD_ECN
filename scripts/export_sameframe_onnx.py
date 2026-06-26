#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import torch
import torch.nn as nn

from train_sameframe_small import build_model

class FloatIoHalfModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.half()
    def forward(self, x):
        return self.model(x.to(torch.float16)).to(torch.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, default=Path('results/paper/full_retrain_sameframe/checkpoint_epoch_0375.pt'))
    ap.add_argument('--output', type=Path, default=Path('models/zdecn_sameframe_fp16.onnx'))
    ap.add_argument('--metadata', type=Path, default=Path('results/paper/final_validation/onnx_sameframe_export.json'))
    args=ap.parse_args()
    ckpt=torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model=build_model()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    wrapped=FloatIoHalfModel(model).eval()
    dummy=torch.zeros(1,144,dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, dummy, args.output,
        input_names=['input'], output_names=['output'],
        opset_version=17,
        dynamic_axes=None,
        do_constant_folding=True,
    )
    import onnx
    m=onnx.load(str(args.output))
    onnx.checker.check_model(m)
    meta={
      'checkpoint': str(args.checkpoint),
      'output': str(args.output),
      'input_names': [i.name for i in m.graph.input],
      'output_names': [o.name for o in m.graph.output],
      'bytes': args.output.stat().st_size,
      'precision': 'fp16 weights/activations internally, float32 external input/output casts',
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=2))
if __name__=='__main__': main()

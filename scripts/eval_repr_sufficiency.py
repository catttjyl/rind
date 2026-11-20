#!/usr/bin/env python3
"""CPU/CUDA synthetic representation sufficiency probe for frozen RIND v2.3.0."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from rind.inverse_design import build_layered_occlusion_benchmark_scene, evaluate_representation_sufficiency
from rind.models import build_model, load_rind_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='RIND unified checkpoint; omitted uses a seeded untrained model')
    parser.add_argument('--device', default='cuda', help='torch device (default cuda; cpu supported)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261114)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu for a CPU probe')
    if args.checkpoint:
        model, provenance = load_rind_checkpoint(args.checkpoint, device=device)
        sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        mode = 'checkpoint'
    else:
        model = build_model('rind_full_geom', hidden_dim=32, max_candidates=8).to(device)
        provenance, sha, mode = None, None, 'seeded_untrained_model'
    model.eval()
    scene = build_layered_occlusion_benchmark_scene()
    with torch.no_grad():
        result = evaluate_representation_sufficiency(
            scene['receivers_xyz'], scene['sun_directions_xyz'], scene['triangles_xyz'],
            rind_model=model, source_instance_ids=scene['source_instance_ids'],
            ground_mask=scene['ground_mask'], device=str(device))
    receipt = {'schema': 'rind.stage1.synthetic_repr_sufficiency.v1',
               'scope': 'synthetic diagnostic only; not a trained-model performance claim',
               'seed': args.seed, 'device': str(device), 'model_mode': mode,
               'checkpoint_sha256': sha, 'checkpoint_provenance': provenance,
               'representation_sufficiency_result': result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False, default=str) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

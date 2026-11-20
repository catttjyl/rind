"""Forward-only transversal intersections on a per-query candidate bank."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def forward_candidate_hits(vertices_m, mask, suns):
    """Candidate-wise Möller–Trumbore, including edges and strictly positive t."""
    tri = vertices_m.double()
    sun = suns.double()
    if not torch.isfinite(tri).all() or not torch.isfinite(sun).all() or (sun.norm(dim=-1)==0).any():
        raise ValueError("Finite geometry and nonzero directions required")
    sun = torch.nn.functional.normalize(sun,dim=-1)[:,None,None,:].expand_as(tri[...,0,:])
    a,b,c = tri.unbind(-2)
    e1,e2 = b-a,c-a
    normal = torch.cross(e1,e2,dim=-1)
    area = normal.norm(dim=-1)
    edge2 = torch.stack([v.square().sum(-1) for v in (e1,e2,c-b)]).amax(0)
    p = torch.cross(sun,e2,dim=-1)
    det = (e1*p).sum(-1)
    active = mask.bool() & (area > 1e-12*edge2) & (det.abs() > 1e-12*area)
    divisor = torch.where(active,det,1.)
    q = torch.cross(-a,e1,dim=-1)
    u = (-a*p).sum(-1)/divisor
    v = (sun*q).sum(-1)/divisor
    t = (e2*q).sum(-1)/divisor
    return active & (t > 0.) & (u >= -1e-10) & (v >= -1e-10) & (u+v <= 1.+1e-10)


def exact_physical_sunlight_rate(tokens_raw_v, mask, suns, shaded_mask):
    if not shaded_mask.any():
        raise ValueError("No selected shadow queries")
    vertices_m = tokens_raw_v.reshape(*tokens_raw_v.shape[:-1],3,3).double()*1000.
    return (~forward_candidate_hits(vertices_m,mask,suns).any(-1))[shaded_mask].double().mean().item()


def evaluate_exports(folder):
    rows = []
    for path in sorted(folder.glob("*.npz")):
        with np.load(path,allow_pickle=False) as data:
            sun = torch.from_numpy(data["sun"])
            mask = torch.from_numpy(data["mask"])
            selected = torch.from_numpy(data["shaded_mask"]).bool()
            rates = {}
            for name in ("baseline","autograd25","blackbox25","blackbox200"):
                vertices = torch.from_numpy(data["vertices_"+name])
                hits = forward_candidate_hits(vertices,mask,sun)
                rates[name] = float((~hits.any(-1))[selected].double().mean())
            rows.append(dict(clip=path.stem,selected=int(selected.sum()),rates=rates,
                             signed_gain={k:v-rates["baseline"] for k,v in rates.items() if k!="baseline"}))
    if not rows:
        raise ValueError("No exported slices")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    report = evaluate_exports(args.exports)
    with args.output.open("x") as out:
        json.dump(report,out,indent=2,allow_nan=False)

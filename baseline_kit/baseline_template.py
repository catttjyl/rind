#!/usr/bin/env python3
"""Starter template for external SOTA baselines (Tracks A-F) on the 93-tile RIND benchmark.

Enforces:
1. Strict IEEE FP32 precision (`allow_tf32=False`, no AMP/BF16).
2. Read-only `fit_67` -> `calibration` -> `val_spatial_clean` split separation.
3. Deterministic `label_budget_fraction` (`0.01`, `0.05`, `1.0`, seed `20261112`) subsampling.
4. Standard output contract: `pred_probs_{tile_id}.npy` (`float32`, `[T, N]`) and
   `perf_receipt.json` evaluated directly by `baseline_kit/evaluate_predictions.py`.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

KIT_ROOT = Path(__file__).resolve().parents[1]
if str(KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(KIT_ROOT))
if str(KIT_ROOT / "baseline_kit") not in sys.path:
    sys.path.insert(0, str(KIT_ROOT / "baseline_kit"))

from evaluate_predictions import calibrate_logits_or_probs, compute_metrics_bundle, load_split_tile_ids
from rind.label_budget import TileLabelDomain, coordinate_keys


def load_shard_dict(data_dir: Path, tile_id: str, portable_index: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    """Load one tile shard from extracted directory, `.npz`, or `.tar.gz` archive."""
    npz_path = data_dir / f"{tile_id}.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    rel = portable_index.get("tiles", {}).get(tile_id, {}).get("path") if portable_index else None
    cand_dirs = [data_dir / tile_id]
    if rel:
        cand_dirs.insert(0, data_dir / rel)
    for d in cand_dirs:
        if d.is_dir() and (d / "truth.npy").exists():
            return {p.stem: np.load(p, allow_pickle=False) for p in sorted(d.glob("*.npy"))}

    tar_path = data_dir / f"{tile_id}.tar.gz"
    if tar_path.exists():
        out: Dict[str, np.ndarray] = {}
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tf.getmembers():
                if member.name.endswith(".npy"):
                    fobj = tf.extractfile(member)
                    if fobj is not None:
                        out[Path(member.name).stem] = np.load(io.BytesIO(fobj.read()), allow_pickle=False)
        return out

    raise FileNotFoundError(f"Tile shard for {tile_id} not found under {data_dir}")


def subsample_receivers_for_budget(
    tile_id: str,
    shard: Dict[str, np.ndarray],
    fraction: float,
    seed: int = 20261112,
) -> np.ndarray:
    """Return boolean receiver mask `[N]` matching the official cold-start label budget."""
    n_pts = int(shard["receivers_xyz"].shape[0])
    if fraction >= 1.0 - 1e-9:
        return np.ones((n_pts,), dtype=bool)
    rc = shard.get("receiver_rc")
    if rc is None:
        rc = np.stack([np.zeros(n_pts, dtype=np.int64), np.arange(n_pts, dtype=np.int64)], axis=1)
    domain = TileLabelDomain(tile_id, rc)
    keys = coordinate_keys(rc)
    return domain.select(keys, tile_id, seed, float(fraction))


class StarterBaselineModel(nn.Module):
    """Minimal template model mapping `(receiver_xyz_rel, sun_dir)` to sunlit logit."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rec_xyz: torch.Tensor, sun_dirs: torch.Tensor) -> torch.Tensor:
        """Predict `[T, N]` logits from `[N, 3]` receivers and `[T, 3]` solar vectors."""
        rec_norm = (rec_xyz - rec_xyz.mean(dim=0, keepdim=True)) / 100.0
        t_len, n_pts = sun_dirs.shape[0], rec_xyz.shape[0]
        feat = torch.cat(
            [rec_norm[None, :, :].expand(t_len, n_pts, 3), sun_dirs[:, None, :].expand(t_len, n_pts, 3)],
            dim=-1,
        )
        return self.net(feat).squeeze(-1)


def train_starter_model(
    fit_shards: Sequence[Tuple[str, Dict[str, np.ndarray]]],
    *,
    epochs: int,
    label_budget_fraction: float,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, float]:
    """Train starter model strictly on `fit` shards using pure IEEE FP32."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = StarterBaselineModel(hidden_dim=64).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(fit_shards))
        for idx in order:
            tid, shard = fit_shards[int(idx)]
            keep = subsample_receivers_for_budget(tid, shard, label_budget_fraction, seed=20261112)
            rec = torch.from_numpy(shard["receivers_xyz"][keep]).to(device=device, dtype=torch.float32)
            suns = torch.from_numpy(shard["sun_dirs"]).to(device=device, dtype=torch.float32)
            truth = torch.from_numpy(shard["truth"][:, keep] == 1).to(device=device, dtype=torch.float32)
            vmask = torch.from_numpy(
                np.asarray(shard["valid_mask"][:, keep], dtype=bool) & (shard["truth"][:, keep] >= 0)
            ).to(device=device)
            if not bool(vmask.any().item()):
                continue
            opt.zero_grad(set_to_none=True)
            logits = model(rec, suns)
            loss = bce(logits[vmask], truth[vmask])
            loss.backward()
            opt.step()
    return model, time.perf_counter() - t0


@torch.no_grad()
def export_predictions_and_receipt(
    model: nn.Module,
    shards: Sequence[Tuple[str, Dict[str, np.ndarray]]],
    out_dir: Path,
    *,
    method_name: str,
    train_wall_sec: float,
    device: torch.device,
) -> Path:
    """Write `pred_probs_{tile_id}.npy` (`float32`, `[T, N]`) and `perf_receipt.json`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for tid, shard in shards:
        rec = torch.from_numpy(shard["receivers_xyz"]).to(device=device, dtype=torch.float32)
        suns = torch.from_numpy(shard["sun_dirs"]).to(device=device, dtype=torch.float32)
        probs = torch.sigmoid(model(rec, suns)).to("cpu", dtype=torch.float32).numpy()
        np.save(out_dir / f"pred_probs_{tid}.npy", probs.astype(np.float32))

    latencies_ms: Dict[str, float] = {}
    dummy_sun = torch.tensor([[0.2, 0.8, 0.5]], dtype=torch.float32, device=device)
    for n_pts in (256, 1024, 4096):
        dummy_rec = torch.zeros((n_pts, 3), dtype=torch.float32, device=device)
        t_s = time.perf_counter()
        _ = model(dummy_rec, dummy_sun)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms[f"N_{n_pts}_ms_per_frame"] = round((time.perf_counter() - t_s) * 1000.0, 4)

    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 2)
        if device.type == "cuda"
        else 0.0
    )
    receipt = {
        "method_name": method_name,
        "precision": "PURE_IEEE_FP32",
        "allow_tf32": False,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "train_wall_seconds": round(train_wall_sec, 3),
        "peak_vram_mb": peak_vram_mb,
        "inference_latency_ms": latencies_ms,
    }
    receipt_path = out_dir / "perf_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt_path


def _make_synthetic_shard(seed: int, t_steps: int = 32, n_pts: int = 64) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rec = rng.uniform(-40.0, 40.0, size=(n_pts, 3)).astype(np.float32)
    rec[:, 1] = 0.0
    rc = np.stack([np.arange(n_pts, dtype=np.int64) // 8, np.arange(n_pts, dtype=np.int64) % 8], axis=1)
    ang = np.linspace(0.2, 2.9, t_steps, dtype=np.float32)
    suns = np.stack([np.cos(ang), np.sin(ang), np.full_like(ang, 0.35)], axis=1)
    suns /= np.linalg.norm(suns, axis=1, keepdims=True)
    score = suns[:, 0:1] * rec[None, :, 0] * 0.05 + suns[:, 1:2] - 0.5
    truth = (score > 0.0).astype(np.int8)
    return {
        "receivers_xyz": rec,
        "receiver_rc": rc,
        "sun_dirs": suns.astype(np.float32),
        "truth": truth,
        "valid_mask": np.ones((t_steps, n_pts), dtype=bool),
        "day_ids": np.zeros((t_steps,), dtype=np.int64),
        "timestamps_unix_s": np.arange(t_steps, dtype=np.int64) * 120,
        "boundary_distance_m": np.full((t_steps, n_pts), 0.3, dtype=np.float32),
        "boundary_distance_valid": np.ones((t_steps, n_pts), dtype=bool),
        "shared_az0_cand_nonempty": np.ones((t_steps, n_pts), dtype=bool),
    }


def run_synthetic_smoke() -> None:
    """Exercise full train -> export -> calibrate -> referee evaluation on synthetic shards."""
    fit_shards = [(f"syn_fit_{i}", _make_synthetic_shard(100 + i)) for i in range(4)]
    cal_shards = [(f"syn_cal_{i}", _make_synthetic_shard(200 + i)) for i in range(2)]
    val_shards = [(f"syn_val_{i}", _make_synthetic_shard(300 + i)) for i in range(2)]

    dev = torch.device("cpu")
    model, wall_s = train_starter_model(fit_shards, epochs=5, label_budget_fraction=0.25, seed=20261114, device=dev)
    with tempfile.TemporaryDirectory(prefix="rind_baseline_smoke_") as tmp:
        out_dir = Path(tmp)
        receipt = export_predictions_and_receipt(
            model, [*cal_shards, *val_shards], out_dir,
            method_name="starter_smoke", train_wall_sec=wall_s, device=dev,
        )
        cal_p = [np.load(out_dir / f"pred_probs_{tid}.npy") for tid, _ in cal_shards]
        _, th, _ = calibrate_logits_or_probs(
            cal_p, [s["truth"] for _, s in cal_shards], [s["valid_mask"] for _, s in cal_shards]
        )
        accs = []
        for tid, sh in val_shards:
            pr = np.load(out_dir / f"pred_probs_{tid}.npy")
            mb = compute_metrics_bundle(
                pr, sh["truth"], sh["valid_mask"], th, sh["day_ids"], sh["timestamps_unix_s"],
                sh["boundary_distance_m"], sh["boundary_distance_valid"], sh["shared_az0_cand_nonempty"],
            )
            accs.append(mb["state_acc"])
        assert receipt.exists() and len(accs) == 2
        print(f"[baseline_template --synthetic-smoke] PASS (th={th:.3f}, mean_val_acc={np.mean(accs)*100:.2f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description="RIND external baseline starter template.")
    ap.add_argument("--synthetic-smoke", action="store_true", help="Run 5-second end-to-end synthetic check.")
    ap.add_argument("--data-dir", type=Path, help="Directory holding 93-tile shards.")
    ap.add_argument("--out-dir", type=Path, default=KIT_ROOT / "outputs" / "my_baseline")
    ap.add_argument("--method-name", type=str, default="starter_mlp")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--label-budget", type=float, default=1.0, help="1.0, 0.05, or 0.01.")
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    if args.synthetic_smoke:
        run_synthetic_smoke()
        return

    if not args.data_dir:
        ap.error("--data-dir is required unless --synthetic-smoke is set.")

    idx_path = KIT_ROOT / "meta" / "PORTABLE_INDEX.json"
    pidx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else None
    buckets = load_split_tile_ids(KIT_ROOT / "meta" / "split.json", KIT_ROOT / "meta" / "manifest.json")

    fit_shards = [(tid, load_shard_dict(args.data_dir, tid, pidx)) for tid in buckets["fit_67"]]
    cal_shards = [(tid, load_shard_dict(args.data_dir, tid, pidx)) for tid in buckets["calibration"]]
    val_shards = [(tid, load_shard_dict(args.data_dir, tid, pidx)) for tid in buckets["val_spatial_clean"]]

    dev = torch.device(args.device)
    model, wall_s = train_starter_model(
        fit_shards, epochs=args.epochs, label_budget_fraction=args.label_budget, seed=args.seed, device=dev
    )
    export_predictions_and_receipt(
        model, [*cal_shards, *val_shards], args.out_dir,
        method_name=args.method_name, train_wall_sec=wall_s, device=dev,
    )
    print(f"Exported predictions and perf_receipt.json -> {args.out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 4 Dataset & Input-Tier Fairness Controlled 93-Tile Training Suite (`run_p4_fairness_baselines_93tile.py`).

Trains & evaluates 4 controlled fairness models under the EXACT same 93-tile protocol as Phase 1/2:
- Split: `fit_67` (67 tiles, N=256, 3 epochs = 804 updates, seed=20261114, rng=default_rng(20261114 + ep*1009))
  -> `calibration` (12 tiles, 81+61 temp grid + 181-point `[0.05, 0.95]` threshold grid)
  -> `val_spatial_clean` (14 tiles, 5,119,233 valid point-frames + Shared External `az0` Hard-Sun Mask).
- Precision: Strictly pure IEEE FP32 (`allow_tf32=False`, `autocast(enabled=False)`), logging per-module
  gradient L2 norms and latent norms to `collapse_diagnostics.jsonl` (Redline #11123/#11124).

Models trained:
  1. `az24_tier_g0_pure_xyz`: Official `RindCandidateWorldModel` (`zero_margin=True`, `use_local_context=False`,
     `enable_spatial_knn_readout=False`, `az24` static sector routing). Strips ALL analytical edge margins
     (`slots 22..24`) and ALL 16D local raster features — pure raw 3D triangle vertices `(x,y,z)` + sun vector!
  2. `ext_pointnet_solar_film_gru`: External 3D Point-Cloud + Solar-FiLM + GRU baseline (`~261K` params, matched
     to RIND's `263.8K`). Consumes the EXACT same `az24` `K=32` raw 3D relative triangles (`9D` vertices + `3D`
     centroid + `3D` unit normal + `1D` log-area = `16D` raw geometry, zero analytical margins) + 27D Solar
     Spherical-Harmonic FiLM modulation + PointNet Max/Mean set pooling + `96D` GRU + `bce_head`!
  3. `ext_dsm25d_solar_knn_gru`: External 2.5D Urban DSM + Solar-FiLM + Spatial-8NN Graph Recurrent baseline
     (*Deep Umbra* / 2.5D raster + neighborhood horizon propagation + `96D` GRU).
  4. `probe_canonical37_mlp`: Official `CanonicalProbe("canonical37_mlp", hidden_dim=96)` consuming the COMPLETE
     `az24 Tier-G2` 37D candidate features (including `slots 22..24` analytical signed projected edge margins),
     directly testing whether 37D features alone solve urban shadow casting without ray cross-attention & recurrence.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
PORTABLE_PKG = Path(os.environ.get("RIND_PORTABLE_PKG", str(BUNDLE_ROOT)))
P4_ROOT = Path(os.environ.get("RIND_OUT_ROOT", str(BUNDLE_ROOT / "outputs" / "internal_baselines")))
SPLIT_FILE = Path(os.environ.get("RIND_SPLIT_FILE", str(BUNDLE_ROOT / "meta" / "split.json")))
MANIFEST_FILE = Path(os.environ.get("RIND_MANIFEST_FILE", str(BUNDLE_ROOT / "meta" / "manifest.json")))

for _p in (str(PORTABLE_PKG), str(BUNDLE_ROOT / "baseline_kit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rind import fast_data
from rind.fast_train import compute_fullday_loss
from rind.models import CanonicalProbe, build_model, eval_real_spherical_harmonics_l4
from rind.split_contract import SplitContract

SPLIT_SHA = hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest() if SPLIT_FILE.exists() else ""


def _solar_sh27(sun_dirs: torch.Tensor) -> torch.Tensor:
    s = sun_dirs.to(torch.float32)
    s_u = s / torch.linalg.norm(s, dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.cat([s_u, eval_real_spherical_harmonics_l4(s_u)], dim=-1)


def _day_spans(day_ids: np.ndarray, ts: np.ndarray) -> List[Tuple[int, int, int]]:
    spans: List[Tuple[int, int, int]] = []
    if len(day_ids) == 0:
        return spans
    s = 0
    for i in range(1, len(day_ids)):
        if day_ids[i] != day_ids[i - 1] or (ts[i] - ts[i - 1]) != 120:
            spans.append((int(day_ids[s]), s, i))
            s = i
    spans.append((int(day_ids[s]), s, len(day_ids)))
    return spans


class PointNetSolarFiLMGRU(nn.Module):
    """External 3D Point-Cloud Strong Baseline (~261K params, matched to RIND 263.8K):
    Reads the EXACT same `az24` `K=32` raw 3D relative triangle geometry (`v0,v1,v2, centroid, normal, log_area`)
    without sphere-tangent projection, analytical margins, or ray-anisotropic cross-attention.
    Uses shared PointNet MLP + Solar-FiLM conditioning + symmetric Max+Mean set pooling + 96D GRU.
    """
    has_event_gate = False

    def __init__(self, hidden_dim: int = 96) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        sol_dim = 27  # 3 raw unit sun + 24 Real Spherical Harmonics (l=1..4)

        # Static PointNet per-triangle encoder (16D raw 3D triangle -> 96D)
        self.tri_mlp1 = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Solar-FiLM modulation on 3D candidate set
        self.solar_film = nn.Sequential(
            nn.Linear(sol_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.tri_mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Symmetric Max + Mean set readout + solar fusion -> 96D
        self.set_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + sol_dim + 1, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=False)
        self.bce_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _raw_tri16(rel_tris: torch.Tensor) -> torch.Tensor:
        """Compute strictly raw 3D coordinates `(v0,v1,v2)/200m`, centroid `/200m`, unit normal, and log(1+area)."""
        # rel_tris: (N, B, K, 3, 3)
        v_scaled = torch.clamp(rel_tris / 200.0, -20.0, 20.0)
        cent = v_scaled.mean(dim=-2)  # (..., 3)
        e1 = rel_tris[..., 1, :] - rel_tris[..., 0, :]
        e2 = rel_tris[..., 2, :] - rel_tris[..., 0, :]
        cr = torch.cross(e1, e2, dim=-1)
        area2 = torch.linalg.norm(cr, dim=-1, keepdim=True)
        unit_n = cr / area2.clamp_min(1e-6)
        log_area = torch.log1p(0.5 * area2) / 10.0
        return torch.cat([v_scaled.flatten(-2, -1), cent, unit_n, log_area], dim=-1)

    def encode_static_bank(self, az_tris: torch.Tensor) -> torch.Tensor:
        raw16 = self._raw_tri16(az_tris)
        return self.tri_mlp1(raw16)

    def forward_day(
        self,
        az_tris_or_bank: torch.Tensor,
        az_mask: torch.Tensor,
        az_bin_idx: torch.Tensor,
        sun_dirs: torch.Tensor,
        is_preencoded_bank: bool = False,
    ) -> Dict[str, torch.Tensor]:
        t_steps = sun_dirs.shape[0]
        b_idx = az_bin_idx.long()
        mk_perm = az_mask.permute(1, 0, 2)         # (24, N, K)
        m_tk = mk_perm[b_idx]                      # (T, N, K)

        if is_preencoded_bank:
            n_pts, _, k_cands, _ = az_tris_or_bank.shape
            h_tk = az_tris_or_bank.permute(1, 0, 2, 3)[b_idx]
        else:
            # Encode ONLY the unique azimuth sectors visited in this day (~8 instead of all 24 -> 3x faster, exact!)
            n_pts, _, k_cands, _, _ = az_tris_or_bank.shape
            uniq_b, inv_b = torch.unique(b_idx, sorted=True, return_inverse=True)
            sub_tris = az_tris_or_bank[:, uniq_b]  # (N, U, K, 3, 3)
            sub_bank = self.encode_static_bank(sub_tris)  # (N, U, K, D)
            h_tk = sub_bank.permute(1, 0, 2, 3)[inv_b]    # (T, N, K, D)

        sol_emb = _solar_sh27(sun_dirs)                             # (T, 27)
        gb = self.solar_film(sol_emb)                               # (T, 2D)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)[:, None, None, :]                 # (T, 1, 1, D)
        beta = beta[:, None, None, :]

        h_mod = self.tri_mlp2(h_tk * (1.0 + gamma) + beta)          # (T, N, K, D)
        m_f = m_tk.unsqueeze(-1)
        cnt = m_f.sum(dim=-2).clamp_min(1.0)
        mean_pool = h_mod.masked_fill(~m_f, 0.0).sum(dim=-2) / cnt
        max_pool = h_mod.masked_fill(~m_f, -1e9).amax(dim=-2)
        has_any = m_tk.any(dim=-1, keepdim=True)
        max_pool = torch.where(has_any, max_pool, torch.zeros_like(max_pool))

        sol_exp = sol_emb[:, None, :].expand(t_steps, n_pts, -1)
        cnt_feat = (m_tk.float().sum(dim=-1, keepdim=True) / float(k_cands))
        x_in = self.set_proj(torch.cat([mean_pool, max_pool, sol_exp, cnt_feat], dim=-1))  # (T, N, D)

        z_seq, _ = self.gru(x_in)
        logits = self.bce_head(z_seq).squeeze(-1)
        logits_b = self.boundary_head(z_seq).squeeze(-1)
        return {
            "logits": logits,
            "logits_bce": logits,
            "logits_boundary": logits_b,
            "states": z_seq,
            "candidate_mask": m_tk,
        }


class UrbanDSM25DSolarKNNGRU(nn.Module):
    """External 2.5D Urban DSM + Solar-FiLM + Spatial-8NN Graph Recurrent Baseline (~215K params):
    Models standard urban solar 2.5D height/sky-view raster + local neighborhood horizon propagation + GRU,
    with ZERO 3D far-field triangle candidates.
    """
    has_event_gate = False

    def __init__(self, hidden_dim: int = 96, k_neighbors: int = 8) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        sol_dim = 27

        self.register_buffer("feat_mean", torch.zeros(16, dtype=torch.float32))
        self.register_buffer("feat_std", torch.ones(16, dtype=torch.float32))
        self.register_buffer("norm_count", torch.zeros((), dtype=torch.long))

        self.static_enc = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # Spatial 8-NN directional edge MLP (delta_xyz/50m + neighbor height features)
        self.edge_mpnn = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.solar_film = nn.Sequential(
            nn.Linear(sol_dim, hidden_dim * 3),
            nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
        )
        self.step_mlp = nn.Sequential(
            nn.Linear(hidden_dim + sol_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=False)
        self.bce_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @torch.no_grad()
    def fit_normalizer(self, fit_feats: List[np.ndarray]) -> None:
        cat = np.concatenate(fit_feats, axis=0).astype(np.float64)
        m = cat.mean(axis=0)
        s = np.maximum(cat.std(axis=0), 1e-4)
        self.feat_mean.copy_(torch.from_numpy(m.astype(np.float32)))
        self.feat_std.copy_(torch.from_numpy(s.astype(np.float32)))
        self.norm_count.fill_(len(cat))

    def encode_static_graph(self, local_features: torch.Tensor, rec_xyz: torch.Tensor) -> torch.Tensor:
        x_norm = (local_features - self.feat_mean) / self.feat_std.clamp_min(1e-4)
        h0 = self.static_enc(x_norm)  # (N, D)
        n_pts = rec_xyz.shape[0]
        k = min(self.k_neighbors, max(n_pts - 1, 1))
        dist = torch.cdist(rec_xyz, rec_xyz)
        dist.fill_diagonal_(1e9)
        knn_d, knn_idx = torch.topk(dist, k=k, dim=-1, largest=False)
        rel = (rec_xyz[knn_idx] - rec_xyz[:, None, :]) / 50.0
        d_feat = (knn_d.unsqueeze(-1) / 50.0).clamp_max(5.0)
        h_nb = h0[knn_idx]
        h_self = h0[:, None, :].expand(n_pts, k, -1)
        msg = self.edge_mpnn(torch.cat([h_self, h_nb, rel, d_feat], dim=-1)).mean(dim=1)
        return F.layer_norm(h0 + msg, (self.hidden_dim,))

    def forward_day(self, h_static: torch.Tensor, sun_dirs: torch.Tensor) -> Dict[str, torch.Tensor]:
        t_steps = sun_dirs.shape[0]
        n_pts = h_static.shape[0]
        sol_emb = _solar_sh27(sun_dirs)                             # (T, 27)
        gb = self.solar_film(sol_emb)
        gamma, beta = gb.chunk(2, dim=-1)
        h_mod = h_static[None, :, :] * (1.0 + torch.tanh(gamma)[:, None, :]) + beta[:, None, :]
        sol_exp = sol_emb[:, None, :].expand(t_steps, n_pts, -1)
        x_in = self.step_mlp(torch.cat([h_mod, sol_exp], dim=-1))
        z_seq, _ = self.gru(x_in)
        logits = self.bce_head(z_seq).squeeze(-1)
        logits_b = self.boundary_head(z_seq).squeeze(-1)
        return {
            "logits": logits,
            "logits_bce": logits,
            "logits_boundary": logits_b,
            "states": z_seq,
            "candidate_mask": torch.ones((t_steps, n_pts, 1), dtype=torch.bool, device=sun_dirs.device),
        }


def calibrate_logits_wide(
    cal_logits: List[np.ndarray],
    cal_truth: List[np.ndarray],
    cal_valid: List[np.ndarray],
) -> Tuple[float, float, float]:
    """Exact Phase 2 calibration protocol: 81 coarse + 61 fine temperature BCE NLL grid + 181-point [0.05, 0.95] threshold grid."""
    lg = np.concatenate([x[v & (t >= 0)] for x, t, v in zip(cal_logits, cal_truth, cal_valid)]).astype(np.float64)
    gt = np.concatenate([(t[v & (t >= 0)] == 1) for t, v in zip(cal_truth, cal_valid)]).astype(np.float64)

    def _nll(temp: float) -> float:
        z = np.clip(lg / max(temp, 1e-4), -60.0, 60.0)
        # Stable BCE with logits: max(z, 0) - z*y + log(1 + exp(-|z|))
        return float(np.mean(np.maximum(z, 0.0) - z * gt + np.log1p(np.exp(-np.abs(z)))))

    coarse_t = np.logspace(np.log10(0.05), np.log10(10.0), 81)
    best_t = float(min(coarse_t, key=_nll))
    lo = max(0.02, best_t * 0.65)
    hi = min(15.0, best_t * 1.45)
    fine_t = np.linspace(lo, hi, 61)
    best_t = float(min(fine_t, key=_nll))
    best_nll = _nll(best_t)

    probs = 1.0 / (1.0 + np.exp(-np.clip(lg / best_t, -60.0, 60.0)))
    gt_b = gt > 0.5
    tot = float(len(gt_b))
    sun_tot = int(gt_b.sum())

    best_th = 0.5
    best_score = -1.0
    for th in np.linspace(0.05, 0.95, 181):
        pr = probs >= th
        tp = int((pr & gt_b).sum())
        fp = int((pr & (~gt_b)).sum())
        tn = int(((~pr) & (~gt_b)).sum())
        fn = sun_tot - tp
        acc = (tp + tn) / tot
        f1 = (2.0 * tp) / max(2 * tp + fp + fn, 1)
        sc = 0.5 * (acc + f1)
        if sc > best_score:
            best_score = sc
            best_th = float(round(float(th), 4))

    return best_t, best_th, best_nll


def _grad_l2(model: nn.Module) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            sq += float((g * g).sum().item())
    return math.sqrt(sq)


def run_one_arm(
    arm_key: str,
    fit_tiles: List[Dict[str, np.ndarray]],
    cal_tiles: List[Dict[str, np.ndarray]],
    val_tiles: List[Dict[str, np.ndarray]],
    dev: torch.device,
    epochs: int = 3,
    seed: int = 20261114,
) -> Dict[str, Any]:
    out_ckpt = P4_ROOT / "checkpoints" / f"{arm_key}_calibrated.pt"
    out_json = P4_ROOT / "results" / f"fairness_{arm_key}.json"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists() and out_ckpt.exists():
        print(f"[{arm_key}] Already completed, loading cached receipt...", flush=True)
        return json.loads(out_json.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    contract = SplitContract(SPLIT_FILE, MANIFEST_FILE)
    firewall = contract.issue_fit_firewall_token(
        [str(z["tile_id"].item()) for z in fit_tiles], subset_name="fit_67"
    )

    if arm_key == "az24_tier_g0_pure_xyz":
        model = build_model(
            "rind_zero_margin",
            hidden_dim=96,
            num_heads=4,
            max_candidates=32,
            enable_spatial_knn_readout=False,
            use_static_mpnn=True,
            use_fourier102=True,
            reanchor_every=0,
        ).to(device=dev, dtype=torch.float32)
        has_gate = True
    elif arm_key == "probe_canonical37_mlp":
        model = CanonicalProbe("canonical37_mlp", hidden_dim=96).to(device=dev, dtype=torch.float32)
        has_gate = False
    elif arm_key == "ext_pointnet_solar_film_gru":
        model = PointNetSolarFiLMGRU(hidden_dim=96).to(device=dev, dtype=torch.float32)
        has_gate = False
    elif arm_key == "ext_dsm25d_solar_knn_gru":
        model = UrbanDSM25DSolarKNNGRU(hidden_dim=96, k_neighbors=8).to(device=dev, dtype=torch.float32)
        model.fit_normalizer([z["local_features"] for z in fit_tiles])
        has_gate = False
    else:
        raise ValueError(f"Unknown arm_key: {arm_key}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Training Fairness Arm `{arm_key}` (params={n_params:,}, Pure FP32 on {dev}) ===", flush=True)

    # Fit normalizers for official RIND models strictly on `fit_67`
    if hasattr(model, "update_normalizers"):
        for z in fit_tiles:
            tid = str(z["tile_id"].item())
            b_norm = {
                "receivers_xyz": torch.from_numpy(z["receivers_xyz"]).to(dev),
                "local_features": torch.from_numpy(z["local_features"]).to(dev),
                "az_rel_triangles": torch.from_numpy(z["az_rel_triangles"]).to(dev),
                "az_opaque": torch.from_numpy(z["az_opaque"]).to(dev),
                "az_mask": torch.from_numpy(z["az_mask"]).to(dev),
                "az_source_ids": torch.from_numpy(z["az_source_ids"]).to(dev),
                "azimuth_bin_idx": torch.from_numpy(z["azimuth_bin_idx"][:8].astype(np.int64)).to(dev),
                "sun_dirs": torch.from_numpy(z["sun_dirs"][:8]).to(dev),
            }
            b_norm = fast_data.materialize_gpu_tokens_from_azimuth_bank(b_norm)
            model.update_normalizers(b_norm, firewall_token=firewall, tile_ids=[tid], expected_split_sha256=SPLIT_SHA)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1.5e-4)

    diag_file = P4_ROOT / "checkpoints" / f"{arm_key}_diagnostics.jsonl"
    if diag_file.exists():
        diag_file.unlink()

    total_steps = 0
    total_pf = 0
    nonfinite_steps = 0
    max_grad_seen = 0.0
    epoch_logs: List[Dict[str, Any]] = []

    def _forward_tile_day(z: Dict[str, np.ndarray], f0: int, f1: int, static_cache: Any = None):
        d_len = f1 - f0
        same_adj = torch.ones((max(d_len - 1, 0),), dtype=torch.bool, device=dev)
        suns_d = torch.from_numpy(z["sun_dirs"][f0:f1]).to(dev)
        az_b = torch.from_numpy(z["azimuth_bin_idx"][f0:f1].astype(np.int64)).to(dev)

        if arm_key in ("az24_tier_g0_pure_xyz", "probe_canonical37_mlp"):
            b_in = {
                "receivers_xyz": torch.from_numpy(z["receivers_xyz"]).to(dev),
                "local_features": torch.from_numpy(z["local_features"]).to(dev),
                "az_rel_triangles": torch.from_numpy(z["az_rel_triangles"]).to(dev),
                "az_opaque": torch.from_numpy(z["az_opaque"]).to(dev),
                "az_mask": torch.from_numpy(z["az_mask"]).to(dev),
                "az_source_ids": torch.from_numpy(z["az_source_ids"]).to(dev),
                "azimuth_bin_idx": az_b,
                "sun_dirs": suns_d,
                "same_day_adjacent_mask": same_adj,
            }
            b_in = fast_data.materialize_gpu_tokens_from_azimuth_bank(b_in)
            out = model(b_in, use_calibrated_temperature=False)
            cmask = b_in["candidate_mask"]
            return out, cmask, same_adj, None

        if arm_key == "ext_pointnet_solar_film_gru":
            if static_cache is not None:
                out = model.forward_day(static_cache, torch.from_numpy(z["az_mask"]).to(dev), az_b, suns_d, is_preencoded_bank=True)
                return out, out["candidate_mask"], same_adj, static_cache
            if not model.training:
                sb = model.encode_static_bank(torch.from_numpy(z["az_rel_triangles"]).to(dev))
                out = model.forward_day(sb, torch.from_numpy(z["az_mask"]).to(dev), az_b, suns_d, is_preencoded_bank=True)
                return out, out["candidate_mask"], same_adj, sb
            out = model.forward_day(torch.from_numpy(z["az_rel_triangles"]).to(dev), torch.from_numpy(z["az_mask"]).to(dev), az_b, suns_d, is_preencoded_bank=False)
            return out, out["candidate_mask"], same_adj, None

        if arm_key == "ext_dsm25d_solar_knn_gru":
            hs = model.encode_static_graph(
                torch.from_numpy(z["local_features"]).to(dev),
                torch.from_numpy(z["receivers_xyz"]).to(dev),
            ) if static_cache is None else static_cache
            out = model.forward_day(hs, suns_d)
            return out, out["candidate_mask"], same_adj, (hs if not model.training else None)

        raise RuntimeError(arm_key)

    model.train()
    for ep in range(epochs):
        ep_t0 = time.perf_counter()
        rng = np.random.default_rng(seed + ep * 1009)
        order = np.arange(len(fit_tiles))
        rng.shuffle(order)
        ep_loss = 0.0
        ep_steps = 0
        ep_pf = 0

        for idx_pos in order:
            z = fit_tiles[int(idx_pos)]
            spans = _day_spans(z["day_ids"], z["timestamps_unix_s"])
            rec_dev = torch.from_numpy(z["receivers_xyz"]).to(dev)
            for _, f0, f1 in spans:
                optimizer.zero_grad(set_to_none=True)
                out, cmask, same_adj, _ = _forward_tile_day(z, f0, f1, static_cache=None)
                b_loss = {
                    "truth": torch.from_numpy(z["truth"][f0:f1]).to(dev),
                    "valid_mask": torch.from_numpy(z["valid_mask"][f0:f1]).to(dev),
                    "same_day_adjacent_mask": same_adj,
                    "candidate_mask": cmask,
                    "boundary_distance_m": torch.from_numpy(z["boundary_distance_m"][f0:f1]).to(dev),
                    "boundary_distance_valid": torch.from_numpy(z["boundary_distance_valid"][f0:f1]).to(dev),
                    "receivers_xyz": rec_dev,
                }
                loss, lstats = compute_fullday_loss(out, b_loss, has_event_gate=has_gate)
                if not bool(torch.isfinite(loss).item()):
                    nonfinite_steps += 1
                    raise RuntimeError(f"Non-finite loss in {arm_key} at step {total_steps}")
                loss.backward()
                gnorm = _grad_l2(model)
                if not math.isfinite(gnorm):
                    nonfinite_steps += 1
                    raise RuntimeError(f"Non-finite grad norm in {arm_key} at step {total_steps}")
                max_grad_seen = max(max_grad_seen, gnorm)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_steps += 1
                ep_steps += 1
                spf = int(lstats["valid_point_frames"])
                ep_pf += spf
                total_pf += spf
                ep_loss += float(lstats["total_loss"])

                if total_steps % 20 == 0 or total_steps == 1:
                    with diag_file.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({
                            "step": total_steps,
                            "epoch": ep + 1,
                            "total_loss": round(float(lstats["total_loss"]), 5),
                            "state_bce": round(float(lstats["state_bce_loss"]), 5),
                            "trans_focal": round(float(lstats["transition_focal_loss"]), 5),
                            "grad_l2": round(gnorm, 4),
                        }) + "\n")

        scheduler.step()
        ep_sec = time.perf_counter() - ep_t0
        epoch_logs.append({
            "epoch": ep + 1,
            "steps": ep_steps,
            "mean_loss": ep_loss / max(ep_steps, 1),
            "valid_point_frames": ep_pf,
            "wall_sec": round(ep_sec, 2),
        })
        print(f"  [{arm_key}] Epoch {ep+1}/{epochs}: steps={ep_steps} loss={ep_loss/max(ep_steps,1):.4f} max_grad={max_grad_seen:.2f} ({ep_sec:.1f}s)", flush=True)

    train_wall = time.perf_counter() - t0
    model.eval()

    @torch.no_grad()
    def _predict_split_logits(tiles_list: List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
        out_lgs: List[np.ndarray] = []
        for z in tiles_list:
            spans = _day_spans(z["day_ids"], z["timestamps_unix_s"])
            lg_tile = np.empty(z["truth"].shape, dtype=np.float32)
            scache = None
            for _, f0, f1 in spans:
                out, _, _, scache = _forward_tile_day(z, f0, f1, static_cache=scache)
                lg_tile[f0:f1] = out["logits"].detach().to("cpu", dtype=torch.float32).numpy()
            out_lgs.append(lg_tile)
        return out_lgs

    cal_lgs = _predict_split_logits(cal_tiles)
    best_temp, best_thresh, cal_nll = calibrate_logits_wide(
        cal_lgs, [z["truth"] for z in cal_tiles], [z["valid_mask"] for z in cal_tiles]
    )
    print(f"  [{arm_key}] Calibrated on 12 tiles: T={best_temp:.4f}, tau*={best_thresh:.3f}, cal_NLL={cal_nll:.5f}", flush=True)

    val_lgs = _predict_split_logits(val_tiles)

    # Evaluate using the unified standalone referee metrics bundle
    from evaluate_predictions import compute_metrics_bundle

    tp_s = fp_s = tn_s = fn_s = pf_s = 0
    ed_pf = ed_ok = hs_pf = hs_ok = hs_sun = hs_hit = 0
    ev_tot = ev_hit = 0
    lags_all: List[float] = []
    brier_sum = 0.0
    per_tile_rows: List[Dict[str, Any]] = []

    for z, lg in zip(val_tiles, val_lgs):
        probs = (1.0 / (1.0 + np.exp(-np.clip(lg.astype(np.float64) / max(best_temp, 1e-4), -60.0, 60.0)))).astype(np.float32)
        mb = compute_metrics_bundle(
            probs, z["truth"], z["valid_mask"], best_thresh,
            z["day_ids"], z["timestamps_unix_s"],
            z["boundary_distance_m"], z["boundary_distance_valid"],
            z["shared_az0_cand_nonempty"],
        )
        tp_s += mb["tp"]; fp_s += mb["fp"]; tn_s += mb["tn"]; fn_s += mb["fn"]; pf_s += mb["point_frames"]
        brier_sum += mb["brier"] * mb["point_frames"]
        ed_pf += mb["edge_lt0p5m_pf"]
        ed_ok += int(round((mb["edge_lt0p5m_acc"] or 0.0) * mb["edge_lt0p5m_pf"]))

        vm = z["valid_mask"] & (z["truth"] >= 0)
        hm = vm & z["shared_az0_cand_nonempty"]
        hs_pf += int(hm.sum())
        hs_ok += int(((probs >= best_thresh) == (z["truth"] == 1))[hm].sum())
        hms = hm & (z["truth"] == 1)
        hs_sun += int(hms.sum())
        hs_hit += int((probs[hms] >= best_thresh).sum())

        ev_tot += mb["true_events"]
        ev_hit += int(round((mb["event_recall_pm2min"] or 0.0) * mb["true_events"]))
        if mb["median_lag_min"] is not None:
            lags_all.append(mb["median_lag_min"])
        per_tile_rows.append({
            "tile_id": str(z["tile_id"].item()),
            "historical_exposure_status": str(z["historical_exposure_status"].item()),
            "state_acc": mb["state_acc"],
            "state_error": mb["state_error"],
            "insolation_bias": mb["insolation_bias"],
            "edge_lt0p5m_acc": mb["edge_lt0p5m_acc"],
            "event_recall_pm2min": mb["event_recall_pm2min"],
        })

    torch.save({
        "arm_key": arm_key,
        "parameter_count": n_params,
        "state_dict": model.state_dict(),
        "calibrated_temperature": best_temp,
        "calibrated_threshold": best_thresh,
        "seed": seed,
        "total_steps": total_steps,
        "trained_point_frames": total_pf,
    }, out_ckpt)
    ckpt_sha = hashlib.sha256(out_ckpt.read_bytes()).hexdigest()

    overall_acc = (tp_s + tn_s) / pf_s
    summary = {
        "arm_key": arm_key,
        "parameter_count": n_params,
        "checkpoint_path": str(out_ckpt),
        "checkpoint_sha256": ckpt_sha,
        "precision": "PURE_IEEE_FP32",
        "bf16_amp_active": False,
        "nonfinite_steps": nonfinite_steps,
        "total_steps": total_steps,
        "trained_point_frames": total_pf,
        "max_grad_l2": max_grad_seen,
        "train_wall_seconds": round(train_wall, 2),
        "epoch_logs": epoch_logs,
        "calibrated_temperature": best_temp,
        "calibrated_threshold": best_thresh,
        "calibration_bce_nll": cal_nll,
        "val_spatial_clean": {
            "valid_point_frames": pf_s,
            "state_acc": overall_acc,
            "state_error": 1.0 - overall_acc,
            "state_f1": (2 * tp_s) / max(2 * tp_s + fp_s + fn_s, 1),
            "sun_iou": tp_s / max(tp_s + fp_s + fn_s, 1),
            "brier": brier_sum / pf_s,
            "mean_tile_insolation_bias": float(np.mean([r["insolation_bias"] for r in per_tile_rows])),
            "edge_lt0p5m_pf": ed_pf,
            "edge_lt0p5m_acc": ed_ok / max(ed_pf, 1),
            "shared_az0_hardsun_pf": hs_pf,
            "shared_az0_hardsun_acc": hs_ok / max(hs_pf, 1),
            "shared_az0_hardsun_recall": hs_hit / max(hs_sun, 1),
            "true_events": ev_tot,
            "event_recall_pm2min": ev_hit / max(ev_tot, 1),
            "median_tile_lag_min": float(np.median(lags_all)) if lags_all else None,
            "per_tile": per_tile_rows,
        },
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    v = summary["val_spatial_clean"]
    print(
        f"==> DONE [{arm_key}]: State Err={v['state_error']*100:.2f}% | Sun IoU={v['sun_iou']*100:.2f}% | "
        f"<0.5m Edge={v['edge_lt0p5m_acc']*100:.2f}% | Bias={v['mean_tile_insolation_bias']*100:.2f}% | "
        f"Recall@±2m={v['event_recall_pm2min']*100:.1f}%",
        flush=True,
    )
    return summary


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=str, default="probe_canonical37_mlp,ext_dsm25d_solar_knn_gru,ext_pointnet_solar_film_gru,az24_tier_g0_pure_xyz")
    ap.add_argument("--cache-dir", type=str, default=str(BUNDLE_ROOT / "data" / "az24_r256"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--finalize-only", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    torch.set_num_threads(int(args.threads))
    dev = torch.device("cpu")

    all_four = [
        "probe_canonical37_mlp",
        "ext_dsm25d_solar_knn_gru",
        "ext_pointnet_solar_film_gru",
        "az24_tier_g0_pure_xyz",
    ]
    if not args.finalize_only:
        r256_dir = Path(args.cache_dir)
        all_npz = sorted(r256_dir.glob("core_*.npz"))
        assert len(all_npz) == 93, f"Expected 93 cached shards in {r256_dir}, found {len(all_npz)}"

        fit_tiles: List[Dict[str, np.ndarray]] = []
        cal_tiles: List[Dict[str, np.ndarray]] = []
        val_tiles: List[Dict[str, np.ndarray]] = []
        for p in all_npz:
            z = {k: v for k, v in np.load(p, allow_pickle=False).items()}
            role = str(z["role"].item())
            if role == "fit":
                fit_tiles.append(z)
            elif role == "calibration":
                cal_tiles.append(z)
            elif role == "val_spatial_clean":
                val_tiles.append(z)

        fit_tiles.sort(key=lambda x: str(x["tile_id"].item()))
        cal_tiles.sort(key=lambda x: str(x["tile_id"].item()))
        val_tiles.sort(key=lambda x: str(x["tile_id"].item()))
        assert (len(fit_tiles), len(cal_tiles), len(val_tiles)) == (67, 12, 14)

        req_arms = [x.strip() for x in args.arms.split(",") if x.strip()]
        for ak in req_arms:
            run_one_arm(ak, fit_tiles, cal_tiles, val_tiles, dev=dev, epochs=3, seed=20261114)
            gc.collect()

    if all((P4_ROOT / "results" / f"fairness_{ak}.json").exists() for ak in all_four):
        results = {
            ak: json.loads((P4_ROOT / "results" / f"fairness_{ak}.json").read_text(encoding="utf-8"))
            for ak in all_four
        }
        master_out = P4_ROOT / "results" / "FAIRNESS_BASELINES_93TILE_SUMMARY.json"
        master_out.write_text(json.dumps({
            "protocol": "MANHATTAN_93TILE_FIT67_CAL12_VAL14_PURE_FP32_804STEPS_SEED20261114",
            "device": str(dev),
            "arms": results,
            "total_wall_seconds": round(time.perf_counter() - t0, 2),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nALL 4 FAIRNESS ARMS MERGED -> {master_out}", flush=True)


if __name__ == "__main__":
    main()

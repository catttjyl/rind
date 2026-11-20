"""Unified Self-Contained Model Zoo & Checkpoint Contract (`MANHATTAN_4445_ANNUAL_18808_V1`).

Zero-Shrinkage Unified Architecture combining the ST-JEPA Highlight Book (`V1..V5`),
Generation-1 (`RIND` Shared-Geometry JEPA), Generation-2 (`rind_v09_clean`), and the
`MANHATTAN_4445_ANNUAL_18808_V1` (`FitFirewallToken`, 52-week `18,808`-frame) benchmark contract:

1. `rind_full_geom` (`Tier-G2_full_geom`):
   Full 37D `canonical37` candidate tokens (including projected edge signed margins `slots 22..24`)
   + globally smooth `C^inf` south-pole minimal-rotation hemisphere tangent frame `(e1, e2)`
   + real spherical harmonics `Y_l^m(s_t)` (`l <= 4`) + 2nd-order relative chord outer products
   (`FourierSphereTangentEncoder102`) + ray-conditioned anisotropic candidate attention bias `b_ray`
   + `StaticGeometryMemory` (`k`-NN spatial MPNN modulated by **Solar-FiLM**)
   + kinematic solar angular-velocity gated `BoundedSphereTangentCell` (`z_direct` + `z_roll`,
     label-free `receiver_obs_mask` per-receiver gap reset, `reanchor_every`, strided rollout)
   + `bce_head` + `boundary_head` + 7-cue detached `FocalEventTransitionGate`
   + valid-only `LocalValidNeighborhoodReadout`.
2. `rind_no_edge_margin` (`Tier-G1_no_edge_margin`):
   Strict single-factor projected edge-margin ablation: identical architecture and parameter count
   as `rind_full_geom`, EXCEPT `NoEdgeMarginSanitizer` slices around `slots 22..24` and fills
   `slots 22..24` with fresh exact zeros without reading them (100% immune to `NaN`/`Inf` in `22..24`).
3. `rind_zero_margin` (`Tier-G0_pure_raw_vertices`):
   Pure raw-vertex geometry ablation: `ZeroMarginSanitizer` retains ONLY static material/area
   `slots 10..11`, raw 3D triangle vertices `slots 13..21`, raw sun vector `slots 25..27`, and
   recomputes the face normal purely from raw vertices, while strictly disabling `local_features`,
   `StaticGeometryMemory`, and `LocalValidNeighborhoodReadout` (`0.0` sensitivity to all disabled slots).
4. `local_convgru_grid` (`Tier-B1_local_convgru`):
   Recurrent spatial 2D `ConvGRU` supporting both `"full_tile_512_halo"` (`O(1)` full-tile pass
   gathering exact `[rows, cols]` for all `N` receivers in `<1s/step`, `<0.6 GiB` VRAM even at
   `N = 27,000`) and `"per_receiver_crop"` (`64x64` or `21x21` 9-channel crops).
5. `pointwise_mlp` (`Tier-B0_pointwise_mlp`):
   Memoryless local receiver context + smooth Sphere-Tangent & Spherical-Harmonics MLP baseline.
6. `rind_coarse_aabb` (`Tier-G0b_coarse_aabb`, optional extended ablation in `ALL_MODEL_TYPES`):
   Coarse axis-aligned bounding-box candidate ablation retaining only `slots 10..11`, `25..27`,
   and AABB bounds `28..33` (`CoarseAABBSanitizer`).
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from . import BENCHMARK_CONTRACT_ID, FEATURE_SCHEMA_VERSION, PACKAGE_VERSION
from .split_contract import FitFirewallToken

CANONICAL_TOKEN_DIM = 37
LOCAL_CONTEXT_DIM = 16
GRID_CROP_CHANNELS = 9
LEGACY_GRID_CROP_CHANNELS = 8
DEFAULT_METERS_PER_PIXEL = 100.0 / 512.0  # 0.1953125 m on 100m / 512 grid

MODEL_TYPES: Tuple[str, ...] = (
    "rind_full_geom",
    "rind_no_edge_margin",
    "rind_zero_margin",
    "local_convgru_grid",
    "pointwise_mlp",
)

ALL_MODEL_TYPES: Tuple[str, ...] = (
    *MODEL_TYPES,
    "canonical37_linear",
    "canonical37_mlp",
    "rind_coarse_aabb",
    "full_tile_convgru",
)

ATTENTION_MODES: Tuple[str, ...] = ("ray_conditioned", "solar_gating")
INFERENCE_MODES: Tuple[str, ...] = ("recurrent", "static_direct", "causal_rollout", "offline_interp")


def compute_smooth_hemisphere_tangent_basis(
    sun_dirs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Globally `C^inf` smooth orthonormal tangent basis `(s_unit, e1, e2)` on upper hemisphere `S^2_+`.

    Uses the minimal-rotation (south-pole stereographic) transport from `+e_y = (0, 1, 0)` to `s_unit`:
        e1 = (1 - sx^2 / (1 + sy), -sx, -sx * sz / (1 + sy))
        e2 = (-sx * sz / (1 + sy), -sz, 1 - sz^2 / (1 + sy))
    Since `1 + sy >= 1` for all `sy >= 0` (including exact zenith `sy = 1` and horizon `sy = 0`),
    this frame has ZERO branch switches at zenith and ZERO `abs(sy)` derivative kinks at the horizon.
    """
    work_dtype = (
        sun_dirs.dtype
        if (sun_dirs.is_floating_point() and sun_dirs.dtype in (torch.float32, torch.float64))
        else torch.float32
    )
    eps = 1e-12 if work_dtype == torch.float64 else 1e-6
    s = sun_dirs.to(work_dtype)
    s_unit = s / torch.linalg.norm(s, dim=-1, keepdim=True).clamp_min(eps)
    sx = s_unit[..., 0:1]
    sy = s_unit[..., 1:2]
    sz = s_unit[..., 2:3]
    denom = (1.0 + sy).clamp_min(1e-4)
    inv_d = 1.0 / denom

    e1 = torch.cat([1.0 - (sx * sx) * inv_d, -sx, -(sx * sz) * inv_d], dim=-1)
    e2 = torch.cat([(sx * sz) * inv_d, sz, (sz * sz) * inv_d - 1.0], dim=-1)
    # Re-normalize defensively for extreme sub-horizon inputs `sy -> -1` (`e1 x e2 == +s_unit`)
    e1 = e1 / torch.linalg.norm(e1, dim=-1, keepdim=True).clamp_min(eps)
    e2 = e2 - (e2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = e2 / torch.linalg.norm(e2, dim=-1, keepdim=True).clamp_min(eps)
    return s_unit, e1, e2


def eval_real_spherical_harmonics_l4(s_unit: torch.Tensor) -> torch.Tensor:
    """Evaluate 24 non-constant real spherical harmonic polynomials `l = 1..4` on unit `s_unit` `(..., 3)`.

    All polynomial basis functions are globally `C^inf` on `S^2` (Y-up convention `(x, y, z)`).
    """
    x = s_unit[..., 0]
    y = s_unit[..., 1]
    z = s_unit[..., 2]
    x2 = x * x
    y2 = y * y
    z2 = z * z

    # l = 1 (3)
    l1 = [x, y, z]
    # l = 2 (5)
    l2 = [
        x * z,
        x * y,
        z * y,
        x2 - z2,
        3.0 * y2 - 1.0,
    ]
    # l = 3 (7)
    l3 = [
        z * (3.0 * x2 - z2),
        x * y * z,
        z * (5.0 * y2 - 1.0),
        y * (5.0 * y2 - 3.0),
        x * (5.0 * y2 - 1.0),
        y * (x2 - z2),
        x * (x2 - 3.0 * z2),
    ]
    # l = 4 (9)
    l4 = [
        x * z * (x2 - z2),
        z * y * (3.0 * x2 - z2),
        x * z * (7.0 * y2 - 1.0),
        z * y * (7.0 * y2 - 3.0),
        35.0 * y2 * y2 - 30.0 * y2 + 3.0,
        x * y * (7.0 * y2 - 3.0),
        (x2 - z2) * (7.0 * y2 - 1.0),
        x * y * (x2 - 3.0 * z2),
        x2 * (x2 - 3.0 * z2) - z2 * (3.0 * x2 - z2),
    ]
    return torch.stack([*l1, *l2, *l3, *l4], dim=-1)


class RunningNormalizer(nn.Module):
    """Streaming chunked FP64 running normalizer strictly gated by `FitFirewallToken`.

    Processes at most `max_rows_per_chunk` rows (`65,536` rows < 20 MiB) per chunk in `float64`,
    maintaining persistent `mean64`, `m2_64`, and `count` buffers for exact streaming parity
    without allocating multi-GiB monolithic `float64` copies.
    """

    def __init__(self, dim: int, eps: float = 1e-5, max_rows_per_chunk: int = 65536) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.max_rows_per_chunk = max(int(max_rows_per_chunk), 256)
        self.register_buffer("mean", torch.zeros(self.dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(self.dim, dtype=torch.float32))
        self.register_buffer("mean64", torch.zeros(self.dim, dtype=torch.float64))
        self.register_buffer("m2_64", torch.zeros(self.dim, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def _merge_batch_stats(self, batch_n: int, batch_mean: torch.Tensor, batch_m2: torch.Tensor) -> None:
        if batch_n <= 0:
            return
        b_n = float(batch_n)
        if not bool(self.initialized.item()) or float(self.count.item()) == 0.0:
            self.mean64.copy_(batch_mean)
            self.m2_64.copy_(batch_m2)
            self.count.fill_(b_n)
            self.initialized.fill_(True)
        else:
            old_n = float(self.count.item())
            new_n = old_n + b_n
            delta = batch_mean - self.mean64
            self.mean64.add_(delta * (b_n / new_n))
            self.m2_64.add_(batch_m2 + delta.square() * (old_n * b_n / new_n))
            self.count.fill_(new_n)

        cur_n = max(float(self.count.item()), 1.0)
        cur_var = (self.m2_64 / cur_n).clamp_min(self.eps)
        self.mean.copy_(self.mean64.to(torch.float32))
        self.var.copy_(cur_var.to(torch.float32))

    @torch.no_grad()
    def update(
        self,
        x: torch.Tensor,
        *,
        firewall_token: FitFirewallToken,
        tile_ids: Sequence[str],
        expected_split_sha256: str,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if firewall_token is None:
            raise RuntimeError("RunningNormalizer.update requires a valid FitFirewallToken!")
        firewall_token.verify_tiles(tile_ids, expected_split_sha256)

        if x.numel() == 0:
            return
        if x.shape[-1] != self.dim:
            raise ValueError(f"RunningNormalizer expected last dim {self.dim}, got {x.shape[-1]}")

        flat_view = x.detach().reshape(-1, self.dim)
        total_rows = int(flat_view.shape[0])

        vm_flat: Optional[torch.Tensor] = None
        if valid_mask is not None:
            vm = valid_mask.detach().to(dtype=torch.bool)
            while vm.ndim < x.ndim - 1:
                vm = vm.unsqueeze(-1)
            vm_expanded = vm.expand(x.shape[:-1]).reshape(-1)
            vm_flat = vm_expanded

        step = self.max_rows_per_chunk
        for s_idx in range(0, total_rows, step):
            e_idx = min(total_rows, s_idx + step)
            sub = flat_view[s_idx:e_idx]
            if vm_flat is not None:
                sub_mask = vm_flat[s_idx:e_idx]
                if not bool(sub_mask.any().item()):
                    continue
                sub = sub[sub_mask]
            sub64 = sub.to(dtype=torch.float64)
            finite_rows = torch.isfinite(sub64).all(dim=-1)
            if not bool(finite_rows.all().item()):
                sub64 = sub64[finite_rows]
            n_sub = int(sub64.shape[0])
            if n_sub == 0:
                continue
            sub_mean = sub64.mean(dim=0)
            sub_m2 = (sub64 - sub_mean).square().sum(dim=0)
            self._merge_batch_stats(n_sub, sub_mean, sub_m2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized.item()):
            return x
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        std = torch.sqrt(self.var.clamp_min(self.eps))
        out = (x32 - self.mean) / std
        return out.to(orig_dtype)


class SphereTangentSolarEncoder(nn.Module):
    """Globally `C^inf` Sphere-Tangent + Real Spherical Harmonics (`l<=4`) + Multi-Scale Fourier encoder."""

    def __init__(self, out_dim: int, num_frequencies: int = 4) -> None:
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        freqs = (2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("frequencies", freqs, persistent=True)
        # 3 (s_unit) + 2 (smooth stereographic tangent) + 6 (e1, e2) + 24 (Y_l^m l<=4) + 6 * F
        in_dim = 35 + 6 * self.num_frequencies
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, sun_dirs: torch.Tensor) -> torch.Tensor:
        s_unit, e1, e2 = compute_smooth_hemisphere_tangent_basis(sun_dirs.to(torch.float32))
        sx = s_unit[..., 0:1]
        sy = s_unit[..., 1:2]
        sz = s_unit[..., 2:3]
        denom = (1.0 + sy).clamp_min(1e-4)
        tangent_xz = torch.cat([sx / denom, sz / denom], dim=-1)
        ylm = eval_real_spherical_harmonics_l4(s_unit)
        angles = s_unit.unsqueeze(-1) * self.frequencies
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(-2)
        raw = torch.cat([s_unit, tangent_xz, e1, e2, ylm, fourier], dim=-1)
        return self.proj(raw)


class NoEdgeMarginSanitizer(nn.Module):
    """IEEE-safe single-factor sanitizer for `rind_no_edge_margin` (`Tier-G1`).

    Slices `slots 0..21` and `slots 25..36` directly and inserts fresh exact zeros into
    `slots 22..24` (projected edge signed margins). Never reads or multiplies `slots 22..24`,
    so even `NaN`/`Inf` in `slots 22..24` cannot affect forward outputs or gradients.
    """

    def forward(self, tokens: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] != CANONICAL_TOKEN_DIM:
            raise ValueError(f"Expected {CANONICAL_TOKEN_DIM} token slots, got {tokens.shape[-1]}")
        t32 = tokens.to(torch.float32)
        head_0_21 = t32[..., :22]
        zeros_22_24 = torch.zeros(*t32.shape[:-1], 3, dtype=torch.float32, device=tokens.device)
        tail_25_36 = t32[..., 25:]
        sanitized = torch.cat([head_0_21, zeros_22_24, tail_25_36], dim=-1)
        mask_f = candidate_mask.unsqueeze(-1).to(torch.float32)
        return (sanitized * mask_f).to(tokens.dtype)


class ZeroMarginSanitizer(nn.Module):
    """IEEE-safe pure raw-vertex sanitizer for `rind_zero_margin` (`Tier-G0`).

    Slices ONLY:
      - `slots 10..11`: static material opacity & log triangle area
      - `slots 13..21`: raw 3D receiver-relative triangle vertices `(v0, v1, v2) / scale_m`
      - `slots 25..27`: raw incident sun unit vector `s_t`
    Recomputes the geometric triangle unit normal purely from `(v1 - v0) x (v2 - v0)` into `0..2`,
    and fills all ray-projected / margin / bbox slots (`3..9`, `12`, `22..24`, `28..36`) with
    fresh exact zeros without ever reading them.
    """

    def forward(self, tokens: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] != CANONICAL_TOKEN_DIM:
            raise ValueError(f"Expected {CANONICAL_TOKEN_DIM} token slots, got {tokens.shape[-1]}")

        t32 = tokens.to(torch.float32)
        static_mat_area = t32[..., 10:12]
        raw_rel_verts = t32[..., 13:22]
        raw_sun = t32[..., 25:28]

        verts3 = raw_rel_verts.reshape(*raw_rel_verts.shape[:-1], 3, 3)
        e1 = verts3[..., 1, :] - verts3[..., 0, :]
        e2 = verts3[..., 2, :] - verts3[..., 0, :]
        normal = torch.cross(e1, e2, dim=-1)
        normal_unit = normal / torch.linalg.norm(normal, dim=-1, keepdim=True).clamp_min(1e-8)

        zeros_7 = torch.zeros(*t32.shape[:-1], 7, dtype=torch.float32, device=tokens.device)
        zeros_1 = torch.zeros(*t32.shape[:-1], 1, dtype=torch.float32, device=tokens.device)
        zeros_3_margin = torch.zeros(*t32.shape[:-1], 3, dtype=torch.float32, device=tokens.device)
        zeros_9_tail = torch.zeros(*t32.shape[:-1], 9, dtype=torch.float32, device=tokens.device)

        sanitized = torch.cat(
            [
                normal_unit,      # 0..2  (3): pure static face normal from raw vertices
                zeros_7,          # 3..9  (7): along/radial/edge/incidence strictly zeroed
                static_mat_area,  # 10..11(2): static opacity + log area
                zeros_1,          # 12    (1): duplicate s_y zeroed
                raw_rel_verts,    # 13..21(9): raw 3D building triangle vertices
                zeros_3_margin,   # 22..24(3): projected edge signed margins STRICTLY ZEROED
                raw_sun,          # 25..27(3): incident sun unit vector
                zeros_9_tail,     # 28..36(9): bbox/degeneracy shortcuts zeroed
            ],
            dim=-1,
        )
        mask_f = candidate_mask.unsqueeze(-1).to(torch.float32)
        return (sanitized * mask_f).to(tokens.dtype)


class CoarseAABBSanitizer(nn.Module):
    """IEEE-safe coarse axis-aligned bounding-box sanitizer for `rind_coarse_aabb` (`Tier-G0b`).

    Slices ONLY static opacity/area `10..11`, incident sun `25..27`, and AABB bounds `28..33`,
    replacing exact vertices `13..21`, ray projections `0..9`, and edge margins `22..24` with fresh zeros.
    """

    def forward(self, tokens: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] != CANONICAL_TOKEN_DIM:
            raise ValueError(f"Expected {CANONICAL_TOKEN_DIM} token slots, got {tokens.shape[-1]}")
        t32 = tokens.to(torch.float32)
        static_mat_area = t32[..., 10:12]
        raw_sun = t32[..., 25:28]
        aabb_6 = t32[..., 28:34]
        z10 = torch.zeros(*t32.shape[:-1], 10, dtype=torch.float32, device=tokens.device)
        z13 = torch.zeros(*t32.shape[:-1], 13, dtype=torch.float32, device=tokens.device)
        z3 = torch.zeros(*t32.shape[:-1], 3, dtype=torch.float32, device=tokens.device)
        sanitized = torch.cat([z10, static_mat_area, z13, raw_sun, aabb_6, z3], dim=-1)
        mask_f = candidate_mask.unsqueeze(-1).to(torch.float32)
        return (sanitized * mask_f).to(tokens.dtype)


class FourierSphereTangentEncoder102(nn.Module):
    """Highlight 2 & 4 Candidate Encoder: Smooth Ray-Tangent Frame + 2nd-Order Chord Outer Products.

    Operates STRICTLY on post-sanitizer tokens `raw_sanitized_tok` and normalized `norm_tok`,
    so `Tier-G1` (`no_edge_margin`) and `Tier-G0` (`zero_margin`) slot immunities hold with `0.0` leakage!
    Extracts:
      - Smooth `(s_unit, e1, e2)` projected vertex coordinates `(u_i, v_i, w_i)` (`i = 0,1,2`, `9D`)
      - Tangent centroid `(u_c, v_c, w_c)` + radial distance `r_perp` (`4D`)
      - 2nd-order relative chord outer products in the `(e1, e2)` shadow plane (`6D` symmetric covariance
        capturing roof overhang span, facade aspect ratio, and edge orientation)
      - Roof vs vertical-facade normal cue (`|n_y|`, `1 - |n_y|`, `n . s_unit`, `3D`)
      - Metric triangle edge lengths, perimeter & isoperimetric compactness (`5D`)
      - Sub-meter ray-plane vertex polar phase cues `(sin(25 u_i), cos(25 v_i))` (`6D`)
      - Multi-scale log-compressed Fourier features on `(u_c, v_c, w_c, r_perp)` (`32D`)
      - Normalized `canonical37` features (`37D`)
      Total rich relational input = `102D` -> projected to `hidden_dim`.
    """

    RAW_ENC_DIM: int = 102

    def __init__(self, hidden_dim: int, num_frequencies: int = 4, far_perspective: bool = False) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_frequencies = int(num_frequencies)
        freqs = (2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("frequencies", freqs, persistent=True)
        in_dim = CANONICAL_TOKEN_DIM + 9 + 4 + 6 + 3 + 5 + 6 + 8 * self.num_frequencies
        self.in_dim = int(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.angle_projection = nn.Parameter(torch.zeros(self.hidden_dim, 6)) if far_perspective else None

    def forward(
        self,
        raw_sanitized_tok: torch.Tensor,
        norm_tok: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        enable_tangent_chords: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(cand_emb: (M, K, D), ray_cues: (M, K, 6))`."""
        m_batch, k_cands, _ = raw_sanitized_tok.shape
        mask_f = candidate_mask.unsqueeze(-1).to(torch.float32)

        if not enable_tangent_chords:
            zeros_extra = torch.zeros(
                (m_batch, k_cands, 33 + 8 * self.num_frequencies),
                dtype=norm_tok.dtype,
                device=norm_tok.device,
            )
            emb = self.net(torch.cat([norm_tok, zeros_extra], dim=-1)) * mask_f.to(norm_tok.dtype)
            zero_cues = torch.zeros((m_batch, k_cands, 10), dtype=torch.float32, device=norm_tok.device)
            return emb, zero_cues

        t32 = raw_sanitized_tok.to(torch.float32)
        verts = t32[..., 13:22].reshape(m_batch, k_cands, 3, 3)  # scaled by 1/1000m
        sun_raw = t32[..., 25:28]
        s_u, e1, e2 = compute_smooth_hemisphere_tangent_basis(sun_raw)

        u = (verts * e1.unsqueeze(-2)).sum(dim=-1)  # (M, K, 3)
        v = (verts * e2.unsqueeze(-2)).sum(dim=-1)  # (M, K, 3)
        w = (verts * s_u.unsqueeze(-2)).sum(dim=-1)  # along-ray depth (M, K, 3)
        uvw_flat = torch.cat([u, v, w], dim=-1)      # (M, K, 9)

        uc = u.mean(dim=-1, keepdim=True)
        vc = v.mean(dim=-1, keepdim=True)
        wc = w.mean(dim=-1, keepdim=True)
        r_perp = torch.sqrt(uc * uc + vc * vc + 1e-12)

        # Log-scaled metric coordinates (1.0 in `verts` == 1000m, so `* 50.0` resolves sub-meter ~20m canyons)
        def _slog(val: torch.Tensor) -> torch.Tensor:
            return torch.sign(val) * torch.log1p(val.abs() * 50.0)

        centroid_4 = torch.cat([_slog(uc), _slog(vc), _slog(wc), _slog(r_perp)], dim=-1)

        du = (u - uc) * 50.0
        dv = (v - vc) * 50.0
        dw = (w - wc) * 50.0
        cov_uu = _slog((du * du).mean(dim=-1, keepdim=True))
        cov_vv = _slog((dv * dv).mean(dim=-1, keepdim=True))
        cov_uv = _slog((du * dv).mean(dim=-1, keepdim=True))
        cov_uw = _slog((du * dw).mean(dim=-1, keepdim=True))
        cov_vw = _slog((dv * dw).mean(dim=-1, keepdim=True))
        cov_ww = _slog((dw * dw).mean(dim=-1, keepdim=True))
        chord_outer_6 = torch.cat([cov_uu, cov_vv, cov_uv, cov_uw, cov_vw, cov_ww], dim=-1)

        edge_a = verts[..., 1, :] - verts[..., 0, :]
        edge_b = verts[..., 2, :] - verts[..., 0, :]
        edge_c = verts[..., 2, :] - verts[..., 1, :]
        n_vec = torch.cross(edge_a, edge_b, dim=-1)
        tri_area2 = torch.linalg.norm(n_vec, dim=-1, keepdim=True)
        n_unit = n_vec / tri_area2.clamp_min(1e-8)
        abs_ny = n_unit[..., 1:2].abs()
        facade_ness = 1.0 - abs_ny
        signed_n_dot_s = (n_unit * s_u).sum(dim=-1, keepdim=True)
        cos_inc = signed_n_dot_s.abs()
        roof_facade_3 = torch.cat([abs_ny, facade_ness, cos_inc], dim=-1)

        len_01 = torch.linalg.norm(edge_a, dim=-1, keepdim=True)
        len_12 = torch.linalg.norm(edge_c, dim=-1, keepdim=True)
        len_20 = torch.linalg.norm(edge_b, dim=-1, keepdim=True)
        perim = len_01 + len_12 + len_20
        compactness = ((2.0 * math.sqrt(3.0) * tri_area2) / (perim * perim + 1e-8)).clamp(0.0, 1.5)
        metric_shape_5 = torch.cat(
            [_slog(len_01), _slog(len_12), _slog(len_20), _slog(perim), compactness], dim=-1
        )
        vertex_ray_polar_6 = torch.cat([torch.sin(u * 25.0), torch.cos(v * 25.0)], dim=-1)

        angles = centroid_4.unsqueeze(-1) * self.frequencies
        fourier_32 = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(-2)

        rich_in = torch.cat(
            [
                norm_tok.to(torch.float32),
                uvw_flat,
                centroid_4,
                chord_outer_6,
                roof_facade_3,
                metric_shape_5,
                vertex_ray_polar_6,
                fourier_32,
            ],
            dim=-1,
        ) * mask_f
        emb = self.net(rich_in.to(norm_tok.dtype))
        if self.angle_projection is not None:
            depth = w.clamp_min(1e-7)
            angles = torch.cat([torch.atan2(u, depth), torch.atan2(v, depth)], dim=-1)
            angles = angles * torch.cat([w > 0, w > 0], dim=-1)
            emb = emb + torch.nn.functional.linear(angles.to(norm_tok.dtype), self.angle_projection)
        emb = emb * mask_f.to(norm_tok.dtype)

        # Full 10D ray-geometry relation cues `\rho_{ij}(u)` for anisotropic attention bias:
        # `[wc_pos, -r_perp, cos_inc, margin_min, margin_mid, facade_ness, delta_az_cos, delta_az_sin, delta_elev, signed_n_dot_s]`
        c_xyz = verts.mean(dim=-2)  # (M, K, 3)
        horiz_c = torch.sqrt(c_xyz[..., 0:1].square() + c_xyz[..., 2:3].square() + 1e-12)
        horiz_s = torch.sqrt(s_u[..., 0:1].square() + s_u[..., 2:3].square() + 1e-12)
        cx_u = c_xyz[..., 0:1] / horiz_c
        cz_u = c_xyz[..., 2:3] / horiz_c
        sx_u = s_u[..., 0:1] / horiz_s
        sz_u = s_u[..., 2:3] / horiz_s
        delta_az_cos = cx_u * sx_u + cz_u * sz_u
        delta_az_sin = cx_u * sz_u - cz_u * sx_u
        dist_3d = torch.linalg.norm(c_xyz, dim=-1, keepdim=True).clamp_min(1e-8)
        delta_elev = (c_xyz[..., 1:2] / dist_3d).clamp(-1.0, 1.0) - s_u[..., 1:2].clamp(-1.0, 1.0)

        margins_3 = t32[..., 22:25]
        ray_cues = torch.cat(
            [
                F.relu(_slog(wc)),
                -_slog(r_perp),
                cos_inc,
                margins_3[..., 0:1],
                margins_3[..., 1:2],
                facade_ness,
                delta_az_cos,
                delta_az_sin,
                delta_elev,
                signed_n_dot_s,
            ],
            dim=-1,
        ) * mask_f
        return emb, ray_cues


class RayAnisotropicAttentionBias(nn.Module):
    """Highlight 4 (`04_st_jepa_feedback_v260807.pdf`): Ray-Conditioned Anisotropic Attention Logit Bias."""

    RAY_CUE_DIM: int = 10

    def __init__(self, num_heads: int, cue_dim: int = 10) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.cue_dim = int(cue_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.cue_dim, 24),
            nn.GELU(),
            nn.Linear(24, self.num_heads),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        ray_cues: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        enable_ray_bias: bool = True,
    ) -> torch.Tensor:
        """Compute `(M * H, 1, K + 1)` float32 additive attention mask with `0` on null slot."""
        m_batch, k_cands, c_dim = ray_cues.shape
        if not enable_ray_bias:
            raw_b = torch.zeros((m_batch, self.num_heads, k_cands), dtype=torch.float32, device=ray_cues.device)
        else:
            cues32 = ray_cues.to(torch.float32)
            if c_dim < self.cue_dim:
                pad = torch.zeros((m_batch, k_cands, self.cue_dim - c_dim), dtype=torch.float32, device=cues32.device)
                cues32 = torch.cat([cues32, pad], dim=-1)
            elif c_dim > self.cue_dim:
                cues32 = cues32[..., : self.cue_dim]
            raw_b = 4.0 * torch.tanh(self.mlp(cues32) / 4.0).permute(0, 2, 1)  # (M, H, K)
        neg_inf = torch.full_like(raw_b, -1e9)
        masked_b = torch.where(candidate_mask.unsqueeze(1), raw_b, neg_inf)
        null_col = torch.zeros((m_batch, self.num_heads, 1), dtype=torch.float32, device=ray_cues.device)
        full_b = torch.cat([masked_b, null_col], dim=-1)  # (M, H, K + 1)
        return full_b.reshape(m_batch * self.num_heads, 1, k_cands + 1)


def _chunked_spatial_knn_3d(
    pos: torch.Tensor,
    obs: torch.Tensor,
    k_eff: int,
    *,
    query_chunk: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact row-chunked 3D Euclidean `k`-NN (`< 55 MiB` at `N = 27,000`, `0` `UNKNOWN` contamination)."""
    bsz, n_pts, _ = pos.shape
    dev = pos.device
    all_idx = torch.arange(n_pts, device=dev)
    tie = all_idx.to(torch.float32).view(1, 1, n_pts) * (1e-6 / float(max(n_pts, 1)))

    dist_parts: List[torch.Tensor] = []
    idx_parts: List[torch.Tensor] = []
    obs_parts: List[torch.Tensor] = []
    q_step = max(1, int(query_chunk))
    for q0 in range(0, n_pts, q_step):
        q1 = min(n_pts, q0 + q_step)
        q_len = q1 - q0
        d_blk = torch.cdist(pos[:, q0:q1].to(torch.float32), pos.to(torch.float32), p=2.0)
        self_mask = (
            torch.arange(q0, q1, device=dev)[:, None] == all_idx[None, :]
        ).unsqueeze(0)
        invalid_nbr = self_mask | (~obs.unsqueeze(1))
        d_sort = torch.where(invalid_nbr, torch.full_like(d_blk, 1e9), d_blk) + tie
        _, idx_b = torch.topk(d_sort, k=k_eff, dim=-1, largest=False)
        dist_b = torch.gather(d_blk, -1, idx_b)
        obs_b = torch.gather(obs.unsqueeze(1).expand(bsz, q_len, n_pts), -1, idx_b)
        dist_parts.append(dist_b)
        idx_parts.append(idx_b)
        obs_parts.append(obs_b)

    return (
        torch.cat(dist_parts, dim=1),
        torch.cat(idx_parts, dim=1),
        torch.cat(obs_parts, dim=1),
    )


def compute_tile_global_knn_pack_torch(
    receivers_xyz: torch.Tensor,
    receiver_obs_mask: torch.Tensor,
    local_features: torch.Tensor,
    *,
    k_neighbors: int = 8,
    query_chunk: int = 512,
) -> Dict[str, torch.Tensor]:
    """Precompute tile-global valid `k`-NN static geometry descriptors so spatial modules are 100% chunk-invariant."""
    pos = receivers_xyz.to(torch.float32)
    obs = receiver_obs_mask.to(torch.bool)
    lf = local_features.to(torch.float32)
    n_pts = int(pos.shape[0])
    l_dim = int(lf.shape[-1])
    k_req = max(1, int(k_neighbors))
    dev = pos.device

    out_lf = torch.zeros((n_pts, k_req, l_dim), dtype=torch.float32, device=dev)
    out_rel = torch.zeros((n_pts, k_req, 3), dtype=torch.float32, device=dev)
    out_dist = torch.full((n_pts, k_req), 1e9, dtype=torch.float32, device=dev)
    out_obs = torch.zeros((n_pts, k_req), dtype=torch.bool, device=dev)
    if n_pts <= 1:
        return {
            "knn_local_features": out_lf,
            "knn_rel_xyz": out_rel,
            "knn_dist": out_dist,
            "knn_obs_valid": out_obs,
        }

    k_eff = min(k_req, n_pts - 1)
    with torch.no_grad():
        knn_d, knn_i, knn_o = _chunked_spatial_knn_3d(
            pos.unsqueeze(0), obs.unsqueeze(0), k_eff, query_chunk=query_chunk
        )
    idx2 = knn_i[0]  # (N, k_eff)
    nbr_pos = pos[idx2]  # (N, k_eff, 3)
    out_lf[:, :k_eff, :] = lf[idx2]
    out_rel[:, :k_eff, :] = nbr_pos - pos.unsqueeze(1)
    out_dist[:, :k_eff] = knn_d[0]
    out_obs[:, :k_eff] = knn_o[0]
    return {
        "knn_local_features": out_lf,
        "knn_rel_xyz": out_rel,
        "knn_dist": out_dist,
        "knn_obs_valid": out_obs,
    }


class StaticGeometryMemory(nn.Module):
    """Highlight 1: Static Receiver Geometry Memory (`k`-NN MPNN) + Dynamic **Solar-FiLM** Modulation.

    Encodes static local urban geometry ONCE per receiver chunk (`O(N)` instead of `O(T*N)`),
    masks unobserved points via label-free `receiver_obs_mask`, supports exact tile-global `k`-NN
    neighbor packs for 100% `receiver_chunk_size` invariance, and modulates static memory `Z_G`
    at each solar step `t` via `tilde_Z_G(u_t) = (1 + tanh(gamma(u_t))) * Z_G + beta(u_t)`.
    """

    def __init__(self, local_dim: int, hidden_dim: int, k_neighbors: int = 8, film_beta_bound: float = 0.0) -> None:
        super().__init__()
        self.local_dim = int(local_dim)
        self.hidden_dim = int(hidden_dim)
        self.k_neighbors = int(k_neighbors)
        self.film_beta_bound = float(film_beta_bound)
        if not math.isfinite(self.film_beta_bound) or self.film_beta_bound < 0.0:
            raise ValueError("film_beta_bound must be finite and non-negative")

        self.node_stem = nn.Sequential(
            nn.Linear(self.local_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.edge_msg = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.edge_gate = nn.Linear(self.hidden_dim * 2 + 4, 1)
        self.out_ln = nn.LayerNorm(self.hidden_dim)

        self.solar_film = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
        )
        nn.init.zeros_(self.solar_film[-1].weight)
        nn.init.zeros_(self.solar_film[-1].bias)
        nn.init.zeros_(self.edge_msg[-1].weight)
        nn.init.zeros_(self.edge_msg[-1].bias)

    def encode_static(
        self,
        norm_local: torch.Tensor,
        *,
        receivers_xyz: Optional[torch.Tensor] = None,
        receiver_rc: Optional[torch.Tensor] = None,
        receiver_obs_static: Optional[torch.Tensor] = None,
        global_knn_norm_local: Optional[torch.Tensor] = None,
        global_knn_rel_xyz: Optional[torch.Tensor] = None,
        global_knn_dist: Optional[torch.Tensor] = None,
        global_knn_obs_valid: Optional[torch.Tensor] = None,
        meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
        enable_mpnn: bool = True,
    ) -> torch.Tensor:
        """Compute static memory `Z_G` `(B, N, D)` once per tile/chunk with exact spatial `k`-NN."""
        bsz, n_pts, _ = norm_local.shape
        h0 = self.node_stem(norm_local)
        if (not enable_mpnn) or self.k_neighbors <= 0 or n_pts == 0:
            return self.out_ln(h0)

        if (
            global_knn_norm_local is not None
            and global_knn_rel_xyz is not None
            and global_knn_dist is not None
            and global_knn_obs_valid is not None
        ):
            gnl = global_knn_norm_local.to(dtype=norm_local.dtype, device=norm_local.device)
            grx = global_knn_rel_xyz.to(dtype=torch.float32, device=norm_local.device)
            gds = global_knn_dist.to(dtype=torch.float32, device=norm_local.device)
            gob = global_knn_obs_valid.to(dtype=torch.bool, device=norm_local.device)
            if gnl.ndim == 3:
                gnl = gnl.unsqueeze(0).expand(bsz, -1, -1, -1)
            if grx.ndim == 3:
                grx = grx.unsqueeze(0).expand(bsz, -1, -1, -1)
            if gds.ndim == 2:
                gds = gds.unsqueeze(0).expand(bsz, -1, -1)
            if gob.ndim == 2:
                gob = gob.unsqueeze(0).expand(bsz, -1, -1)
            k_eff = min(self.k_neighbors, int(gnl.shape[2]))
            if k_eff <= 0:
                return self.out_ln(h0)
            nbr_h_g = self.node_stem(gnl[:, :, :k_eff, :])
            rel_xyz = grx[:, :, :k_eff, :] / 20.0
            knn_dist = gds[:, :, :k_eff]
            nbr_obs = gob[:, :, :k_eff]
        else:
            if n_pts <= 1:
                return self.out_ln(h0)
            if receivers_xyz is not None:
                pos = receivers_xyz.to(dtype=torch.float32, device=norm_local.device)
                if pos.ndim == 2:
                    pos = pos.unsqueeze(0).expand(bsz, -1, -1)
            elif receiver_rc is not None:
                rc = receiver_rc.to(dtype=torch.float32, device=norm_local.device)
                if rc.ndim == 2:
                    rc = rc.unsqueeze(0).expand(bsz, -1, -1)
                zeros_y = torch.zeros((bsz, n_pts, 1), dtype=torch.float32, device=norm_local.device)
                pos = torch.cat(
                    [rc[..., 1:2] * meters_per_pixel, zeros_y, rc[..., 0:1] * meters_per_pixel], dim=-1
                )
            else:
                return self.out_ln(h0)

            if receiver_obs_static is not None:
                obs = receiver_obs_static.to(dtype=torch.bool, device=norm_local.device)
                if obs.ndim == 1:
                    obs = obs.unsqueeze(0).expand(bsz, -1)
            else:
                obs = torch.ones((bsz, n_pts), dtype=torch.bool, device=norm_local.device)

            k_eff = min(self.k_neighbors, n_pts - 1)
            knn_dist, knn_idx, nbr_obs = _chunked_spatial_knn_3d(pos, obs, k_eff)
            flat_idx = knn_idx.reshape(bsz, n_pts * k_eff, 1)
            nbr_pos_g = torch.gather(pos, 1, flat_idx.expand(bsz, n_pts * k_eff, 3)).reshape(
                bsz, n_pts, k_eff, 3
            )
            nbr_h_g = torch.gather(
                h0, 1, flat_idx.expand(bsz, n_pts * k_eff, self.hidden_dim)
            ).reshape(bsz, n_pts, k_eff, self.hidden_dim)
            rel_xyz = (nbr_pos_g - pos.unsqueeze(2)) / 20.0

        valid_edge = (knn_dist < 50.0) & nbr_obs
        rel_d = torch.log1p(knn_dist.clamp_min(0.0)).unsqueeze(-1) / 4.0
        edge_in = torch.cat(
            [
                h0.unsqueeze(2).expand(bsz, n_pts, k_eff, self.hidden_dim),
                nbr_h_g,
                rel_xyz.to(h0.dtype),
                rel_d.to(h0.dtype),
            ],
            dim=-1,
        )
        msg = self.edge_msg(edge_in)
        w = torch.sigmoid(self.edge_gate(edge_in)) * valid_edge.unsqueeze(-1).to(h0.dtype)
        agg = (msg * w).sum(dim=2) / w.sum(dim=2).clamp_min(1.0)
        return self.out_ln(h0 + agg)

    def modulate_solar_film(self, z_static: torch.Tensor, solar_emb: torch.Tensor) -> torch.Tensor:
        """Apply Solar-FiLM `(B, N, D)` x `(B, T, D)` -> `(B, T, N, D)`."""
        film = self.solar_film(solar_emb)  # (B, T, 2D)
        gamma_raw, beta = torch.chunk(film, 2, dim=-1)
        scale = 1.0 + torch.tanh(gamma_raw).unsqueeze(2)
        if self.film_beta_bound > 0.0:
            beta = self.film_beta_bound * torch.tanh(beta / self.film_beta_bound)
        shift = beta.unsqueeze(2)
        return scale * z_static.unsqueeze(1) + shift

    def solar_film_scale(self, solar_emb: torch.Tensor) -> torch.Tensor:
        """Return Solar-FiLM multiplicative gain `1 + tanh(gamma(u_t))` `(B, T, 1, D)`."""
        film = self.solar_film(solar_emb)
        gamma_raw, _ = torch.chunk(film, 2, dim=-1)
        return 1.0 + torch.tanh(gamma_raw).unsqueeze(2)


class BoundedSphereTangentCell(nn.Module):
    """Mathematically bounded convex-gated recurrence cell with kinematic solar angular-velocity gate."""

    def __init__(self, hidden_dim: int, max_coord_bound: float = 1.0) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_coord_bound = float(max_coord_bound)
        self.input_ln = nn.LayerNorm(self.hidden_dim)
        self.init_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proposal_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate_proj = nn.Linear(self.hidden_dim * 2 + 2, self.hidden_dim)

    @property
    def theoretical_l2_bound(self) -> float:
        return self.max_coord_bound * math.sqrt(float(self.hidden_dim))

    def init_state(self, h0: torch.Tensor) -> torch.Tensor:
        h0_32 = self.input_ln(h0.to(torch.float32))
        return self.max_coord_bound * torch.tanh(self.init_proj(h0_32))

    def step(
        self,
        h_t: torch.Tensor,
        z_prev: torch.Tensor,
        solar_kinematics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h32 = self.input_ln(h_t.to(torch.float32))
        z_prev32 = z_prev.to(torch.float32)
        if solar_kinematics is None:
            kin = torch.zeros((*h32.shape[:-1], 2), dtype=torch.float32, device=h32.device)
        else:
            kin = solar_kinematics.to(dtype=torch.float32, device=h32.device)
            while kin.ndim < h32.ndim:
                kin = kin.unsqueeze(-2)
            kin = kin.expand(*h32.shape[:-1], 2)
        proposal = self.max_coord_bound * torch.tanh(self.proposal_proj(h32))
        gate = torch.sigmoid(self.gate_proj(torch.cat([h32, z_prev32, kin], dim=-1)))
        return (1.0 - gate) * z_prev32 + gate * proposal


class FocalEventTransitionGate(nn.Module):
    """Zero-Initialized Detached Event-Sensitive Readout Gate (`P0-4` Calibration-Preserving Theorem).

    Combines detached boundary/transition cues `xi_t` so sparse rare-event Focal gradients never
    corrupt the recurrent trunk while `boundary_head` retains clean metric supervision:
      1. `z_det` (`D`)
      2. `delta_z_det` (`D`, strictly zeroed across non-120s / cross-day / unobserved gaps)
      3. `1 - 2 * |p_bce - 0.5|` (`1`: interior uncertainty)
      4. `tanh(|logits_boundary_det| / 4)` (`1`: boundary proximity confidence)
      5. `logits_boundary_det` (`1`: detached signed boundary proxy)
      6. `||delta_z_det||_2` (`1`: latent velocity norm)
      7. `solar_angular_step` (`1`: physical sun step speed)
      8. `local_height_relief` (`1`: static canyon relief norm)
    The residual linear head `(w_res, b_res)` is initialized strictly to `0.0`, guaranteeing
    `logits_fused == logits_bce` (`max |diff| == 0.0`) and `d(logits_fused)/d(logits_bce) == 1.0`
    at initialization (`Proposition: Exact Calibration-Preserving Initialization`).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        in_dim = hidden_dim * 2 + 6
        self.in_dim = int(in_dim)
        self.gate_net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.res_linear = nn.Linear(in_dim, 1)
        # Conservative zero-weight + negative-bias gate probability (`sigmoid(-1.0) ~= 0.269`)
        # paired with strictly zero-initialized `(w_res = 0, b_res = 0)` residual correction!
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, -1.0)
        nn.init.zeros_(self.res_linear.weight)
        nn.init.zeros_(self.res_linear.bias)

    def forward(
        self,
        z_seq: torch.Tensor,
        logits_bce: torch.Tensor,
        logits_boundary: torch.Tensor,
        same_day_adjacent_mask: Optional[torch.Tensor] = None,
        prev_chunk_latent: Optional[torch.Tensor] = None,
        solar_angular_speed: Optional[torch.Tensor] = None,
        local_relief: Optional[torch.Tensor] = None,
        *,
        obs_mask: Optional[torch.Tensor] = None,
        prev_chunk_obs_mask: Optional[torch.Tensor] = None,
        disable_gate: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if disable_gate:
            fused_off = logits_bce.to(torch.float32)
            return fused_off, torch.zeros_like(fused_off)

        z32 = z_seq.to(torch.float32)
        has_batch = z32.ndim == 4
        if not has_batch:
            z32 = z32.unsqueeze(0)
            logits_bce = logits_bce.unsqueeze(0)
            logits_boundary = logits_boundary.unsqueeze(0)

        bsz, t_steps, n_pts, _ = z32.shape
        obs_btn: Optional[torch.Tensor] = None
        if obs_mask is not None:
            om = obs_mask.detach().to(dtype=torch.bool, device=z32.device)
            if om.ndim == 1:
                obs_btn = om.view(1, 1, n_pts).expand(bsz, t_steps, n_pts)
            elif om.ndim == 2:
                if int(om.shape[0]) == t_steps and int(om.shape[1]) == n_pts and bsz == 1:
                    obs_btn = om.unsqueeze(0)
                else:
                    obs_btn = om.unsqueeze(1).expand(bsz, t_steps, n_pts)
            elif om.ndim == 3:
                obs_btn = om.expand(bsz, t_steps, n_pts)

        z_det = z32.detach()
        delta_z = torch.zeros_like(z_det)
        if prev_chunk_latent is not None and t_steps >= 1:
            p32 = prev_chunk_latent.detach().to(dtype=torch.float32, device=z32.device)
            if p32.ndim == 2:
                p32 = p32.unsqueeze(0)
            dz0 = z_det[:, 0] - p32
            if obs_btn is not None:
                valid0 = obs_btn[:, 0]
                if prev_chunk_obs_mask is not None:
                    pm = prev_chunk_obs_mask.detach().to(dtype=torch.bool, device=z32.device)
                    if pm.ndim == 1:
                        pm = pm.unsqueeze(0).expand(bsz, n_pts)
                    valid0 = valid0 & pm
                dz0 = dz0 * valid0.unsqueeze(-1).to(torch.float32)
            delta_z[:, 0] = dz0

        if t_steps >= 2:
            raw_diff = z_det[:, 1:] - z_det[:, :-1]
            if same_day_adjacent_mask is not None:
                sdm = same_day_adjacent_mask.to(dtype=torch.float32, device=z32.device)
                if sdm.ndim == 1:
                    sdm = sdm.view(1, -1, 1, 1)
                elif sdm.ndim == 2:
                    sdm = sdm.unsqueeze(-1).unsqueeze(-1)
                raw_diff = raw_diff * sdm
            if obs_btn is not None:
                obs_pair = (obs_btn[:, 1:] & obs_btn[:, :-1]).unsqueeze(-1).to(torch.float32)
                raw_diff = raw_diff * obs_pair
            delta_z[:, 1:] = raw_diff

        lbnd_det = logits_boundary.detach().to(torch.float32)
        p_bce_det = torch.sigmoid(logits_bce.detach().to(torch.float32))
        uncertainty = (1.0 - 2.0 * (p_bce_det - 0.5).abs()).unsqueeze(-1)
        b_conf = torch.tanh(lbnd_det.abs() * 0.25).unsqueeze(-1)
        b_signed = torch.tanh(lbnd_det * 0.25).unsqueeze(-1)
        dz_norm = torch.linalg.norm(delta_z, dim=-1, keepdim=True)

        if solar_angular_speed is not None:
            sol_sp = solar_angular_speed.detach().to(dtype=torch.float32, device=z32.device)
            if sol_sp.ndim == 1:
                sol_sp = sol_sp.view(1, t_steps, 1, 1)
            elif sol_sp.ndim == 2:
                sol_sp = sol_sp.view(bsz, t_steps, 1, 1)
            sol_sp = sol_sp.expand(bsz, t_steps, n_pts, 1)
        else:
            sol_sp = torch.zeros((bsz, t_steps, n_pts, 1), dtype=torch.float32, device=z32.device)

        if local_relief is not None:
            rel = local_relief.detach().to(dtype=torch.float32, device=z32.device)
            if rel.ndim == 1:
                rel = rel.view(1, 1, n_pts, 1)
            elif rel.ndim == 2:
                rel = rel.view(bsz, 1, n_pts, 1)
            rel = rel.expand(bsz, t_steps, n_pts, 1)
        else:
            rel = torch.zeros((bsz, t_steps, n_pts, 1), dtype=torch.float32, device=z32.device)

        gate_in = torch.cat([z_det, delta_z, uncertainty, b_conf, b_signed, dz_norm, sol_sp, rel], dim=-1)
        gate_raw = self.gate_net(gate_in).squeeze(-1)
        gate_prob = torch.sigmoid(gate_raw)
        res_corr = self.res_linear(gate_in).squeeze(-1)
        fused = logits_bce.to(torch.float32) + gate_prob * res_corr
        if not has_batch:
            return fused.squeeze(0), gate_prob.squeeze(0)
        return fused, gate_prob


class LocalValidNeighborhoodReadout(nn.Module):
    """Kaifeng `<0.5m` Narrow-Shadow Refinement: Valid-Only Spatial `k`-NN Logit Residual Head.

    Strictly masked by label-free `receiver_obs_mask` (`0` `UNKNOWN` contamination), active at full
    annual scale (`N = 27,000+`), and 100% invariant to `receiver_chunk_size` when tile-global valid
    `k`-NN static descriptors (`global_knn_*`) are provided.
    """

    def __init__(self, hidden_dim: int, k_neighbors: int = 4) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.k_neighbors = int(k_neighbors)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2 + 1),
            nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        z_seq: torch.Tensor,
        logits_base: torch.Tensor,
        *,
        receivers_xyz: Optional[torch.Tensor] = None,
        receiver_rc: Optional[torch.Tensor] = None,
        receiver_obs_static: Optional[torch.Tensor] = None,
        norm_local: Optional[torch.Tensor] = None,
        node_stem: Optional[Any] = None,
        solar_film_scale: Optional[torch.Tensor] = None,
        global_knn_norm_local: Optional[torch.Tensor] = None,
        global_knn_dist: Optional[torch.Tensor] = None,
        global_knn_obs_valid: Optional[torch.Tensor] = None,
        meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
    ) -> torch.Tensor:
        bsz, t_steps, n_pts, d_dim = z_seq.shape
        if n_pts <= 0 or self.k_neighbors <= 0:
            return logits_base

        if receiver_obs_static is not None:
            obs = receiver_obs_static.to(dtype=torch.bool, device=z_seq.device)
            if obs.ndim == 1:
                obs = obs.unsqueeze(0).expand(bsz, -1)
        else:
            obs = torch.ones((bsz, n_pts), dtype=torch.bool, device=z_seq.device)

        # Path A: Tile-global chunk-invariant static geometry + Solar-FiLM neighborhood contrast
        if norm_local is not None and node_stem is not None:
            nl = norm_local.to(dtype=torch.float32, device=z_seq.device)
            if nl.ndim == 2:
                nl = nl.unsqueeze(0).expand(bsz, -1, -1)

            if (
                global_knn_norm_local is not None
                and global_knn_dist is not None
                and global_knn_obs_valid is not None
            ):
                g_nl = global_knn_norm_local.to(dtype=torch.float32, device=z_seq.device)
                g_dist = global_knn_dist.to(dtype=torch.float32, device=z_seq.device)
                g_obs = global_knn_obs_valid.to(dtype=torch.bool, device=z_seq.device)
                if g_nl.ndim == 3:
                    g_nl = g_nl.unsqueeze(0).expand(bsz, -1, -1, -1)
                if g_dist.ndim == 2:
                    g_dist = g_dist.unsqueeze(0).expand(bsz, -1, -1)
                if g_obs.ndim == 2:
                    g_obs = g_obs.unsqueeze(0).expand(bsz, -1, -1)
                k_eff = min(self.k_neighbors, int(g_nl.shape[2]))
                if k_eff <= 0:
                    return logits_base
                knn_nl = g_nl[:, :, :k_eff, :]
                knn_dist = g_dist[:, :, :k_eff]
                knn_valid = g_obs[:, :, :k_eff] & (knn_dist < 15.0)
            else:
                if n_pts <= 1:
                    return logits_base
                if receivers_xyz is not None:
                    pos = receivers_xyz.to(dtype=torch.float32, device=z_seq.device)
                    if pos.ndim == 2:
                        pos = pos.unsqueeze(0).expand(bsz, -1, -1)
                elif receiver_rc is not None:
                    rc = receiver_rc.to(dtype=torch.float32, device=z_seq.device)
                    if rc.ndim == 2:
                        rc = rc.unsqueeze(0).expand(bsz, -1, -1)
                    zeros_y = torch.zeros((bsz, n_pts, 1), dtype=torch.float32, device=z_seq.device)
                    pos = torch.cat(
                        [rc[..., 1:2] * meters_per_pixel, zeros_y, rc[..., 0:1] * meters_per_pixel], dim=-1
                    )
                else:
                    return logits_base

                k_eff = min(self.k_neighbors, n_pts - 1)
                knn_dist, knn_idx, knn_obs = _chunked_spatial_knn_3d(pos, obs, k_eff=k_eff, query_chunk=512)
                l_dim = int(nl.shape[-1])
                knn_nl = torch.gather(
                    nl,
                    1,
                    knn_idx.reshape(bsz, n_pts * k_eff, 1).expand(bsz, n_pts * k_eff, l_dim),
                ).reshape(bsz, n_pts, k_eff, l_dim)
                knn_valid = knn_obs & (knn_dist < 15.0)

            valid_w = knn_valid.to(torch.float32) * torch.exp(-knn_dist.clamp(0.0, 50.0) / 3.0)
            norm_w = valid_w / valid_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # (B, N, k_eff)

            with torch.amp.autocast(device_type="cuda" if z_seq.is_cuda else "cpu", enabled=False):
                # Evaluate `node_stem` in float32/module dtype and cast cleanly to `z_seq.dtype`
                stem_dtype = next(node_stem.parameters()).dtype
                self_geo = node_stem(nl.to(stem_dtype)).to(torch.float32)  # (B, N, D)
                nbr_geo = node_stem(knn_nl.to(stem_dtype)).to(torch.float32)  # (B, N, k, D)
                nbr_geo_mean = (nbr_geo * norm_w.unsqueeze(-1)).sum(dim=2)  # (B, N, D)
                geo_diff_static = self_geo - nbr_geo_mean  # (B, N, D)
                if solar_film_scale is not None:
                    sf = solar_film_scale.to(dtype=torch.float32, device=z_seq.device)
                    geo_contrast = geo_diff_static.unsqueeze(1) * sf  # (B, T, N, D)
                else:
                    geo_contrast = geo_diff_static.unsqueeze(1).expand(bsz, t_steps, n_pts, d_dim)

                h_col = min(1, int(nl.shape[-1]) - 1)
                h_self = nl[..., h_col : h_col + 1]  # (B, N, 1)
                h_nbr = (knn_nl[..., h_col] * norm_w).sum(dim=-1, keepdim=True)  # (B, N, 1)
                h_relief_step = (h_self - h_nbr).unsqueeze(1).expand(bsz, t_steps, n_pts, 1)
                p_base = torch.sigmoid(logits_base.detach().to(torch.float32)).reshape(bsz, t_steps, n_pts, 1)
                edge_cue = h_relief_step + 0.5 * (2.0 * p_base - 1.0)

                delta_logit = self.mlp(
                    torch.cat([z_seq.to(torch.float32), geo_contrast, edge_cue], dim=-1)
                ).squeeze(-1)
                has_any_nbr = ((valid_w.sum(dim=-1) > 1e-5) & obs).unsqueeze(1).to(logits_base.dtype)
                return logits_base + delta_logit.to(logits_base.dtype) * has_any_nbr

        # Path B: Standalone fallback when called directly without `norm_local` (`_chunked_spatial_knn_3d`, all `N`)
        if n_pts <= 1:
            return logits_base
        if receivers_xyz is not None:
            pos = receivers_xyz.to(dtype=torch.float32, device=z_seq.device)
            if pos.ndim == 2:
                pos = pos.unsqueeze(0).expand(bsz, -1, -1)
        elif receiver_rc is not None:
            rc = receiver_rc.to(dtype=torch.float32, device=z_seq.device)
            if rc.ndim == 2:
                rc = rc.unsqueeze(0).expand(bsz, -1, -1)
            zeros_y = torch.zeros((bsz, n_pts, 1), dtype=torch.float32, device=z_seq.device)
            pos = torch.cat([rc[..., 1:2] * meters_per_pixel, zeros_y, rc[..., 0:1] * meters_per_pixel], dim=-1)
        else:
            return logits_base

        k_eff = min(self.k_neighbors, n_pts - 1)
        knn_dist, knn_idx, knn_obs = _chunked_spatial_knn_3d(pos, obs, k_eff=k_eff, query_chunk=512)
        valid_w = (knn_obs & (knn_dist < 15.0)).to(torch.float32) * torch.exp(-knn_dist.clamp(0.0, 50.0) / 3.0)
        norm_w = valid_w / valid_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        z_perm = z_seq.to(torch.float32).permute(0, 2, 1, 3)  # (B, N, T, D)
        nbr_mean_chunks: List[torch.Tensor] = []
        p_base = torch.sigmoid(logits_base.detach().to(torch.float32)).reshape(bsz, t_steps, n_pts)
        p_perm = p_base.permute(0, 2, 1)  # (B, N, T)
        nbr_p_chunks: List[torch.Tensor] = []
        q_step = 512
        for q0 in range(0, n_pts, q_step):
            q1 = min(n_pts, q0 + q_step)
            idx_q = knn_idx[:, q0:q1]  # (B, Q, k)
            w_q = norm_w[:, q0:q1]     # (B, Q, k)
            ql = q1 - q0
            z_g = torch.gather(
                z_perm.reshape(bsz, n_pts, t_steps * d_dim),
                1,
                idx_q.reshape(bsz, ql * k_eff, 1).expand(bsz, ql * k_eff, t_steps * d_dim),
            ).reshape(bsz, ql, k_eff, t_steps, d_dim)
            nbr_mean_chunks.append((z_g * w_q.unsqueeze(-1).unsqueeze(-1)).sum(dim=2))
            p_g = torch.gather(
                p_perm,
                1,
                idx_q.reshape(bsz, ql * k_eff, 1).expand(bsz, ql * k_eff, t_steps),
            ).reshape(bsz, ql, k_eff, t_steps)
            nbr_p_chunks.append((p_g * w_q.unsqueeze(-1)).sum(dim=2))

        nbr_mean = torch.cat(nbr_mean_chunks, dim=1).permute(0, 2, 1, 3)  # (B, T, N, D)
        nbr_p_mean = torch.cat(nbr_p_chunks, dim=1).permute(0, 2, 1)     # (B, T, N)
        p_diff = (p_base - nbr_p_mean).unsqueeze(-1)

        delta_logit = self.mlp(torch.cat([z_seq.to(torch.float32), z_seq.to(torch.float32) - nbr_mean, p_diff], dim=-1)).squeeze(-1)
        has_any_nbr = ((valid_w.sum(dim=-1) > 1e-5) & obs).unsqueeze(1).to(logits_base.dtype)
        return logits_base + delta_logit.to(logits_base.dtype) * has_any_nbr


def _compute_solar_kinematics(
    sun_dirs: torch.Tensor,
    same_day_adj: Optional[torch.Tensor],
    prev_frame_sun_dir: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute physical solar angular step `||s_t - s_{t-1}||_2` and altitude rate `(B, T, 2)` & `(B, T)`."""
    s = sun_dirs.to(torch.float32)
    s_u = s / torch.linalg.norm(s, dim=-1, keepdim=True).clamp_min(1e-6)
    bsz, t_steps, _ = s_u.shape
    speed = torch.zeros((bsz, t_steps), dtype=torch.float32, device=s_u.device)
    dsy = torch.zeros((bsz, t_steps), dtype=torch.float32, device=s_u.device)
    if prev_frame_sun_dir is not None and t_steps >= 1:
        ps = prev_frame_sun_dir.to(device=s_u.device, dtype=torch.float32)
        if ps.ndim == 1:
            ps = ps.unsqueeze(0).expand(bsz, -1)
        ps_u = ps / torch.linalg.norm(ps, dim=-1, keepdim=True).clamp_min(1e-6)
        speed[:, 0] = torch.linalg.norm(s_u[:, 0] - ps_u, dim=-1) * 20.0
        dsy[:, 0] = (s_u[:, 0, 1] - ps_u[:, 1]) * 20.0
    if t_steps >= 2:
        raw_sp = torch.linalg.norm(s_u[:, 1:] - s_u[:, :-1], dim=-1) * 20.0
        raw_dsy = (s_u[:, 1:, 1] - s_u[:, :-1, 1]) * 20.0
        if same_day_adj is not None:
            sdm = same_day_adj.to(dtype=torch.float32, device=s_u.device)
            if sdm.ndim == 1:
                sdm = sdm.unsqueeze(0).expand(bsz, -1)
            raw_sp = raw_sp * sdm
            raw_dsy = raw_dsy * sdm
        speed[:, 1:] = raw_sp
        dsy[:, 1:] = raw_dsy
    return torch.stack([speed, dsy], dim=-1), speed


def _carry_observation_mask(current, batch):
    previous = batch.get("prev_chunk_obs_mask")
    if previous is not None:
        return current & previous.to(device=current.device, dtype=torch.bool)
    # A 1-D availability mask is static. Time-varying availability needs both endpoints.
    mask = batch.get("receiver_obs_mask")
    return current if mask is None or mask.ndim == 1 else torch.zeros_like(current)


def _resolve_label_free_obs_mask(
    batch: Mapping[str, Any],
    *,
    bsz: int,
    t_steps: int,
    n_pts: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract label-free `receiver_obs_mask` `(B, T, N)` and static `(B, N)` WITHOUT reading `valid_mask`."""
    raw = batch.get("receiver_obs_mask")
    if raw is None:
        return None, None
    obs = raw.to(device=device, dtype=torch.bool)
    if obs.ndim == 1:
        obs_static = obs.unsqueeze(0).expand(bsz, n_pts)
        obs_btn = obs_static.unsqueeze(1).expand(bsz, t_steps, n_pts)
    elif obs.ndim == 2:
        if int(obs.shape[0]) == t_steps and int(obs.shape[1]) == n_pts and bsz == 1:
            obs_btn = obs.unsqueeze(0)
            obs_static = obs_btn[:, 0]
        else:
            obs_static = obs.expand(bsz, n_pts)
            obs_btn = obs_static.unsqueeze(1).expand(bsz, t_steps, n_pts)
    elif obs.ndim == 3:
        obs_btn = obs.expand(bsz, t_steps, n_pts)
        obs_static = obs_btn[:, 0]
    else:
        return None, None
    return obs_btn, obs_static


def _require_continuity_mask(
    t_steps: int,
    same_day_adj: Optional[torch.Tensor],
    batch: Mapping[str, Any],
) -> None:
    if t_steps > 1 and same_day_adj is None and not bool(batch.get("allow_implicit_continuity", False)):
        raise ValueError(
            "Explicit `same_day_adjacent_mask` is required whenever T > 1 so cross-day or "
            "non-120s gaps never leak recurrent state or event-gate deltas."
        )


class _BaseRindModel(nn.Module):
    """Shared calibration buffers, normalizer provenance tracking, and metadata interface."""

    model_type: str = "base"
    ablation_tier: str = "unspecified"
    input_tier: str = "unspecified"
    has_event_gate: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("calibrated_temperature", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("calibrated_threshold", torch.tensor(0.5, dtype=torch.float32))
        self._consumed_normalizer_tile_ids: Set[str] = set()

    @property
    def consumed_normalizer_tile_ids(self) -> List[str]:
        return sorted(self._consumed_normalizer_tile_ids)

    def record_normalizer_tiles(self, tile_ids: Sequence[str]) -> None:
        for tid in tile_ids:
            self._consumed_normalizer_tile_ids.add(str(tid))

    def set_calibration(self, temperature: float, decision_threshold: float) -> None:
        self.calibrated_temperature.fill_(max(float(temperature), 1e-4))
        self.calibrated_threshold.fill_(float(min(max(decision_threshold, 1e-4), 1.0 - 1e-4)))

    def get_model_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    def forward_with_mode(
        self,
        batch: Mapping[str, Any],
        *,
        inference_mode: str = "causal_rollout",
        stride: int = 1,
        use_calibrated_temperature: bool = True,
    ) -> Dict[str, Any]:
        """Unified mode-explicit inference dispatcher (`static_direct | causal_rollout | offline_interp`)."""
        if inference_mode not in INFERENCE_MODES:
            raise ValueError(f"Unsupported inference_mode {inference_mode!r}; expected {INFERENCE_MODES}")
        if hasattr(self, "evaluate_strided_rollout") and inference_mode in (
            "causal_rollout",
            "offline_interp",
            "static_direct",
        ):
            return getattr(self, "evaluate_strided_rollout")(
                batch,
                stride=stride,
                mode=inference_mode,
                use_calibrated_temperature=use_calibrated_temperature,
            )
        b_copy = dict(batch)
        b_copy["inference_mode"] = inference_mode
        out = dict(self(b_copy, use_calibrated_temperature=use_calibrated_temperature))
        out.setdefault("inference_mode", inference_mode)
        out.setdefault("is_causal", inference_mode != "offline_interp")
        return out


class RindCandidateWorldModel(_BaseRindModel):
    """Zero-Shrinkage Relational Incident-Normal Candidate World Model across 4 geometry tiers."""

    has_event_gate = True

    def __init__(
        self,
        *,
        ablation_mode: Optional[str] = None,
        zero_margin: bool = False,
        hidden_dim: int = 96,
        num_heads: int = 4,
        max_candidates: int = 8,
        local_dim: int = LOCAL_CONTEXT_DIM,
        solar_frequencies: int = 4,
        max_coord_bound: float = 1.0,
        max_token_chunk_queries: int = 8192,
        reanchor_every: int = 0,
        enable_spatial_knn_readout: bool = True,
        use_static_mpnn: bool = True,
        use_local_knn_readout: Optional[bool] = None,
        use_fourier102: bool = True,
        far_perspective: bool = False,
        attention_mode: str = "ray_conditioned",
        film_beta_bound: float = 0.0,
        attn_qk_norm: bool = False,
        cand_readout: str = "attn_only",
        disable_event_gate: bool = False,
    ) -> None:
        super().__init__()
        if ablation_mode is None:
            ablation_mode = "zero_margin" if bool(zero_margin) else "full_geom"
        if ablation_mode not in ("full_geom", "no_edge_margin", "zero_margin", "coarse_aabb"):
            raise ValueError(
                f"Invalid ablation_mode {ablation_mode!r}; expected 'full_geom', "
                "'no_edge_margin', 'zero_margin', or 'coarse_aabb'."
            )
        if attention_mode not in ATTENTION_MODES:
            raise ValueError(
                f"Invalid attention_mode {attention_mode!r}; expected {ATTENTION_MODES}"
            )
        self.ablation_mode = str(ablation_mode)
        self.attention_mode = str(attention_mode)
        self.film_beta_bound = float(film_beta_bound)
        self.attn_qk_norm = bool(attn_qk_norm)
        if cand_readout not in ("attn_only", "attn_plus_lse"):
            raise ValueError("cand_readout must be attn_only or attn_plus_lse")
        self.cand_readout = str(cand_readout)
        self.disable_event_gate = bool(disable_event_gate)
        self.zero_margin = self.ablation_mode == "zero_margin"
        self.no_edge_margin = self.ablation_mode == "no_edge_margin"
        self.coarse_aabb = self.ablation_mode == "coarse_aabb"
        self.use_local_context = self.ablation_mode not in ("zero_margin",)
        self.use_static_mpnn = bool(use_static_mpnn)
        self.use_fourier102 = bool(use_fourier102)
        self.enable_tangent_chords = (not self.coarse_aabb) and self.use_fourier102
        if far_perspective and not self.enable_tangent_chords:
            raise ValueError("far_perspective requires Fourier tangent encoding and triangle vertices")
        self.reanchor_every = max(int(reanchor_every), 0)
        eff_knn = enable_spatial_knn_readout if use_local_knn_readout is None else bool(use_local_knn_readout)
        self.enable_spatial_knn_readout = bool(eff_knn) and self.use_local_context

        if self.ablation_mode == "full_geom":
            self.model_type = "rind_full_geom"
            self.ablation_tier = "Tier-G2_full_geom"
            self.input_tier = "canonical37_plus_local16"
            self.sanitizer: Optional[nn.Module] = None
        elif self.ablation_mode == "no_edge_margin":
            self.model_type = "rind_no_edge_margin"
            self.ablation_tier = "Tier-G1_no_edge_margin"
            self.input_tier = "canonical37_zeroed_22_24_plus_local16"
            self.sanitizer = NoEdgeMarginSanitizer()
        elif self.ablation_mode == "coarse_aabb":
            self.model_type = "rind_coarse_aabb"
            self.ablation_tier = "Tier-G0b_coarse_aabb"
            self.input_tier = "coarse_aabb_28_33_plus_local16"
            self.sanitizer = CoarseAABBSanitizer()
        else:
            self.model_type = "rind_zero_margin"
            self.ablation_tier = "Tier-G0_pure_raw_vertices"
            self.input_tier = "raw_vertices_and_sun_only_no_local16"
            self.sanitizer = ZeroMarginSanitizer()

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.max_candidates = int(max_candidates)
        self.local_dim = int(local_dim)
        self.solar_frequencies = int(solar_frequencies)
        self.max_coord_bound = float(max_coord_bound)
        self.max_token_chunk_queries = max(int(max_token_chunk_queries), 256)

        self.token_normalizer = RunningNormalizer(CANONICAL_TOKEN_DIM)
        self.local_normalizer = RunningNormalizer(self.local_dim)

        self.solar_encoder = SphereTangentSolarEncoder(
            self.hidden_dim, num_frequencies=self.solar_frequencies
        )
        self.static_geom_memory = StaticGeometryMemory(
            self.local_dim, self.hidden_dim, k_neighbors=8, film_beta_bound=self.film_beta_bound
        )
        self.candidate_encoder = FourierSphereTangentEncoder102(
            self.hidden_dim, num_frequencies=self.solar_frequencies, far_perspective=far_perspective
        )
        self.ray_attn_bias = RayAnisotropicAttentionBias(self.num_heads)
        self.solar_attn_gate: Optional[nn.Linear] = None
        if self.attention_mode == "solar_gating":
            self.solar_attn_gate = nn.Linear(self.hidden_dim, self.hidden_dim)
            nn.init.zeros_(self.solar_attn_gate.weight)
            nn.init.zeros_(self.solar_attn_gate.bias)
        self.query_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.cand_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.num_heads, batch_first=True
        )
        self.cand_face_score: Optional[nn.Module] = None
        if self.cand_readout == "attn_plus_lse":
            self.cand_face_score = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim // 2, 1),
            )
            nn.init.zeros_(self.cand_face_score[-1].weight)
            nn.init.zeros_(self.cand_face_score[-1].bias)
        self.step_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.recurrence = BoundedSphereTangentCell(
            self.hidden_dim, max_coord_bound=self.max_coord_bound
        )

        self.bce_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.focal_event_gate = FocalEventTransitionGate(self.hidden_dim)
        self.spatial_knn_readout = LocalValidNeighborhoodReadout(self.hidden_dim, k_neighbors=4)

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "ablation_mode": self.ablation_mode,
            "ablation_tier": self.ablation_tier,
            "input_tier": self.input_tier,
            "attention_mode": self.attention_mode,
            "film_beta_bound": self.film_beta_bound,
            "attn_qk_norm": self.attn_qk_norm,
            "cand_readout": self.cand_readout,
            "far_perspective": self.candidate_encoder.angle_projection is not None,
            "disable_event_gate": self.disable_event_gate,
            "use_local_context": self.use_local_context,
            "zero_margin": self.zero_margin,
            "no_edge_margin": self.no_edge_margin,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "max_candidates": self.max_candidates,
            "local_dim": self.local_dim,
            "solar_frequencies": self.solar_frequencies,
            "max_coord_bound": self.max_coord_bound,
            "max_token_chunk_queries": self.max_token_chunk_queries,
            "reanchor_every": self.reanchor_every,
            "enable_spatial_knn_readout": self.enable_spatial_knn_readout,
            "use_static_mpnn": self.use_static_mpnn,
            "use_fourier102": self.use_fourier102,
            "theoretical_latent_l2_bound": self.recurrence.theoretical_l2_bound,
        }

    @torch.no_grad()
    def update_normalizers(
        self,
        batch: Mapping[str, Any],
        *,
        firewall_token: FitFirewallToken,
        tile_ids: Sequence[str],
        expected_split_sha256: str,
    ) -> None:
        firewall_token.verify_tiles(tile_ids, expected_split_sha256)
        tokens = batch["candidate_tokens"]
        cmask = batch["candidate_mask"].to(torch.bool)
        flat_tokens = tokens.detach().reshape(-1, tokens.shape[-2], CANONICAL_TOKEN_DIM)
        flat_cmask = cmask.detach().reshape(-1, cmask.shape[-1])

        step = self.max_token_chunk_queries
        for s_idx in range(0, int(flat_tokens.shape[0]), step):
            e_idx = min(int(flat_tokens.shape[0]), s_idx + step)
            sub_t = flat_tokens[s_idx:e_idx]
            sub_m = flat_cmask[s_idx:e_idx]
            sub_t = torch.where(sub_m.unsqueeze(-1), sub_t, torch.zeros_like(sub_t))
            if self.sanitizer is not None:
                sub_t = self.sanitizer(sub_t, sub_m)
            self.token_normalizer.update(
                sub_t,
                firewall_token=firewall_token,
                tile_ids=tile_ids,
                expected_split_sha256=expected_split_sha256,
                valid_mask=sub_m,
            )

        if self.use_local_context and "local_features" in batch:
            self.local_normalizer.update(
                batch["local_features"],
                firewall_token=firewall_token,
                tile_ids=tile_ids,
                expected_split_sha256=expected_split_sha256,
            )
        self.record_normalizer_tiles(tile_ids)

    def _encode_candidate_cross_attention(
        self,
        tokens: torch.Tensor,
        cmask: torch.Tensor,
        q_in: torch.Tensor,
        *,
        solar_exp: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
        return_readout: bool = False,
    ) -> Any:
        """Chunked sanitization + Fourier-Tangent encoding + Ray-Anisotropic / Solar-Gated Cross-Attention."""
        bsz, t_steps, n_pts, k_cands, _ = tokens.shape
        flat_bt_n = bsz * t_steps * n_pts
        flat_tokens = tokens.reshape(flat_bt_n, k_cands, CANONICAL_TOKEN_DIM)
        flat_cmask = cmask.reshape(flat_bt_n, k_cands)
        flat_q_in = q_in.reshape(flat_bt_n, self.hidden_dim * 2)
        use_ray_bias = self.attention_mode == "ray_conditioned"

        max_attn_batch = min(
            self.max_token_chunk_queries,
            max(1, 32768 // max(int(self.cand_attn.num_heads), 1)),
        )
        attn_chunks: List[torch.Tensor] = []
        weight_chunks: List[torch.Tensor] = []
        readout_chunks: List[torch.Tensor] = []
        for s_idx in range(0, flat_bt_n, max_attn_batch):
            e_idx = min(flat_bt_n, s_idx + max_attn_batch)
            c_len = e_idx - s_idx
            tok_sub = flat_tokens[s_idx:e_idx]
            cm_sub = flat_cmask[s_idx:e_idx]
            # Remove padding before any geometry arithmetic: NaN * 0 is NaN.
            tok_sub = torch.where(cm_sub.unsqueeze(-1), tok_sub, torch.zeros_like(tok_sub))
            if self.sanitizer is not None:
                tok_sub = self.sanitizer(tok_sub, cm_sub)
            else:
                tok_sub = tok_sub * cm_sub.unsqueeze(-1).to(tok_sub.dtype)

            if self.cand_face_score is not None and k_cands > 0:
                # A face is an element of a set. Deduplicate only after the sanitizer,
                # so removed ablation fields cannot influence either readout branch.
                equal = torch.ones((c_len, k_cands, k_cands), dtype=torch.bool, device=tok_sub.device)
                for feature in tok_sub.unbind(dim=-1):
                    equal = equal & (feature[:, :, None] == feature[:, None, :])
                earlier = torch.tril(torch.ones((k_cands, k_cands), dtype=torch.bool, device=cm_sub.device), diagonal=-1)
                duplicate = (equal & earlier & cm_sub[:, None, :]).any(dim=-1)
                cm_sub = cm_sub & ~duplicate
                tok_sub = tok_sub * cm_sub.unsqueeze(-1).to(tok_sub.dtype)

            norm_tok_sub = self.token_normalizer(tok_sub) * cm_sub.unsqueeze(-1).to(tok_sub.dtype)
            cand_h_sub, ray_cues = self.candidate_encoder(
                tok_sub,
                norm_tok_sub,
                cm_sub,
                enable_tangent_chords=self.enable_tangent_chords,
            )
            if self.cand_face_score is not None:
                face_scores = self.cand_face_score(cand_h_sub).squeeze(-1).to(torch.float32)
                count = cm_sub.sum(dim=-1)
                # The null contributes exactly zero only for an empty set, avoiding
                # all-negative-infinity LSE and its undefined backward derivative.
                scores = face_scores.masked_fill(~cm_sub, float("-inf"))
                null = torch.where(count == 0, scores.new_zeros(count.shape), scores.new_full(count.shape, float("-inf")))
                lse = torch.logsumexp(torch.cat([scores, null[:, None]], dim=-1), dim=-1)
                readout_chunks.append(lse - torch.log(count.clamp_min(1).to(lse.dtype)))
            else:
                readout_chunks.append(cand_h_sub.new_zeros((c_len,)))
            q_sub = self.query_proj(flat_q_in[s_idx:e_idx]).unsqueeze(1)

            null_slot = torch.zeros(
                (c_len, 1, self.hidden_dim), dtype=cand_h_sub.dtype, device=cand_h_sub.device
            )
            kv_sub = torch.cat([cand_h_sub, null_slot], dim=1)
            attn_bias = self.ray_attn_bias(
                ray_cues, cm_sub, enable_ray_bias=use_ray_bias
            ).to(torch.float32)

            # Explicit multi-head cross-attention using `self.cand_attn` weights so `grad` and
            # `no_grad` execute the exact same floating-point ops (`0.0` ULP fastpath divergence!).
            w_in = self.cand_attn.in_proj_weight
            b_in = self.cand_attn.in_proj_bias
            d_m = self.hidden_dim
            n_h = self.num_heads
            h_d = d_m // n_h
            q_proj = F.linear(q_sub, w_in[:d_m], b_in[:d_m] if b_in is not None else None)
            k_proj = F.linear(kv_sub, w_in[d_m : 2 * d_m], b_in[d_m : 2 * d_m] if b_in is not None else None)
            v_proj = F.linear(kv_sub, w_in[2 * d_m :], b_in[2 * d_m :] if b_in is not None else None)

            q_mh = q_proj.to(torch.float32).view(c_len, 1, n_h, h_d).permute(0, 2, 1, 3)
            k_mh = k_proj.to(torch.float32).view(c_len, k_cands + 1, n_h, h_d).permute(0, 2, 3, 1)
            v_mh = v_proj.to(torch.float32).view(c_len, k_cands + 1, n_h, h_d).permute(0, 2, 1, 3)

            if self.attn_qk_norm:
                # Parameter-free per-head RMS normalization controls content logits
                # while preserving the learned additive ray bias scale (+/-4).
                q_mh = q_mh * torch.rsqrt(q_mh.square().mean(dim=-1, keepdim=True) + 1e-6)
                k_mh = k_mh * torch.rsqrt(k_mh.square().mean(dim=-2, keepdim=True) + 1e-6)

            attn_scores = torch.matmul(q_mh, k_mh) * (1.0 / math.sqrt(float(h_d))) + attn_bias.view(
                c_len, n_h, 1, k_cands + 1
            )
            attn_weights = torch.softmax(attn_scores, dim=-1)
            if return_attn_weights:
                # Head-averaged attention probability over the K real candidate slots `(c_len, K)`
                weight_chunks.append(attn_weights[:, :, 0, :k_cands].mean(dim=1).detach())
            ctx_mh = torch.matmul(attn_weights, v_mh).permute(0, 2, 1, 3).reshape(c_len, 1, d_m)
            out_sub = self.cand_attn.out_proj(ctx_mh.to(q_sub.dtype))
            attn_chunks.append(out_sub.squeeze(1))

        pooled = torch.cat(attn_chunks, dim=0).reshape(bsz, t_steps, n_pts, self.hidden_dim)
        if self.solar_attn_gate is not None and solar_exp is not None:
            gate = 1.0 + torch.tanh(self.solar_attn_gate(solar_exp.to(pooled.dtype)))
            pooled = pooled * gate
        cand_w = (
            torch.cat(weight_chunks, dim=0).reshape(bsz, t_steps, n_pts, k_cands)
            if return_attn_weights and weight_chunks
            else None
        )
        if return_readout:
            readout = torch.cat(readout_chunks, dim=0).reshape(bsz, t_steps, n_pts)
            return pooled, cand_w, readout
        return pooled, cand_w

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        use_calibrated_temperature: bool = False,
        inference_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run memory-bounded rollout or static direct query over `(B, T, N, K, 37)` or `(T, N, K, 37)` inputs."""
        eff_inf_mode = str(inference_mode or batch.get("inference_mode") or "recurrent")
        if eff_inf_mode not in INFERENCE_MODES:
            raise ValueError(f"Invalid inference_mode {eff_inf_mode!r}; expected {INFERENCE_MODES}")

        tokens = batch["candidate_tokens"]
        cmask = batch["candidate_mask"].to(torch.bool)
        sun_dirs = batch["sun_dirs"]
        same_day_adj = batch.get("same_day_adjacent_mask")
        carry_prev = bool(batch.get("carry_state_from_prev_chunk", False) and batch.get("prev_frame_same_day", True))
        init_latent = batch.get("initial_latent_state")
        reanchor_k = int(batch.get("reanchor_every", self.reanchor_every) or 0)
        eff_disable_gate = bool(batch.get("disable_event_gate", self.disable_event_gate))
        want_cand_attn = bool(batch.get("return_candidate_attention", False))

        squeeze_batch = False
        if tokens.ndim == 4:
            tokens = tokens.unsqueeze(0)
            cmask = cmask.unsqueeze(0)
            squeeze_batch = True
        if sun_dirs.ndim == 2:
            sun_dirs = sun_dirs.unsqueeze(0)

        bsz, t_steps, n_pts, _k_cands, tok_dim = tokens.shape
        if tok_dim != CANONICAL_TOKEN_DIM:
            raise ValueError(f"Expected {CANONICAL_TOKEN_DIM} token features, got {tok_dim}")
        if eff_inf_mode != "static_direct":
            _require_continuity_mask(t_steps, same_day_adj, batch)

        # Optional candidate-level dropout mask (for Dominant-Occluder Dropout / Greedy K=1 tests)
        cand_drop_mask = batch.get("candidate_keep_mask")
        if cand_drop_mask is not None:
            ckm = cand_drop_mask.to(device=tokens.device, dtype=torch.bool)
            if ckm.ndim == 3:
                ckm = ckm.unsqueeze(0)
            cmask = cmask & ckm

        # Strictly label-free per-receiver observability mask (`0` reads of `truth` or `valid_mask`!)
        obs_btn, obs_static = _resolve_label_free_obs_mask(
            batch, bsz=bsz, t_steps=t_steps, n_pts=n_pts, device=tokens.device
        )
        if eff_inf_mode == "static_direct":
            solar_kin_bt2 = torch.zeros((bsz, t_steps, 2), dtype=torch.float32, device=tokens.device)
            solar_speed_bt = torch.zeros((bsz, t_steps), dtype=torch.float32, device=tokens.device)
        else:
            solar_kin_bt2, solar_speed_bt = _compute_solar_kinematics(
                sun_dirs,
                same_day_adj,
                prev_frame_sun_dir=batch.get("prev_frame_sun_dir") if carry_prev else None,
            )
        solar_emb = self.solar_encoder(sun_dirs)  # (B, T, D)
        solar_exp = solar_emb.unsqueeze(2).expand(bsz, t_steps, n_pts, self.hidden_dim)

        if self.use_local_context:
            local_feat = batch["local_features"]
            if local_feat.ndim == 2:
                local_feat = local_feat.unsqueeze(0)
            norm_local = self.local_normalizer(local_feat)
            raw_g_knn_lf = batch.get("global_knn_local_features")
            g_knn_norm_local = (
                self.local_normalizer(raw_g_knn_lf.to(device=tokens.device, dtype=local_feat.dtype))
                if raw_g_knn_lf is not None
                else None
            )
            frozen_static = batch.get("frozen_static_geometry") if not self.training and not torch.is_grad_enabled() else None
            if frozen_static is not None:
                static_zg = frozen_static.reshape(bsz,n_pts,self.hidden_dim)
            else:
                static_zg = self.static_geom_memory.encode_static(
                    norm_local,
                    receivers_xyz=batch.get("receivers_xyz"),
                    receiver_rc=batch.get("receiver_rc"),
                    receiver_obs_static=obs_static,
                    global_knn_norm_local=g_knn_norm_local,
                    global_knn_rel_xyz=batch.get("global_knn_rel_xyz"),
                    global_knn_dist=batch.get("global_knn_dist"),
                    global_knn_obs_valid=batch.get("global_knn_obs_valid"),
                    enable_mpnn=self.use_static_mpnn,
                )
            local_exp = self.static_geom_memory.modulate_solar_film(static_zg, solar_emb)
            solar_film_scale = self.static_geom_memory.solar_film_scale(solar_emb)
            local_relief = torch.linalg.norm(norm_local.detach().to(torch.float32), dim=-1) / math.sqrt(
                float(self.local_dim)
            )
        else:
            # Tier-G0 strictly disables local_features, receivers_xyz, and StaticGeometryMemory!
            norm_local = None
            g_knn_norm_local = None
            solar_film_scale = None
            static_zg = torch.zeros(
                (bsz, n_pts, self.hidden_dim), dtype=solar_emb.dtype, device=tokens.device
            )
            local_exp = static_zg.unsqueeze(1).expand(bsz, t_steps, n_pts, self.hidden_dim)
            local_relief = None

        q_in = torch.cat([local_exp, solar_exp], dim=-1)
        cand_summary, cand_attn_w, cand_readout = self._encode_candidate_cross_attention(
            tokens,
            cmask,
            q_in,
            solar_exp=solar_exp,
            return_attn_weights=want_cand_attn,
            return_readout=True,
        )
        step_h = self.step_fusion(torch.cat([local_exp, solar_exp, cand_summary], dim=-1))

        device_type = "cuda" if step_h.is_cuda else "cpu"
        recurrent_step_count = 0
        reanchor_reset_count = 0
        with torch.amp.autocast(device_type=device_type, enabled=False):
            step_h32 = step_h.to(torch.float32)
            z_direct = self.recurrence.max_coord_bound * torch.tanh(
                self.recurrence.init_proj(self.recurrence.input_ln(step_h32))
            )

            if eff_inf_mode == "static_direct":
                z_seq = z_direct
                logits_bce = self.bce_head(z_seq).squeeze(-1) + cand_readout
                logits_boundary = self.boundary_head(z_seq).squeeze(-1)
                gate_prob = torch.zeros_like(logits_bce)
                logits_fused = logits_bce
            else:
                z_list: List[torch.Tensor] = []
                init_offset = int(batch.get("intra_day_step_offset", 0) or 0) if carry_prev else 0
                intra_step = torch.full((bsz,), init_offset, dtype=torch.long, device=step_h32.device)
                reanchor_at_t0 = bool(reanchor_k > 0 and init_offset > 0 and (init_offset % reanchor_k) == 0)

                prev_chunk_z32: Optional[torch.Tensor] = None
                if carry_prev and init_latent is not None:
                    prev_chunk_z32 = init_latent.to(dtype=torch.float32, device=step_h32.device)
                    if prev_chunk_z32.ndim == 2:
                        prev_chunk_z32 = prev_chunk_z32.unsqueeze(0)
                    if reanchor_at_t0:
                        z_cur = z_direct[:, 0]
                        reanchor_reset_count += 1
                    else:
                        z_cur = self.recurrence.step(
                            step_h32[:, 0], prev_chunk_z32, solar_kinematics=solar_kin_bt2[:, 0]
                        )
                        recurrent_step_count += 1
                        if obs_btn is not None:
                            z_fresh0 = z_direct[:, 0]
                            keep = _carry_observation_mask(obs_btn[:, 0], batch)
                            z_cur = torch.where(keep.unsqueeze(-1), z_cur, z_fresh0)
                else:
                    z_cur = z_direct[:, 0]
                z_list.append(z_cur)

                if same_day_adj is not None:
                    sdm = same_day_adj.to(dtype=torch.bool, device=step_h32.device)
                    if sdm.ndim == 1:
                        sdm = sdm.unsqueeze(0).expand(bsz, -1)
                else:
                    sdm = None

                use_parallel_day_segments = (
                    bsz == 1
                    and (not carry_prev)
                    and sdm is not None
                    and t_steps >= 2
                    and bool((~sdm[0]).any().item())
                )
                if use_parallel_day_segments:
                    breaks = torch.nonzero(~sdm[0], as_tuple=False).reshape(-1).tolist()
                    seg_bounds: List[Tuple[int, int]] = []
                    s_cur = 0
                    for brk in breaks:
                        seg_bounds.append((s_cur, int(brk) + 1))
                        s_cur = int(brk) + 1
                    seg_bounds.append((s_cur, t_steps))
                    m_segs = len(seg_bounds)
                    seg_lens = [e_i - s_i for s_i, e_i in seg_bounds]
                    l_max = max(seg_lens)

                    h_pad = step_h32.new_zeros((m_segs, l_max, n_pts, self.hidden_dim))
                    zd_pad = z_direct.new_zeros((m_segs, l_max, n_pts, self.hidden_dim))
                    sk_pad = solar_kin_bt2.new_zeros((m_segs, l_max, 2))
                    obs_pad = (
                        torch.ones((m_segs, l_max, n_pts), dtype=torch.bool, device=step_h32.device)
                        if obs_btn is not None
                        else None
                    )
                    for m_i, (s_i, e_i) in enumerate(seg_bounds):
                        l_i = e_i - s_i
                        h_pad[m_i, :l_i] = step_h32[0, s_i:e_i]
                        zd_pad[m_i, :l_i] = z_direct[0, s_i:e_i]
                        sk_pad[m_i, :l_i] = solar_kin_bt2[0, s_i:e_i]
                        if obs_pad is not None and obs_btn is not None:
                            obs_pad[m_i, :l_i] = obs_btn[0, s_i:e_i]

                    z_cur_m = zd_pad[:, 0]
                    z_pad_list: List[torch.Tensor] = [z_cur_m]
                    for k_idx in range(1, l_max):
                        z_step_m = self.recurrence.step(
                            h_pad[:, k_idx], z_cur_m, solar_kinematics=sk_pad[:, k_idx]
                        )
                        is_reanchor_k = bool(reanchor_k > 0 and (k_idx % reanchor_k) == 0)
                        if is_reanchor_k:
                            reanchor_reset_count += sum(1 for l_i in seg_lens if l_i > k_idx)
                            z_cur_m = zd_pad[:, k_idx]
                        elif obs_pad is not None:
                            obs_pair_m = (obs_pad[:, k_idx] & obs_pad[:, k_idx - 1]).unsqueeze(-1)
                            z_cur_m = torch.where(obs_pair_m, z_step_m, zd_pad[:, k_idx])
                        else:
                            z_cur_m = z_step_m
                        z_pad_list.append(z_cur_m)

                    z_pad_seq = torch.stack(z_pad_list, dim=1)
                    z_seq = torch.cat(
                        [z_pad_seq[m_i, : (e_i - s_i)] for m_i, (s_i, e_i) in enumerate(seg_bounds)],
                        dim=0,
                    ).unsqueeze(0)
                    recurrent_step_count = t_steps - 1
                else:
                    for t_idx in range(1, t_steps):
                        if sdm is not None:
                            intra_step = torch.where(
                                sdm[:, t_idx - 1], intra_step + 1, torch.zeros_like(intra_step)
                            )
                        else:
                            intra_step = intra_step + 1

                        z_step = self.recurrence.step(
                            step_h32[:, t_idx], z_cur, solar_kinematics=solar_kin_bt2[:, t_idx]
                        )
                        recurrent_step_count += 1
                        keep_mask: Optional[torch.Tensor] = None
                        if sdm is not None:
                            keep_mask = sdm[:, t_idx - 1].view(bsz, 1, 1)
                        if reanchor_k > 0:
                            is_reanchor_step = (intra_step > 0) & ((intra_step % reanchor_k) == 0)
                            if bool(is_reanchor_step.any().item()):
                                reanchor_reset_count += 1
                            not_reanchor = (~is_reanchor_step).view(bsz, 1, 1)
                            keep_mask = not_reanchor if keep_mask is None else (keep_mask & not_reanchor)
                        if obs_btn is not None:
                            obs_pair = (obs_btn[:, t_idx] & obs_btn[:, t_idx - 1]).unsqueeze(-1)
                            keep_mask = obs_pair if keep_mask is None else (keep_mask & obs_pair)

                        if keep_mask is not None:
                            z_cur = torch.where(keep_mask, z_step, z_direct[:, t_idx])
                        else:
                            z_cur = z_step
                        z_list.append(z_cur)

                    z_seq = torch.stack(z_list, dim=1)  # (B, T, N, D)
                logits_bce = self.bce_head(z_seq).squeeze(-1) + cand_readout
                logits_boundary = self.boundary_head(z_seq).squeeze(-1)
                logits_fused, gate_prob = self.focal_event_gate(
                    z_seq,
                    logits_bce,
                    logits_boundary,
                    same_day_adjacent_mask=same_day_adj,
                    prev_chunk_latent=prev_chunk_z32,
                    solar_angular_speed=solar_speed_bt,
                    local_relief=local_relief,
                    obs_mask=obs_btn,
                    prev_chunk_obs_mask=batch.get("prev_chunk_obs_mask") if carry_prev else None,
                    disable_gate=eff_disable_gate,
                )

            if self.enable_spatial_knn_readout:
                logits_fused = self.spatial_knn_readout(
                    z_seq,
                    logits_fused,
                    receivers_xyz=batch.get("receivers_xyz"),
                    receiver_rc=batch.get("receiver_rc"),
                    receiver_obs_static=obs_static,
                    norm_local=norm_local,
                    node_stem=self.static_geom_memory.node_stem,
                    solar_film_scale=solar_film_scale,
                    global_knn_norm_local=g_knn_norm_local,
                    global_knn_dist=batch.get("global_knn_dist"),
                    global_knn_obs_valid=batch.get("global_knn_obs_valid"),
                )

            if use_calibrated_temperature:
                logits_eval = logits_fused / self.calibrated_temperature.clamp_min(1e-4)
            else:
                logits_eval = logits_fused

            logits_direct = logits_bce if eff_inf_mode == "static_direct" else self.bce_head(z_direct).squeeze(-1) + cand_readout
            if self.enable_spatial_knn_readout:
                if eff_inf_mode == "static_direct":
                    logits_direct = logits_fused
                else:
                    logits_direct = self.spatial_knn_readout(
                        z_direct, logits_direct, receivers_xyz=batch.get("receivers_xyz"),
                        receiver_rc=batch.get("receiver_rc"), receiver_obs_static=obs_static,
                        norm_local=norm_local, node_stem=self.static_geom_memory.node_stem,
                        solar_film_scale=solar_film_scale, global_knn_norm_local=g_knn_norm_local,
                        global_knn_dist=batch.get("global_knn_dist"),
                        global_knn_obs_valid=batch.get("global_knn_obs_valid"))
            nonfinite_count = int((~torch.isfinite(logits_eval)).sum().item()) + int(
                (~torch.isfinite(z_seq)).sum().item()
            )
            latent_norms = torch.linalg.norm(z_seq, dim=-1)
            max_latent_norm = float(latent_norms.max().item()) if latent_norms.numel() > 0 else 0.0
            probs = torch.sigmoid(logits_eval)
            final_latent = z_seq[:, -1]

        counters = {
            "forward_calls": 1,
            "total_frames": int(t_steps),
            "anchor_frames": int(t_steps),
            "candidate_materializations": 0,
            "recurrent_steps": int(recurrent_step_count),
            "reanchor_resets": int(reanchor_reset_count),
        }
        if squeeze_batch:
            res_s: Dict[str, Any] = {
                "logits": logits_eval.squeeze(0),
                "logits_bce": logits_bce.squeeze(0),
                "logits_direct": logits_direct.squeeze(0),
                "logits_boundary": logits_boundary.squeeze(0),
                "event_gate": gate_prob.squeeze(0),
                "probs": probs.squeeze(0),
                "latent_roll": z_seq.squeeze(0),
                "latent_direct": z_direct.squeeze(0),
                "static_geom_memory": static_zg.squeeze(0),
                "solar_emb": solar_emb.squeeze(0),
                "final_latent_state": final_latent.squeeze(0),
                "max_latent_norm": max_latent_norm,
                "nonfinite_count": nonfinite_count,
                "inference_mode": eff_inf_mode,
                "is_causal": True,
                "execution_counters": counters,
            }
            if cand_attn_w is not None:
                res_s["candidate_attn_weights"] = cand_attn_w.squeeze(0)
            return res_s
        res_b: Dict[str, Any] = {
            "logits": logits_eval,
            "logits_bce": logits_bce,
            "logits_direct": logits_direct,
            "logits_boundary": logits_boundary,
            "event_gate": gate_prob,
            "probs": probs,
            "latent_roll": z_seq,
            "latent_direct": z_direct,
            "static_geom_memory": static_zg,
            "solar_emb": solar_emb,
            "final_latent_state": final_latent,
            "max_latent_norm": max_latent_norm,
            "nonfinite_count": nonfinite_count,
            "inference_mode": eff_inf_mode,
            "is_causal": True,
            "execution_counters": counters,
        }
        if cand_attn_w is not None:
            res_b["candidate_attn_weights"] = cand_attn_w
        return res_b

    @torch.no_grad()
    def evaluate_strided_rollout(
        self,
        batch: Mapping[str, Any],
        *,
        stride: int = 4,
        mode: str = "causal_rollout",
        use_calibrated_temperature: bool = True,
    ) -> Dict[str, Any]:
        """Mode-explicit inference (`static_direct | causal_rollout | offline_interp`).

        - `static_direct`: independent per-frame static direct readout (`0` recurrent steps, `0` cross-frame coupling).
        - `causal_rollout` (default): strictly causal open-loop rollout (`stride in {1, 2, 4, 8}`).
          Anchors ONLY at segment starts and past multiples of `stride` (`anchor <= t`), NEVER reading
          any right-hand future anchor (`> t`), so future inputs at `t' > t` have `0.0` influence on `<= t`!
        - `offline_interp`: non-causal offline two-sided anchor interpolation (`[left_pos, right_pos]`).
        """
        if mode not in ("causal_rollout", "offline_interp", "static_direct"):
            raise ValueError(
                f"evaluate_strided_rollout mode must be 'causal_rollout', 'offline_interp', or 'static_direct', got {mode!r}"
            )
        eff_stride = 1 if mode == "static_direct" else max(int(stride), 1)
        if "candidate_anchor_stride" in batch and int(batch["candidate_anchor_stride"]) != eff_stride:
            raise ValueError("Candidate anchor stride differs from model rollout stride")
        if eff_stride > 1 and "candidate_anchor_mode" in batch and batch["candidate_anchor_mode"] != mode:
            raise ValueError("Candidate anchor mode differs from model rollout mode")
        if mode == "static_direct" or eff_stride == 1:
            cand_mat_cnt = 0
            if "candidate_tokens" not in batch and "az_rel_triangles" in batch:
                from .fast_data import materialize_gpu_tokens_from_azimuth_bank

                mat_batch = materialize_gpu_tokens_from_azimuth_bank(dict(batch))
                cand_mat_cnt = int(mat_batch["candidate_tokens"].shape[-4])
            else:
                mat_batch = batch
            fwd_mode = "static_direct" if mode == "static_direct" else "recurrent"
            out = dict(
                self.forward(
                    mat_batch,
                    use_calibrated_temperature=use_calibrated_temperature,
                    inference_mode=fwd_mode,
                )
            )
            out["inference_mode"] = mode
            out["is_causal"] = mode != "offline_interp"
            out["anchor_stride"] = 1
            out["candidate_attention_fraction"] = 1.0
            if "execution_counters" in out:
                ec = dict(out["execution_counters"])
                ec["candidate_materializations"] = cand_mat_cnt
                out["execution_counters"] = ec
            if "candidate_mask" in mat_batch:
                out["has_active_candidates"] = mat_batch["candidate_mask"].to(torch.bool).any(dim=-1)
            return out

        sun_dirs = batch["sun_dirs"]
        same_day_adj = batch.get("same_day_adjacent_mask")
        carry_prev = bool(batch.get("carry_state_from_prev_chunk", False) and batch.get("prev_frame_same_day", True))
        init_latent = batch.get("initial_latent_state")
        reanchor_k = int(batch.get("reanchor_every", self.reanchor_every) or 0)
        eff_disable_gate = bool(batch.get("disable_event_gate", self.disable_event_gate))
        want_cand_attn = bool(batch.get("return_candidate_attention", False))

        has_pre_tokens = "candidate_tokens" in batch and batch["candidate_tokens"] is not None
        if has_pre_tokens:
            tokens = batch["candidate_tokens"]
            cmask = batch["candidate_mask"].to(torch.bool)
            squeeze_batch = tokens.ndim == 4
            if squeeze_batch:
                tokens = tokens.unsqueeze(0)
                cmask = cmask.unsqueeze(0)
            if sun_dirs.ndim == 2:
                sun_dirs = sun_dirs.unsqueeze(0)
            bsz, t_steps, n_pts, _k_cands, _ = tokens.shape
            dev = tokens.device
            has_cands_all: Optional[torch.Tensor] = cmask.any(dim=-1)
        elif "az_rel_triangles" in batch:
            az_tris = batch["az_rel_triangles"]
            squeeze_batch = az_tris.ndim == 5 and sun_dirs.ndim == 2
            if sun_dirs.ndim == 2:
                sun_dirs = sun_dirs.unsqueeze(0)
            bsz = int(sun_dirs.shape[0])
            t_steps = int(sun_dirs.shape[1])
            n_pts = int(az_tris.shape[0] if az_tris.ndim == 5 else az_tris.shape[1])
            dev = az_tris.device
            has_cands_all = None
        else:
            raise KeyError("evaluate_strided_rollout requires `candidate_tokens` or `az_rel_triangles`.")

        _require_continuity_mask(t_steps, same_day_adj, batch)
        obs_btn, obs_static = _resolve_label_free_obs_mask(
            batch, bsz=bsz, t_steps=t_steps, n_pts=n_pts, device=dev
        )
        solar_kin_bt2, solar_speed_bt = _compute_solar_kinematics(
            sun_dirs,
            same_day_adj,
            prev_frame_sun_dir=batch.get("prev_frame_sun_dir") if carry_prev else None,
        )
        solar_emb = self.solar_encoder(sun_dirs)
        solar_exp = solar_emb.unsqueeze(2).expand(bsz, t_steps, n_pts, self.hidden_dim)

        if self.use_local_context:
            local_feat = batch["local_features"]
            if local_feat.ndim == 2:
                local_feat = local_feat.unsqueeze(0)
            norm_local = self.local_normalizer(local_feat)
            raw_g_knn_lf = batch.get("global_knn_local_features")
            g_knn_norm_local = (
                self.local_normalizer(raw_g_knn_lf.to(device=dev, dtype=local_feat.dtype))
                if raw_g_knn_lf is not None
                else None
            )
            frozen_static = batch.get("frozen_static_geometry") if not self.training and not torch.is_grad_enabled() else None
            if frozen_static is not None:
                static_zg = frozen_static.reshape(bsz,n_pts,self.hidden_dim)
            else:
                static_zg = self.static_geom_memory.encode_static(
                    norm_local,
                    receivers_xyz=batch.get("receivers_xyz"),
                    receiver_rc=batch.get("receiver_rc"),
                    receiver_obs_static=obs_static,
                    global_knn_norm_local=g_knn_norm_local,
                    global_knn_rel_xyz=batch.get("global_knn_rel_xyz"),
                    global_knn_dist=batch.get("global_knn_dist"),
                    global_knn_obs_valid=batch.get("global_knn_obs_valid"),
                    enable_mpnn=self.use_static_mpnn,
                )
            local_exp = self.static_geom_memory.modulate_solar_film(static_zg, solar_emb)
            solar_film_scale = self.static_geom_memory.solar_film_scale(solar_emb)
            local_relief = torch.linalg.norm(norm_local.to(torch.float32), dim=-1) / math.sqrt(
                float(self.local_dim)
            )
        else:
            norm_local = None
            g_knn_norm_local = None
            solar_film_scale = None
            static_zg = torch.zeros(
                (bsz, n_pts, self.hidden_dim), dtype=solar_emb.dtype, device=dev
            )
            local_exp = static_zg.unsqueeze(1).expand(bsz, t_steps, n_pts, self.hidden_dim)
            local_relief = None

        prior = batch.get("stride_anchor_state") if carry_prev else None
        if carry_prev and mode == "causal_rollout" and prior is None:
            raise ValueError("causal strided continuation requires stride_anchor_state from preceding output")
        geometry_identity = batch.get("candidate_geometry_identity")
        if prior is not None and prior.get("geometry_identity") != geometry_identity:
            raise ValueError("Candidate geometry changed; invalidate stride anchor cache and reanchor")
        if carry_prev and mode == "offline_interp":
            raise ValueError("offline interpolation requires complete uninterrupted day segments")
        phase = int(prior["next_phase"]) if prior is not None else 0
        if batch.get("candidate_anchor_mask") is not None:
            anchor_mask = batch["candidate_anchor_mask"].to(device=dev, dtype=torch.bool).reshape(-1)
            if int(anchor_mask.numel()) != t_steps:
                raise ValueError("candidate_anchor_mask must match the time dimension")
            next_phase = int(batch["candidate_anchor_next_phase"])
        else:
            # Legacy/custom tensor batches use the same planner as the data layer.
            from .ray_candidates import plan_anchor_frames
            import numpy as np

            if same_day_adj is not None:
                adjacent = same_day_adj.detach().to(device="cpu", dtype=torch.bool)
                if adjacent.ndim == 2:
                    adjacent = adjacent.all(dim=0)
                day_ids = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(~adjacent.numpy())])
            else:
                day_ids = np.zeros(t_steps, dtype=np.int64)
            planned, next_phase = plan_anchor_frames(
                day_ids, np.arange(t_steps, dtype=np.int64) * 120, eff_stride,
                initial_phase=phase, carry_from_previous=prior is not None,
                prev_frame_same_day=carry_prev, offline_interp=mode == "offline_interp",
            )
            anchor_mask = torch.as_tensor(planned, device=dev, dtype=torch.bool)
        anchor_idx = torch.nonzero(anchor_mask, as_tuple=False).squeeze(-1)
        if t_steps and not bool(anchor_mask[0]) and prior is None:
            raise ValueError("A chunk before its first anchor requires previous stride_anchor_state")

        cand_mat_frames = 0
        if has_pre_tokens:
            tok_anch = tokens[:, anchor_idx]
            cm_anch = cmask[:, anchor_idx]
        else:
            from .fast_data import (
                compute_canonical_azimuth_bins_torch,
                materialize_gpu_tokens_from_azimuth_bank,
            )

            anch_in: Dict[str, Any] = {
                "az_rel_triangles": batch["az_rel_triangles"],
                "az_opaque": batch["az_opaque"],
                "az_mask": batch["az_mask"],
                "sun_dirs": sun_dirs[:, anchor_idx].squeeze(0) if squeeze_batch else sun_dirs[:, anchor_idx],
            }
            if "azimuth_bin_idx" in batch and batch["azimuth_bin_idx"] is not None:
                ab_all = batch["azimuth_bin_idx"].to(device=dev, dtype=torch.long)
                anch_in["azimuth_bin_idx"] = ab_all[anchor_idx] if ab_all.ndim == 1 else ab_all[:, anchor_idx]
            anch_mat = materialize_gpu_tokens_from_azimuth_bank(anch_in)
            tok_anch = anch_mat["candidate_tokens"]
            cm_anch = anch_mat["candidate_mask"].to(torch.bool)
            if tok_anch.ndim == 4:
                tok_anch = tok_anch.unsqueeze(0)
                cm_anch = cm_anch.unsqueeze(0)
            cand_mat_frames = int(anchor_idx.numel())

            az_m = batch["az_mask"].to(device=dev, dtype=torch.bool)
            if az_m.ndim == 3:
                az_m = az_m.unsqueeze(0)
            az_has = az_m.any(dim=-1)  # (B, N, AzBins)
            num_bins = int(az_has.shape[-1])
            if "azimuth_bin_idx" in batch and batch["azimuth_bin_idx"] is not None:
                ab_full = batch["azimuth_bin_idx"].to(device=dev, dtype=torch.long)
                if ab_full.ndim == 1:
                    ab_full = ab_full.unsqueeze(0).expand(bsz, -1)
                bin_full = torch.remainder(ab_full, num_bins)
            else:
                bin_full = compute_canonical_azimuth_bins_torch(sun_dirs, num_azimuth_bins=num_bins)
            has_cands_all = torch.gather(
                az_has, 2, bin_full.unsqueeze(1).expand(bsz, n_pts, t_steps)
            ).permute(0, 2, 1)

        q_in_anch = torch.cat([local_exp[:, anchor_idx], solar_exp[:, anchor_idx]], dim=-1)
        if anchor_idx.numel():
            cand_anch, cand_attn_anch, readout_anch = self._encode_candidate_cross_attention(
                tok_anch, cm_anch, q_in_anch, solar_exp=solar_exp[:,anchor_idx],
                return_attn_weights=want_cand_attn, return_readout=True)
        else:
            readout_anch = q_in_anch.new_empty((bsz, 0, n_pts))
            cand_anch = q_in_anch.new_empty((bsz,0,n_pts,self.hidden_dim))
            cand_attn_anch = None
        has_cands_anch = cm_anch.any(dim=-1)
        local_anchor_count = int(anchor_idx.numel())
        if prior is not None and mode=="causal_rollout":
            cand_anch = torch.cat([prior["embedding"][:,None],cand_anch],dim=1)
            old_readout = prior.get("readout", cand_anch.new_zeros((bsz, n_pts)))
            old_has_cands = prior.get("has_cands", torch.zeros((bsz, n_pts), dtype=torch.bool, device=dev))
            has_cands_anch = torch.cat([old_has_cands[:, None], has_cands_anch], dim=1)
            readout_anch = torch.cat([old_readout[:, None], readout_anch], dim=1)
            anchor_idx = torch.cat([anchor_idx.new_tensor([-1]),anchor_idx])
            if want_cand_attn:
                old_attention = prior.get("attention")
                if old_attention is None:
                    old_attention = tok_anch.new_zeros((bsz, n_pts, tok_anch.shape[-2]))
                old_weight = old_attention[:, None]
                cand_attn_anch = torch.cat([old_weight,cand_attn_anch],dim=1) if cand_attn_anch is not None else old_weight
        anchor_state = {"embedding":cand_anch[:,-1].detach(),"next_phase":next_phase,
                        "readout":readout_anch[:,-1].detach(),
                        "has_cands":has_cands_anch[:,-1].detach(),
                        "geometry_identity":geometry_identity,
                        "attention":cand_attn_anch[:,-1].detach() if cand_attn_anch is not None else None}
        arange_t = torch.arange(t_steps, device=dev)
        if mode == "causal_rollout":
            # Strictly causal zero-order hold from most recent past anchor `anchor_idx[left_pos] <= t` (`0` future reads!)
            left_pos = (torch.searchsorted(anchor_idx, arange_t, right=True) - 1).clamp(
                0, max(int(anchor_idx.numel()) - 1, 0)
            )
            cand_step = cand_anch[:, left_pos]
            cand_readout = readout_anch[:, left_pos]
            has_cands_all = has_cands_anch[:, left_pos]
            cand_attn_all = cand_attn_anch[:, left_pos] if cand_attn_anch is not None else None
        else:
            # Explicit `offline_interp`: two-sided linear interpolation within same-day segments
            pos_in_anch = torch.searchsorted(anchor_idx, arange_t, right=False)
            right_pos = pos_in_anch.clamp(0, int(anchor_idx.numel()) - 1)
            is_exact_anchor = anchor_idx[right_pos] == arange_t
            left_pos = torch.where(is_exact_anchor, right_pos, (right_pos - 1).clamp_min(0))
            t_left = anchor_idx[left_pos].to(torch.float32)
            t_right = anchor_idx[right_pos].to(torch.float32)
            denom = (t_right - t_left).clamp_min(1.0)
            alpha = torch.where(
                is_exact_anchor,
                torch.zeros_like(t_left),
                ((arange_t.to(torch.float32) - t_left) / denom).clamp(0.0, 1.0),
            )
            cand_step = (1.0 - alpha.view(1, t_steps, 1, 1)) * cand_anch[:, left_pos] + alpha.view(
                1, t_steps, 1, 1
            ) * cand_anch[:, right_pos]
            has_cands_all = has_cands_anch[:, left_pos]
            cand_readout = ((1.0 - alpha.view(1, t_steps, 1)) * readout_anch[:, left_pos]
                            + alpha.view(1, t_steps, 1) * readout_anch[:, right_pos])
            cand_attn_all = cand_attn_anch[:, left_pos] if cand_attn_anch is not None else None

        step_h = self.step_fusion(torch.cat([local_exp, solar_exp, cand_step], dim=-1))
        device_type = "cuda" if step_h.is_cuda else "cpu"
        recurrent_step_count = 0
        reanchor_reset_count = 0
        with torch.amp.autocast(device_type=device_type, enabled=False):
            step_h32 = step_h.to(torch.float32)
            z_direct = self.recurrence.max_coord_bound * torch.tanh(
                self.recurrence.init_proj(self.recurrence.input_ln(step_h32))
            )
            z_list: List[torch.Tensor] = []

            init_offset = int(batch.get("intra_day_step_offset", 0) or 0) if carry_prev else 0
            intra_step = torch.full((bsz,), init_offset, dtype=torch.long, device=step_h32.device)
            reanchor_at_t0 = bool(reanchor_k > 0 and init_offset > 0 and (init_offset % reanchor_k) == 0)

            prev_chunk_z32: Optional[torch.Tensor] = None
            if carry_prev and init_latent is not None:
                prev_chunk_z32 = init_latent.to(dtype=torch.float32, device=step_h32.device)
                if prev_chunk_z32.ndim == 2:
                    prev_chunk_z32 = prev_chunk_z32.unsqueeze(0)
                if reanchor_at_t0:
                    z_cur = z_direct[:, 0]
                    reanchor_reset_count += 1
                else:
                    z_cur = self.recurrence.step(
                        step_h32[:, 0], prev_chunk_z32, solar_kinematics=solar_kin_bt2[:, 0]
                    )
                    recurrent_step_count += 1
                    if obs_btn is not None:
                        keep = _carry_observation_mask(obs_btn[:, 0], batch)
                        z_cur = torch.where(keep.unsqueeze(-1), z_cur, z_direct[:, 0])
            else:
                z_cur = z_direct[:, 0]
            z_list.append(z_cur)

            sdm = (
                same_day_adj.to(dtype=torch.bool, device=dev).unsqueeze(0).expand(bsz, -1)
                if (same_day_adj is not None and same_day_adj.ndim == 1)
                else (same_day_adj.to(dtype=torch.bool, device=dev) if same_day_adj is not None else None)
            )
            for t_i in range(1, t_steps):
                if sdm is not None:
                    intra_step = torch.where(
                        sdm[:, t_i - 1], intra_step + 1, torch.zeros_like(intra_step)
                    )
                else:
                    intra_step = intra_step + 1

                z_step = self.recurrence.step(step_h32[:, t_i], z_cur, solar_kinematics=solar_kin_bt2[:, t_i])
                recurrent_step_count += 1
                keep = sdm[:, t_i - 1].view(bsz, 1, 1) if sdm is not None else None
                if reanchor_k > 0:
                    is_reanchor_step = (intra_step > 0) & ((intra_step % reanchor_k) == 0)
                    if bool(is_reanchor_step.any().item()):
                        reanchor_reset_count += 1
                    not_reanchor = (~is_reanchor_step).view(bsz, 1, 1)
                    keep = not_reanchor if keep is None else (keep & not_reanchor)
                if obs_btn is not None:
                    obs_p = (obs_btn[:, t_i] & obs_btn[:, t_i - 1]).unsqueeze(-1)
                    keep = obs_p if keep is None else (keep & obs_p)
                z_cur = torch.where(keep, z_step, z_direct[:, t_i]) if keep is not None else z_step
                z_list.append(z_cur)

            z_seq = torch.stack(z_list, dim=1)
            logits_bce = self.bce_head(z_seq).squeeze(-1) + cand_readout
            logits_boundary = self.boundary_head(z_seq).squeeze(-1)
            logits_fused, gate_prob = self.focal_event_gate(
                z_seq,
                logits_bce,
                logits_boundary,
                same_day_adjacent_mask=same_day_adj,
                prev_chunk_latent=prev_chunk_z32,
                solar_angular_speed=solar_speed_bt,
                local_relief=local_relief,
                obs_mask=obs_btn,
                prev_chunk_obs_mask=batch.get("prev_chunk_obs_mask") if carry_prev else None,
                disable_gate=eff_disable_gate,
            )
            if self.enable_spatial_knn_readout:
                logits_fused = self.spatial_knn_readout(
                    z_seq,
                    logits_fused,
                    receivers_xyz=batch.get("receivers_xyz"),
                    receiver_rc=batch.get("receiver_rc"),
                    receiver_obs_static=obs_static,
                    norm_local=norm_local,
                    node_stem=self.static_geom_memory.node_stem,
                    solar_film_scale=solar_film_scale,
                    global_knn_norm_local=g_knn_norm_local,
                    global_knn_dist=batch.get("global_knn_dist"),
                    global_knn_obs_valid=batch.get("global_knn_obs_valid"),
                )
            logits_eval = (
                logits_fused / self.calibrated_temperature.clamp_min(1e-4)
                if use_calibrated_temperature
                else logits_fused
            )
            probs = torch.sigmoid(logits_eval)
            nonfinite_count = int((~torch.isfinite(logits_eval)).sum().item()) + int(
                (~torch.isfinite(z_seq)).sum().item()
            )
            latent_norms = torch.linalg.norm(z_seq, dim=-1)
            max_latent_norm = float(latent_norms.max().item()) if latent_norms.numel() > 0 else 0.0
            final_latent = z_seq[:, -1]

        frac = float(local_anchor_count) / float(max(t_steps, 1))
        counters = {
            "forward_calls": 1,
            "total_frames": int(t_steps),
            "anchor_frames": int(local_anchor_count),
            "actual_anchor_frames": int(batch.get("candidate_query_frames", local_anchor_count)),
            "bvh_savings_rate": 1.0 - float(batch.get("candidate_query_frames", local_anchor_count)) / max(t_steps, 1),
            "candidate_materializations": int(cand_mat_frames),
            "recurrent_steps": int(recurrent_step_count),
            "reanchor_resets": int(reanchor_reset_count),
        }
        if squeeze_batch:
            res_sq: Dict[str, Any] = {
                "logits": logits_eval.squeeze(0),
                "probs": probs.squeeze(0),
                "event_gate": gate_prob.squeeze(0),
                "latent_roll": z_seq.squeeze(0),
                "final_latent_state": final_latent.squeeze(0),
                "max_latent_norm": max_latent_norm,
                "nonfinite_count": nonfinite_count,
                "static_geom_memory": static_zg.squeeze(0),
                "anchor_stride": eff_stride,
                "stride_anchor_state": anchor_state,
                "candidate_attention_fraction": frac,
                "inference_mode": mode,
                "is_causal": mode != "offline_interp",
                "execution_counters": counters,
            }
            if has_cands_all is not None:
                res_sq["has_active_candidates"] = has_cands_all.squeeze(0)
            if cand_attn_all is not None:
                res_sq["candidate_attn_weights"] = cand_attn_all.squeeze(0)
            return res_sq
        res_full: Dict[str, Any] = {
            "logits": logits_eval,
            "probs": probs,
            "event_gate": gate_prob,
            "latent_roll": z_seq,
            "final_latent_state": final_latent,
            "max_latent_norm": max_latent_norm,
            "nonfinite_count": nonfinite_count,
            "static_geom_memory": static_zg,
            "anchor_stride": eff_stride,
                "stride_anchor_state": anchor_state,
            "candidate_attention_fraction": frac,
            "inference_mode": mode,
            "is_causal": mode != "offline_interp",
            "execution_counters": counters,
        }
        if has_cands_all is not None:
            res_full["has_active_candidates"] = has_cands_all
        if cand_attn_all is not None:
            res_full["candidate_attn_weights"] = cand_attn_all
        return res_full


class BoundedConvGRU2dCell(nn.Module):
    """Spatial 2D ConvGRU cell with bounded state `|H_{t,c,y,x}| <= max_coord_bound`."""

    def __init__(self, in_channels: int, hidden_channels: int, max_coord_bound: float = 1.0) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.max_coord_bound = float(max_coord_bound)
        self.conv_gates = nn.Conv2d(
            in_channels + hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1
        )
        self.conv_can = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1
        )

    def init_state(self, x0: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = x0.shape
        zeros_h = torch.zeros(
            (bsz, self.hidden_channels, h, w), dtype=torch.float32, device=x0.device
        )
        return self.step(x0, zeros_h)

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        hp32 = h_prev.to(torch.float32)
        h_in = hp32.to(x_t.dtype) if x_t.dtype != hp32.dtype else hp32
        combined = torch.cat([x_t, h_in], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined).to(torch.float32))
        update_gate, reset_gate = torch.chunk(gates, 2, dim=1)
        rh = reset_gate * hp32
        rh_in = rh.to(x_t.dtype) if x_t.dtype != rh.dtype else rh
        cand_in = torch.cat([x_t, rh_in], dim=1)
        proposal = self.max_coord_bound * torch.tanh(self.conv_can(cand_in).to(torch.float32))
        return (1.0 - update_gate) * hp32 + update_gate * proposal

    def step_preprojected(
        self,
        gate_x: torch.Tensor,
        can_x: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Exact ConvGRU step from pre-convolved input terms `gate_x` and `can_x` (zero `cat` copies)."""
        c_x = int(self.conv_gates.in_channels) - self.hidden_channels
        w_gh = self.conv_gates.weight[:, c_x:, :, :]
        w_ch = self.conv_can.weight[:, c_x:, :, :]
        hp32 = h_prev.to(torch.float32)
        h_cast = hp32.to(gate_x.dtype) if gate_x.dtype != hp32.dtype else hp32
        gate_h = F.conv2d(h_cast, w_gh, None, stride=1, padding=1)
        gate_sum = gate_x + gate_h
        gates = torch.sigmoid(gate_sum.to(torch.float32))
        update_gate, reset_gate = torch.chunk(gates, 2, dim=1)
        rh = reset_gate * hp32
        rh_cast = rh.to(can_x.dtype) if can_x.dtype != rh.dtype else rh
        can_h = F.conv2d(rh_cast, w_ch, None, stride=1, padding=1)
        can_sum = can_x + can_h
        proposal = self.max_coord_bound * torch.tanh(can_sum.to(torch.float32))
        return (1.0 - update_gate) * hp32 + update_gate * proposal


class LocalConvGRUGridModel(_BaseRindModel):
    """Recurrent spatial 2D ConvGRU baseline supporting strict `per_receiver_crop` (`L0-21`/`L0-64`) & `full_tile_512_halo`.

    - `"per_receiver_crop"` (default for `local_convgru_grid`): processes strictly isolated centered
      `(N, 9, P, P)` receiver crops (`P in {21, 64}`, `4.10m` or `12.50m`) with zero leakage from
      any tile pixel outside the `P x P` receiver window.
    - `"full_tile_512_halo"` (default for `full_tile_convgru`): executes `spatial_stem` and
      `BoundedConvGRU2dCell` ONCE on the `(B, 9, H, W)` tile grid with single-level temporal segment
      checkpointing and gathers exact `[rows, cols]` for all `N` receivers simultaneously.
    """

    model_type = "local_convgru_grid"
    ablation_tier = "Tier-B1_local_convgru"
    input_tier = "centered_local_grid9_plus_local16"
    has_event_gate = True

    def __init__(
        self,
        *,
        crop_size: int = 64,
        in_channels: int = GRID_CROP_CHANNELS,
        hidden_dim: int = 64,
        meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
        solar_frequencies: int = 4,
        max_coord_bound: float = 1.0,
        receiver_subbatch: Optional[int] = None,
        temporal_ckpt_segment: int = 32,
        spatial_execution_mode: str = "per_receiver_crop",
        is_full_tile_variant: bool = False,
        disable_event_gate: bool = False,
    ) -> None:
        super().__init__()
        if is_full_tile_variant:
            spatial_execution_mode = "full_tile_512_halo"
        if crop_size not in (21, 64) and not is_full_tile_variant:
            raise ValueError(f"crop_size must be 21 or 64, got {crop_size}")
        if spatial_execution_mode not in ("full_tile_512_halo", "per_receiver_crop"):
            raise ValueError(f"Invalid spatial_execution_mode: {spatial_execution_mode!r}")
        self.is_full_tile_variant = bool(is_full_tile_variant)
        if self.is_full_tile_variant:
            self.model_type = "full_tile_convgru"
            self.ablation_tier = "Tier-B1b_full_tile_convgru"
            self.input_tier = "full_tile_grid9_plus_local16"
        self.crop_size = int(crop_size)
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.meters_per_pixel = float(meters_per_pixel)
        self.solar_frequencies = int(solar_frequencies)
        self.max_coord_bound = float(max_coord_bound)
        default_subbatch = 128 if self.crop_size <= 32 else 32
        self.receiver_subbatch = max(
            int(default_subbatch if receiver_subbatch is None else receiver_subbatch), 1
        )
        self.temporal_ckpt_segment = max(int(temporal_ckpt_segment), 4)
        self.spatial_execution_mode = str(spatial_execution_mode)
        self.disable_event_gate = bool(disable_event_gate)
        if self.disable_event_gate:
            self.has_event_gate = False

        self.grid_normalizer = RunningNormalizer(self.in_channels)
        self.local_normalizer = RunningNormalizer(LOCAL_CONTEXT_DIM)
        self.solar_encoder = SphereTangentSolarEncoder(
            self.hidden_dim // 2, num_frequencies=self.solar_frequencies
        )

        self.spatial_stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_dim // 2, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim // 2, self.hidden_dim // 2, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim // 2, self.hidden_dim // 2, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.conv_gru = BoundedConvGRU2dCell(
            self.hidden_dim, self.hidden_dim, max_coord_bound=self.max_coord_bound
        )
        self.receiver_proj = nn.Sequential(
            nn.Linear(self.hidden_dim + LOCAL_CONTEXT_DIM, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.bce_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.focal_event_gate = FocalEventTransitionGate(self.hidden_dim)

    @property
    def patch_extent_m(self) -> float:
        if self.spatial_execution_mode == "full_tile_512_halo":
            return 512.0 * self.meters_per_pixel
        return float(self.crop_size) * self.meters_per_pixel

    @property
    def receptive_field_m(self) -> float:
        if self.spatial_execution_mode == "full_tile_512_halo":
            return 512.0 * self.meters_per_pixel
        return float(self.crop_size) * self.meters_per_pixel

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "ablation_tier": self.ablation_tier,
            "input_tier": self.input_tier,
            "crop_size": self.crop_size,
            "in_channels": self.in_channels,
            "hidden_dim": self.hidden_dim,
            "meters_per_pixel": self.meters_per_pixel,
            "patch_extent_m": self.patch_extent_m,
            "receptive_field_m": self.receptive_field_m,
            "solar_frequencies": self.solar_frequencies,
            "max_coord_bound": self.max_coord_bound,
            "receiver_subbatch": self.receiver_subbatch,
            "temporal_ckpt_segment": self.temporal_ckpt_segment,
            "spatial_execution_mode": self.spatial_execution_mode,
            "is_full_tile_variant": self.is_full_tile_variant,
            "disable_event_gate": self.disable_event_gate,
        }

    def _adapt_crop_channels(self, crop: torch.Tensor) -> torch.Tensor:
        c = int(crop.shape[-3])
        if c == self.in_channels:
            return crop
        if c == LEGACY_GRID_CROP_CHANNELS and self.in_channels == GRID_CROP_CHANNELS:
            ones = torch.ones(
                (*crop.shape[:-3], 1, crop.shape[-2], crop.shape[-1]),
                dtype=crop.dtype,
                device=crop.device,
            )
            return torch.cat([crop, ones], dim=-3)
        if c == GRID_CROP_CHANNELS and self.in_channels == LEGACY_GRID_CROP_CHANNELS:
            return crop[..., :LEGACY_GRID_CROP_CHANNELS, :, :]
        raise ValueError(
            f"Grid crop channel count {c} incompatible with model in_channels={self.in_channels}"
        )

    def _build_full_tile_9chw(self, batch: Mapping[str, Any]) -> Optional[torch.Tensor]:
        sf = batch.get("static_fields_8hw")
        gk = batch.get("geometry_known_hw")
        rc = batch.get("receiver_rc")
        if sf is None or gk is None or rc is None:
            return None
        sf_t = sf.to(torch.float32)
        gk_t = gk.to(dtype=torch.float32, device=sf_t.device)
        if sf_t.ndim == 3:
            sf_t = sf_t.unsqueeze(0)
        if gk_t.ndim == 2:
            gk_t = gk_t.unsqueeze(0)
        halo = gk_t.unsqueeze(1)  # (B, 1, H, W)
        cleaned_8 = torch.nan_to_num(sf_t, nan=0.0, posinf=0.0, neginf=0.0) * halo
        if self.in_channels == GRID_CROP_CHANNELS:
            return torch.cat([cleaned_8, halo], dim=1)
        return cleaned_8[:, : self.in_channels]

    def _extract_per_receiver_crops_from_tile(
        self,
        tile_9chw: torch.Tensor,
        rc_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Extract strictly isolated `(B, N, C, P, P)` receiver crops around `(row, col)`.

        Note: matches `fullday_data.extract_receiver_grid_crops` exact centered convention:
        `r0 = r - (P // 2)`, `c0 = c - (P // 2)` with zero-padded halo outside `[0, H) x [0, W)`.
        """
        bsz, c_in, h_tile, w_tile = tile_9chw.shape
        p = int(self.crop_size)
        pad = p // 2
        padded = F.pad(tile_9chw, (pad, p - pad, pad, p - pad), mode="constant", value=0.0)
        rc_t = rc_raw.to(device=tile_9chw.device, dtype=torch.long)
        if rc_t.ndim == 2:
            rc_t = rc_t.unsqueeze(0).expand(bsz, -1, -1)
        n_pts = int(rc_t.shape[1])
        offs = torch.arange(p, device=tile_9chw.device, dtype=torch.long)
        # In padded coordinates (offset by `pad`), `r_pad_start = rc_t[..., 0]` (`= r - pad + pad = r`)
        rows = (rc_t[..., 0].unsqueeze(-1) + offs).clamp(0, padded.shape[-2] - 1)  # (B, N, P)
        cols = (rc_t[..., 1].unsqueeze(-1) + offs).clamp(0, padded.shape[-1] - 1)  # (B, N, P)
        b_idx = torch.arange(bsz, device=tile_9chw.device, dtype=torch.long)[:, None, None, None]
        c_idx = torch.arange(c_in, device=tile_9chw.device, dtype=torch.long)[None, None, :, None, None]
        r_idx = rows[:, :, None, :, None]
        col_idx = cols[:, :, None, None, :]
        return padded[b_idx.unsqueeze(-1), c_idx, r_idx, col_idx]

    def _resolve_batch_crop(self, batch: Mapping[str, Any]) -> torch.Tensor:
        if self.spatial_execution_mode == "full_tile_512_halo":
            tile_9 = self._build_full_tile_9chw(batch)
            if tile_9 is not None:
                return tile_9
        for key in (
            f"receiver_crops_{self.crop_size}",
            "receiver_crops",
            f"grid_crop_{self.crop_size}",
            "grid_crop",
        ):
            val = batch.get(key)
            if val is not None:
                return self._adapt_crop_channels(val)
        tile_9 = self._build_full_tile_9chw(batch)
        if tile_9 is not None:
            if self.spatial_execution_mode == "per_receiver_crop" and batch.get("receiver_rc") is not None:
                return self._extract_per_receiver_crops_from_tile(tile_9, batch["receiver_rc"])
            return tile_9
        raise KeyError(
            "LocalConvGRUGridModel requires `static_fields_8hw`+`receiver_rc` or `receiver_crops` in batch."
        )

    @torch.no_grad()
    def update_normalizers(
        self,
        batch: Mapping[str, Any],
        *,
        firewall_token: FitFirewallToken,
        tile_ids: Sequence[str],
        expected_split_sha256: str,
    ) -> None:
        firewall_token.verify_tiles(tile_ids, expected_split_sha256)
        grid_in = self._resolve_batch_crop(batch)
        perm = grid_in.movedim(-3, -1)
        self.grid_normalizer.update(
            perm,
            firewall_token=firewall_token,
            tile_ids=tile_ids,
            expected_split_sha256=expected_split_sha256,
        )
        if "local_features" in batch:
            self.local_normalizer.update(
                batch["local_features"],
                firewall_token=firewall_token,
                tile_ids=tile_ids,
                expected_split_sha256=expected_split_sha256,
            )
        self.record_normalizer_tiles(tile_ids)

    def _rollout_full_tile_segment(
        self,
        g_static_b: torch.Tensor,
        c_static_b: torch.Tensor,
        sol_seg_bt: torch.Tensor,
        keep_seg_bt: torch.Tensor,
        h_init: torch.Tensor,
        rc_bn2: torch.Tensor,
        use_step_on_first: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run `O(1)` full-tile ConvGRU over `T_seg` steps and gather `[rows, cols]` for all `N` receivers."""
        bsz, _, hc, wc = g_static_b.shape
        t_seg = int(sol_seg_bt.shape[1])
        c_sol = int(sol_seg_bt.shape[2])
        c_stat = self.hidden_dim // 2
        c_x = c_stat + c_sol
        n_pts = int(rc_bn2.shape[1])

        sol_1x1 = sol_seg_bt.to(g_static_b.dtype).reshape(bsz * t_seg, c_sol, 1, 1)
        use_3x3_template = hc >= 3 and wc >= 3
        if use_3x3_template:
            # Exact border-aware 3x3 zero-padded convolution of spatially uniform `s_t`:
            # convolving a constant `(H, W)` field with a 3x3 padding=1 kernel produces only 9 spatial
            # zones `(top/interior/bottom) x (left/interior/right)`, avoiding multi-GiB cuDNN workspaces!
            sol_3x3 = sol_1x1.expand(bsz * t_seg, c_sol, 3, 3).contiguous()
            g_sol_3x3 = F.conv2d(
                sol_3x3,
                self.conv_gru.conv_gates.weight[:, c_stat:c_x, :, :],
                None,
                stride=1,
                padding=1,
            ).reshape(bsz, t_seg, 2 * self.hidden_dim, 3, 3)
            c_sol_3x3 = F.conv2d(
                sol_3x3,
                self.conv_gru.conv_can.weight[:, c_stat:c_x, :, :],
                None,
                stride=1,
                padding=1,
            ).reshape(bsz, t_seg, self.hidden_dim, 3, 3)
            r_zone = torch.ones((hc,), dtype=torch.long, device=g_static_b.device)
            r_zone[0] = 0
            r_zone[-1] = 2
            c_zone = torch.ones((wc,), dtype=torch.long, device=g_static_b.device)
            c_zone[0] = 0
            c_zone[-1] = 2
            g_sol_all = None
            c_sol_all = None
        else:
            sol_maps = sol_1x1.expand(bsz * t_seg, c_sol, hc, wc).contiguous()
            g_sol_all = F.conv2d(
                sol_maps,
                self.conv_gru.conv_gates.weight[:, c_stat:c_x, :, :],
                None,
                stride=1,
                padding=1,
            ).reshape(bsz, t_seg, 2 * self.hidden_dim, hc, wc)
            c_sol_all = F.conv2d(
                sol_maps,
                self.conv_gru.conv_can.weight[:, c_stat:c_x, :, :],
                None,
                stride=1,
                padding=1,
            ).reshape(bsz, t_seg, self.hidden_dim, hc, wc)
            g_sol_3x3 = None
            c_sol_3x3 = None
            r_zone = None
            c_zone = None

        b_idx = torch.arange(bsz, device=g_static_b.device)[:, None].expand(bsz, n_pts)
        rows = rc_bn2[..., 0]
        cols = rc_bn2[..., 1]

        zeros_h = torch.zeros_like(h_init)
        if use_3x3_template:
            assert g_sol_3x3 is not None and c_sol_3x3 is not None and r_zone is not None and c_zone is not None
            gx0 = g_static_b + g_sol_3x3[:, 0][:, :, r_zone, :][:, :, :, c_zone]
            cx0 = c_static_b + c_sol_3x3[:, 0][:, :, r_zone, :][:, :, :, c_zone]
        else:
            assert g_sol_all is not None and c_sol_all is not None
            gx0 = g_static_b + g_sol_all[:, 0]
            cx0 = c_static_b + c_sol_all[:, 0]
        if use_step_on_first:
            h_in0 = torch.where(keep_seg_bt[:, 0].view(bsz, 1, 1, 1), h_init, zeros_h)
        else:
            h_in0 = zeros_h
        h_cur = self.conv_gru.step_preprojected(gx0, cx0, h_in0)
        gathered: List[torch.Tensor] = [h_cur[b_idx, :, rows, cols]]

        for t_i in range(1, t_seg):
            if use_3x3_template:
                assert g_sol_3x3 is not None and c_sol_3x3 is not None and r_zone is not None and c_zone is not None
                gxt = g_static_b + g_sol_3x3[:, t_i][:, :, r_zone, :][:, :, :, c_zone]
                cxt = c_static_b + c_sol_3x3[:, t_i][:, :, r_zone, :][:, :, :, c_zone]
            else:
                assert g_sol_all is not None and c_sol_all is not None
                gxt = g_static_b + g_sol_all[:, t_i]
                cxt = c_static_b + c_sol_all[:, t_i]
            h_in = torch.where(keep_seg_bt[:, t_i].view(bsz, 1, 1, 1), h_cur, zeros_h)
            h_cur = self.conv_gru.step_preprojected(gxt, cxt, h_in)
            gathered.append(h_cur[b_idx, :, rows, cols])

        return torch.stack(gathered, dim=1), h_cur  # (B, T_seg, N, D), (B, D, H, W)

    def _rollout_segment(
        self,
        g_static: torch.Tensor,
        c_static: torch.Tensor,
        sol_seg_bt: torch.Tensor,
        keep_seg_bt: torch.Tensor,
        h_init: torch.Tensor,
        use_step_on_first: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run ConvGRU over `T_seg` steps on per-receiver crops `(B, sub_n, C, Hc, Wc)`."""
        bsz, sub_n, _, hc, wc = g_static.shape
        eff_b = bsz * sub_n
        t_seg = int(sol_seg_bt.shape[1])
        c_sol = int(sol_seg_bt.shape[2])
        c_stat = self.hidden_dim // 2
        c_x = c_stat + c_sol
        cy, cx = hc // 2, wc // 2
        centers: List[torch.Tensor] = []

        sol_1x1 = sol_seg_bt.to(g_static.dtype).reshape(bsz * t_seg, c_sol, 1, 1)
        use_3x3_template = hc >= 3 and wc >= 3
        if use_3x3_template:
            sol_3x3 = sol_1x1.expand(bsz * t_seg, c_sol, 3, 3).contiguous()
            g_sol_3x3 = F.conv2d(
                sol_3x3,
                self.conv_gru.conv_gates.weight[:, c_stat:c_x, :, :],
                None,
                stride=1,
                padding=1,
            ).reshape(bsz, t_seg, 1, 2 * self.hidden_dim, 3, 3)
            c_sol_3x3 = F.conv2d(
                sol_3x3,
                self.conv_gru.conv_can.weight[:, c_stat:c_x, :, :],
                None,
                stride=1,
                padding=1,
            ).reshape(bsz, t_seg, 1, self.hidden_dim, 3, 3)
            r_zone = torch.ones((hc,), dtype=torch.long, device=g_static.device)
            r_zone[0] = 0
            r_zone[-1] = 2
            c_zone = torch.ones((wc,), dtype=torch.long, device=g_static.device)
            c_zone[0] = 0
            c_zone[-1] = 2
            g_sol_all = None
            c_sol_all = None
        else:
            sol_maps = sol_1x1.expand(bsz * t_seg, c_sol, hc, wc).contiguous()
            g_sol_all = F.conv2d(
                sol_maps,
                self.conv_gru.conv_gates.weight[:, c_stat:c_x, :, :],
                None,
                stride=1,
                padding=1,
            ).reshape(bsz, t_seg, 1, 2 * self.hidden_dim, hc, wc)
            c_sol_all = F.conv2d(
                sol_maps,
                self.conv_gru.conv_can.weight[:, c_stat:c_x, :, :],
                None,
                stride=1,
                padding=1,
            ).reshape(bsz, t_seg, 1, self.hidden_dim, hc, wc)
            g_sol_3x3 = None
            c_sol_3x3 = None
            r_zone = None
            c_zone = None

        zeros_h = torch.zeros_like(h_init)
        if use_3x3_template:
            assert g_sol_3x3 is not None and c_sol_3x3 is not None and r_zone is not None and c_zone is not None
            gx0 = (g_static + g_sol_3x3[:, 0][:, :, :, r_zone, :][:, :, :, :, c_zone]).reshape(
                eff_b, 2 * self.hidden_dim, hc, wc
            )
            cx0 = (c_static + c_sol_3x3[:, 0][:, :, :, r_zone, :][:, :, :, :, c_zone]).reshape(
                eff_b, self.hidden_dim, hc, wc
            )
        else:
            assert g_sol_all is not None and c_sol_all is not None
            gx0 = (g_static + g_sol_all[:, 0]).reshape(eff_b, 2 * self.hidden_dim, hc, wc)
            cx0 = (c_static + c_sol_all[:, 0]).reshape(eff_b, self.hidden_dim, hc, wc)
        if use_step_on_first:
            keep0 = keep_seg_bt[:, 0].view(bsz, 1, 1, 1, 1).expand(bsz, sub_n, 1, 1, 1).reshape(eff_b, 1, 1, 1)
            h_in0 = torch.where(keep0, h_init, zeros_h)
        else:
            h_in0 = zeros_h
        h_cur = self.conv_gru.step_preprojected(gx0, cx0, h_in0)
        centers.append(h_cur[:, :, cy, cx])

        for t_i in range(1, t_seg):
            if use_3x3_template:
                assert g_sol_3x3 is not None and c_sol_3x3 is not None and r_zone is not None and c_zone is not None
                gxt = (g_static + g_sol_3x3[:, t_i][:, :, :, r_zone, :][:, :, :, :, c_zone]).reshape(
                    eff_b, 2 * self.hidden_dim, hc, wc
                )
                cxt = (c_static + c_sol_3x3[:, t_i][:, :, :, r_zone, :][:, :, :, :, c_zone]).reshape(
                    eff_b, self.hidden_dim, hc, wc
                )
            else:
                assert g_sol_all is not None and c_sol_all is not None
                gxt = (g_static + g_sol_all[:, t_i]).reshape(eff_b, 2 * self.hidden_dim, hc, wc)
                cxt = (c_static + c_sol_all[:, t_i]).reshape(eff_b, self.hidden_dim, hc, wc)
            keep_t = keep_seg_bt[:, t_i].view(bsz, 1, 1, 1, 1).expand(bsz, sub_n, 1, 1, 1).reshape(eff_b, 1, 1, 1)
            h_in = torch.where(keep_t, h_cur, zeros_h)
            h_cur = self.conv_gru.step_preprojected(gxt, cxt, h_in)
            centers.append(h_cur[:, :, cy, cx])

        return torch.stack(centers, dim=1), h_cur

    def _rollout_subbatch(
        self,
        crops_sub: torch.Tensor,
        solar_bt: torch.Tensor,
        sdm_transitions: torch.Tensor,
        init_h_sub: Optional[torch.Tensor],
        carry_prev: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process `(B, sub_n, C, Hc, Wc)` receiver crops across all `T` frames (single-level ckpt)."""
        bsz, sub_n, c_in, hc, wc = crops_sub.shape
        t_steps = int(solar_bt.shape[1])
        eff_b = bsz * sub_n

        norm_g = self.grid_normalizer(crops_sub.movedim(2, -1)).movedim(-1, 2)
        if c_in == GRID_CROP_CHANNELS:
            halo_mask = crops_sub[:, :, -1:, :, :].to(norm_g.dtype)
            norm_g = torch.cat([norm_g[:, :, :-1, :, :] * halo_mask, halo_mask], dim=2)
        static_feat = self.spatial_stem(norm_g.reshape(eff_b, c_in, hc, wc))
        c_stat = int(static_feat.shape[1])
        g_static = F.conv2d(
            static_feat,
            self.conv_gru.conv_gates.weight[:, :c_stat, :, :],
            self.conv_gru.conv_gates.bias,
            stride=1,
            padding=1,
        ).reshape(bsz, sub_n, 2 * self.hidden_dim, hc, wc)
        c_static = F.conv2d(
            static_feat,
            self.conv_gru.conv_can.weight[:, :c_stat, :, :],
            self.conv_gru.conv_can.bias,
            stride=1,
            padding=1,
        ).reshape(bsz, sub_n, self.hidden_dim, hc, wc)

        first_keep = torch.full(
            (bsz, 1), bool(carry_prev and init_h_sub is not None), dtype=torch.bool, device=crops_sub.device
        )
        full_keep_bt = torch.cat([first_keep, sdm_transitions], dim=1) if t_steps > 1 else first_keep

        if init_h_sub is not None and carry_prev:
            h_cur = init_h_sub.reshape(eff_b, self.hidden_dim, hc, wc).to(torch.float32)
        else:
            h_cur = torch.zeros(
                (eff_b, self.hidden_dim, hc, wc), dtype=torch.float32, device=crops_sub.device
            )

        seg_len = (
            max(self.temporal_ckpt_segment, 64)
            if (hc * wc >= 2048 and t_steps >= 128)
            else self.temporal_ckpt_segment
        )
        center_segs: List[torch.Tensor] = []
        use_ckpt = torch.is_grad_enabled() and t_steps > seg_len

        for t0 in range(0, t_steps, seg_len):
            t1 = min(t_steps, t0 + seg_len)
            sol_s = solar_bt[:, t0:t1]
            keep_s = full_keep_bt[:, t0:t1]
            step_first = (t0 > 0) or bool(carry_prev and init_h_sub is not None)
            if use_ckpt:
                c_seg, h_cur = grad_checkpoint(
                    self._rollout_segment,
                    g_static,
                    c_static,
                    sol_s,
                    keep_s,
                    h_cur,
                    step_first,
                    use_reentrant=False,
                )
            else:
                c_seg, h_cur = self._rollout_segment(
                    g_static, c_static, sol_s, keep_s, h_cur, step_first
                )
            center_segs.append(c_seg)

        centers_all = torch.cat(center_segs, dim=1)
        centers_btnd = centers_all.reshape(bsz, sub_n, t_steps, self.hidden_dim).permute(0, 2, 1, 3)
        h_final = h_cur.reshape(bsz, sub_n, self.hidden_dim, hc, wc)
        return centers_btnd, h_final

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        use_calibrated_temperature: bool = False,
        inference_mode: str = "recurrent",
    ) -> Dict[str, Any]:
        if inference_mode not in ("recurrent", "static_direct", "causal_rollout", "offline_interp"):
            raise ValueError(f"Unsupported inference_mode: {inference_mode!r}")
        local_feat = batch["local_features"]
        sun_dirs = batch["sun_dirs"]
        same_day_adj = batch.get("same_day_adjacent_mask")
        carry_prev = bool(batch.get("carry_state_from_prev_chunk", False) and batch.get("prev_frame_same_day", True)) and (inference_mode != "static_direct")
        init_conv_state = batch.get("initial_conv_state")
        init_latent = batch.get("initial_latent_state")

        squeeze_batch = False
        if sun_dirs.ndim == 2:
            sun_dirs = sun_dirs.unsqueeze(0)
            squeeze_batch = True
        if local_feat.ndim == 2:
            local_feat = local_feat.unsqueeze(0)

        bsz, t_steps, _ = sun_dirs.shape
        n_pts = int(local_feat.shape[1])
        _require_continuity_mask(t_steps, same_day_adj, batch)

        if same_day_adj is not None:
            sdm_bt = same_day_adj.to(dtype=torch.bool, device=sun_dirs.device)
            if sdm_bt.ndim == 1:
                sdm_bt = sdm_bt.unsqueeze(0).expand(bsz, -1)
        else:
            sdm_bt = torch.ones((bsz, max(t_steps - 1, 0)), dtype=torch.bool, device=sun_dirs.device)
        if inference_mode == "static_direct":
            sdm_bt = torch.zeros_like(sdm_bt)

        obs_btn, _obs_static = _resolve_label_free_obs_mask(
            batch, bsz=bsz, t_steps=t_steps, n_pts=n_pts, device=sun_dirs.device
        )
        prev_obs_bn = batch.get("prev_chunk_obs_mask") if carry_prev else None

        _, solar_speed_bt = _compute_solar_kinematics(
            sun_dirs,
            same_day_adj,
            prev_frame_sun_dir=batch.get("prev_frame_sun_dir") if carry_prev else None,
        )
        norm_local = self.local_normalizer(local_feat)
        local_relief = torch.linalg.norm(norm_local.detach().to(torch.float32), dim=-1) / math.sqrt(
            float(LOCAL_CONTEXT_DIM)
        )
        solar_emb = self.solar_encoder(sun_dirs)

        full_tile_9 = (
            self._build_full_tile_9chw(batch)
            if self.spatial_execution_mode == "full_tile_512_halo"
            else None
        )
        final_conv_state: Optional[torch.Tensor] = None
        static_encode_passes = 0
        gru_steps_executed = 0

        if full_tile_9 is not None:
            if int(full_tile_9.shape[0]) == 1 and bsz > 1:
                full_tile_9 = full_tile_9.expand(bsz, -1, -1, -1)
            _, c_in, hc, wc = full_tile_9.shape
            rc_t = batch["receiver_rc"].to(device=sun_dirs.device, dtype=torch.long)
            if rc_t.ndim == 2:
                rc_t = rc_t.unsqueeze(0).expand(bsz, -1, -1)
            if bool(((rc_t[..., 0] < 0) | (rc_t[..., 0] >= hc) | (rc_t[..., 1] < 0) | (rc_t[..., 1] >= wc)).any().item()):
                raise ValueError(f"receiver_rc out of bounds for full tile ({hc}, {wc})")

            norm_g = self.grid_normalizer(full_tile_9.movedim(1, -1)).movedim(-1, 1)
            if c_in == GRID_CROP_CHANNELS:
                halo_m = full_tile_9[:, -1:, :, :].to(norm_g.dtype)
                norm_g = torch.cat([norm_g[:, :-1, :, :] * halo_m, halo_m], dim=1)
            static_feat = self.spatial_stem(norm_g)
            static_encode_passes += 1
            c_stat = int(static_feat.shape[1])
            g_static_b = F.conv2d(
                static_feat,
                self.conv_gru.conv_gates.weight[:, :c_stat, :, :],
                self.conv_gru.conv_gates.bias,
                stride=1,
                padding=1,
            )
            c_static_b = F.conv2d(
                static_feat,
                self.conv_gru.conv_can.weight[:, :c_stat, :, :],
                self.conv_gru.conv_can.bias,
                stride=1,
                padding=1,
            )

            first_keep = torch.full(
                (bsz, 1),
                bool(carry_prev and init_conv_state is not None),
                dtype=torch.bool,
                device=sun_dirs.device,
            )
            full_keep_bt = torch.cat([first_keep, sdm_bt], dim=1) if t_steps > 1 else first_keep

            if carry_prev and init_conv_state is not None:
                h_cur = init_conv_state.to(dtype=torch.float32, device=sun_dirs.device)
                if h_cur.ndim == 3:
                    h_cur = h_cur.unsqueeze(0)
            else:
                h_cur = torch.zeros(
                    (bsz, self.hidden_dim, hc, wc), dtype=torch.float32, device=sun_dirs.device
                )

            seg_len = (
                min(self.temporal_ckpt_segment, 6)
                if (hc * wc >= 16384)
                else self.temporal_ckpt_segment
            )
            use_ckpt = torch.is_grad_enabled() and t_steps > seg_len
            seg_outs: List[torch.Tensor] = []
            for t0 in range(0, t_steps, seg_len):
                t1 = min(t_steps, t0 + seg_len)
                sol_s = solar_emb[:, t0:t1]
                keep_s = full_keep_bt[:, t0:t1]
                step_first = (t0 > 0) or bool(carry_prev and init_conv_state is not None)
                if use_ckpt:
                    z_s, h_cur = grad_checkpoint(
                        self._rollout_full_tile_segment,
                        g_static_b,
                        c_static_b,
                        sol_s,
                        keep_s,
                        h_cur,
                        rc_t,
                        step_first,
                        use_reentrant=False,
                    )
                else:
                    z_s, h_cur = self._rollout_full_tile_segment(
                        g_static_b, c_static_b, sol_s, keep_s, h_cur, rc_t, step_first
                    )
                seg_outs.append(z_s)
            raw_z = torch.cat(seg_outs, dim=1)  # (B, T, N, D)
            final_conv_state = h_cur.detach()
            gru_steps_executed += int(t_steps)
        else:
            grid_crop = self._resolve_batch_crop(batch)
            crop_rc = batch.get(f"crop_receiver_rc_{self.crop_size}", batch.get("crop_receiver_rc"))
            if grid_crop.ndim == 4 and int(grid_crop.shape[0]) == n_pts and bsz == 1:
                grid_crop = grid_crop.unsqueeze(0)
            elif grid_crop.ndim == 3:
                grid_crop = grid_crop.unsqueeze(0)

            per_receiver_crops = grid_crop.ndim == 5
            if per_receiver_crops:
                raw_z_chunks: List[torch.Tensor] = []
                h_final_chunks: List[torch.Tensor] = []
                if init_conv_state is not None and init_conv_state.ndim == 4:
                    init_conv_state = init_conv_state.unsqueeze(0)

                crop_area = int(grid_crop.shape[-2]) * int(grid_crop.shape[-1])
                if torch.is_grad_enabled():
                    eff_sub = min(self.receiver_subbatch, 8) if crop_area >= 2048 else self.receiver_subbatch
                else:
                    eff_sub = max(self.receiver_subbatch, 128)
                sub_chunk = max(1, eff_sub // max(bsz, 1))
                for s_n in range(0, n_pts, sub_chunk):
                    e_n = min(n_pts, s_n + sub_chunk)
                    crops_sub = grid_crop[:, s_n:e_n]
                    init_sub = init_conv_state[:, s_n:e_n] if init_conv_state is not None else None
                    static_encode_passes += 1
                    z_sub, hf_sub = self._rollout_subbatch(
                        crops_sub, solar_emb, sdm_bt, init_sub, carry_prev
                    )
                    raw_z_chunks.append(z_sub)
                    h_final_chunks.append(hf_sub.detach())
                raw_z = torch.cat(raw_z_chunks, dim=2)
                final_conv_state = torch.cat(h_final_chunks, dim=1)
                gru_steps_executed += int(t_steps)
            else:
                hc, wc = int(grid_crop.shape[-2]), int(grid_crop.shape[-1])
                if crop_rc is not None:
                    rc = crop_rc.to(torch.long)
                    if rc.ndim == 2:
                        rc = rc.unsqueeze(0).expand(bsz, -1, -1)
                    if bool(((rc[..., 0] < 0) | (rc[..., 0] >= hc) | (rc[..., 1] < 0) | (rc[..., 1] >= wc)).any().item()):
                        raise ValueError(
                            f"crop_receiver_rc out of bounds for shared crop ({hc}, {wc}); "
                            "pass per-receiver centered `receiver_crops` instead of clamping!"
                        )
                norm_g = self.grid_normalizer(grid_crop.movedim(1, -1)).movedim(-1, 1)
                static_feat = self.spatial_stem(norm_g)
                static_encode_passes += 1
                sol_cast = solar_emb.to(static_feat.dtype)

                if carry_prev and init_conv_state is not None:
                    ic = init_conv_state.to(torch.float32)
                    if ic.ndim == 3:
                        ic = ic.unsqueeze(0)
                    s0 = sol_cast[:, 0, :, None, None].expand(bsz, -1, hc, wc)
                    h_cur = self.conv_gru.step(torch.cat([static_feat, s0], dim=1), ic)
                else:
                    s0 = sol_cast[:, 0, :, None, None].expand(bsz, -1, hc, wc)
                    h_cur = self.conv_gru.init_state(torch.cat([static_feat, s0], dim=1))

                def _extract_shared(h_map: torch.Tensor) -> torch.Tensor:
                    if crop_rc is not None:
                        rc_l = crop_rc.to(torch.long)
                        if rc_l.ndim == 2:
                            rc_l = rc_l.unsqueeze(0).expand(bsz, -1, -1)
                        b_idx = torch.arange(bsz, device=h_map.device)[:, None].expand(bsz, n_pts)
                        return h_map[b_idx, :, rc_l[..., 0], rc_l[..., 1]]
                    return h_map[:, :, hc // 2, wc // 2].unsqueeze(1).expand(bsz, n_pts, self.hidden_dim)

                zeros_shared = torch.zeros_like(h_cur)
                rec_latents = [_extract_shared(h_cur)]
                for t_i in range(1, t_steps):
                    st = sol_cast[:, t_i, :, None, None].expand(bsz, -1, hc, wc)
                    xt = torch.cat([static_feat, st], dim=1)
                    keep = sdm_bt[:, t_i - 1].view(bsz, 1, 1, 1)
                    h_in = torch.where(keep, h_cur, zeros_shared)
                    h_cur = self.conv_gru.step(xt, h_in)
                    rec_latents.append(_extract_shared(h_cur))
                raw_z = torch.stack(rec_latents, dim=1)
                final_conv_state = h_cur.detach()
                gru_steps_executed += int(t_steps)

        loc_exp = norm_local.to(raw_z.dtype).unsqueeze(1).expand(bsz, t_steps, n_pts, -1)
        proj_out = self.receiver_proj(torch.cat([raw_z, loc_exp], dim=-1))

        device_type = "cuda" if sun_dirs.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            z_seq = self.max_coord_bound * torch.tanh(proj_out.to(torch.float32))

            prev_z32: Optional[torch.Tensor] = None
            if carry_prev and init_latent is not None:
                prev_z32 = init_latent.to(dtype=torch.float32, device=z_seq.device)
                if prev_z32.ndim == 2:
                    prev_z32 = prev_z32.unsqueeze(0)

            logits_bce = self.bce_head(z_seq).squeeze(-1)
            logits_boundary = self.boundary_head(z_seq).squeeze(-1)
            logits_fused, gate_prob = self.focal_event_gate(
                z_seq,
                logits_bce,
                logits_boundary,
                same_day_adjacent_mask=same_day_adj,
                prev_chunk_latent=prev_z32,
                solar_angular_speed=solar_speed_bt,
                local_relief=local_relief,
                obs_mask=obs_btn,
                prev_chunk_obs_mask=prev_obs_bn,
                disable_gate=self.disable_event_gate,
            )
            logits_eval = (
                logits_fused / self.calibrated_temperature.clamp_min(1e-4)
                if use_calibrated_temperature
                else logits_fused
            )
            nonfinite_count = int((~torch.isfinite(logits_eval)).sum().item()) + int(
                (~torch.isfinite(z_seq)).sum().item()
            )
            latent_norms = torch.linalg.norm(z_seq, dim=-1)
            max_latent_norm = float(latent_norms.max().item()) if latent_norms.numel() > 0 else 0.0
            probs = torch.sigmoid(logits_eval)
            final_latent = z_seq[:, -1]

        counters = {
            "inference_mode": str(inference_mode),
            "spatial_execution_mode": str(self.spatial_execution_mode),
            "static_encode_passes": int(static_encode_passes),
            "attention_steps": 0,
            "gru_steps": int(gru_steps_executed),
            "total_frames": int(t_steps),
        }

        if squeeze_batch:
            return {
                "logits": logits_eval.squeeze(0),
                "logits_bce": logits_bce.squeeze(0),
                "logits_boundary": logits_boundary.squeeze(0),
                "event_gate": gate_prob.squeeze(0),
                "probs": probs.squeeze(0),
                "latent_roll": z_seq.squeeze(0),
                "latent_direct": z_seq.squeeze(0),
                "final_latent_state": final_latent.squeeze(0),
                "final_conv_state": final_conv_state.squeeze(0) if final_conv_state is not None else None,
                "max_latent_norm": max_latent_norm,
                "nonfinite_count": nonfinite_count,
                "execution_counters": counters,
            }
        return {
            "logits": logits_eval,
            "logits_bce": logits_bce,
            "logits_boundary": logits_boundary,
            "event_gate": gate_prob,
            "probs": probs,
            "latent_roll": z_seq,
            "latent_direct": z_seq,
            "final_latent_state": final_latent,
            "final_conv_state": final_conv_state,
            "max_latent_norm": max_latent_norm,
            "nonfinite_count": nonfinite_count,
            "execution_counters": counters,
        }


class PointwiseMLPBaseline(_BaseRindModel):
    """Memoryless local + smooth Sphere-Tangent & Spherical-Harmonics MLP baseline (`pointwise_mlp`)."""

    model_type = "pointwise_mlp"
    ablation_tier = "Tier-B0_pointwise_mlp"
    input_tier = "local16_plus_solar_only"
    has_event_gate = False

    def __init__(
        self,
        *,
        local_dim: int = LOCAL_CONTEXT_DIM,
        hidden_dim: int = 96,
        solar_frequencies: int = 4,
        max_coord_bound: float = 1.0,
    ) -> None:
        super().__init__()
        self.local_dim = int(local_dim)
        self.hidden_dim = int(hidden_dim)
        self.solar_frequencies = int(solar_frequencies)
        self.max_coord_bound = float(max_coord_bound)

        self.local_normalizer = RunningNormalizer(self.local_dim)
        self.solar_encoder = SphereTangentSolarEncoder(
            self.hidden_dim, num_frequencies=self.solar_frequencies
        )
        self.trunk = nn.Sequential(
            nn.Linear(self.local_dim + self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.bce_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "ablation_tier": self.ablation_tier,
            "input_tier": self.input_tier,
            "local_dim": self.local_dim,
            "hidden_dim": self.hidden_dim,
            "solar_frequencies": self.solar_frequencies,
            "max_coord_bound": self.max_coord_bound,
        }

    @torch.no_grad()
    def update_normalizers(
        self,
        batch: Mapping[str, Any],
        *,
        firewall_token: FitFirewallToken,
        tile_ids: Sequence[str],
        expected_split_sha256: str,
    ) -> None:
        firewall_token.verify_tiles(tile_ids, expected_split_sha256)
        self.local_normalizer.update(
            batch["local_features"],
            firewall_token=firewall_token,
            tile_ids=tile_ids,
            expected_split_sha256=expected_split_sha256,
        )
        self.record_normalizer_tiles(tile_ids)

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        use_calibrated_temperature: bool = False,
        inference_mode: str = "recurrent",
    ) -> Dict[str, Any]:
        if inference_mode not in ("recurrent", "static_direct", "causal_rollout", "offline_interp"):
            raise ValueError(f"Unsupported inference_mode: {inference_mode!r}")
        local_feat = batch["local_features"]
        sun_dirs = batch["sun_dirs"]
        same_day_adj = batch.get("same_day_adjacent_mask")

        squeeze_batch = False
        if sun_dirs.ndim == 2:
            sun_dirs = sun_dirs.unsqueeze(0)
            squeeze_batch = True
        if local_feat.ndim == 2:
            local_feat = local_feat.unsqueeze(0)

        bsz, t_steps, _ = sun_dirs.shape
        n_pts = int(local_feat.shape[1])
        _require_continuity_mask(t_steps, same_day_adj, batch)

        norm_local = self.local_normalizer(local_feat)
        solar_emb = self.solar_encoder(sun_dirs)

        loc_exp = norm_local.unsqueeze(1).expand(bsz, t_steps, n_pts, -1)
        sol_exp = solar_emb.unsqueeze(2).expand(bsz, t_steps, n_pts, -1).to(loc_exp.dtype)
        raw_h = self.trunk(torch.cat([loc_exp, sol_exp], dim=-1))

        device_type = "cuda" if sun_dirs.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            z_seq = self.max_coord_bound * torch.tanh(raw_h.to(torch.float32))
            logits_bce = self.bce_head(z_seq).squeeze(-1)
            logits_boundary = self.boundary_head(z_seq).squeeze(-1)
            logits_eval = (
                logits_bce / self.calibrated_temperature.clamp_min(1e-4)
                if use_calibrated_temperature
                else logits_bce
            )
            nonfinite_count = int((~torch.isfinite(logits_eval)).sum().item()) + int(
                (~torch.isfinite(z_seq)).sum().item()
            )
            latent_norms = torch.linalg.norm(z_seq, dim=-1)
            max_latent_norm = float(latent_norms.max().item()) if latent_norms.numel() > 0 else 0.0
            probs = torch.sigmoid(logits_eval)
            gate_zeros = torch.zeros_like(logits_eval)
            final_latent = z_seq[:, -1]

        counters = {
            "inference_mode": str(inference_mode),
            "static_encode_passes": 1,
            "attention_steps": 0,
            "gru_steps": 0,
            "total_frames": int(t_steps),
        }

        if squeeze_batch:
            return {
                "logits": logits_eval.squeeze(0),
                "logits_bce": logits_bce.squeeze(0),
                "logits_boundary": logits_boundary.squeeze(0),
                "event_gate": gate_zeros.squeeze(0),
                "probs": probs.squeeze(0),
                "latent_roll": z_seq.squeeze(0),
                "latent_direct": z_seq.squeeze(0),
                "final_latent_state": final_latent.squeeze(0),
                "max_latent_norm": max_latent_norm,
                "nonfinite_count": nonfinite_count,
                "execution_counters": counters,
            }
        return {
            "logits": logits_eval,
            "logits_bce": logits_bce,
            "logits_boundary": logits_boundary,
            "event_gate": gate_zeros,
            "probs": probs,
            "latent_roll": z_seq,
            "latent_direct": z_seq,
            "final_latent_state": final_latent,
            "max_latent_norm": max_latent_norm,
            "nonfinite_count": nonfinite_count,
            "execution_counters": counters,
        }


class CanonicalProbe(_BaseRindModel):
    """Memoryless readout of masked candidate mean and maximum in each canonical37 channel."""
    has_event_gate = False
    input_tier = "canonical37_masked_mean_max"
    ablation_tier = "shallow_probe"

    def __init__(self, model_type, hidden_dim):
        super().__init__()
        self.model_type = model_type
        self.hidden_dim = 74 if model_type == "canonical37_linear" else int(hidden_dim)
        self.encoder = (nn.Identity() if model_type == "canonical37_linear"
                        else nn.Sequential(nn.Linear(74,self.hidden_dim),nn.GELU()))
        self.bce_head = nn.Linear(self.hidden_dim,1)

    def get_model_config(self):
        return dict(hidden_dim=self.hidden_dim,input_tier=self.input_tier,ablation_tier=self.ablation_tier)

    def update_normalizers(self,batch,*,firewall_token,tile_ids,expected_split_sha256):
        firewall_token.verify_tiles(tile_ids,expected_split_sha256)
        self.record_normalizer_tiles(tile_ids)

    def forward(self,batch,*,use_calibrated_temperature=False,inference_mode="recurrent"):
        tokens = batch["candidate_tokens"].float()
        mask = batch["candidate_mask"].bool().unsqueeze(-1)
        mean = tokens.masked_fill(~mask,0).sum(-2)/mask.sum(-2).clamp_min(1)
        maximum = tokens.masked_fill(~mask,-torch.inf).amax(-2)
        maximum = torch.where(mask.any(-2),maximum,0.)
        z = self.encoder(torch.cat((mean,maximum),dim=-1))
        logits = self.bce_head(z).squeeze(-1)
        pred = logits/self.calibrated_temperature if use_calibrated_temperature else logits
        return dict(logits=pred,logits_bce=logits,probs=pred.sigmoid(),event_gate=torch.zeros_like(pred),
                    latent_roll=z,latent_direct=z,final_latent_state=z.select(-3,-1),
                    max_latent_norm=float(z.detach().norm(dim=-1).max()),nonfinite_count=int((~torch.isfinite(pred)).sum()),
                    execution_counters={"attention_steps":0,"gru_steps":0})


TIER_G2_TOKEN_INDICES: Tuple[int, ...] = tuple(range(CANONICAL_TOKEN_DIM))
TIER_G1_TOKEN_INDICES: Tuple[int, ...] = tuple(i for i in range(CANONICAL_TOKEN_DIM) if i not in (22, 23, 24))
TIER_G0_RAW_INPUT_TOKEN_INDICES: Tuple[int, ...] = (
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    25,
    26,
    27,
)


def build_model(model_type: str, **kwargs: Any) -> _BaseRindModel:
    kwargs = dict(kwargs)
    kwargs.pop("zero_margin", None)
    kwargs.pop("ablation_mode", None)
    kwargs.pop("is_full_tile_variant", None)
    if model_type in ("canonical37_linear","canonical37_mlp"):
        return CanonicalProbe(model_type,kwargs.get("hidden_dim",96))
    if model_type in ("rind_full_geom", "rind_no_edge_margin", "rind_zero_margin", "rind_coarse_aabb"):
        kwargs.pop("crop_size", None)
        kwargs.pop("meters_per_pixel", None)
        kwargs.pop("spatial_execution_mode", None)
        mode_map = {
            "rind_full_geom": "full_geom",
            "rind_no_edge_margin": "no_edge_margin",
            "rind_zero_margin": "zero_margin",
            "rind_coarse_aabb": "coarse_aabb",
        }
        return RindCandidateWorldModel(ablation_mode=mode_map[model_type], **kwargs)
    if model_type in ("local_convgru_grid", "full_tile_convgru"):
        kwargs.pop("num_heads", None)
        kwargs.pop("max_candidates", None)
        kwargs.pop("local_dim", None)
        kwargs.pop("reanchor_every", None)
        kwargs.pop("enable_spatial_knn_readout", None)
        kwargs.pop("attention_mode", None)
        kwargs.pop("film_beta_bound", None)
        kwargs.pop("attn_qk_norm", None)
        kwargs.pop("cand_readout", None)
        kwargs.pop("use_static_mpnn", None)
        kwargs.pop("use_fourier102", None)
        if model_type == "full_tile_convgru":
            kwargs["spatial_execution_mode"] = "full_tile_512_halo"
            return LocalConvGRUGridModel(is_full_tile_variant=True, **kwargs)
        return LocalConvGRUGridModel(is_full_tile_variant=False, **kwargs)
    if model_type == "pointwise_mlp":
        kwargs.pop("num_heads", None)
        kwargs.pop("max_candidates", None)
        kwargs.pop("crop_size", None)
        kwargs.pop("meters_per_pixel", None)
        kwargs.pop("spatial_execution_mode", None)
        kwargs.pop("reanchor_every", None)
        kwargs.pop("enable_spatial_knn_readout", None)
        kwargs.pop("attention_mode", None)
        kwargs.pop("film_beta_bound", None)
        kwargs.pop("attn_qk_norm", None)
        kwargs.pop("cand_readout", None)
        kwargs.pop("disable_event_gate", None)
        kwargs.pop("use_static_mpnn", None)
        kwargs.pop("use_fourier102", None)
        return PointwiseMLPBaseline(**kwargs)
    raise ValueError(f"Unknown model_type {model_type!r}. Supported: {ALL_MODEL_TYPES}")


def compute_head_parameter_norms(
    model: nn.Module,
    aux_jepa_module: Optional[nn.Module] = None,
) -> Dict[str, float]:
    norms: Dict[str, float] = {}
    for head_name in ("bce_head", "boundary_head", "focal_event_gate", "spatial_knn_readout"):
        sub = getattr(model, head_name, None)
        if sub is not None:
            sq = sum(float(p.detach().to(torch.float64).square().sum().item()) for p in sub.parameters())
            norms[head_name] = math.sqrt(sq)
    if aux_jepa_module is not None:
        for aux_name in ("predictor", "context_aggregator", "reg_projector", "response_teacher", "ema_teacher"):
            sub = getattr(aux_jepa_module, aux_name, None)
            if sub is not None:
                sq = sum(float(p.detach().to(torch.float64).square().sum().item()) for p in sub.parameters())
                norms[aux_name] = math.sqrt(sq)
    return norms


def save_rind_checkpoint(
    path: Path | str,
    model: _BaseRindModel,
    *,
    split_contract_sha256: str,
    manifest_sha256: str,
    fit_subset_name: str,
    fit_tile_ids: Sequence[str],
    trained_from_scratch_fit_only: bool = True,
    seed: int = 0,
    calibrated: bool = False,
    calibration_tile_ids: Optional[Sequence[str]] = None,
    normalizer_tile_ids: Optional[Sequence[str]] = None,
    selection_rule: str = "fixed_final_epoch",
    num_train_updates: int = 0,
    trained_point_frames: int = 0,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize self-contained `.pt` checkpoint (v2) with complete provenance and head guards."""
    head_norms = compute_head_parameter_norms(model)
    if "bce_head" not in head_norms or head_norms["bce_head"] <= 0.0:
        raise RuntimeError(
            f"Refusing to save checkpoint with missing or zero `bce_head`: {head_norms}"
        )

    state_dict = model.state_dict()
    if not any(k.startswith("bce_head.") for k in state_dict.keys()):
        raise RuntimeError("`bce_head.*` keys missing from `model.state_dict()`!")

    fit_set = sorted({str(t) for t in fit_tile_ids})
    if not fit_set:
        raise RuntimeError("Refusing to save checkpoint with empty `fit_tile_ids`!")

    norm_tiles = sorted(
        {str(t) for t in (normalizer_tile_ids if normalizer_tile_ids is not None else model.consumed_normalizer_tile_ids)}
    )
    if norm_tiles and not (set(norm_tiles) <= set(fit_set)):
        raise RuntimeError(
            f"Normalizer consumed tiles outside `fit_tile_ids`: {sorted(set(norm_tiles) - set(fit_set))}"
        )

    cal_set = sorted({str(t) for t in (calibration_tile_ids or [])})
    if calibrated and not cal_set:
        raise RuntimeError("Calibrated checkpoint requires non-empty `calibration_tile_ids`!")
    if set(cal_set) & set(fit_set):
        raise RuntimeError(
            f"`calibration_tile_ids` overlap `fit_tile_ids`: {sorted(set(cal_set) & set(fit_set))}"
        )

    param_count = int(sum(p.numel() for p in model.parameters()))
    plain_config = json.loads(json.dumps(model.get_model_config(), allow_nan=False))
    provenance = json.loads(
        json.dumps(
            {
                "benchmark_contract": BENCHMARK_CONTRACT_ID,
                "package_version": PACKAGE_VERSION,
                "feature_version": FEATURE_SCHEMA_VERSION,
                "model_type": model.model_type,
                "model_config": plain_config,
                "ablation_tier": model.ablation_tier,
                "input_tier": model.input_tier,
                "parameter_count": param_count,
                "split_contract_sha256": str(split_contract_sha256),
                "manifest_sha256": str(manifest_sha256),
                "fit_subset_name": str(fit_subset_name),
                "fit_tile_ids": fit_set,
                "normalizer_tile_ids": norm_tiles,
                "calibration_tile_ids": cal_set,
                "trained_from_scratch_fit_only": bool(trained_from_scratch_fit_only),
                "seed": int(seed),
                "calibrated": bool(calibrated),
                "calibrated_temperature": float(model.calibrated_temperature.item()),
                "calibrated_threshold": float(model.calibrated_threshold.item()),
                "selection_rule": str(selection_rule),
                "num_train_updates": int(num_train_updates),
                "trained_point_frames": int(trained_point_frames),
                "extra": dict(extra_metadata or {}),
            },
            allow_nan=False,
        )
    )

    payload = {
        "schema": "rind.unified.checkpoint.v2",
        "benchmark_contract": BENCHMARK_CONTRACT_ID,
        "model_type": model.model_type,
        "model_config": plain_config,
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "head_parameter_norms": head_norms,
        "provenance": provenance,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(out_path)
    return provenance


def load_rind_checkpoint(
    path: Path | str,
    *,
    device: str | torch.device = "cpu",
    expected_split_sha256: Optional[str] = None,
    expected_manifest_sha256: Optional[str] = None,
    require_calibrated: bool = False,
) -> Tuple[_BaseRindModel, Dict[str, Any]]:
    """Load self-contained `.pt` checkpoint strictly (`strict=True`) and verify provenance."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    schema = payload.get("schema")
    if schema not in ("rind.unified.checkpoint.v2", "rind.unified.checkpoint.v1"):
        raise RuntimeError(f"Unsupported checkpoint schema: {schema!r}")
    if payload.get("benchmark_contract") != BENCHMARK_CONTRACT_ID:
        raise RuntimeError(
            f"Checkpoint benchmark contract mismatch: {payload.get('benchmark_contract')!r}"
        )

    provenance: Dict[str, Any] = dict(payload["provenance"])
    if (
        expected_split_sha256 is not None
        and provenance.get("split_contract_sha256") != expected_split_sha256
    ):
        raise RuntimeError(
            f"Checkpoint split_contract_sha256 ({provenance.get('split_contract_sha256')}) "
            f"!= expected {expected_split_sha256}"
        )
    if (
        expected_manifest_sha256 is not None
        and provenance.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise RuntimeError(
            f"Checkpoint manifest_sha256 ({provenance.get('manifest_sha256')}) "
            f"!= expected {expected_manifest_sha256}"
        )
    if schema == "rind.unified.checkpoint.v2":
        if provenance.get("feature_version") != FEATURE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Checkpoint feature_version ({provenance.get('feature_version')}) "
                f"!= {FEATURE_SCHEMA_VERSION}"
            )
    if require_calibrated and provenance.get("calibrated") is not True:
        raise RuntimeError(
            f"Checkpoint {ckpt_path.name} is NOT calibrated (`calibrated={provenance.get('calibrated')}`); "
            "run `calibrate` on the `calibration` split before formal evaluation."
        )

    model_type = str(payload["model_type"])
    raw_cfg = dict(payload["model_config"])
    if provenance.get("model_config") is not None:
        config_defaults = {"film_beta_bound": 0.0, "attn_qk_norm": False, "cand_readout": "attn_only"}
        declared = {**config_defaults, **raw_cfg}
        recorded = {**config_defaults, **dict(provenance["model_config"])}
        if declared != recorded:
            raise RuntimeError("Checkpoint model_config mismatch between payload and provenance")
    ctor_kwargs: Dict[str, Any] = {}
    if model_type in ("rind_full_geom", "rind_no_edge_margin", "rind_zero_margin", "rind_coarse_aabb"):
        for k in (
            "hidden_dim",
            "num_heads",
            "max_candidates",
            "local_dim",
            "solar_frequencies",
            "max_coord_bound",
            "max_token_chunk_queries",
            "reanchor_every",
            "enable_spatial_knn_readout",
            "use_static_mpnn",
            "use_fourier102",
            "attention_mode",
            "film_beta_bound",
            "attn_qk_norm",
            "cand_readout",
            "far_perspective",
            "disable_event_gate",
        ):
            if k in raw_cfg:
                ctor_kwargs[k] = raw_cfg[k]
    elif model_type in ("local_convgru_grid", "full_tile_convgru"):
        for k in (
            "crop_size",
            "in_channels",
            "hidden_dim",
            "meters_per_pixel",
            "solar_frequencies",
            "max_coord_bound",
            "receiver_subbatch",
            "temporal_ckpt_segment",
            "spatial_execution_mode",
            "disable_event_gate",
        ):
            if k in raw_cfg:
                ctor_kwargs[k] = raw_cfg[k]
    elif model_type in ("pointwise_mlp","canonical37_linear","canonical37_mlp"):
        for k in ("local_dim", "hidden_dim", "solar_frequencies", "max_coord_bound"):
            if k in raw_cfg:
                ctor_kwargs[k] = raw_cfg[k]

    model = build_model(model_type, **ctor_kwargs)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.record_normalizer_tiles(provenance.get("normalizer_tile_ids") or [])
    extra = provenance.get("extra")
    proto = (
        extra.get("calibration_candidate_protocol") or extra.get("candidate_protocol")
        if isinstance(extra, Mapping)
        else None
    )
    if isinstance(proto, Mapping) and "num_azimuth_bins" in proto:
        model.num_azimuth_bins = int(proto["num_azimuth_bins"])
    elif model_type.startswith("rind_"):
        warnings.warn(
            f"Checkpoint {ckpt_path.name} is missing candidate_protocol.num_azimuth_bins; "
            "downstream reconstruction will fall back to num_azimuth_bins=24.",
            UserWarning,
            stacklevel=2,
        )
    head_norms = compute_head_parameter_norms(model)
    if head_norms.get("bce_head", 0.0) <= 0.0:
        raise RuntimeError(f"Loaded checkpoint has zero `bce_head` norm: {head_norms}")

    model.to(torch.device(device))
    return model, provenance

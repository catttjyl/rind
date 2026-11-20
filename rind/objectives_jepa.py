"""Zero-Shrinkage ST-JEPA (`Mask + X`), Anti-Collapse Regularizers & Representation-Sufficiency Objectives.

Implements every mathematical and physical mechanism from:
  - `ST-JEPA五项核心方法改进报告-20260807.pdf` (Upgrades 1–5: `Mask + X` restricted response context,
    `D_X` bypass diagnostic, bounded `tanh` teacher + L1 + SIGReg/VICReg, `L_trans` increment matching),
  - `ST-JEPA与RIND核心亮点书及故事线_20261112.md`,
  - `UNIFIED_STANDARD_SPEC_V1.md`, and
  - Representation Sufficiency under Driver Changes and Structural Interventions (bidirectional
    local candidate-face perturbations and dropout for layered occlusion).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .label_budget import TileLabelDomain, coordinate_keys, coordinate_scores
from .fast_data import build_canonical37_torch
from .models import CANONICAL_TOKEN_DIM, DEFAULT_METERS_PER_PIXEL, RunningNormalizer
from .split_contract import FitFirewallToken

JEPA_MODES: Tuple[str, ...] = ("none", "geometry_mae", "ema_geom", "response", "response_teacher", "hybrid")
REG_MODES: Tuple[str, ...] = ("sigreg", "vicreg", "vicreg_sigreg", "none")
STRUCTURED_MASK_MODES: Tuple[str, ...] = (
    "spatial",
    "temporal",
    "spacetime_tube",
    "driver",
    "boundary",
)
ALL_MASK_MODES: Tuple[str, ...] = (
    "none",
    "random",
    "spatial",
    "boundary",
    "spatial_boundary",
    "contiguous_block",
    "buffered_block",
    "temporal",
    "spacetime_tube",
    "driver",
    "cycle_all",
)
JEPA_CONTEXT_MODES: Tuple[str, ...] = ("with_x", "without_x", "shuffled_x")
TRANSITION_MODES: Tuple[str, ...] = ("none", "driver_smooth", "unweighted_increment", "driver_smooth_l2", "event_aware")
EMA_TARGET_SCOPES: Tuple[str, ...] = ("head", "full_backbone")
DEFAULT_JEPA_LEADS: Tuple[int, ...] = (1, 2, 5, 10, 15)


@dataclass
class ObjectiveConfig:
    """Complete serializable objective configuration (`P0-2` & `P1` single source of truth)."""

    jepa_mode: str = "hybrid"
    ema_target_scope: str = "head"
    mask_mode: str = "spatial_boundary"
    mask_ratio: float = 0.25
    spatial_block_size: int = 24
    buffer_m: float = 1.5
    guard_buffer_m: float = 0.0
    jepa_context_mode: str = "with_x"
    reg_mode: str = "sigreg"
    transition_mode: str = "event_aware"
    jepa_weight: float = 0.25
    teacher_align_weight: float = 0.15
    reg_weight: float = 1.0
    transition_weight: float = 0.20
    dual_path_weight: float = 0.05
    counterfactual_weight: float = 0.05
    dominant_occluder_dropout_prob: float = 0.25
    boundary_metric_weight: float = 0.25
    hard_sun_weight: float = 0.20
    focal_event_weight: float = 0.35
    label_budget_fraction: float = 1.0
    label_budget_seed: int = 20261112
    pure_supervised: bool = False
    leads: Tuple[int, ...] = DEFAULT_JEPA_LEADS

    def __post_init__(self) -> None:
        if self.jepa_mode not in JEPA_MODES:
            raise ValueError(f"Invalid jepa_mode {self.jepa_mode!r}; expected {JEPA_MODES}")
        if self.ema_target_scope not in EMA_TARGET_SCOPES:
            raise ValueError(
                f"Invalid ema_target_scope {self.ema_target_scope!r}; expected {EMA_TARGET_SCOPES}"
            )
        if self.mask_mode not in ALL_MASK_MODES:
            raise ValueError(f"Invalid mask_mode {self.mask_mode!r}; expected {ALL_MASK_MODES}")
        if self.jepa_context_mode not in JEPA_CONTEXT_MODES:
            raise ValueError(
                f"Invalid jepa_context_mode {self.jepa_context_mode!r}; expected {JEPA_CONTEXT_MODES}"
            )
        if self.reg_mode not in REG_MODES:
            raise ValueError(f"Invalid reg_mode {self.reg_mode!r}; expected {REG_MODES}")
        if self.transition_mode not in TRANSITION_MODES:
            raise ValueError(
                f"Invalid transition_mode {self.transition_mode!r}; expected {TRANSITION_MODES}"
            )
        self.mask_ratio = float(min(max(self.mask_ratio, 0.05), 0.90))
        self.spatial_block_size = max(int(self.spatial_block_size), 2)
        if not math.isfinite(self.buffer_m) or self.buffer_m <= 0:
            raise ValueError("buffer_m must be finite and positive")
        self.guard_buffer_m = float(self.guard_buffer_m)
        if not math.isfinite(self.guard_buffer_m) or self.guard_buffer_m < 0.0:
            raise ValueError("guard_buffer_m must be finite and non-negative")
        self.label_budget_fraction = float(self.label_budget_fraction)
        if not (0.0 < self.label_budget_fraction <= 1.0):
            raise ValueError(
                f"label_budget_fraction must be in (0, 1], got {self.label_budget_fraction}"
            )
        self.leads = tuple(int(x) for x in self.leads)
        if self.pure_supervised:
            self.jepa_mode = "none"
            self.reg_mode = "none"
            self.transition_mode = "none"
            self.jepa_weight = 0.0
            self.teacher_align_weight = 0.0
            self.reg_weight = 0.0
            self.transition_weight = 0.0
            self.dual_path_weight = 0.0
            self.counterfactual_weight = 0.0
            self.dominant_occluder_dropout_prob = 0.0
            self.guard_buffer_m = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["leads"] = list(self.leads)
        return d

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObjectiveConfig":
        kw = dict(raw)
        if "leads" in kw and kw["leads"] is not None:
            kw["leads"] = tuple(int(x) for x in kw["leads"])
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in kw.items() if k in valid_fields}
        return cls(**filtered)


def transition_distance(student_delta, teacher_delta, mode):
    """Original driver smoothness is the squared L2 norm of the student change."""
    if mode == "driver_smooth_l2":
        return student_delta.square().sum(dim=-1)
    return (student_delta - teacher_delta.detach()).abs().mean(dim=-1)


def compute_label_budget_mask(
    n_pts: int,
    *,
    label_budget_fraction: float,
    label_budget_seed: int = 20261112,
    tile_id: str = "",
    receiver_rc: Optional[torch.Tensor] = None,
    domain: Optional[TileLabelDomain] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Select exact tile quotas on a declared domain. Standalone coordinates
    without a domain use a stable hash threshold (fraction in expectation).
    """
    frac = float(label_budget_fraction)
    if not 0 < frac <= 1:
        raise ValueError("label fraction must be in (0, 1]")
    if frac == 1 or n_pts == 0:
        return torch.ones(n_pts, dtype=torch.bool, device=device)
    if receiver_rc is None:
        raise ValueError("low-label budgets need stable receiver coordinates")
    rc = receiver_rc.detach().cpu()
    if rc.ndim == 3 and rc.shape[0] == 1:
        rc = rc[0]
    keys = coordinate_keys(rc.numpy())
    if keys.shape != (n_pts,):
        raise ValueError("receiver count does not match label support")
    if domain is None:
        selected = coordinate_scores(keys, tile_id, label_budget_seed) < frac * 2**31
    else:
        selected = domain.select(keys, tile_id, label_budget_seed, frac)
    return torch.as_tensor(selected, dtype=torch.bool, device=device)


def previous_label_validity(batch):
    """Read the v5 producer alias without accepting contradictory masks."""
    current = batch.get("prev_frame_valid_mask")
    legacy = batch.get("prev_frame_valid")
    if current is not None and legacy is not None and not torch.equal(current, legacy):
        raise ValueError("conflicting previous-frame label validity masks")
    return current if current is not None else legacy


def _authorized_grid_boundaries(batch, truth, valid):
    """Certify the capped Euclidean distance using only authorized raster cells."""
    dist = torch.zeros_like(truth, dtype=torch.float32)
    support = torch.zeros_like(truth, dtype=torch.bool)
    rc = batch.get("receiver_rc")
    if rc is None or not rc.numel():
        return dist, support
    if rc.ndim == 3:
        if truth.ndim < 2 or rc.shape[0] != truth.shape[0]:
            raise ValueError("batched coordinates require matching batch dimension")
        parts = [_authorized_grid_boundaries(dict(batch,receiver_rc=coords), y, v)
                 for coords,y,v in zip(rc,truth,valid)]
        return tuple(torch.stack(items) for items in zip(*parts))
    rc = rc.to(device=truth.device,dtype=torch.long)
    spacing = float(batch.get("grid_spacing_m",batch.get("meters_per_pixel",DEFAULT_METERS_PER_PIXEL)))
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("grid spacing must be finite and positive")
    radius = 2.25
    limit = int(math.ceil(radius/spacing))
    pitch = int(rc[:,1].max()-rc[:,1].min()) + 2*limit + 1
    keys = rc[:,0]*pitch+rc[:,1]
    ordered, order = keys.sort()
    observed = valid & ((truth==0)|(truth==1))
    shells = {}
    for dr in range(-limit,limit+1):
        for dc in range(-limit,limit+1):
            d2 = dr*dr+dc*dc
            if d2 and math.sqrt(d2)*spacing <= radius:
                shells.setdefault(d2,[]).append(dr*pitch+dc)
    # Process concentric rings. Once a nearer unknown is found no more distant
    # opposite can certify this receiver. An equally distant opposite fixes the
    # distance even if other cells in its ring are unknown or outside the tile.
    flat_y, flat_v = truth.reshape(-1,len(keys)), observed.reshape(-1,len(keys))
    flat_d, flat_s = dist.reshape_as(flat_y),support.reshape_as(flat_v)
    for n0 in range(0,len(keys),128):
        ns = slice(n0,n0+128)
        for t0 in range(0,len(flat_y),32):
            ts = slice(t0,t0+32)
            searching = flat_v[ts,ns].clone()
            for d2, offsets in sorted(shells.items()):
                if not searching.any():
                    break
                target_keys = keys[ns,None] + keys.new_tensor(offsets)
                pos = torch.searchsorted(ordered,target_keys).clamp_max(len(keys)-1)
                neighbors = order[pos]
                authorized = flat_v[ts,neighbors] & (ordered[pos]==target_keys)
                opposite = (authorized & (flat_y[ts,neighbors]!=flat_y[ts,ns,None])).any(-1)
                hit = searching & opposite
                flat_d[ts,ns].masked_fill_(hit,math.sqrt(d2)*spacing)
                flat_s[ts,ns] |= hit
                searching &= ~opposite & authorized.all(-1)
            flat_d[ts,ns].masked_fill_(searching,radius)
            flat_s[ts,ns] |= searching
    return dist,support


def apply_label_budget_view(
    batch: Mapping[str, Any],
    *,
    label_budget_fraction: float,
    label_budget_seed: int = 20261112,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]:
    """Sanitize `batch` under `label_budget_fraction < 1.0` so hidden `1 - frac` labels CANNOT be read!

    Zeroes hidden truth values, marks hidden `valid_mask` and `boundary_distance_valid` False, and
    strips pre-packed `response_5x5_bitpack` (which would otherwise carry hidden neighbor bits from
    disk) so `unpack_response_5x5_features_torch` reconstructs `5x5` patches strictly from labeled points.
    """
    if float(label_budget_fraction) >= 1.0 - 1e-9:
        return dict(batch), None

    out = dict(batch)
    ref = batch.get("truth", batch.get("local_features"))
    if ref is None:
        return out, None
    dev = device if device is not None else ref.device
    n_pts = int(ref.shape[-1]) if batch.get("truth") is not None else int(ref.shape[-2])
    raw_tid = batch.get("tile_id")
    if not raw_tid:
        tids_seq = batch.get("tile_ids")
        raw_tid = tids_seq[0] if (tids_seq and len(tids_seq) > 0) else ""
    tile_id = str(raw_tid)
    bmask_n = compute_label_budget_mask(
        n_pts,
        label_budget_fraction=label_budget_fraction,
        label_budget_seed=label_budget_seed,
        tile_id=tile_id,
        receiver_rc=batch.get("receiver_rc"),
        domain=batch.get("tile_label_domain"),
        device=dev,
    )

    def _mask_bt_n(t: torch.Tensor, fill_val: float | int | bool) -> torch.Tensor:
        t_dev = t.to(device=dev)
        m = bmask_n.view(*([1] * (t_dev.ndim - 1)), n_pts)
        if isinstance(fill_val, bool):
            return t_dev.to(torch.bool) & m
        return torch.where(m, t_dev, torch.full_like(t_dev, fill_val))

    if out.get("valid_mask") is not None:
        out["valid_mask"] = _mask_bt_n(out["valid_mask"], False)
    if out.get("truth") is not None:
        out["truth"] = _mask_bt_n(out["truth"], 0)
    prev_valid = previous_label_validity(out)
    if prev_valid is not None:
        out["prev_frame_valid_mask"] = _mask_bt_n(prev_valid, False)
        out.pop("prev_frame_valid", None)
    if out.get("prev_frame_truth") is not None:
        out["prev_frame_truth"] = _mask_bt_n(out["prev_frame_truth"], 0)
    for prefix in ("", "prev_frame_"):
        truth = out.get(prefix+"truth")
        valid = out.get(prefix+"valid_mask")
        if truth is not None and valid is not None:
            dist, cert = _authorized_grid_boundaries(out,truth,valid)
            out[prefix+"boundary_distance_m"] = dist
            out[prefix+"boundary_distance_valid"] = cert
    # Strip pre-packed 5x5 bitpacks so hidden neighbor labels can NEVER leak into 5x5 descriptors
    out.pop("response_5x5_bitpack", None)
    out.pop("prev_frame_response_5x5_bitpack", None)
    out["label_budget_mask"] = bmask_n
    return out, bmask_n


def unpack_response_5x5_features_torch(
    batch: Mapping[str, Any],
    *,
    bsz: int,
    t_steps: int,
    n_pts: int,
    device: torch.device,
) -> torch.Tensor:
    """Build `54D` spatial-temporal response descriptor `(B, T, N, 54)` for `ResponseTargetTeacher` / `X`.

    Slots:
      - `0..24`  (25): `5x5` local sunlit patch (`0.0` if invalid/unknown/masked)
      - `25..49` (25): `5x5` local valid mask (`1.0` if observed, `0.0` if `UNKNOWN`/OOB/masked)
      - `50`     (1):  center signed boundary distance `tanh(signed_d / 2.0)`
      - `51`     (1):  `exp(-|d| / 0.5)` sub-half-meter boundary indicator
      - `52`     (1):  temporal response delta `y_t - y_{t-1}` (with cross-subchunk `prev_frame_*` carry!)
      - `53`     (1):  center valid flag
    """
    truth = batch["truth"].to(device=device)
    valid = batch["valid_mask"].to(device=device, dtype=torch.bool)
    if truth.ndim == 2:
        truth = truth.unsqueeze(0)
    if valid.ndim == 2:
        valid = valid.unsqueeze(0)
    valid = valid & ((truth == 0) | (truth == 1))
    y_f = torch.where(valid, (truth > 0).to(torch.float32), torch.zeros_like(truth, dtype=torch.float32))
    v_f = valid.to(torch.float32)

    bpack = batch.get("response_5x5_bitpack")
    if bpack is not None:
        bp = bpack.to(device=device, dtype=torch.int64)
        if bp.ndim == 2:
            bp = bp.unsqueeze(0)
        shifts = torch.arange(25, device=device, dtype=torch.int64).view(1, 1, 1, 25)
        sun_25 = ((bp.unsqueeze(-1) >> shifts) & 1).to(torch.float32)
        val_25 = ((bp.unsqueeze(-1) >> (shifts + 25)) & 1).to(torch.float32) * v_f.unsqueeze(-1)
        sun_25 = sun_25 * val_25
    elif "receiver_rc" in batch and batch["receiver_rc"] is not None:
        rc = batch["receiver_rc"].to(device=device, dtype=torch.long)
        if rc.ndim == 3:
            rc = rc[0]
        r_min = int(rc[:, 0].min().item()) if rc.numel() > 0 else 0
        r_max = int(rc[:, 0].max().item()) if rc.numel() > 0 else 0
        c_min = int(rc[:, 1].min().item()) if rc.numel() > 0 else 0
        c_max = int(rc[:, 1].max().item()) if rc.numel() > 0 else 0
        h_box = max(r_max - r_min + 5, 5)
        w_box = max(c_max - c_min + 5, 5)
        r_loc = (rc[:, 0] - r_min + 2).clamp(2, h_box - 3)
        c_loc = (rc[:, 1] - c_min + 2).clamp(2, w_box - 3)

        sun_chunks: List[torch.Tensor] = []
        val_chunks: List[torch.Tensor] = []
        t_step_chunk = max(1, min(64, t_steps))
        dr = torch.tensor([-2, -1, 0, 1, 2], device=device, dtype=torch.long)
        rr_off, cc_off = torch.meshgrid(dr, dr, indexing="ij")
        rr_25 = (r_loc[:, None] + rr_off.reshape(1, 25)).clamp(0, h_box - 1)  # (N, 25)
        cc_25 = (c_loc[:, None] + cc_off.reshape(1, 25)).clamp(0, w_box - 1)  # (N, 25)
        flat_lin_n25 = rr_25 * w_box + cc_25
        flat_center_n = r_loc * w_box + c_loc

        for t0 in range(0, t_steps, t_step_chunk):
            t1 = min(t_steps, t0 + t_step_chunk)
            tc = t1 - t0
            canvas_s = torch.zeros((bsz * tc, h_box * w_box), dtype=torch.float32, device=device)
            canvas_v = torch.zeros((bsz * tc, h_box * w_box), dtype=torch.float32, device=device)
            idx_c = flat_center_n.unsqueeze(0).expand(bsz * tc, n_pts)
            canvas_s.scatter_(1, idx_c, y_f[:, t0:t1].reshape(bsz * tc, n_pts))
            canvas_v.scatter_(1, idx_c, v_f[:, t0:t1].reshape(bsz * tc, n_pts))
            idx_p = flat_lin_n25.reshape(1, n_pts * 25).expand(bsz * tc, n_pts * 25)
            sun_chunks.append(canvas_s.gather(1, idx_p).reshape(bsz, tc, n_pts, 25))
            val_chunks.append(canvas_v.gather(1, idx_p).reshape(bsz, tc, n_pts, 25))
        sun_25 = torch.cat(sun_chunks, dim=1)
        val_25 = torch.cat(val_chunks, dim=1)
    else:
        sun_25 = y_f.unsqueeze(-1).expand(bsz, t_steps, n_pts, 25)
        val_25 = v_f.unsqueeze(-1).expand(bsz, t_steps, n_pts, 25)

    b_dist = batch.get("boundary_distance_m")
    b_val = batch.get("boundary_distance_valid")
    if b_dist is not None and b_val is not None:
        bd = b_dist.to(device=device, dtype=torch.float32)
        bv = b_val.to(device=device, dtype=torch.bool)
        if bd.ndim == 2:
            bd = bd.unsqueeze(0)
        if bv.ndim == 2:
            bv = bv.unsqueeze(0)
        bv_eff = bv & valid
        bd_clean = torch.where(bv_eff & torch.isfinite(bd), bd, torch.full_like(bd, 4.0))
        sign = torch.where(y_f > 0.5, torch.ones_like(y_f), -torch.ones_like(y_f))
        signed_d = torch.tanh((sign * bd_clean) / 2.0) * bv_eff.to(torch.float32)
        near_05 = torch.exp(-bd_clean.abs() / 0.5) * bv_eff.to(torch.float32)
    else:
        signed_d = (2.0 * y_f - 1.0) * v_f
        near_05 = torch.zeros_like(y_f)

    dy = torch.zeros_like(y_f)
    prev_valid = previous_label_validity(batch)
    if (
        bool(batch.get("carry_state_from_prev_chunk", False))
        and batch.get("prev_frame_truth") is not None
        and prev_valid is not None
        and t_steps >= 1
    ):
        prev_t = batch["prev_frame_truth"].to(device=device).reshape(bsz, n_pts)
        prev_v = prev_valid.to(device=device, dtype=torch.bool).reshape(bsz, n_pts)
        prev_v = prev_v & ((prev_t == 0) | (prev_t == 1))
        prev_sd = bool(batch.get("prev_frame_same_day", False))
        if prev_sd:
            prev_y = torch.where(prev_v, (prev_t > 0).to(torch.float32), torch.zeros_like(y_f[:, 0]))
            dy[:, 0] = (y_f[:, 0] - prev_y) * (valid[:, 0] & prev_v).to(torch.float32)

    if t_steps >= 2:
        raw_dy = (y_f[:, 1:] - y_f[:, :-1]) * (valid[:, 1:] & valid[:, :-1]).to(torch.float32)
        sdm = batch.get("same_day_adjacent_mask")
        if sdm is not None:
            sdm_f = sdm.to(device=device, dtype=torch.float32)
            if sdm_f.ndim == 1:
                sdm_f = sdm_f.view(1, -1, 1)
            elif sdm_f.ndim == 2:
                sdm_f = sdm_f.unsqueeze(-1)
            raw_dy = raw_dy * sdm_f
        dy[:, 1:] = raw_dy

    return torch.cat(
        [
            sun_25,
            val_25,
            signed_d.unsqueeze(-1),
            near_05.unsqueeze(-1),
            dy.unsqueeze(-1),
            v_f.unsqueeze(-1),
        ],
        dim=-1,
    )


def build_restricted_context_response_features_torch(
    batch: Mapping[str, Any],
    context_mask: torch.Tensor,
    *,
    bsz: int,
    t_steps: int,
    n_pts: int,
    device: torch.device,
) -> torch.Tensor:
    """Construct restricted response observation `X = S_u|_{Omega_C}` with **0.0 leakage** from `Omega_T`.

    In `Omega_T` (where `context_mask == False`) and `~valid_mask`:
      - `truth` is zeroed BEFORE any `5x5` spatial scatter or temporal difference,
      - `valid_mask` is set to `False`,
      - `boundary_distance_m` / `boundary_distance_valid` are cleared,
      - `response_5x5_bitpack` is stripped so `5x5` local patches around `r in Omega_C` NEVER read
        adjacent `Omega_T` bits,
      - final `54D` features are multiplied by `context_mask` so `X[Omega_T] == 0.0` identically.
    """
    cm = context_mask.to(device=device, dtype=torch.bool).reshape(bsz, t_steps, n_pts)
    raw_valid = batch["valid_mask"].to(device=device, dtype=torch.bool)
    if raw_valid.ndim == 2:
        raw_valid = raw_valid.unsqueeze(0)
    ctx_valid = raw_valid & cm

    raw_truth = batch["truth"].to(device=device)
    if raw_truth.ndim == 2:
        raw_truth = raw_truth.unsqueeze(0)
    ctx_truth = torch.where(ctx_valid, raw_truth, torch.zeros_like(raw_truth))

    restricted_batch: Dict[str, Any] = {
        "truth": ctx_truth,
        "valid_mask": ctx_valid,
        "receiver_rc": batch.get("receiver_rc"),
        "same_day_adjacent_mask": batch.get("same_day_adjacent_mask"),
        # Deliberately omit `response_5x5_bitpack` and cross-chunk `prev_frame_truth` if masked
        "carry_state_from_prev_chunk": False,
    }
    restricted_batch["boundary_distance_m"], restricted_batch["boundary_distance_valid"] = (
        _authorized_grid_boundaries(batch,ctx_truth,ctx_valid))
    feats = unpack_response_5x5_features_torch(
        restricted_batch, bsz=bsz, t_steps=t_steps, n_pts=n_pts, device=device
    )
    return feats * ctx_valid.unsqueeze(-1).to(feats.dtype)


class ResponseTargetTeacher(nn.Module):
    """Highlight Book `08-07 Upgrade 5` + Gen-2 `ResponseTargetTeacher` (`FitFirewallToken` gated)."""

    RESPONSE_DIM = 54

    def __init__(self, hidden_dim: int, ema_decay: float = 0.996) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ema_decay = float(ema_decay)
        self.response_normalizer = RunningNormalizer(self.RESPONSE_DIM)
        self.online_encoder = nn.Sequential(
            nn.Linear(self.RESPONSE_DIM + self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_normalizer(
        self,
        batch: Mapping[str, Any],
        *,
        firewall_token: FitFirewallToken,
        tile_ids: Sequence[str],
        expected_split_sha256: str,
        device: torch.device,
    ) -> None:
        firewall_token.verify_tiles(tile_ids, expected_split_sha256)
        truth = batch["truth"]
        bsz = 1 if truth.ndim == 2 else int(truth.shape[0])
        t_steps = int(truth.shape[-2])
        n_pts = int(truth.shape[-1])
        feats = unpack_response_5x5_features_torch(
            batch, bsz=bsz, t_steps=t_steps, n_pts=n_pts, device=device
        )
        valid = batch["valid_mask"].to(device=device, dtype=torch.bool)
        if valid.ndim == 2:
            valid = valid.unsqueeze(0)
        self.response_normalizer.update(
            feats,
            firewall_token=firewall_token,
            tile_ids=tile_ids,
            expected_split_sha256=expected_split_sha256,
            valid_mask=valid,
        )

    @torch.no_grad()
    def step_ema(self) -> None:
        d = self.ema_decay
        for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            p_t.data.mul_(d).add_(p_o.data, alpha=1.0 - d)

    def encode_online_and_target(
        self,
        batch: Mapping[str, Any],
        solar_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode full unmasked valid field `S_u` (NEVER receiving `mask_token`!) into `(z_on, z_tg)`."""
        sol = solar_emb.to(torch.float32)
        if sol.ndim == 2:
            sol = sol.unsqueeze(0)
        bsz, t_steps, _ = sol.shape
        n_pts = int(batch["truth"].shape[-1])
        raw_r = unpack_response_5x5_features_torch(
            batch, bsz=bsz, t_steps=t_steps, n_pts=n_pts, device=sol.device
        )
        norm_r = self.response_normalizer(raw_r)
        sol_exp = sol.detach().unsqueeze(2).expand(bsz, t_steps, n_pts, self.hidden_dim)
        inp = torch.cat([norm_r, sol_exp], dim=-1)
        z_online = torch.tanh(self.online_encoder(inp))
        with torch.no_grad():
            z_target = torch.tanh(self.target_encoder(inp)).detach()
        return z_online, z_target

    def encode_restricted_context_x(
        self,
        batch: Mapping[str, Any],
        context_mask: torch.Tensor,
        solar_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Encode restricted context `X = S_u|_{Omega_C}` on student side (`0.0` leakage from `Omega_T`)."""
        sol = solar_emb.to(torch.float32)
        if sol.ndim == 2:
            sol = sol.unsqueeze(0)
        bsz, t_steps, _ = sol.shape
        n_pts = int(batch["truth"].shape[-1])
        cm = context_mask.to(device=sol.device, dtype=torch.bool).reshape(bsz, t_steps, n_pts)
        raw_x = build_restricted_context_response_features_torch(
            batch, cm, bsz=bsz, t_steps=t_steps, n_pts=n_pts, device=sol.device
        )
        norm_x = raw_x * cm.unsqueeze(-1).to(torch.float32)  # Public bounded features; teacher statistics include hidden responses.
        sol_exp = sol.unsqueeze(2).expand(bsz, t_steps, n_pts, self.hidden_dim)
        inp = torch.cat([norm_x, sol_exp], dim=-1)
        z_ctx_x = torch.tanh(self.online_encoder(inp)) * cm.unsqueeze(-1).to(torch.float32)
        return z_ctx_x


class EMAGeometricTarget(nn.Module):
    """Label-Free EMA Geometric-Driver Teacher (`ema_geom` & `hybrid` modes — `0` label reads)."""

    def __init__(self, hidden_dim: int, ema_decay: float = 0.996) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ema_decay = float(ema_decay)
        self.online_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.target_proj = copy.deepcopy(self.online_proj)
        for p in self.target_proj.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def step_ema(self) -> None:
        d = self.ema_decay
        for p_t, p_o in zip(self.target_proj.parameters(), self.online_proj.parameters()):
            p_t.data.mul_(d).add_(p_o.data, alpha=1.0 - d)

    def encode_online_and_target(
        self,
        z_direct: torch.Tensor,
        solar_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zd = z_direct.to(torch.float32)
        sol = solar_emb.to(torch.float32)
        if zd.ndim == 3:
            zd = zd.unsqueeze(0)
        if sol.ndim == 2:
            sol = sol.unsqueeze(0)
        bsz, t_steps, n_pts, _ = zd.shape
        sol_exp = sol.unsqueeze(2).expand(bsz, t_steps, n_pts, self.hidden_dim)
        inp = torch.cat([zd, sol_exp], dim=-1)
        z_on = torch.tanh(self.online_proj(inp))
        with torch.no_grad():
            z_tg = torch.tanh(self.target_proj(inp.detach())).detach()
        return z_on, z_tg


class CrossReceiverContextAttention(nn.Module):
    """Highlight 1 Cross-Receiver Context Aggregator (`Omega_C -> Omega_T` fusion for `Mask + X`).

    Aggregates restricted context representations `e_X` from visible context receivers `r' in Omega_C`
    into target queries `r in Omega_T` using spatial coordinate embeddings + distance kernel bias.
    When `context_mask` is empty or `context_mode == "without_x"`, outputs `null_context_token` (`D_X = 0`).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pos_proj = nn.Linear(3, self.hidden_dim)
        self.q_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.null_context_token = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_dim))
        self.dist_log_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _resolve_pos_xyz(
        self,
        *,
        bsz: int,
        n_pts: int,
        device: torch.device,
        receivers_xyz: Optional[torch.Tensor] = None,
        receiver_rc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if receivers_xyz is not None:
            rx = receivers_xyz.to(device=device, dtype=torch.float32)
            if rx.ndim == 2:
                rx = rx.unsqueeze(0).expand(bsz, -1, -1)
            rx_centered = rx - rx.mean(dim=1, keepdim=True)
            return rx_centered / 50.0
        if receiver_rc is not None:
            rc = receiver_rc.to(device=device, dtype=torch.float32)
            if rc.ndim == 2:
                rc = rc.unsqueeze(0).expand(bsz, -1, -1)
            xz_m = (rc - rc.mean(dim=1, keepdim=True)) * DEFAULT_METERS_PER_PIXEL / 50.0
            zeros_y = torch.zeros((bsz, n_pts, 1), dtype=torch.float32, device=device)
            return torch.cat([xz_m[..., :1], zeros_y, xz_m[..., 1:2]], dim=-1)
        lin = torch.linspace(-0.5, 0.5, steps=max(n_pts, 1), device=device, dtype=torch.float32)
        zeros = torch.zeros_like(lin)
        return torch.stack([lin, zeros, zeros], dim=-1).unsqueeze(0).expand(bsz, -1, -1)

    def forward(
        self,
        z_geom_query: torch.Tensor,
        z_geom_key: torch.Tensor,
        context_x_emb: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        receivers_xyz: Optional[torch.Tensor] = None,
        receiver_rc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return aggregated cross-receiver context `(B, T, N, D)`."""
        bsz, t_steps, n_pts, d_dim = z_geom_query.shape
        cm = context_mask.to(device=z_geom_query.device, dtype=torch.bool).reshape(bsz, t_steps, n_pts)
        null_tok = self.null_context_token.to(dtype=torch.float32, device=z_geom_query.device).expand(
            bsz, t_steps, n_pts, d_dim
        )
        if not bool(cm.any().item()) or n_pts == 0:
            return null_tok

        pos_norm = self._resolve_pos_xyz(
            bsz=bsz,
            n_pts=n_pts,
            device=z_geom_query.device,
            receivers_xyz=receivers_xyz,
            receiver_rc=receiver_rc,
        )
        pos_emb = self.pos_proj(pos_norm).unsqueeze(1).expand(bsz, t_steps, n_pts, d_dim)

        # Select among legal keys in physical lexicographic order, separately
        # for every frame. Even a singleton at the last storage index is retained.
        width = min(128, n_pts)
        order = torch.arange(n_pts,device=cm.device).expand(bsz,n_pts)
        for axis in (2,1,0):
            order = order.gather(1, pos_norm[...,axis].gather(1,order).argsort(dim=1,stable=True))
        order = order[:,None].expand(-1,t_steps,-1)
        legal = cm.gather(2,order)
        packed = order.gather(2,(~legal).long().argsort(dim=2,stable=True))
        count = legal.sum(2,keepdim=True)
        slots = torch.arange(width,device=cm.device)[None,None,:]
        rank = slots * count // width
        rank = torch.where(count <= width, slots, rank)
        key_idx = packed.gather(2,rank)
        def gather_keys(values):
            return values.gather(2,key_idx[...,None].expand(-1,-1,-1,values.shape[-1]))
        zk_sub = gather_keys(z_geom_key)
        vx_sub = gather_keys(context_x_emb)
        cm_sub = cm.gather(2,key_idx)
        pos_k_emb = gather_keys(pos_emb)
        pos_k_norm = gather_keys(pos_norm[:,None].expand(-1,t_steps,-1,-1))

        if not bool(cm_sub.any().item()):
            return null_tok

        q = self.q_proj(torch.cat([z_geom_query, pos_emb], dim=-1)) / math.sqrt(float(d_dim))
        k = self.k_proj(torch.cat([zk_sub, pos_k_emb], dim=-1))
        v = self.v_proj(vx_sub * cm_sub.unsqueeze(-1).to(torch.float32))

        dist_mat = torch.cdist(pos_norm[:,None], pos_k_norm, p=2.0)  # (B, 1, N, M)
        dist_scale = F.softplus(self.dist_log_scale)
        logits = torch.matmul(q, k.transpose(-1, -2)) - dist_scale * dist_mat  # (B, T, N, M)
        logits_masked = logits.masked_fill(~cm_sub.unsqueeze(2), -1e9)
        has_any_ctx_bt = cm_sub.any(dim=-1, keepdim=True).unsqueeze(-1)  # (B, T, 1, 1)
        attn = torch.softmax(logits_masked, dim=-1) * cm_sub.unsqueeze(2).to(torch.float32)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        agg = self.out_proj(torch.matmul(attn, v))
        return torch.where(has_any_ctx_bt, agg, null_tok)


def permute_context_values_across_receivers(
    context_x_emb: torch.Tensor,
    context_mask: torch.Tensor,
    *,
    step_seed: int = 0,
) -> torch.Tensor:
    """Permute `X` strictly within active context slots `Omega_C` (`pi(X)`), leaving `Omega_T` at `0`!"""
    bsz, t_steps, n_pts, d_dim = context_x_emb.shape
    cm = context_mask.to(device=context_x_emb.device, dtype=torch.bool).reshape(bsz, t_steps, n_pts)
    if n_pts < 2 or not bool(cm.any().item()):
        return context_x_emb

    out = torch.zeros_like(context_x_emb)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(step_seed) + 31337)
    # Deterministic derangement-like random permutation on CPU indices, gathered within active `cm`
    rand_scores = torch.rand((bsz, t_steps, n_pts), generator=gen).to(context_x_emb.device)
    # Force inactive slots to sort after active slots so active context slots permute strictly among active context slots!
    active_keys = torch.where(cm, rand_scores, torch.full_like(rand_scores, 2.0))
    orig_keys = torch.where(cm, torch.zeros_like(rand_scores), torch.full_like(rand_scores, 2.0))
    perm_idx = torch.argsort(active_keys, dim=-1)
    sorted_orig_idx = torch.argsort(orig_keys, dim=-1, stable=True)
    # Shift by 1 inside active count when `>= 2` active slots exist to avoid fixed-point identity on tiny `N`
    src_vals = torch.gather(context_x_emb, 2, perm_idx.unsqueeze(-1).expand(-1, -1, -1, d_dim))
    out.scatter_(2, sorted_orig_idx.unsqueeze(-1).expand(-1, -1, -1, d_dim), src_vals)
    # Ensure exact shift if accidental identity occurred when `>= 2` context points are active
    same_check = (out - context_x_emb).abs().amax() < 1e-12
    if bool(same_check.item()):
        # Cyclic shift within active context slots per `(b, t)`
        rolled = torch.zeros_like(context_x_emb)
        for b_i in range(bsz):
            for t_i in range(t_steps):
                idx_act = torch.nonzero(cm[b_i, t_i], as_tuple=False).squeeze(-1)
                if int(idx_act.numel()) >= 2:
                    rolled[b_i, t_i, idx_act] = context_x_emb[b_i, t_i, torch.roll(idx_act, shifts=1, dims=0)]
                elif int(idx_act.numel()) == 1:
                    rolled[b_i, t_i, idx_act] = context_x_emb[b_i, t_i, idx_act]
        return rolled * cm.unsqueeze(-1).to(context_x_emb.dtype)
    return out * cm.unsqueeze(-1).to(context_x_emb.dtype)


def permitted_context(mask, valid, mode, receivers_xyz, buffer_m, guard_buffer_m: float = 0.0):
    if mode == "none":
        return torch.zeros_like(valid)
    context = ~mask & valid
    eff_guard = (
        max(float(buffer_m), float(guard_buffer_m))
        if mode == "buffered_block"
        else float(guard_buffer_m)
    )
    if eff_guard <= 0.0:
        return context
    if receivers_xyz is None:
        raise ValueError("Metric guard buffer requires receiver XYZ")
    bsz, _, n_pts = mask.shape
    xy = receivers_xyz.to(device=mask.device, dtype=torch.float64).reshape(-1, n_pts, 3)[..., [0, 2]]
    xy = xy.expand(bsz, -1, -1)
    for b in range(bsz):
        patterns, inverse = mask[b].unique(dim=0, return_inverse=True)
        for j, pattern in enumerate(patterns):
            targets = xy[b, pattern]
            if targets.numel():
                near = torch.cat([torch.cdist(block, targets).amin(1) <= eff_guard
                                  for block in xy[b].split(128)])
                context[b] &= ~((inverse == j)[:, None] & near[None, :])
    return context


class MultiHorizonLatentPredictor(nn.Module):
    """Highlight 1 & 3 Predictor: `Mask + X` Disjoint Context-to-Target L1 Prediction + `D_X` Diagnostic."""

    def __init__(
        self,
        hidden_dim: int,
        leads: Sequence[int] = DEFAULT_JEPA_LEADS,
        mask_ratio: float = 0.25,
        spatial_block_size: int = 24,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.leads: Tuple[int, ...] = tuple(int(x) for x in leads)
        self.mask_ratio = float(mask_ratio)
        self.spatial_block_size = max(int(spatial_block_size), 2)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_dim))
        self.driver_mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_dim))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        nn.init.normal_(self.driver_mask_token, mean=0.0, std=0.02)

        self.context_aggregator = CrossReceiverContextAttention(self.hidden_dim)
        self.lead_embed = nn.Sequential(
            nn.Linear(4, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

    def _sample_2d_spatial_block_mask(
        self,
        bsz: int,
        n_pts: int,
        device: torch.device,
        gen: torch.Generator,
        *,
        receiver_rc: Optional[torch.Tensor] = None,
        receivers_xyz: Optional[torch.Tensor] = None,
        target_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """Sample contiguous 2D spatial blocks on `receiver_rc` (or 2D XZ plane) of shape `(B, N)`."""
        ratio = float(self.mask_ratio if target_ratio is None else target_ratio)
        if n_pts <= 1:
            return torch.ones((bsz, n_pts), dtype=torch.bool, device=device)

        if receiver_rc is not None and receiver_rc.numel() >= 2 * n_pts:
            rc = receiver_rc.detach().to(device="cpu", dtype=torch.float32)
            if rc.ndim == 3:
                rc = rc[0]
            r_span = max(float((rc[:, 0].max() - rc[:, 0].min()).item()), 1.0)
            c_span = max(float((rc[:, 1].max() - rc[:, 1].min()).item()), 1.0)
            # 2 contiguous 2D blocks anchored on sampled receivers so sparse receiver sets always get spatial blocks!
            num_blocks = 2 if n_pts >= 16 else 1
            per_block_frac = max(ratio / float(num_blocks), 0.04)
            half_h = max(1.0, min(float(self.spatial_block_size) * 0.5, 0.5 * r_span * math.sqrt(per_block_frac)))
            half_w = max(1.0, min(float(self.spatial_block_size) * 0.5, 0.5 * c_span * math.sqrt(per_block_frac)))
            mask_cpu = torch.zeros((bsz, n_pts), dtype=torch.bool)
            for b_i in range(bsz):
                m_b = torch.zeros((n_pts,), dtype=torch.bool)
                anchors = torch.randperm(n_pts, generator=gen)[:num_blocks]
                for a_idx in anchors.tolist():
                    ar, ac = float(rc[a_idx, 0].item()), float(rc[a_idx, 1].item())
                    in_blk = ((rc[:, 0] - ar).abs() <= half_h) & ((rc[:, 1] - ac).abs() <= half_w)
                    m_b = m_b | in_blk
                # Guarantee non-trivial split (`1 <= |Omega_T| <= N - 1`) whenever `N >= 2`
                if not bool(m_b.any().item()):
                    m_b[int(anchors[0].item())] = True
                elif bool(m_b.all().item()):
                    # Keep farthest points from first anchor as unmasked context `Omega_C`
                    ar0, ac0 = float(rc[int(anchors[0].item()), 0].item()), float(rc[int(anchors[0].item()), 1].item())
                    d2 = (rc[:, 0] - ar0).square() + (rc[:, 1] - ac0).square()
                    keep_k = max(1, min(n_pts - 1, int(round(float(n_pts) * ratio))))
                    nearest = torch.argsort(d2)[:keep_k]
                    m_b.zero_()
                    m_b[nearest] = True
                mask_cpu[b_i] = m_b
            return mask_cpu.to(device=device)

        if receivers_xyz is not None and receivers_xyz.numel() >= 3 * n_pts:
            rx = receivers_xyz.detach().to(device="cpu", dtype=torch.float32)
            if rx.ndim == 3:
                rx = rx[0]
            mask_cpu = torch.zeros((bsz, n_pts), dtype=torch.bool)
            keep_k = max(1, min(n_pts - 1, int(round(float(n_pts) * ratio))))
            for b_i in range(bsz):
                anchor = int(torch.randint(0, n_pts, (1,), generator=gen).item())
                d2 = (rx[:, 0] - rx[anchor, 0]).square() + (rx[:, 2] - rx[anchor, 2]).square()
                nearest = torch.argsort(d2)[:keep_k]
                mask_cpu[b_i, nearest] = True
            return mask_cpu.to(device=device)

        # Fallback when no spatial coordinates were passed: contiguous slice
        keep_k = max(1, min(n_pts - 1, int(round(float(n_pts) * ratio))))
        mask_cpu = torch.zeros((bsz, n_pts), dtype=torch.bool)
        for b_i in range(bsz):
            start = int(torch.randint(0, max(1, n_pts - keep_k + 1), (1,), generator=gen).item())
            mask_cpu[b_i, start : start + keep_k] = True
        return mask_cpu.to(device=device)

    def build_structured_mask(
        self,
        z_ctx: torch.Tensor,
        *,
        mode: str,
        boundary_distance_m: Optional[torch.Tensor] = None,
        logits_boundary: Optional[torch.Tensor] = None,
        receiver_rc: Optional[torch.Tensor] = None,
        receivers_xyz: Optional[torch.Tensor] = None,
        step_seed: int = 0,
    ) -> torch.Tensor:
        """Build target mask `Omega_T` `(B, T, N)` for any mode in `ALL_MASK_MODES`."""
        if mode not in ALL_MASK_MODES:
            raise ValueError(f"Unknown mask mode {mode!r}; expected {ALL_MASK_MODES}")
        eff_mode = (
            STRUCTURED_MASK_MODES[int(step_seed) % len(STRUCTURED_MASK_MODES)]
            if mode == "cycle_all"
            else mode
        )
        bsz, t_steps, n_pts, _ = z_ctx.shape
        dev = z_ctx.device
        if eff_mode == "none":
            return torch.ones((bsz, t_steps, n_pts), dtype=torch.bool, device=dev)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(step_seed) + 1337)

        if eff_mode == "random":
            m_bt = (torch.rand((bsz, t_steps, n_pts), generator=gen) < self.mask_ratio).to(dev)
            if n_pts >= 2 and not bool(m_bt.any().item()):
                m_bt[:, :, 0] = True
            if n_pts >= 2 and bool(m_bt.all().item()):
                m_bt[:, :, -1] = False
            return m_bt

        if eff_mode in ("contiguous_block","buffered_block"):
            if receivers_xyz is None:
                raise ValueError("Block masks require public metric XYZ")
            coords = receivers_xyz.reshape(-1,n_pts,3).expand(bsz,-1,-1)
            rows = []
            for b in range(bsz):
                anchor = int(torch.randint(n_pts,(1,),generator=gen))
                distance = (coords[b,:,[0,2]] - coords[b,anchor,[0,2]]).abs().amax(-1)
                size = max(1,min(n_pts-1,round(n_pts*self.mask_ratio)))
                rows.append(distance <= distance.kthvalue(size).values)
            return torch.stack(rows)[:,None,:].expand(bsz,t_steps,n_pts)

        if eff_mode == "spatial":
            m_bn = self._sample_2d_spatial_block_mask(
                bsz,
                n_pts,
                dev,
                gen,
                receiver_rc=receiver_rc,
                receivers_xyz=receivers_xyz,
                target_ratio=self.mask_ratio,
            )
            return m_bn.unsqueeze(1).expand(bsz, t_steps, n_pts)

        if eff_mode == "temporal":
            span_len = max(1, min(max(2, t_steps // 2), max(2, int(round(t_steps * self.mask_ratio)))))
            span_len = min(span_len, max(t_steps - 1, 1)) if t_steps >= 2 else 1
            t_start = int(torch.randint(0, max(1, t_steps - span_len + 1), (1,), generator=gen).item())
            if t_steps >= 2 and t_start == 0 and span_len < t_steps:
                t_start = 1  # Ensure at least one `t >= 1` target frame is masked
            mask = torch.zeros((bsz, t_steps, n_pts), dtype=torch.bool, device=dev)
            mask[:, t_start : t_start + span_len, :] = True
            return mask

        if eff_mode == "spacetime_tube":
            tube_len = max(2, min(8, max(2, t_steps // 2))) if t_steps >= 2 else 1
            tube_len = min(tube_len, t_steps)
            t_start = int(torch.randint(0, max(1, t_steps - tube_len + 1), (1,), generator=gen).item())
            m_bn = self._sample_2d_spatial_block_mask(
                bsz,
                n_pts,
                dev,
                gen,
                receiver_rc=receiver_rc,
                receivers_xyz=receivers_xyz,
                target_ratio=min(0.5, self.mask_ratio * 1.5),
            )
            mask = torch.zeros((bsz, t_steps, n_pts), dtype=torch.bool, device=dev)
            mask[:, t_start : t_start + tube_len, :] = m_bn.unsqueeze(1).expand(bsz, tube_len, n_pts)
            return mask

        if eff_mode == "driver":
            span_len = max(1, min(max(2, t_steps // 2), max(2, int(round(t_steps * self.mask_ratio)))))
            span_len = min(span_len, max(t_steps - 1, 1)) if t_steps >= 2 else 1
            t_start = int(torch.randint(0, max(1, t_steps - span_len + 1), (1,), generator=gen).item())
            if t_steps >= 2 and t_start == 0 and span_len == 1:
                t_start = 1
            mask = torch.zeros((bsz, t_steps, n_pts), dtype=torch.bool, device=dev)
            mask[:, t_start : t_start + span_len, :] = True
            return mask

        def _boundary_band_mask(prob_keep: float) -> Optional[torch.Tensor]:
            # Prefer label-free student `logits_boundary.detach()` so mask selection itself has `0` truth dependency!
            cue_src = None
            if logits_boundary is not None:
                lb = logits_boundary.detach().to(device=dev, dtype=torch.float32)
                cue_src = 0.75 * (lb.unsqueeze(0) if lb.ndim == 2 else lb)
            if cue_src is None:
                return None
            near = torch.isfinite(cue_src) & (cue_src.abs() <= 1.25)
            if not bool(near.any().item()):
                return None
            drop = (torch.rand((bsz, t_steps, n_pts), generator=gen).to(dev) < prob_keep) & near
            if bool(drop.any().item()) and not bool(drop.all().item()):
                return drop
            return None

        if eff_mode == "boundary":
            b_m = _boundary_band_mask(0.5)
            if b_m is not None:
                return b_m
            return self._sample_2d_spatial_block_mask(
                bsz, n_pts, dev, gen, receiver_rc=receiver_rc, receivers_xyz=receivers_xyz
            ).unsqueeze(1).expand(bsz, t_steps, n_pts)

        if eff_mode == "spatial_boundary":
            sp_bn = self._sample_2d_spatial_block_mask(
                bsz,
                n_pts,
                dev,
                gen,
                receiver_rc=receiver_rc,
                receivers_xyz=receivers_xyz,
                target_ratio=max(0.10, self.mask_ratio * 0.65),
            ).unsqueeze(1).expand(bsz, t_steps, n_pts)
            b_m = _boundary_band_mask(0.35)
            comb = (sp_bn | b_m) if b_m is not None else sp_bn
            if n_pts >= 2 and bool(comb.all().item()):
                comb = sp_bn
            return comb

        return (torch.rand((bsz, t_steps, n_pts), generator=gen) < self.mask_ratio).to(dev)

    def _encode_lead_scalar(self, lead: int, device: torch.device) -> torch.Tensor:
        lv = float(lead)
        feat = torch.tensor(
            [lv / 15.0, math.log1p(lv) / 3.0, math.sin(lv * 0.4), math.cos(lv * 0.4)],
            dtype=torch.float32,
            device=device,
        )
        return self.lead_embed(feat)  # (D,)

    def forward_multi_horizon(
        self,
        z_ctx: torch.Tensor,
        z_target_detached: torch.Tensor,
        solar_emb: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
        same_day_adjacent_mask: Optional[torch.Tensor],
        boundary_distance_m: Optional[torch.Tensor] = None,
        logits_boundary: Optional[torch.Tensor] = None,
        receiver_rc: Optional[torch.Tensor] = None,
        receivers_xyz: Optional[torch.Tensor] = None,
        static_geom_memory: Optional[torch.Tensor] = None,
        context_x_emb: Optional[torch.Tensor] = None,
        jepa_context_mode: str = "with_x",
        mask_mode: str = "spatial_boundary",
        precomputed_target_mask: Optional[torch.Tensor] = None,
        history: Optional[Dict[str, torch.Tensor]] = None,
        include_zero_lead: bool = False,
        step_seed: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Predict `z_target` strictly on target support `Omega_T` with L1 loss & true `D_X` diagnostic."""
        if jepa_context_mode not in JEPA_CONTEXT_MODES:
            raise ValueError(f"Unsupported jepa_context_mode: {jepa_context_mode!r}")
        zc = z_ctx.to(torch.float32)
        zt = z_target_detached.detach().to(torch.float32)
        sol = solar_emb.to(torch.float32)
        vm = valid_mask.to(device=zc.device, dtype=torch.bool)
        if zc.ndim == 3:
            zc = zc.unsqueeze(0)
            zt = zt.unsqueeze(0)
        if sol.ndim == 2:
            sol = sol.unsqueeze(0)
        if vm.ndim == 2:
            vm = vm.unsqueeze(0)

        bsz, t_steps, n_pts, d_dim = zc.shape
        eff_mask_mode = (
            STRUCTURED_MASK_MODES[int(step_seed) % len(STRUCTURED_MASK_MODES)]
            if mask_mode == "cycle_all"
            else mask_mode
        )
        raw_struct_mask = (
            precomputed_target_mask.to(device=zc.device, dtype=torch.bool).reshape(bsz, t_steps, n_pts)
            if precomputed_target_mask is not None
            else self.build_structured_mask(
                zc,
                mode=eff_mask_mode,
                boundary_distance_m=boundary_distance_m,
                logits_boundary=logits_boundary,
                receiver_rc=receiver_rc,
                receivers_xyz=receivers_xyz,
                step_seed=step_seed,
            )
        )

        # Disjoint partition: `Omega_T = raw_struct_mask & vm`, `Omega_C = (~raw_struct_mask) & vm`
        target_mask = raw_struct_mask & vm
        context_mask = permitted_context(
            raw_struct_mask,
            vm,
            eff_mask_mode,
            receivers_xyz,
            getattr(self, "buffer_m", 1.5),
            guard_buffer_m=float(getattr(self, "guard_buffer_m", 0.0)),
        )
        valid_count = int(vm.sum().item())
        target_count = int(target_mask.sum().item())
        actual_mask_ratio = float(target_count) / float(max(valid_count, 1))

        mask_tok = self.mask_token.to(dtype=torch.float32, device=zc.device).expand(bsz, t_steps, n_pts, d_dim)
        drv_tok = self.driver_mask_token.to(dtype=torch.float32, device=zc.device).expand(bsz, t_steps, n_pts, d_dim)

        if eff_mask_mode == "driver":
            public_geometry = mask_tok
            if static_geom_memory is not None:
                public_geometry = static_geom_memory.to(zc).reshape(bsz,1,n_pts,d_dim) + mask_tok
            zc_masked = torch.where(raw_struct_mask.any(dim=-1,keepdim=True).unsqueeze(-1), public_geometry, zc)
            # Mask solar driver embeddings across the ENTIRE driver blackout window!
            drv_frame_mask = raw_struct_mask.any(dim=-1, keepdim=True)  # (B, T, 1)
            sol_clean = torch.where(drv_frame_mask, drv_tok[:, :, 0, :], sol)
        elif eff_mask_mode == "none":
            zc_masked = zc
            sol_clean = sol
        else:
            # Response masking keeps the deployed geometry query available.
            zc_masked = zc
            sol_clean = sol

        # Build restricted cross-receiver context `X` (`Omega_C -> all queries`)
        if context_x_emb is not None:
            raw_x = context_x_emb.to(dtype=torch.float32, device=zc.device).reshape(bsz, t_steps, n_pts, d_dim)
            raw_x = raw_x * context_mask.unsqueeze(-1).to(torch.float32)
        else:
            # Label-free `ema_geom` fallback: geometric context strictly on `Omega_C`
            raw_x = zc * context_mask.unsqueeze(-1).to(torch.float32)

        if jepa_context_mode == "without_x" or eff_mask_mode == "none":
            eff_ctx_mask = torch.zeros_like(context_mask)
            eff_x = torch.zeros_like(raw_x)
        elif jepa_context_mode == "shuffled_x":
            eff_ctx_mask = context_mask
            eff_x = permute_context_values_across_receivers(raw_x, context_mask, step_seed=step_seed)
        else:
            eff_ctx_mask = context_mask
            eff_x = raw_x

        ctx_agg = self.context_aggregator(
            zc_masked,
            zc * eff_ctx_mask.unsqueeze(-1).to(torch.float32),
            eff_x,
            eff_ctx_mask,
            receivers_xyz=receivers_xyz,
            receiver_rc=receiver_rc,
        )

        # True bypass diagnostic `D_X`: permute ONLY `X` (`pi(X)`) while keeping `(G, u)` 100% fixed!
        with torch.no_grad():
            if jepa_context_mode == "without_x" or eff_mask_mode == "none" or not bool(eff_ctx_mask.any().item()):
                ctx_agg_perm = ctx_agg.detach()
            else:
                perm_x = permute_context_values_across_receivers(
                    eff_x.detach(), eff_ctx_mask, step_seed=step_seed + 997
                )
                ctx_agg_perm = self.context_aggregator(
                    zc_masked.detach(),
                    (zc.detach() * eff_ctx_mask.unsqueeze(-1).to(torch.float32)),
                    perm_x,
                    eff_ctx_mask,
                    receivers_xyz=receivers_xyz,
                    receiver_rc=receiver_rc,
                )

        if same_day_adjacent_mask is not None:
            sdm = same_day_adjacent_mask.to(device=zc.device, dtype=torch.bool)
            if sdm.ndim == 1:
                sdm = sdm.unsqueeze(0).expand(bsz, -1)
        else:
            sdm = torch.ones((bsz, max(t_steps - 1, 0)), dtype=torch.bool, device=zc.device)

        if history is not None:
            # Retain detached source-side features only. New endpoints own their
            # loss; past endpoints are suppressed, so boundary spans count once.
            current = dict(z=zc_masked,ctx=ctx_agg,perm=ctx_agg_perm,valid=vm)
            prefix = history["z"].shape[1] if "z" in history else 0
            if prefix:
                current = {k:torch.cat([history[k],v],dim=1) for k,v in current.items()}
                bridge = torch.ones((bsz,1),dtype=torch.bool,device=zc.device)
                sdm = torch.cat([history["adj"],bridge,sdm],dim=1)
                target_mask = torch.cat([torch.zeros_like(history["valid"]),target_mask],dim=1)
                zt = torch.cat([torch.zeros_like(history["z"]),zt],dim=1)
                sol_clean = torch.cat([torch.zeros_like(history["z"][:,:,0]),sol_clean],dim=1)
            zc_masked,ctx_agg,ctx_agg_perm,vm = (current[k] for k in ("z","ctx","perm","valid"))
            t_steps = vm.shape[1]
            keep = min(max(self.leads,default=0),t_steps)
            history.clear()
            if keep:
                history.update({k:v[:,-keep:].detach().clone() for k,v in current.items()})
                history["adj"] = sdm[:,t_steps-keep:].detach().clone()

        # Prefix sums for O(1) unbroken same-day span AND full `[t .. t + lead]` validity (`P0-3`)!
        broken = (~sdm).to(torch.int32)
        zero_col = torch.zeros((bsz, 1), dtype=torch.int32, device=zc.device)
        broken_pref = torch.cat([zero_col, torch.cumsum(broken, dim=1)], dim=1)  # (B, T)

        inv_int = (~vm).to(torch.int32)
        zero_inv = torch.zeros((bsz, 1, n_pts), dtype=torch.int32, device=zc.device)
        inv_pref = torch.cat([zero_inv, torch.cumsum(inv_int, dim=1)], dim=1)  # (B, T + 1, N)

        candidate_leads: Tuple[int, ...] = (0,) + self.leads if include_zero_lead else self.leads
        horizon_losses: List[torch.Tensor] = []
        dx_vals: List[float] = []
        used_leads = 0

        lead_diag = {}
        for lead in candidate_leads:
            lead_diag.update({f"lead_{lead}_support": 0., f"lead_{lead}_loss": 0., f"lead_{lead}_dx": 0.})
            if lead < 0 or lead >= t_steps:
                continue
            if lead == 0:
                if eff_mask_mode in ("driver", "temporal"):
                    continue
                all_valid_span = vm
            else:
                same_day_span = (broken_pref[:, lead:] - broken_pref[:, :-lead]) == 0  # (B, T - lead)
                # Every frame `s in [t, t + lead]` must be valid (`inv_pref[:, lead+1:] - inv_pref[:, :T-lead] == 0`)
                all_frames_valid = (inv_pref[:, lead + 1 :] - inv_pref[:, : t_steps - lead]) == 0
                all_valid_span = all_frames_valid & same_day_span.unsqueeze(-1)

            if not bool(all_valid_span.any().item()):
                continue

            t_rem = t_steps - lead
            tgt_m_lead = target_mask if lead == 0 else target_mask[:, lead:]
            # Evaluate strictly on target support `Omega_T` (intersected with valid span)!
            eval_mask = all_valid_span & tgt_m_lead
            if not bool(eval_mask.any().item()):
                continue

            sol_fut = (
                sol_clean.unsqueeze(2).expand(bsz, t_rem, n_pts, d_dim)
                if lead == 0
                else sol_clean[:, lead:].unsqueeze(2).expand(bsz, t_rem, n_pts, d_dim)
            )
            lead_vec = self._encode_lead_scalar(lead, zc.device).view(1, 1, 1, d_dim).expand(
                bsz, t_rem, n_pts, d_dim
            )
            zc_src = zc_masked if lead == 0 else zc_masked[:, :-lead]
            ctx_src = ctx_agg if lead == 0 else ctx_agg[:, :-lead]
            pred_in = torch.cat([zc_src, ctx_src, sol_fut, lead_vec], dim=-1)
            pred_z = torch.tanh(self.net(pred_in))
            tgt_z = zt if lead == 0 else zt[:, lead:]

            # Exact L1 loss strictly on `Omega_T` (`eval_mask`)
            l1_per_token = (pred_z - tgt_z).abs().mean(dim=-1)
            l_h = l1_per_token[eval_mask].mean()
            horizon_losses.append(l_h)
            used_leads += 1

            with torch.no_grad():
                ctx_src_perm = ctx_agg_perm if lead == 0 else ctx_agg_perm[:, :-lead]
                pred_perm_in = torch.cat(
                    [zc_src.detach(), ctx_src_perm, sol_fut.detach(), lead_vec.detach()],
                    dim=-1,
                )
                pred_perm_z = torch.tanh(self.net(pred_perm_in))
                # Exact L1 norm `||\hat{z}_Y(G, u, X)(r) - \hat{z}_Y(G, u, \pi(X))(r)||_1` on `r in Omega_T`
                l1_vec_diff = (pred_z.detach() - pred_perm_z).abs().sum(dim=-1)
                dx_vals.append(float(l1_vec_diff[eval_mask].mean().item()))
            lead_diag[f"lead_{lead}_support"] = float(eval_mask.sum().item())
            lead_diag[f"lead_{lead}_loss"] = float(l_h.detach().item())
            lead_diag[f"lead_{lead}_dx"] = dx_vals[-1]

        if not horizon_losses:
            zero = zc.sum() * 0.0
            return zero, {
                **lead_diag, "jepa_active_leads": 0.0,
                "jepa_context_sensitivity_dx": 0.0,
                "jepa_mask_ratio_actual": actual_mask_ratio,
            }

        total = sum(horizon_losses) / float(len(horizon_losses))
        dx_mean = float(sum(dx_vals) / len(dx_vals)) if dx_vals else 0.0
        return total, {
            **lead_diag, "jepa_active_leads": float(used_leads),
            "jepa_context_sensitivity_dx": dx_mean,
            "jepa_mask_ratio_actual": actual_mask_ratio,
        }


def sketched_isotropic_gaussian_reg(
    z_proj: torch.Tensor,
    *,
    num_slices: int = 64,
    num_quad_nodes: int = 17,
    t_max: float = 3.0,
    step_seed: int = 0,
) -> torch.Tensor:
    """Official `galilai-group/lejepa` Epps-Pulley Sketched Isotropic Gaussian Regularizer (`SIGReg`)."""
    m_rows, p_dim = z_proj.shape
    if m_rows < 4 or p_dim < 2:
        return z_proj.sum() * 0.0

    z32 = z_proj.to(torch.float32)
    if m_rows > 2048:
        gen_sub = torch.Generator(device="cpu")
        gen_sub.manual_seed(int(step_seed) + 7919)
        idx = torch.randperm(m_rows, generator=gen_sub)[:2048].to(z32.device)
        z_sub = z32[idx]
    else:
        z_sub = z32
    n_eff = int(z_sub.shape[0])

    gen_dir = torch.Generator(device="cpu")
    gen_dir.manual_seed(int(step_seed) + 4242)
    dirs = torch.randn((p_dim, int(num_slices)), generator=gen_dir, dtype=torch.float32).to(z32.device)
    dirs = dirs / torch.linalg.norm(dirs, dim=0, keepdim=True).clamp_min(1e-8)

    proj = z_sub @ dirs  # (N_eff, S) — uncentered!
    q_steps = max(int(num_quad_nodes), 3)
    taus = torch.linspace(0.0, float(t_max), steps=q_steps, device=z32.device, dtype=torch.float32)
    dt = float(t_max) / float(q_steps - 1)
    trapz = torch.full_like(taus, 2.0 * dt)
    trapz[0] = dt
    trapz[-1] = dt
    gauss_window = torch.exp(-0.5 * taus.square())
    quad_weights = (trapz * gauss_window).view(1, -1)

    phase = proj.unsqueeze(-1) * taus.view(1, 1, -1)
    re_cf = torch.cos(phase).mean(dim=0)
    im_cf = torch.sin(phase).mean(dim=0)
    target_cf = gauss_window.view(1, -1)

    cf_sq_err = (re_cf - target_cf).square() + im_cf.square()
    ep_stat = (cf_sq_err * quad_weights).sum(dim=-1).mean() * float(n_eff)
    return ep_stat


def vicreg_variance_covariance_loss(z_proj: torch.Tensor) -> torch.Tensor:
    """Standard `VICReg` Variance + Off-Diagonal Covariance anti-collapse regularizer."""
    m_rows, p_dim = z_proj.shape
    if m_rows < 4 or p_dim < 2:
        return z_proj.sum() * 0.0
    z32 = z_proj.to(torch.float32)
    zc = z32 - z32.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0, unbiased=False) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()
    cov = (zc.T @ zc) / float(max(m_rows - 1, 1))
    eye = torch.eye(p_dim, dtype=torch.bool, device=z32.device)
    cov_off = cov.masked_select(~eye).square().mean()
    return var_loss + cov_off


@torch.no_grad()
def compute_latent_health_diagnostics(
    z_seq: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    max_samples: int = 1024,
) -> Dict[str, Optional[float] | int]:
    """Compute all 5 mandatory anti-collapse health metrics strictly on valid representations.

    Never falls back to invalid slots when `valid_mask` is all-False, and returns `None` (`null` in JSON)
    with `sample_count < 4` when valid support is empty or insufficient!
    """
    z32 = z_seq.detach().to(torch.float32)
    d_dim = int(z32.shape[-1])
    flat_z = z32.reshape(-1, d_dim)
    if valid_mask is not None:
        vm = valid_mask.detach().to(device=z32.device, dtype=torch.bool).reshape(-1)
        flat_z = flat_z[vm]

    m_rows = int(flat_z.shape[0])
    if m_rows < 4 or d_dim < 2:
        return {
            "sample_count": m_rows,
            "per_dim_std_min": None,
            "min_std": None,
            "effective_rank": None,
            "participation_ratio": None,
            "singular_value_top1_ratio": None,
            "off_diag_cov_rms": None,
            "offdiag_cov_rms": None,
            "pairwise_latent_dist_mean": None,
            "cosine_diversity": None,
        }

    if m_rows > max_samples:
        stride = max(1, m_rows // max_samples)
        flat_z = flat_z[::stride][:max_samples]
        m_rows = int(flat_z.shape[0])

    zc = flat_z - flat_z.mean(dim=0, keepdim=True)
    stds = torch.sqrt(zc.var(dim=0, unbiased=False) + 1e-12)
    per_dim_std_min = float(stds.min().item())

    svals = torch.linalg.svdvals(zc)
    s_sum = svals.sum().clamp_min(1e-12)
    p_sv = (svals / s_sum).clamp_min(1e-12)
    eff_rank = float(torch.exp(-(p_sv * torch.log(p_sv)).sum()).item())
    top1_ratio = float((svals[0] / s_sum).item())
    eig_cov = svals.square()
    part_ratio = float((eig_cov.sum().square() / eig_cov.square().sum().clamp_min(1e-12)).item())

    cov = (zc.T @ zc) / float(max(m_rows - 1, 1))
    eye = torch.eye(d_dim, dtype=torch.bool, device=z32.device)
    off_rms = float(torch.sqrt(cov.masked_select(~eye).square().mean().clamp_min(0.0)).item())

    sub_p = flat_z[: min(m_rows, 256)]
    pdist = torch.cdist(sub_p, sub_p, p=2.0)
    eye_p = torch.eye(int(sub_p.shape[0]), dtype=torch.bool, device=z32.device)
    pair_mean = float(pdist.masked_select(~eye_p).mean().item()) if int(sub_p.shape[0]) >= 2 else 0.0
    sub_u = sub_p / torch.linalg.norm(sub_p, dim=-1, keepdim=True).clamp_min(1e-8)
    cos_mat = sub_u @ sub_u.T
    cos_div = (
        float((1.0 - cos_mat.masked_select(~eye_p).mean()).item()) if int(sub_p.shape[0]) >= 2 else 0.0
    )

    return {
        "sample_count": m_rows,
        "per_dim_std_min": per_dim_std_min,
        "min_std": per_dim_std_min,
        "effective_rank": eff_rank,
        "participation_ratio": part_ratio,
        "singular_value_top1_ratio": top1_ratio,
        "off_diag_cov_rms": off_rms,
        "offdiag_cov_rms": off_rms,
        "pairwise_latent_dist_mean": pair_mean,
        "cosine_diversity": cos_div,
    }


def append_collapse_diagnostics_jsonl(
    jsonl_path: Path | str,
    *,
    step: int,
    epoch: int,
    health: Mapping[str, Any],
    aux_stats: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Append step-by-step collapse & `D_X` diagnostic record to `collapse_diagnostics.jsonl` (`P1`)."""
    p = Path(jsonl_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {
        "step": int(step),
        "epoch": int(epoch),
        **dict(health),
    }
    if aux_stats is not None:
        for k in (
            "jepa_pred_loss",
            "jepa_context_sensitivity_dx",
            "jepa_mask_ratio_actual",
            "modality_alignment_rho_m",
            "anti_collapse_reg_loss",
            "jepa_trans_vel_loss",
        ):
            if k in aux_stats:
                record[k] = aux_stats[k]
    line = json.dumps(record, allow_nan=False) + "\n"
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(line)
    return record


def compute_metric_boundary_loss(
    logits_boundary: torch.Tensor,
    truth_f: torch.Tensor,
    valid: torch.Tensor,
    batch: Mapping[str, Any],
    *,
    same_day_adj: Optional[torch.Tensor] = None,
    far_tau_m: float = 1.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Kaifeng's Full Gen-1 `MetricBoundaryLoss` (`<0.5m` signed distance + XZ gradient + velocity)."""
    lb = logits_boundary.to(torch.float32)
    if lb.ndim == 2:
        lb = lb.unsqueeze(0)
    yf = truth_f.to(torch.float32)
    vm = valid.to(torch.bool)
    if yf.ndim == 2:
        yf = yf.unsqueeze(0)
    if vm.ndim == 2:
        vm = vm.unsqueeze(0)

    b_dist = batch.get("boundary_distance_m")
    b_val = batch.get("boundary_distance_valid")
    if b_dist is None or b_val is None:
        return lb.sum() * 0.0, {"metric_boundary_near05_loss": 0.0, "metric_boundary_grad_loss": 0.0}

    bd = b_dist.to(device=lb.device, dtype=torch.float32)
    bv = b_val.to(device=lb.device, dtype=torch.bool)
    if bd.ndim == 2:
        bd = bd.unsqueeze(0)
    if bv.ndim == 2:
        bv = bv.unsqueeze(0)

    active = vm & bv & torch.isfinite(bd)
    if not bool(active.any().item()):
        return lb.sum() * 0.0, {"metric_boundary_near05_loss": 0.0, "metric_boundary_grad_loss": 0.0}

    bd_safe = torch.where(active, bd, torch.zeros_like(bd)).clamp(0.0, 6.0)
    sign = torch.where(yf > 0.5, torch.ones_like(yf), -torch.ones_like(yf))
    target_signed_m = sign * bd_safe
    pred_signed_m = 0.5 * float(far_tau_m) * lb

    near_05 = active & (bd_safe <= 0.5)
    if bool(near_05.any().item()):
        near_loss = F.smooth_l1_loss(pred_signed_m[near_05], target_signed_m[near_05], beta=0.08)
    else:
        near_loss = lb.sum() * 0.0

    w_sign = torch.exp(-bd_safe / 0.5) * active.to(torch.float32)
    sign_bce = (
        F.binary_cross_entropy_with_logits(lb, yf, reduction="none") * w_sign
    ).sum() / w_sign.sum().clamp_min(1.0)

    far_mask = active & (bd_safe > 0.5)
    if bool(far_mask.any().item()):
        soft_tgt = torch.sigmoid(target_signed_m[far_mask] / float(far_tau_m))
        far_loss = F.smooth_l1_loss(torch.sigmoid(lb[far_mask]), soft_tgt, beta=0.05)
    else:
        far_loss = lb.sum() * 0.0

    grad_loss = lb.sum() * 0.0
    n_pts = int(lb.shape[-1])
    rc = batch.get("receiver_rc")
    if rc is not None and 2 <= n_pts <= 512:
        rc_f = rc.to(device=lb.device, dtype=torch.float32)
        if rc_f.ndim == 3:
            rc_f = rc_f[0]
        pos_xz = rc_f * DEFAULT_METERS_PER_PIXEL
        d_mat = torch.cdist(pos_xz, pos_xz, p=2.0)
        eye = torch.eye(n_pts, dtype=torch.bool, device=lb.device)
        d_mat = torch.where(eye, torch.full_like(d_mat, 1e9), d_mat)
        nn_d, nn_idx = d_mat.min(dim=-1)
        close_nbr = nn_d <= 2.5
        if bool(close_nbr.any().item()):
            pair_act = active & active[:, :, nn_idx] & close_nbr.view(1, 1, n_pts) & (bd_safe <= 1.5)
            if bool(pair_act.any().item()):
                dp_pred = pred_signed_m - pred_signed_m[:, :, nn_idx]
                dp_true = target_signed_m - target_signed_m[:, :, nn_idx]
                grad_loss = F.smooth_l1_loss(dp_pred[pair_act], dp_true[pair_act], beta=0.1)

    vel_loss = lb.sum() * 0.0
    if int(lb.shape[1]) >= 2:
        pair_t = active[:, 1:] & active[:, :-1] & ((bd_safe[:, 1:] <= 1.5) | (bd_safe[:, :-1] <= 1.5))
        if same_day_adj is not None:
            sdm = same_day_adj.to(device=lb.device, dtype=torch.bool)
            if sdm.ndim == 1:
                sdm = sdm.view(1, -1, 1)
            elif sdm.ndim == 2:
                sdm = sdm.unsqueeze(-1)
            pair_t = pair_t & sdm
        if bool(pair_t.any().item()):
            v_pred = pred_signed_m[:, 1:] - pred_signed_m[:, :-1]
            v_true = target_signed_m[:, 1:] - target_signed_m[:, :-1]
            vel_loss = F.smooth_l1_loss(v_pred[pair_t], v_true[pair_t], beta=0.1)

    total = sign_bce + 0.6 * near_loss + 0.25 * far_loss + 0.25 * grad_loss + 0.25 * vel_loss
    return total, {
        "metric_boundary_near05_loss": float(near_loss.detach().item()),
        "metric_boundary_grad_loss": float(grad_loss.detach().item()),
    }


def compute_hard_sunlight_and_contrastive_loss(
    logits: torch.Tensor,
    truth_f: torch.Tensor,
    valid: torch.Tensor,
    candidate_mask: Optional[torch.Tensor],
    *,
    extra_sunlight_weight: float = 1.5,
    margin: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Yushu's Hard-Sunlight Fix: Upweights `truth==1 & has_candidates` + direction-matched margin."""
    if candidate_mask is None:
        return logits.sum() * 0.0, {"hard_sunlight_loss": 0.0, "hard_sunlight_fraction": 0.0}

    cm = candidate_mask.to(device=logits.device, dtype=torch.bool)
    if cm.ndim == 4 and logits.ndim == 2:
        cm = cm.squeeze(0)
    has_cands = cm.any(dim=-1)
    if has_cands.shape != logits.shape:
        has_cands = has_cands.reshape_as(logits)

    hard_sun = valid & (truth_f > 0.5) & has_cands
    occ_with_cands = valid & (truth_f < 0.5) & has_cands
    valid_cnt = float(valid.sum().item())
    hard_frac = float(hard_sun.sum().item()) / max(valid_cnt, 1.0)

    if not bool(hard_sun.any().item()):
        return logits.sum() * 0.0, {"hard_sunlight_loss": 0.0, "hard_sunlight_fraction": 0.0}

    log32 = logits.to(torch.float32)
    bce_hard = F.binary_cross_entropy_with_logits(log32[hard_sun], truth_f.to(torch.float32)[hard_sun]).mean()
    contrast = log32.sum() * 0.0
    if bool(occ_with_cands.any().item()):
        log_3d = log32.unsqueeze(0) if log32.ndim == 2 else log32
        hs_3d = hard_sun.unsqueeze(0) if hard_sun.ndim == 2 else hard_sun
        oc_3d = occ_with_cands.unsqueeze(0) if occ_with_cands.ndim == 2 else occ_with_cands

        hs_f = hs_3d.to(torch.float32)
        oc_f = oc_3d.to(torch.float32)
        hs_cnt = hs_f.sum(dim=-1)
        oc_cnt = oc_f.sum(dim=-1)
        both_bt = (hs_cnt > 0.5) & (oc_cnt > 0.5)
        if bool(both_bt.any().item()):
            safe_log = torch.nan_to_num(log_3d, nan=0.0, posinf=50.0, neginf=-50.0)
            mean_pos_bt = (safe_log * hs_f).sum(dim=-1) / hs_cnt.clamp_min(1.0)
            mean_neg_bt = (safe_log * oc_f).sum(dim=-1) / oc_cnt.clamp_min(1.0)
            contrast = F.relu(float(margin) - (mean_pos_bt[both_bt] - mean_neg_bt[both_bt])).mean()
        else:
            mean_pos = log32[hard_sun].mean()
            mean_neg = log32[occ_with_cands].mean()
            contrast = F.relu(float(margin) - (mean_pos - mean_neg))

    total = float(extra_sunlight_weight) * bce_hard + 0.15 * contrast
    return total, {
        "hard_sunlight_loss": float(total.detach().item()),
        "hard_sunlight_fraction": hard_frac,
    }


def compute_focal_event_hard_negative_loss(
    probs: torch.Tensor,
    event_gate: Optional[torch.Tensor],
    truth_f: torch.Tensor,
    valid: torch.Tensor,
    same_day_adj: Optional[torch.Tensor],
    *,
    prev_chunk_prob: Optional[torch.Tensor] = None,
    prev_chunk_truth: Optional[torch.Tensor] = None,
    prev_chunk_valid: Optional[torch.Tensor] = None,
    carry_prev: bool = False,
    top_k_hard_stable_ratio: float = 0.10,
) -> Tuple[torch.Tensor, int, Dict[str, float]]:
    """Directionally Balanced Switching Focal (`sun->shadow` & `shadow->sun` 0.5/0.5) + Hard-Negative Mining."""
    p = probs.to(torch.float32)
    yf = truth_f.to(torch.float32)
    vm = valid.to(torch.bool)
    eg = event_gate.to(torch.float32) if event_gate is not None else None
    if p.ndim == 2:
        p = p.unsqueeze(0)
        yf = yf.unsqueeze(0)
        vm = vm.unsqueeze(0)
        if eg is not None:
            eg = eg.unsqueeze(0)

    bsz, t_steps, n_pts = p.shape
    def _pair_term(
        p_cur: torch.Tensor,
        p_prev: torch.Tensor,
        y_cur: torch.Tensor,
        y_prev: torch.Tensor,
        pair_m: torch.Tensor,
        eg_cur: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], int, float, float, float]:
        cnt = int(pair_m.sum().item())
        if cnt <= 0:
            return None, 0, 0.0, 0.0, 0.0
        target_sw = (y_cur - y_prev).abs()
        pred_sw = (p_cur * (1.0 - p_prev) + (1.0 - p_cur) * p_prev).clamp(1e-6, 1.0 - 1e-6)
        raw_bce = F.binary_cross_entropy(pred_sw, target_sw, reduction="none")
        pt = torch.where(target_sw > 0.5, pred_sw, 1.0 - pred_sw)
        focal = (1.0 - pt).pow(2.0) * raw_bce

        gate_bce: Optional[torch.Tensor] = None
        if eg_cur is not None:
            eg_clamped = eg_cur.clamp(1e-6, 1.0 - 1e-6)
            gate_bce = F.binary_cross_entropy(eg_clamped, target_sw, reduction="none")

        # Directionally balanced switching (`sun -> shadow` `1 -> 0` vs `shadow -> sun` `0 -> 1`)
        pos_sun_to_sh = pair_m & (y_prev > 0.5) & (y_cur <= 0.5)
        pos_sh_to_sun = pair_m & (y_prev <= 0.5) & (y_cur > 0.5)
        neg_m = pair_m & (target_sw <= 0.5)

        has_10 = bool(pos_sun_to_sh.any().item())
        has_01 = bool(pos_sh_to_sun.any().item())

        term = p.sum() * 0.0
        s10 = 0.0
        s01 = 0.0
        if has_10 and has_01:
            f10 = focal[pos_sun_to_sh].mean()
            f01 = focal[pos_sh_to_sun].mean()
            term = term + 0.5 * f10 + 0.5 * f01
            s10 = float(f10.detach().item())
            s01 = float(f01.detach().item())
            if gate_bce is not None:
                term = term + 0.25 * (0.5 * gate_bce[pos_sun_to_sh].mean() + 0.5 * gate_bce[pos_sh_to_sun].mean())
        elif has_10:
            f10 = focal[pos_sun_to_sh].mean()
            term = term + f10
            s10 = float(f10.detach().item())
            if gate_bce is not None:
                term = term + 0.25 * gate_bce[pos_sun_to_sh].mean()
        elif has_01:
            f01 = focal[pos_sh_to_sun].mean()
            term = term + f01
            s01 = float(f01.detach().item())
            if gate_bce is not None:
                term = term + 0.25 * gate_bce[pos_sh_to_sun].mean()

        if bool(neg_m.any().item()):
            neg_vals = focal[neg_m]
            term = term + 0.5 * neg_vals.mean()
            if gate_bce is not None:
                term = term + 0.25 * gate_bce[neg_m].mean()
            neg_score = (
                raw_bce[neg_m] + 0.5 * gate_bce[neg_m] if gate_bce is not None else raw_bce[neg_m]
            )
            k_hard = max(1, int(round(float(neg_vals.numel()) * float(top_k_hard_stable_ratio))))
            hard_top, _ = torch.topk(neg_score, k=min(k_hard, int(neg_vals.numel())))
            hn_loss = hard_top.mean()
            term = term + 0.5 * hn_loss
            hn_scalar = float(hn_loss.detach().item())
        else:
            hn_scalar = 0.0

        return term, cnt, hn_scalar, s10, s01

    # Mine hard negatives and normalize each direction once over all pairs in this update.
    pair_mask = vm[:, 1:] & vm[:, :-1]
    if same_day_adj is not None and t_steps > 1:
        adj = same_day_adj.to(device=p.device, dtype=torch.bool)
        pair_mask = pair_mask & adj.reshape(-1, t_steps - 1, 1)
    current_p, previous_p = p[:, 1:], p[:, :-1]
    current_y, previous_y = yf[:, 1:], yf[:, :-1]
    gate = eg[:, 1:] if eg is not None else None
    if carry_prev and prev_chunk_prob is not None and prev_chunk_truth is not None and prev_chunk_valid is not None:
        prior_truth = prev_chunk_truth.to(p.device).reshape(bsz, 1, n_pts)
        prior_valid = prev_chunk_valid.to(device=p.device, dtype=torch.bool).reshape(bsz, 1, n_pts)
        prior_valid = prior_valid & ((prior_truth == 0) | (prior_truth == 1))
        pair_mask = torch.cat([vm[:, :1] & prior_valid, pair_mask], dim=1)
        previous_p = torch.cat([prev_chunk_prob.to(p).reshape(bsz, 1, n_pts), previous_p], dim=1)
        previous_y = torch.cat([(prior_truth > 0).to(yf), previous_y], dim=1)
        current_p, current_y, gate = p, yf, eg
    loss, count, hard, to_shadow, to_sun = _pair_term(
        current_p, previous_p, current_y, previous_y, pair_mask, gate)
    return loss if loss is not None else p.sum() * 0.0, count, {
        "hard_negative_stable_loss": hard,
        "focal_sun_to_shadow_loss": to_shadow,
        "focal_shadow_to_sun_loss": to_sun,
    }


def build_height_raised_counterfactual_tokens_torch(
    tokens: torch.Tensor,
    candidate_mask: torch.Tensor,
    sun_dirs: torch.Tensor,
    *,
    delta_height_m: float = 6.0,
    elevated_y_threshold_m: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extrude elevated candidate triangle vertices (`rel_y > 0.25m`) by `+delta_height_m` and rebuild all 37 slots."""
    cm = candidate_mask.to(torch.bool)
    rel_tris_m = tokens[..., 13:22].to(torch.float32).reshape(*tokens.shape[:-1], 3, 3).clone() * 1000.0
    opaque_mask = (tokens[..., 10] > 0.5) & cm
    vy = rel_tris_m[..., :, 1]
    above = (vy > float(elevated_y_threshold_m)) & cm.unsqueeze(-1)
    rel_tris_m[..., :, 1] = torch.where(above, vy + float(delta_height_m), vy)
    cf_tokens = build_canonical37_torch(rel_tris_m, opaque_mask, cm, sun_dirs, scale_m=1000.0)
    return cf_tokens, rel_tris_m


def build_structural_intervention_tokens_torch(
    tokens: torch.Tensor,
    candidate_mask: torch.Tensor,
    sun_dirs: torch.Tensor,
    *,
    delta_height_m: float,
    edit_cand_mask: Optional[torch.Tensor] = None,
    elevated_y_threshold_m: float = 0.25,
    min_roof_y_m: float = 0.30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply signed height intervention (`+dh` raise OR `-dh` lower) to selected candidate subset & rebuild 37 slots."""
    cm = candidate_mask.to(torch.bool)
    edit_m = cm if edit_cand_mask is None else (cm & edit_cand_mask.to(torch.bool))
    rel_tris_m = tokens[..., 13:22].to(torch.float32).reshape(*tokens.shape[:-1], 3, 3).clone() * 1000.0
    opaque_mask = (tokens[..., 10] > 0.5) & cm
    vy = rel_tris_m[..., :, 1]
    above = (vy > float(elevated_y_threshold_m)) & edit_m.unsqueeze(-1)
    dh = float(delta_height_m)
    if dh >= 0.0:
        new_vy = vy + dh
    else:
        new_vy = (vy + dh).clamp_min(float(min_roof_y_m))
    rel_tris_m[..., :, 1] = torch.where(above, new_vy, vy)
    cf_tokens = build_canonical37_torch(rel_tris_m, opaque_mask, cm, sun_dirs, scale_m=1000.0)
    return cf_tokens, rel_tris_m


@torch.no_grad()
def _candidate_ray_distances_torch(
    rel_tris_m: torch.Tensor,
    active_opaque_mask: torch.Tensor,
    sun_dirs: torch.Tensor,
    *,
    t_min_m: float = 1e-2,
    t_max_m: float = 4000.0,
) -> torch.Tensor:
    """Exact FP64 Möller–Trumbore ray-triangle test from origin `(0,0,0)` along `sun_dirs` on `rel_tris_m`.

    Return per-slot forward intersection distance in metres, or infinity for a miss.
    """
    tris = rel_tris_m.to(torch.float64)  # (T, N, K, 3, 3)
    mask = active_opaque_mask.to(torch.bool)  # (T, N, K)
    s = sun_dirs.to(torch.float64)
    if s.ndim == 2:
        s = s[:, None, None, :]  # (T, 1, 1, 3)
    s = s / torch.linalg.norm(s, dim=-1, keepdim=True).clamp_min(1e-12)

    v0 = tris[..., 0, :]
    v1 = tris[..., 1, :]
    v2 = tris[..., 2, :]
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = torch.cross(s.expand_as(e2), e2, dim=-1)
    det = (e1 * pvec).sum(dim=-1)
    non_parallel = det.abs() > 1e-10
    inv_det = torch.where(non_parallel, 1.0 / torch.where(non_parallel, det, torch.ones_like(det)), torch.zeros_like(det))
    tvec = -v0  # ray origin is receiver `(0, 0, 0)` in relative coordinates
    u = (tvec * pvec).sum(dim=-1) * inv_det
    qvec = torch.cross(tvec, e1, dim=-1)
    v = (s.expand_as(qvec) * qvec).sum(dim=-1) * inv_det
    t_hit = (e2 * qvec).sum(dim=-1) * inv_det

    eps_bary = 1e-7
    hit = (
        mask
        & non_parallel
        & (t_hit >= float(t_min_m))
        & (t_hit <= float(t_max_m))
        & (u >= -eps_bary)
        & (v >= -eps_bary)
        & ((u + v) <= (1.0 + eps_bary))
    )
    return torch.where(hit,t_hit,torch.full_like(t_hit,float("inf")))


def exact_candidate_ray_blocked_torch(
    rel_tris_m, active_opaque_mask, sun_dirs, *, t_min_m=1e-2, t_max_m=4000.0
):
    distances = _candidate_ray_distances_torch(
        rel_tris_m, active_opaque_mask, sun_dirs, t_min_m=t_min_m, t_max_m=t_max_m
    )
    return torch.isfinite(distances).any(dim=-1)


def _compute_candidate_face_perturbation_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
    base_probs: torch.Tensor,
    *,
    delta_height_m: float = 6.0,
    dominant_occluder_dropout_prob: float = 0.25,
    max_sub_frames: int = 8,
    max_sub_receivers: int = 32,
    step_seed: int = 0,
    edit_counts: Optional[Dict[str, int]] = None,
    cf_step_stats: Optional[Dict[str, float]] = None,
    cf_tensor_terms: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Local candidate-token auxiliary with fixed static context.

    Edit a candidate slot or a set of face tokens. Building identities are unavailable
    on this path. Full-scene, building-ID interventions run in inverse_design.
    """
    if not getattr(model, "model_type", "").startswith("rind_") or "candidate_tokens" not in batch:
        return base_probs.sum() * 0.0

    tokens = batch["candidate_tokens"]
    cmask = batch["candidate_mask"].to(torch.bool)
    sun_dirs = batch["sun_dirs"]
    squeeze = tokens.ndim == 4
    tok4 = tokens if squeeze else tokens[0]
    cm3 = cmask if squeeze else cmask[0]
    sun2 = sun_dirs if sun_dirs.ndim == 2 else sun_dirs[0]
    _ = base_probs if base_probs.ndim == 2 else base_probs[0]

    t_steps, n_pts, k_cands, _ = tok4.shape
    if t_steps == 0 or n_pts == 0 or k_cands == 0:
        return base_probs.sum() * 0.0

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(step_seed) + 9001)
    t_pick = min(t_steps, int(max_sub_frames))
    n_pick = min(n_pts, int(max_sub_receivers))
    t_idx = torch.sort(torch.randperm(t_steps, generator=gen)[:t_pick])[0].to(tokens.device)
    n_idx = torch.sort(torch.randperm(n_pts, generator=gen)[:n_pick])[0].to(tokens.device)

    tok_orig_sub = tok4.index_select(0, t_idx).index_select(1, n_idx)
    cm_sub = cm3.index_select(0, t_idx).index_select(1, n_idx)
    sun_sub = sun2.index_select(0, t_idx)
    if not bool(cm_sub.any().item()):
        return base_probs.sum() * 0.0

    sanitize_fn = getattr(model, "sanitize_tokens", None)
    tok_clean_san = sanitize_fn(tok_orig_sub) if callable(sanitize_fn) else tok_orig_sub

    base_sub: Dict[str, Any] = {
        "candidate_tokens": tok_clean_san,
        "candidate_mask": cm_sub,
        "sun_dirs": sun_sub,
        "allow_implicit_continuity": True,
        "same_day_adjacent_mask": torch.zeros(
            (max(t_pick - 1, 0),), dtype=torch.bool, device=tokens.device
        ),
    }
    if "local_features" in batch and batch["local_features"] is not None:
        lf = batch["local_features"]
        lf2 = lf if lf.ndim == 2 else lf[0]
        base_sub["local_features"] = lf2.index_select(0, n_idx)
    if "receivers_xyz" in batch and batch["receivers_xyz"] is not None:
        rx = batch["receivers_xyz"]
        rx2 = rx if rx.ndim == 2 else rx[0]
        base_sub["receivers_xyz"] = rx2.index_select(0, n_idx)
    if "receiver_rc" in batch and batch["receiver_rc"] is not None:
        rc = batch["receiver_rc"]
        rc2 = rc if rc.ndim == 2 else rc[0]
        base_sub["receiver_rc"] = rc2.index_select(0, n_idx)
    if "receiver_obs_mask" in batch and batch["receiver_obs_mask"] is not None:
        ro = batch["receiver_obs_mask"]
        if ro.ndim == 1:
            base_sub["receiver_obs_mask"] = ro.index_select(0, n_idx)
        elif ro.ndim == 2:
            if int(ro.shape[0]) == t_steps and int(ro.shape[1]) == n_pts:
                base_sub["receiver_obs_mask"] = ro.index_select(0, t_idx).index_select(1, n_idx)
            else:
                base_sub["receiver_obs_mask"] = ro[0].index_select(0, n_idx)
        elif ro.ndim == 3:
            base_sub["receiver_obs_mask"] = ro[0].index_select(0, t_idx).index_select(1, n_idx)
    for g_key in (
        "global_knn_local_features",
        "global_knn_rel_xyz",
        "global_knn_dist",
        "global_knn_obs_valid",
    ):
        if g_key in batch and batch[g_key] is not None:
            gv = batch[g_key]
            gv0 = gv if gv.ndim in (2, 3) and int(gv.shape[0]) == n_pts else gv[0]
            base_sub[g_key] = gv0.index_select(0, n_idx)

    with torch.no_grad():
        clean_sub_probs = model(base_sub, use_calibrated_temperature=False)["probs"].detach().to(torch.float32)

    # Select single/subset candidate intervention mask (1-of-K or random subset of active candidates)
    cand_scores = torch.rand((1, n_pick, k_cands), generator=gen).to(tokens.device)
    active_any_t = cm_sub.any(dim=0, keepdim=True)  # (1, N_sub, K)
    ranked = torch.where(active_any_t, cand_scores, torch.full_like(cand_scores, -1.0))
    top_slot = ranked.argmax(dim=-1, keepdim=True)  # (1, N_sub, 1)
    single_slot_mask = torch.zeros_like(cm_sub)
    single_slot_mask.scatter_(-1, top_slot.expand(t_pick, n_pick, 1), True)
    # Alternate between single-slot and all-active-slot perturbations
    edit_subset = (single_slot_mask & cm_sub) if (int(step_seed) % 2 == 0 and k_cands >= 2) else cm_sub
    if not bool(edit_subset.any().item()):
        edit_subset = cm_sub

    # Alternate `+dh` (raise) and `-dh` (lower) across steps (or evaluate `+dh` on odd, `-dh` on modulo)
    signed_dh = float(abs(delta_height_m)) if (int(step_seed) % 3 != 2) else -float(abs(delta_height_m))
    tok_cf_raw, edited_triangles_m = build_structural_intervention_tokens_torch(
        tok_orig_sub,
        cm_sub,
        sun_sub,
        delta_height_m=signed_dh,
        edit_cand_mask=edit_subset,
    )
    tok_cf_san = sanitize_fn(tok_cf_raw) if callable(sanitize_fn) else tok_cf_raw

    cf_sub = dict(base_sub)
    cf_sub["candidate_tokens"] = tok_cf_san.to(dtype=tok_orig_sub.dtype)
    # All static features, including global kNN, stay identical in both arms.

    cf_out = model(cf_sub, use_calibrated_temperature=False)
    cf_probs = cf_out["probs"].to(torch.float32)
    has_ed_tn = edit_subset.any(dim=-1)
    zero_t = base_probs.sum() * 0.0
    total_loss = zero_t
    edit_term = zero_t
    drop_mono_term = zero_t
    sec_bce_term = zero_t
    ed_supp_cnt = 0
    flip_supp_cnt = 0
    unch_supp_cnt = 0
    bern_supp_cnt = 0
    sec_supp_cnt = 0
    if bool(has_ed_tn.any().item()):
        opaque = (tok_orig_sub[..., 10] > 0.5) & cm_sub
        original_triangles_m = tok_orig_sub[..., 13:22].float().reshape(*cm_sub.shape, 3, 3) * 1000.0
        old_hit = exact_candidate_ray_blocked_torch(original_triangles_m, opaque, sun_sub)
        new_hit = exact_candidate_ray_blocked_torch(edited_triangles_m, opaque, sun_sub)
        ed_supp_cnt = int(has_ed_tn.sum().item())
        flip_supp_cnt = int(((old_hit != new_hit) & has_ed_tn).sum().item())
        unch_supp_cnt = int(((old_hit == new_hit) & has_ed_tn).sum().item())
        if edit_counts is not None:
            sizes = edit_subset.sum(-1)
            for name,mask in (("single_face",sizes == 1),("multi_face",sizes > 1),
                              ("shadow_to_sun",old_hit & ~new_hit),("sun_to_shadow",~old_hit & new_hit),
                              ("unchanged_ray",old_hit == new_hit)):
                edit_counts[name] = edit_counts.get(name,0) + int((mask & has_ed_tn).sum())
            name = "raised_calls" if signed_dh > 0 else "lowered_calls"
            edit_counts[name] = edit_counts.get(name,0) + 1
        change = cf_probs - clean_sub_probs
        expected = old_hit.float() - new_hit.float()
        # A physical flip requires a nonzero response. The old direction-only
        # hinge had zero loss AND zero gradient when change == 0. Regress the
        # signed physical probability difference on flipped rays; unchanged rays
        # retain the original invariance penalty. The clean reference is detached.
        violation = torch.where(expected == 0, change.abs(), (change - expected).square())
        edit_term = violation[has_ed_tn].mean()
        total_loss = total_loss + edit_term

    # 2. Dominant-Occluder Dropout for Layered Occlusion Sufficiency (`K >= 2`)
    multi_cand_tn = cm_sub.sum(dim=-1) >= 2
    if float(dominant_occluder_dropout_prob) > 0.0 and k_cands >= 2 and bool(multi_cand_tn.any().item()):
        rel_tris_orig_m = tok_orig_sub[...,13:22].float().reshape(*cm_sub.shape,3,3)*1000.
        opaque_orig = (tok_orig_sub[...,10] > .5) & cm_sub
        distances = _candidate_ray_distances_torch(rel_tris_orig_m,opaque_orig,sun_sub)
        hits = distances.isfinite()
        hit_count = hits.sum(-1)
        first = distances.amin(-1,keepdim=True)
        # Duplicated or coplanar triangles at the front depth represent one
        # crossing. Delete their whole group before checking a deeper crossing.
        drop_dom = hits & torch.isclose(distances,first,atol=1e-4,rtol=1e-8)
        deeper = (hits & ~drop_dom).any(-1)
        if edit_counts is not None:
            for key,mask in (("unhit_rays",hit_count==0),("single_hit_rays",hit_count==1),
                             ("multiple_hit_face_rays",hit_count>=2),
                             ("one_distinct_hit_depth",(hit_count>0) & ~deeper),
                             ("tied_front_depth",drop_dom.sum(-1)>1),
                             ("multiple_hit_depth_rays",deeper)):
                edit_counts[key] = edit_counts.get(key,0) + int((multi_cand_tn & mask).sum())
        # Apply dropout with probability `dominant_occluder_dropout_prob` on `multi_cand_tn` slots
        bern = (
            torch.rand((t_pick, n_pick), generator=gen).to(tokens.device)
            < float(dominant_occluder_dropout_prob)
        ) & multi_cand_tn & (hit_count > 0)
        bern_supp_cnt = int(bern.sum().item())
        if bool(bern.any().item()):
            keep_sec = cm_sub & (~(drop_dom & bern.unsqueeze(-1)))
            sec_sub = dict(base_sub)
            sec_sub["candidate_keep_mask"] = keep_sec
            sec_out = model(sec_sub, use_calibrated_temperature=False)
            sec_probs = sec_out["probs"].to(torch.float32)
            sec_logits = sec_out["logits_bce"].to(torch.float32)

            # (a) Deleting dominant occluder `A` cannot make a sunlit receiver become shadowed
            del_mono = F.relu(clean_sub_probs - sec_probs)[bern].mean()
            drop_mono_term = 0.5 * del_mono
            total_loss = total_loss + drop_mono_term

            # (b) Exact 3D secondary occlusion retention (`0` label reads!): if remaining `K-1` candidates
            #     physically intersect the solar ray, `sec_logits` MUST predict shadow (`0.0`)!
            sec_blocked = exact_candidate_ray_blocked_torch(
                rel_tris_orig_m, opaque_orig & keep_sec, sun_sub
            )
            layered_occ = bern & sec_blocked
            sec_supp_cnt = int(layered_occ.sum().item())
            if edit_counts is not None:
                edit_counts["secondary_hit_after_dropout"] = edit_counts.get("secondary_hit_after_dropout",0) + sec_supp_cnt
            if bool(layered_occ.any().item()):
                sec_occ_bce = F.binary_cross_entropy_with_logits(
                    sec_logits[layered_occ], torch.zeros_like(sec_logits[layered_occ])
                )
                sec_bce_term = 0.5 * sec_occ_bce
                total_loss = total_loss + sec_bce_term

    if cf_step_stats is not None:
        cf_step_stats["cf_raw_total_loss"] = float(total_loss.detach().item())
        cf_step_stats["height_edit_raw_loss"] = float(edit_term.detach().item())
        cf_step_stats["dropout_monotonicity_raw_loss"] = float(drop_mono_term.detach().item())
        cf_step_stats["secondary_occlusion_bce_raw_loss"] = float(sec_bce_term.detach().item())
        cf_step_stats["edit_support"] = float(ed_supp_cnt)
        cf_step_stats["flip_support"] = float(flip_supp_cnt)
        cf_step_stats["unchanged_support"] = float(unch_supp_cnt)
        cf_step_stats["dropout_bern_support"] = float(bern_supp_cnt)
        cf_step_stats["secondary_hit_support"] = float(sec_supp_cnt)
    if cf_tensor_terms is not None:
        cf_tensor_terms["cf_raw_total"] = total_loss
        cf_tensor_terms["height_edit_raw"] = edit_term
        cf_tensor_terms["dropout_monotonicity_raw"] = drop_mono_term
        cf_tensor_terms["secondary_occlusion_bce_raw"] = sec_bce_term

    return total_loss


def compute_candidate_face_perturbation_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
    base_probs: torch.Tensor,
    *,
    delta_height_m: float = 6.0,
    dominant_occluder_dropout_prob: float = 0.25,
    max_sub_frames: int = 8,
    max_sub_receivers: int = 32,
    step_seed: int = 0,
    edit_counts: Optional[Dict[str, int]] = None,
    cf_subgroup_size: int = 0,
    cf_step_stats: Optional[Dict[str, float]] = None,
    cf_tensor_terms: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Sample one CF perturbation per receiver subgroup, preserving total weight.

    Zero keeps the legacy single sample for the whole batch. With e.g. 64, every
    group of at most 64 receivers gets its own bounded sample, and group means
    are weighted by their actual receiver counts (including the final short group).
    No labels are read, and the subgroup weights always sum to one.
    """
    subgroup_size = int(cf_subgroup_size)
    if subgroup_size < 0:
        raise ValueError("cf_subgroup_size must be non-negative")
    kwargs: Dict[str, Any] = dict(
        delta_height_m=delta_height_m,
        dominant_occluder_dropout_prob=dominant_occluder_dropout_prob,
        max_sub_frames=max_sub_frames,
        max_sub_receivers=max_sub_receivers,
        step_seed=step_seed,
        edit_counts=edit_counts,
    )
    if cf_step_stats is not None:
        kwargs["cf_step_stats"] = cf_step_stats
    if cf_tensor_terms is not None:
        kwargs["cf_tensor_terms"] = cf_tensor_terms
    tokens = batch.get("candidate_tokens")
    if subgroup_size == 0 or tokens is None or not getattr(model, "model_type", "").startswith("rind_"):
        return _compute_candidate_face_perturbation_loss(model, batch, base_probs, **kwargs)
    n_pts = int(tokens.shape[-3])
    if n_pts <= subgroup_size:
        return _compute_candidate_face_perturbation_loss(model, batch, base_probs, **kwargs)

    zero_t = base_probs.sum() * 0.0
    total = zero_t
    agg_stats = {
        "cf_raw_total_loss": 0.0,
        "height_edit_raw_loss": 0.0,
        "dropout_monotonicity_raw_loss": 0.0,
        "secondary_occlusion_bce_raw_loss": 0.0,
        "edit_support": 0.0,
        "flip_support": 0.0,
        "unchanged_support": 0.0,
        "dropout_bern_support": 0.0,
        "secondary_hit_support": 0.0,
    }
    agg_tensors = {
        "cf_raw_total": zero_t,
        "height_edit_raw": zero_t,
        "dropout_monotonicity_raw": zero_t,
        "secondary_occlusion_bce_raw": zero_t,
    }
    static_keys = (
        "local_features", "receivers_xyz", "receiver_rc", "global_knn_local_features",
        "global_knn_rel_xyz", "global_knn_dist", "global_knn_obs_valid",
    )
    for group_idx, start in enumerate(range(0, n_pts, subgroup_size)):
        stop = min(start + subgroup_size, n_pts)
        # Only the inputs consumed by the CF worker are copied. In particular,
        # response labels in a LabelReadForbiddenBatchView are never traversed.
        sub: Dict[str, Any] = {
            "candidate_tokens": tokens[..., start:stop, :, :],
            "candidate_mask": batch["candidate_mask"][..., start:stop, :],
            "sun_dirs": batch["sun_dirs"],
        }
        for key in static_keys:
            value = batch.get(key)
            if value is not None:
                if key.startswith("global_knn_"):
                    # Packs are (N,K,F)/(N,K) or carry an explicit batch axis.
                    rank = 2 if key in ("global_knn_dist", "global_knn_obs_valid") else 3
                    axis = 0 if value.ndim == rank else 1
                else:
                    axis = 0 if value.ndim == 2 else 1
                sub[key] = value.narrow(axis, start, stop - start)
        obs = batch.get("receiver_obs_mask")
        if obs is not None:
            sub["receiver_obs_mask"] = obs[..., start:stop]
        sub_probs = base_probs[..., start:stop]
        # Multiple of six preserves the raise/lower and single/subset schedule.
        sub_kw = dict(kwargs)
        sub_kw["step_seed"] = int(step_seed) + group_idx * 104730
        grp_stats: Optional[Dict[str, float]] = {} if cf_step_stats is not None else None
        grp_tensors: Optional[Dict[str, torch.Tensor]] = {} if cf_tensor_terms is not None else None
        if grp_stats is not None:
            sub_kw["cf_step_stats"] = grp_stats
        if grp_tensors is not None:
            sub_kw["cf_tensor_terms"] = grp_tensors
        group_loss = _compute_candidate_face_perturbation_loss(model, sub, sub_probs, **sub_kw)
        w_grp = float(stop - start) / float(n_pts)
        total = total + w_grp * group_loss
        if grp_stats is not None:
            for lk in ("cf_raw_total_loss", "height_edit_raw_loss", "dropout_monotonicity_raw_loss", "secondary_occlusion_bce_raw_loss"):
                agg_stats[lk] += w_grp * grp_stats.get(lk, 0.0)
            for sk in ("edit_support", "flip_support", "unchanged_support", "dropout_bern_support", "secondary_hit_support"):
                agg_stats[sk] += grp_stats.get(sk, 0.0)
        if grp_tensors is not None:
            for tk in agg_tensors:
                if tk in grp_tensors:
                    agg_tensors[tk] = agg_tensors[tk] + w_grp * grp_tensors[tk]
    if cf_step_stats is not None:
        cf_step_stats.update(agg_stats)
    if cf_tensor_terms is not None:
        cf_tensor_terms.update(agg_tensors)
    return total


def _resolve_jepa_label_free_valid(
    batch: Mapping[str, Any],
    *,
    bsz: int,
    t_steps: int,
    n_pts: int,
    device: torch.device,
) -> torch.Tensor:
    """Resolve valid mask for `ema_geom` / `none` modes strictly from `receiver_obs_mask` (`0` label reads!)."""
    raw_obs = batch.get("receiver_obs_mask")
    if raw_obs is None:
        return torch.ones((bsz, t_steps, n_pts), dtype=torch.bool, device=device)
    obs = raw_obs.to(device=device, dtype=torch.bool)
    if obs.ndim == 1:
        return obs.view(1, 1, n_pts).expand(bsz, t_steps, n_pts)
    if obs.ndim == 2:
        if int(obs.shape[0]) == t_steps and int(obs.shape[1]) == n_pts and bsz == 1:
            return obs.unsqueeze(0)
        return obs.unsqueeze(1).expand(bsz, t_steps, n_pts)
    if obs.ndim == 3:
        return obs.expand(bsz, t_steps, n_pts)
    return torch.ones((bsz, t_steps, n_pts), dtype=torch.bool, device=device)


# Compatibility for existing Python callers.
compute_counterfactual_height_monotonicity_loss = compute_candidate_face_perturbation_loss


class RindJepaAuxiliaryModule(nn.Module):
    """Complete Zero-Shrinkage Training Auxiliary Module (`Mask + X` JEPA + `SIGReg`/`VICReg` + `L_trans`)."""

    def __init__(
        self,
        hidden_dim: Optional[int] = None,
        *,
        latent_dim: Optional[int] = None,
        solar_dim: Optional[int] = None,
        jepa_mode: str = "hybrid",
        reg_mode: str = "sigreg",
        leads: Sequence[int] = DEFAULT_JEPA_LEADS,
        ema_decay: float = 0.996,
        objective_config: Optional[ObjectiveConfig] = None,
    ) -> None:
        super().__init__()
        eff_dim = hidden_dim if hidden_dim is not None else latent_dim
        if eff_dim is None:
            raise ValueError("RindJepaAuxiliaryModule requires `hidden_dim` (or `latent_dim`).")
        if objective_config is not None:
            self.cfg = objective_config
        else:
            self.cfg = ObjectiveConfig(
                jepa_mode=jepa_mode,
                reg_mode=reg_mode,
                leads=tuple(int(x) for x in leads),
            )
        self.hidden_dim = int(eff_dim)
        self.solar_dim = int(solar_dim) if solar_dim is not None else self.hidden_dim
        self.jepa_mode = (
            "response_teacher" if str(self.cfg.jepa_mode) == "response" else str(self.cfg.jepa_mode)
        )
        self.reg_mode = str(self.cfg.reg_mode)
        self.solar_adapter: Optional[nn.Linear] = (
            nn.Linear(self.solar_dim, self.hidden_dim) if self.solar_dim != self.hidden_dim else None
        )

        self.use_response_teacher = (not self.cfg.pure_supervised) and self.jepa_mode in (
            "response_teacher",
            "hybrid",
        )
        self.use_ema_teacher = (not self.cfg.pure_supervised) and self.jepa_mode in (
            "ema_geom",
            "hybrid",
        )

        self.response_teacher: Optional[ResponseTargetTeacher] = (
            ResponseTargetTeacher(self.hidden_dim, ema_decay=ema_decay)
            if self.use_response_teacher
            else None
        )
        self.ema_decay = float(ema_decay)
        self.ema_teacher: Optional[EMAGeometricTarget] = (
            EMAGeometricTarget(self.hidden_dim, ema_decay=ema_decay)
            if self.use_ema_teacher
            else None
        )
        self.ema_backbone: Optional[nn.Module] = None
        self._student_backbone_ref: List[Optional[nn.Module]] = [None]
        self.predictor: Optional[MultiHorizonLatentPredictor] = (
            MultiHorizonLatentPredictor(
                self.hidden_dim,
                leads=self.cfg.leads,
                mask_ratio=self.cfg.mask_ratio,
                spatial_block_size=self.cfg.spatial_block_size,
            )
            if (not self.cfg.pure_supervised and self.jepa_mode not in ("none","geometry_mae"))
            else None
        )
        if self.predictor is not None:
            self.predictor.buffer_m = self.cfg.buffer_m
            self.predictor.guard_buffer_m = self.cfg.guard_buffer_m
        self.geometry_decoder = nn.Linear(self.hidden_dim,37) if self.jepa_mode == "geometry_mae" else None
        self.reg_projector: Optional[nn.Module] = (
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            if (not self.cfg.pure_supervised and self.reg_mode != "none")
            else None
        )

    def bind_student_backbone(self, student_model: nn.Module) -> None:
        """Attach full-backbone EMA copy when `ema_target_scope == 'full_backbone'` (`JEPA-1`)."""
        self._student_backbone_ref[0] = student_model
        if self.use_ema_teacher and self.cfg.ema_target_scope == "full_backbone":
            ema_copy = copy.deepcopy(student_model)
            ema_copy.eval()
            for p in ema_copy.parameters():
                p.requires_grad_(False)
            self.ema_backbone = ema_copy

    @property
    def context_aggregator(self) -> Optional[nn.Module]:
        return self.predictor.context_aggregator if self.predictor is not None else None

    def get_objective_config(self) -> Dict[str, Any]:
        return self.cfg.to_dict()

    @torch.no_grad()
    def update_normalizers(
        self,
        batch: Mapping[str, Any],
        *,
        firewall_token: FitFirewallToken,
        tile_ids: Sequence[str],
        expected_split_sha256: str,
        device: torch.device,
    ) -> None:
        if self.response_teacher is not None:
            budget_batch, _ = apply_label_budget_view(
                batch,
                label_budget_fraction=self.cfg.label_budget_fraction,
                label_budget_seed=self.cfg.label_budget_seed,
                device=device,
            )
            self.response_teacher.update_normalizer(
                budget_batch,
                firewall_token=firewall_token,
                tile_ids=tile_ids,
                expected_split_sha256=expected_split_sha256,
                device=device,
            )
        if self.ema_backbone is not None and self._student_backbone_ref[0] is not None:
            for b_ema, b_stu in zip(self.ema_backbone.buffers(), self._student_backbone_ref[0].buffers()):
                if b_ema.dtype.is_floating_point or b_ema.dtype == torch.int64:
                    b_ema.copy_(b_stu.to(device=b_ema.device, dtype=b_ema.dtype))

    @torch.no_grad()
    def step_ema(self) -> None:
        if self.response_teacher is not None:
            self.response_teacher.step_ema()
        if self.ema_teacher is not None:
            self.ema_teacher.step_ema()
        if self.ema_backbone is not None and self._student_backbone_ref[0] is not None:
            d = self.ema_decay
            for p_ema, p_stu in zip(self.ema_backbone.parameters(), self._student_backbone_ref[0].parameters()):
                p_ema.mul_(d).add_(p_stu.detach().to(device=p_ema.device, dtype=p_ema.dtype), alpha=1.0 - d)
            for b_ema, b_stu in zip(self.ema_backbone.buffers(), self._student_backbone_ref[0].buffers()):
                b_ema.copy_(b_stu.detach().to(device=b_ema.device, dtype=b_ema.dtype))

    def forward_objectives(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        *,
        step_index: int = 0,
        mask_mode_override: Optional[str] = None,
        jepa_context_mode_override: Optional[str] = None,
        precomputed_target_mask: Optional[torch.Tensor] = None,
        tensor_terms_out: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute `(aux_loss, aux_metrics)` with strict pure-supervised & low-label firewalls."""
        z_roll = outputs["latent_roll"].to(torch.float32)
        z_direct = outputs.get("latent_direct", z_roll).to(torch.float32)
        if z_roll.ndim == 3:
            z_roll = z_roll.unsqueeze(0)
            z_direct = z_direct.unsqueeze(0)

        metrics: Dict[str, Optional[float]] = {
            "jepa_pred_loss": 0.0,
            "jepa_trans_vel_loss": 0.0,
            "dual_path_consistency_loss": 0.0,
            "anti_collapse_reg_loss": 0.0,
            "jepa_context_sensitivity_dx": 0.0,
            "jepa_mask_ratio_actual": 0.0,
            "modality_alignment_rho_m": None,
            "ema_target_scope": str(self.cfg.ema_target_scope),
        }

        # Pure-supervised firewall (`P1`): exact `0.0` aux loss and `0.0 / None` aux parameter gradients!
        if self.cfg.pure_supervised or (
            self.jepa_mode == "none"
            and self.reg_mode == "none"
            and self.cfg.transition_mode == "none"
            and self.cfg.dual_path_weight <= 0.0
        ):
            zero_detached = torch.zeros((), dtype=torch.float32, device=z_roll.device)
            return zero_detached, metrics

        bsz, t_steps, n_pts, _ = z_roll.shape
        same_day_adj = batch.get("same_day_adjacent_mask")
        logits_boundary_out = outputs.get("logits_boundary")

        truth_f: Optional[torch.Tensor] = None
        if self.use_response_teacher:
            eff_batch, _ = apply_label_budget_view(
                batch,
                label_budget_fraction=self.cfg.label_budget_fraction,
                label_budget_seed=self.cfg.label_budget_seed,
                device=z_roll.device,
            )
            raw_valid = eff_batch.get("valid_mask")
            if raw_valid is not None:
                valid = raw_valid.to(device=z_roll.device, dtype=torch.bool)
                if valid.ndim == 2:
                    valid = valid.unsqueeze(0)
            else:
                valid = _resolve_jepa_label_free_valid(
                    eff_batch, bsz=bsz, t_steps=t_steps, n_pts=n_pts, device=z_roll.device
                )
            raw_truth = eff_batch.get("truth")
            if raw_truth is not None:
                labels = raw_truth.to(device=z_roll.device).reshape(bsz, t_steps, n_pts)
                valid = valid & ((labels == 0) | (labels == 1))
                tf = (labels > 0).to(torch.float32)
                truth_f = tf.unsqueeze(0) if tf.ndim == 2 else tf
            boundary_cue_m = eff_batch.get("boundary_distance_m")
        else:
            # Strictly 0 reads of any label tensor in `ema_geom` / `none` modes!
            eff_batch = dict(batch)
            valid = _resolve_jepa_label_free_valid(
                eff_batch, bsz=bsz, t_steps=t_steps, n_pts=n_pts, device=z_roll.device
            )
            boundary_cue_m = None

        solar_emb = outputs.get("solar_emb")
        if solar_emb is None:
            solar_emb = torch.zeros(
                (bsz, t_steps, self.hidden_dim),
                dtype=torch.float32,
                device=z_roll.device,
            )
        else:
            solar_emb = solar_emb.to(torch.float32)
            if solar_emb.ndim == 2:
                solar_emb = solar_emb.unsqueeze(0)
            if self.solar_adapter is not None and int(solar_emb.shape[-1]) == self.solar_dim:
                solar_emb = self.solar_adapter(solar_emb)

        total_aux = z_roll.sum() * 0.0

        if self.geometry_decoder is not None:
            # Hide a randomly selected subset of actual face tokens. The decoder
            # reconstructs the mean transformed canonical37 of the held-out set.
            tokens = batch["candidate_tokens"]
            present = batch["candidate_mask"].bool()
            gen = torch.Generator(device="cpu").manual_seed(9701 + int(step_index))
            drop = (torch.rand(present.shape, generator=gen).to(present.device) < self.cfg.mask_ratio) & present
            support = drop.any(-1)
            metrics["geometry_mae_queries"] = int(support.sum())
            if support.any():
                masked = dict(batch)
                masked["candidate_mask"] = present & ~drop
                masked["candidate_tokens"] = tokens.masked_fill(drop[...,None],0)
                branch = self._student_backbone_ref[0](masked, inference_mode="static_direct",
                                                      use_calibrated_temperature=False)
                prediction = self.geometry_decoder(branch["latent_direct"])
                target = (tokens.detach().float().asinh()*drop[...,None]).sum(-2)
                target = target / drop.sum(-1,keepdim=True).clamp_min(1)
                reconstruction = F.smooth_l1_loss(prediction[support],target[support])
                total_aux = total_aux + self.cfg.jepa_weight*reconstruction
                metrics["geometry_mae_loss"] = float(reconstruction.detach())

        # 1. Dual-Path Rollout <-> Direct Consistency
        if self.cfg.dual_path_weight > 0.0 and bool(valid.any().item()):
            w_dp = float(self.cfg.dual_path_weight)
            dp_lat = F.smooth_l1_loss(z_roll[valid], z_direct.detach()[valid], beta=0.1)
            if "logits_direct" in outputs and "logits_bce" in outputs:
                lbce = outputs["logits_bce"].to(torch.float32).reshape_as(valid)
                ldir = outputs["logits_direct"].detach().to(torch.float32).reshape_as(valid)
                dp_log = F.smooth_l1_loss(lbce[valid], ldir[valid], beta=0.25)
                dp_loss = w_dp * dp_lat + w_dp * dp_log
            else:
                dp_loss = w_dp * dp_lat
            total_aux = total_aux + dp_loss
            metrics["dual_path_consistency_loss"] = float(dp_loss.detach().item())
            if tensor_terms_out is not None:
                tensor_terms_out["dual_path_scaled"] = dp_loss

        # 2. `Mask + X` Restricted-Context ST-JEPA + Teacher Alignment + Increment Matching `L_trans`
        # Decoupled active check (Gate #1): setting `jepa_weight == 0.0` (`J0`) does NOT disable
        # `teacher_alignment_loss` or `L_trans` when their respective weights are positive.
        jepa_branch_active = (
            float(self.cfg.jepa_weight) > 0.0
            or float(self.cfg.teacher_align_weight) > 0.0
            or (self.cfg.transition_mode != "none" and float(self.cfg.transition_weight) > 0.0)
        )
        if self.predictor is not None and jepa_branch_active:
            eff_mask_mode = (
                str(mask_mode_override) if mask_mode_override is not None else str(self.cfg.mask_mode)
            )
            eff_ctx_mode = (
                str(jepa_context_mode_override)
                if jepa_context_mode_override is not None
                else str(self.cfg.jepa_context_mode)
            )

            # Construct disjoint `(Omega_T, Omega_C)` BEFORE building student restricted context `X`!
            struct_target_mask = (
                precomputed_target_mask.to(device=z_roll.device, dtype=torch.bool).reshape(bsz, t_steps, n_pts)
                if precomputed_target_mask is not None
                else self.predictor.build_structured_mask(
                    z_direct,
                    mode=eff_mask_mode,
                    boundary_distance_m=boundary_cue_m,
                    logits_boundary=logits_boundary_out,
                    receiver_rc=eff_batch.get("receiver_rc"),
                    receivers_xyz=eff_batch.get("receivers_xyz"),
                    step_seed=int(step_index),
                )
            )
            context_mask = permitted_context(
                struct_target_mask,
                valid,
                eff_mask_mode,
                eff_batch.get("receivers_xyz"),
                self.cfg.buffer_m,
                guard_buffer_m=float(self.cfg.guard_buffer_m),
            )

            targets: List[torch.Tensor] = []
            teacher_align_terms: List[torch.Tensor] = []
            context_x_emb: Optional[torch.Tensor] = None
            need_align_grad = float(self.cfg.teacher_align_weight) > 0.0
            need_pred_grad = float(self.cfg.jepa_weight) > 0.0

            if self.response_teacher is not None:
                # Teacher encodes full unmasked valid field `S_u` (stop-grad on `z_r_tg`)
                with torch.set_grad_enabled(torch.is_grad_enabled() and need_align_grad):
                    z_r_on, z_r_tg = self.response_teacher.encode_online_and_target(eff_batch, solar_emb)
                    if bool(valid.any().item()):
                        reconstruction = F.binary_cross_entropy_with_logits(
                            z_r_on[..., 0][valid] * 4.0, truth_f[valid]
                        )
                        metrics["teacher_response_reconstruction_bce"] = float(reconstruction.detach())
                        teacher_align_terms.append(
                            reconstruction
                            + F.l1_loss(z_r_on[valid], z_roll.detach()[valid])
                            + F.l1_loss(z_roll[valid], z_r_tg[valid])
                        )
                targets.append(z_r_tg)
                if bool(valid.any().item()):
                    with torch.no_grad():
                        metrics["modality_alignment_rho_m"] = float(
                            F.cosine_similarity(z_roll.detach()[valid], z_r_tg[valid], dim=-1).mean().item()
                        )
                # Student restricted context `X = S_u|_{Omega_C}` (`0.0` dependency on `Omega_T` truth!)
                with torch.set_grad_enabled(torch.is_grad_enabled() and need_pred_grad):
                    context_x_emb = self.response_teacher.encode_restricted_context_x(
                        eff_batch, context_mask, solar_emb
                    )

            if self.ema_teacher is not None:
                z_ema_src = z_direct
                if self.ema_backbone is not None and self.cfg.ema_target_scope == "full_backbone":
                    lf_ema_in = {
                        k: v
                        for k, v in eff_batch.items()
                        if k
                        not in (
                            "truth",
                            "valid_mask",
                            "boundary_distance_m",
                            "boundary_distance_valid",
                            "response_5x5_bitpack",
                            "label_budget_mask",
                        )
                    }
                    if "sun_dirs" in lf_ema_in:
                        with torch.no_grad():
                            ema_out = self.ema_backbone(lf_ema_in, use_calibrated_temperature=False)
                        z_ema_bb = ema_out.get("latent_direct", ema_out["latent_roll"]).to(torch.float32)
                        if z_ema_bb.ndim == 3:
                            z_ema_bb = z_ema_bb.unsqueeze(0)
                        z_ema_src = z_ema_bb.detach()
                with torch.set_grad_enabled(torch.is_grad_enabled() and need_align_grad):
                    z_e_on, z_e_tg = self.ema_teacher.encode_online_and_target(z_ema_src, solar_emb)
                    if bool(valid.any().item()):
                        teacher_align_terms.append(F.l1_loss(z_e_on[valid], z_roll.detach()[valid]))
                targets.append(z_e_tg)

            if targets:
                z_teacher = sum(targets) / float(len(targets))
                history = None
                if self.training and eff_batch.get("frame_range") is not None:
                    identity = (tuple(eff_batch.get("tile_ids",())),tuple(eff_batch.get("receiver_range",())))
                    start, stop = eff_batch["frame_range"]
                    continuation = (eff_batch.get("carry_state_from_prev_chunk",False)
                                    and eff_batch.get("prev_frame_same_day",False)
                                    and getattr(self,"history_key",None)==(identity,start))
                    if not continuation:
                        self.horizon_history = {}
                    self.history_key = (identity,stop)
                    history = self.horizon_history
                with torch.set_grad_enabled(torch.is_grad_enabled() and need_pred_grad):
                    pred_loss, pred_diag = self.predictor.forward_multi_horizon(
                        z_direct,
                        z_teacher,
                        solar_emb,
                        valid_mask=valid,
                        same_day_adjacent_mask=same_day_adj,
                        boundary_distance_m=boundary_cue_m,
                        logits_boundary=logits_boundary_out,
                        receiver_rc=eff_batch.get("receiver_rc"),
                        receivers_xyz=eff_batch.get("receivers_xyz"),
                        static_geom_memory=outputs.get("static_geom_memory"),
                        context_x_emb=context_x_emb,
                        jepa_context_mode=eff_ctx_mode,
                        mask_mode=eff_mask_mode,
                        precomputed_target_mask=struct_target_mask,
                        history=history,
                        include_zero_lead=(context_x_emb is not None and eff_mask_mode != "none"),
                        step_seed=int(step_index),
                    )
                align_loss = (
                    sum(teacher_align_terms) / float(len(teacher_align_terms))
                    if teacher_align_terms
                    else z_roll.sum() * 0.0
                )
                pred_scaled = float(self.cfg.jepa_weight) * pred_loss if need_pred_grad else z_roll.new_zeros(())
                align_scaled = (
                    float(self.cfg.teacher_align_weight) * align_loss
                    if (need_align_grad and bool(teacher_align_terms))
                    else z_roll.new_zeros(())
                )
                jepa_comb = pred_scaled + align_scaled
                total_aux = total_aux + jepa_comb
                if tensor_terms_out is not None:
                    tensor_terms_out["jepa_pred_scaled"] = pred_scaled
                    tensor_terms_out["teacher_align_scaled"] = align_scaled
                metrics.update(pred_diag)
                metrics["teacher_alignment_loss"] = float(align_loss.detach())
                metrics["multi_horizon_prediction_loss"] = float(pred_loss.detach())
                metrics["jepa_pred_loss"] = float(jepa_comb.detach().item())
                metrics["jepa_context_sensitivity_dx"] = float(pred_diag["jepa_context_sensitivity_dx"])
                metrics["jepa_mask_ratio_actual"] = float(pred_diag["jepa_mask_ratio_actual"])
                metrics["last_teacher_final_latent_norm"] = float(
                    z_teacher[:, -1].detach().norm().item()
                )
                metrics["guard_buffer_m"] = float(self.cfg.guard_buffer_m)
                metrics["context_receiver_fraction_actual"] = float(context_mask.sum().item()) / float(
                    max(int(valid.sum().item()), 1)
                )

                if self.cfg.transition_mode != "none" and self.cfg.transition_weight > 0.0:
                    z_pair, teacher_pair, valid_pair = z_roll, z_teacher, valid
                    signal = truth_f if self.use_response_teacher and truth_f is not None else outputs.get("probs")
                    if signal is not None:
                        signal = signal.detach().reshape(bsz,t_steps,n_pts)
                    adj = (same_day_adj.reshape(-1,t_steps-1).expand(bsz,-1)
                           if same_day_adj is not None and t_steps>1 else
                           torch.ones((bsz,max(t_steps-1,0)),dtype=torch.bool,device=z_roll.device))
                    if history is not None:
                        previous = getattr(self,"transition_history",{}) if continuation else {}
                        self.transition_history = {
                            "z": z_roll[:,-1:].detach(), "teacher": z_teacher[:,-1:].detach(),
                            "valid": valid[:,-1:].detach(),
                        }
                        if signal is not None:
                            self.transition_history["signal"] = signal[:,-1:].detach()
                        if previous:
                            z_pair = torch.cat((previous["z"],z_pair),dim=1)
                            teacher_pair = torch.cat((previous["teacher"],teacher_pair),dim=1)
                            valid_pair = torch.cat((previous["valid"],valid_pair),dim=1)
                            adj = torch.cat((torch.ones_like(valid[:,:1,0]),adj),dim=1)
                            if signal is not None:
                                signal = torch.cat((previous["signal"],signal),dim=1)
                    pair_v = valid_pair[:,1:] & valid_pair[:,:-1] & adj.unsqueeze(-1)
                    metrics["teacher_increment_pair_support"] = int(pair_v.sum())
                    if bool(pair_v.any().item()):
                        dz_stu = z_pair[:,1:] - z_pair[:,:-1]
                        dz_tea = (teacher_pair[:,1:] - teacher_pair[:,:-1]).detach()
                        w_ev = pair_v.float()
                        if self.cfg.transition_mode == "event_aware":
                            is_sw = ((signal[:,1:]-signal[:,:-1]).abs() if signal is not None
                                     else torch.linalg.norm(dz_tea,dim=-1).clamp(0,1))
                            w_ev = w_ev * (1. + 4. * is_sw)
                        # driver_smooth is the legacy unweighted teacher-increment experiment.
                        distance = transition_distance(dz_stu, dz_tea, self.cfg.transition_mode)
                        l_trans = (
                            float(self.cfg.transition_weight)
                            * (distance * w_ev).sum()
                            / w_ev.sum().clamp_min(1.0)
                        )
                        total_aux = total_aux + l_trans
                        metrics["jepa_trans_vel_loss"] = float(l_trans.detach().item())
                        if tensor_terms_out is not None:
                            tensor_terms_out["trans_vel_scaled"] = l_trans

        # 3. Online Projected Anti-Collapse Regularization (`SIGReg`, `VICReg`, or joint `vicreg_sigreg`)
        if self.reg_projector is not None and self.cfg.reg_weight > 0.0 and bool(valid.any().item()):
            z_online_valid = z_direct[valid]
            if int(z_online_valid.shape[0]) > 2048:
                stride = max(1, int(z_online_valid.shape[0]) // 2048)
                z_online_valid = z_online_valid[::stride][:2048]
            metrics["regularizer_sample_count"] = int(z_online_valid.shape[0])
            metrics["regularizer_skipped"] = int(z_online_valid.shape[0] < 8)
            if int(z_online_valid.shape[0]) >= 8:
                proj_z = self.reg_projector(z_online_valid)
                if self.reg_mode == "sigreg":
                    reg_raw = 0.02 * sketched_isotropic_gaussian_reg(
                        proj_z, step_seed=int(step_index)
                    )
                elif self.reg_mode == "vicreg":
                    reg_raw = 0.08 * vicreg_variance_covariance_loss(proj_z)
                else:
                    reg_raw = 0.02 * sketched_isotropic_gaussian_reg(
                        proj_z, step_seed=int(step_index)
                    ) + 0.04 * vicreg_variance_covariance_loss(proj_z)
                reg_scaled = float(self.cfg.reg_weight) * reg_raw
                total_aux = total_aux + reg_scaled
                metrics["anti_collapse_reg_loss"] = float(reg_scaled.detach().item())
                if tensor_terms_out is not None:
                    tensor_terms_out["anti_collapse_reg_scaled"] = reg_scaled

        return total_aux, metrics

    def compute_auxiliary_losses(
        self,
        z_seq: torch.Tensor,
        *,
        solar_emb: Optional[torch.Tensor] = None,
        gate_probs: Optional[torch.Tensor] = None,
        dual_latents: Optional[torch.Tensor] = None,
        logits_boundary: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        receiver_obs_mask: Optional[torch.Tensor] = None,
        receiver_rc: Optional[torch.Tensor] = None,
        truth: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        boundary_distance_m: Optional[torch.Tensor] = None,
        boundary_distance_valid: Optional[torch.Tensor] = None,
        response_5x5_bitpack: Optional[torch.Tensor] = None,
        same_day_adjacent_mask: Optional[torch.Tensor] = None,
        receivers_xyz: Optional[torch.Tensor] = None,
        tile_ids: Optional[Sequence[str]] = None,
        firewall_token: Optional[FitFirewallToken] = None,
        expected_split_sha256: Optional[str] = None,
        mask_mode: Optional[str] = None,
        jepa_context_mode: Optional[str] = None,
        step_index: int = 0,
    ) -> Dict[str, Any]:
        """Functional wrapper verifying `FitFirewallToken` for `ResponseTargetTeacher` and returning dict."""
        batch_dict: Dict[str, Any] = {
            "same_day_adjacent_mask": same_day_adjacent_mask,
            "receivers_xyz": receivers_xyz,
            "receiver_rc": receiver_rc,
            "receiver_obs_mask": receiver_obs_mask,
        }
        if self.use_response_teacher:
            batch_dict.update(
                {
                    "truth": truth,
                    "valid_mask": valid_mask,
                    "boundary_distance_m": boundary_distance_m,
                    "boundary_distance_valid": boundary_distance_valid,
                    "response_5x5_bitpack": response_5x5_bitpack,
                }
            )
        if self.response_teacher is not None:
            if firewall_token is not None or tile_ids is not None:
                if firewall_token is None or tile_ids is None or expected_split_sha256 is None:
                    raise RuntimeError(
                        "ResponseTargetTeacher requires firewall_token, tile_ids, and expected_split_sha256!"
                    )
                self.update_normalizers(
                    batch_dict,
                    firewall_token=firewall_token,
                    tile_ids=tile_ids,
                    expected_split_sha256=expected_split_sha256,
                    device=z_seq.device,
                )
        outputs_dict: Dict[str, Any] = {
            "latent_roll": z_seq,
            "latent_direct": dual_latents if dual_latents is not None else z_seq,
            "solar_emb": solar_emb,
            "event_gate": gate_probs,
            "logits_boundary": logits_boundary,
            "probs": probs,
        }
        total_aux, stats = self.forward_objectives(
            outputs_dict,
            batch_dict,
            step_index=step_index,
            mask_mode_override=mask_mode,
            jepa_context_mode_override=jepa_context_mode,
        )
        return {"total_aux_loss": total_aux, "stats": stats}

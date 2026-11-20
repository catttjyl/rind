"""Matched-Denominator State, Transition, Boundary-Distance & Paired Bootstrap Metrics
Across All 5 Physical Solar Regimes (`MANHATTAN_4445_ANNUAL_18808_V1`).

Enforces Astra P0/P1 measurement rigor:
- Exact integer confusion counts (`tp, fp, tn, fn`) on matched valid support (`point_frames`
  in true 120s frame units).
- Zero-support buckets emit `None` (JSON `null` / table `"NA"`), NEVER `0.0`!
- Adjacent transition events computed strictly on `common_valid` pairs WITHIN the same day AND
  exact `dt == 120s` cadence (`build_same_day_adjacent_mask`), rejecting `0s`, `119s`, `121s`,
  `240s`, negative `dt`, and cross-day / cross-tile jumps!
- Reports Sun IoU, Shadow IoU, Hard-Sun accuracy/IoU/F1, undirected & directed transition F1,
  all 5 physical solar regimes, and shadow-boundary distance bands (`<0.5m`, `0.5-2.0m`, `>2.0m`)
  strictly gated on `distance_valid == True`.
- Explicitly counts `nonfinite_point_frame_count` & `nonfinite_window_count` BEFORE clamping,
  plus `rejected_or_unavailable_slots`, `max_latent_norm`, paired spatial-cluster bootstrap CIs,
  and multi-seed `Mean ± Std` summaries.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SOLAR_REGIME_ORDER: Tuple[str, ...] = (
    "sunrise_low",
    "morning_mid",
    "solar_noon",
    "afternoon_west",
    "sunset_low",
)

DISTANCE_BAND_ORDER: Tuple[str, ...] = (
    "lt_0p5m",
    "0p5m_to_2p0m",
    "gt_2p0m",
)

BLOCKER_DISTANCE_BUCKET_ORDER: Tuple[str, ...] = (
    "lt_50m",
    "50m_to_300m",
    "300m_to_4000m",
)


def format_metric_or_na(val: Optional[float], decimals: int = 4) -> str:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return "NA"
    return f"{val:.{decimals}f}"


def classify_solar_regime(solar_altitude_deg: float, solar_azimuth_deg: float) -> str:
    """Map `(solar_altitude_deg, solar_azimuth_deg)` to one of the 5 physical solar regimes.

    Compass convention (`East-Up-North` world basis):
      `0° = North, 90° = East, 180° = South, 270° = West` (`[0, 360)`).

    Mutually exclusive & collectively exhaustive partition:
      1. `sunrise_low`: `alt <= 15.0°` and `az < 180.0°` (eastern low sun)
      2. `sunset_low`:  `alt <= 15.0°` and `az >= 180.0°` (western low sun)
      3. `morning_mid`: `alt > 15.0°`  and `az < 165.0°` (eastern mid/high sun)
      4. `solar_noon`:  `alt > 15.0°`  and `165.0° <= az <= 195.0°` (meridian window)
      5. `afternoon_west`: `alt > 15.0°` and `az > 195.0°` (western mid/high sun)
    """
    alt = float(solar_altitude_deg)
    az = float(solar_azimuth_deg) % 360.0
    if alt <= 15.0:
        return "sunrise_low" if az < 180.0 else "sunset_low"
    if 165.0 <= az <= 195.0:
        return "solar_noon"
    if az < 165.0:
        return "morning_mid"
    return "afternoon_west"


def compute_solar_angles_and_regimes(
    sun_directions_eun: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute `(solar_altitude_deg, solar_azimuth_deg, regimes)` from East-Up-North unit sun vectors."""
    s = np.asarray(sun_directions_eun, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError(f"sun_directions_eun must have shape (T, 3), got {s.shape}")
    norms = np.linalg.norm(s, axis=-1, keepdims=True)
    unit = np.where(norms > 1e-12, s / np.maximum(norms, 1e-12), 0.0)
    sx_east = unit[:, 0]
    sy_up = np.clip(unit[:, 1], -1.0, 1.0)
    sz_north = unit[:, 2]

    alt_deg = np.degrees(np.arcsin(sy_up))
    az_deg = (np.degrees(np.arctan2(sx_east, sz_north)) + 360.0) % 360.0
    regimes = [classify_solar_regime(float(a), float(z)) for a, z in zip(alt_deg, az_deg)]
    return alt_deg.astype(np.float32), az_deg.astype(np.float32), regimes


def build_same_day_adjacent_mask(
    day_ids: np.ndarray,
    timestamps_unix_s: Optional[np.ndarray] = None,
    *,
    tile_ids: Optional[Sequence[Any]] = None,
    expected_cadence_s: int = 120,
    allow_missing_timestamps: bool = False,
) -> np.ndarray:
    """Return `(T - 1,)` boolean mask `True` iff frame `t-1` and `t` belong to the same continuous
    day AND tile with exact `dt == expected_cadence_s` (`120s`).

    Fixes Astra P1-1:
    - Requires `timestamps_unix_s` unless `allow_missing_timestamps=True` is explicitly requested.
    - Enforces exact `dt == 120`, strictly rejecting `0s` duplicates, `119s` jitter, `121s`,
      `240s` frame drops, negative `dt`, cross-day jumps, and cross-tile concatenations!
    """
    d = np.asarray(day_ids)
    if d.ndim != 1:
        raise ValueError(f"day_ids must be 1D, got {d.shape}")
    if d.size <= 1:
        return np.zeros((0,), dtype=bool)
    if timestamps_unix_s is None and not allow_missing_timestamps:
        raise ValueError(
            "timestamps_unix_s is required for `build_same_day_adjacent_mask` unless "
            "`allow_missing_timestamps=True` is explicitly set."
        )
    same_day = d[1:] == d[:-1]
    if tile_ids is not None:
        t_arr = np.asarray(list(tile_ids))
        if t_arr.shape != d.shape:
            raise ValueError(f"tile_ids shape {t_arr.shape} != day_ids shape {d.shape}")
        same_day = same_day & (t_arr[1:] == t_arr[:-1])
    if timestamps_unix_s is not None:
        ts = np.asarray(timestamps_unix_s, dtype=np.int64)
        if ts.shape != d.shape:
            raise ValueError(f"timestamps_unix_s shape {ts.shape} != day_ids shape {d.shape}")
        dt = ts[1:] - ts[:-1]
        same_day = same_day & (dt == int(expected_cadence_s))
    return np.asarray(same_day, dtype=bool)


def wilson_interval_95(
    successes: int, total: int, z: float = 1.959963984540054
) -> Optional[Tuple[float, float]]:
    if total <= 0:
        return None
    n = float(total)
    p = float(successes) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def _safe_prf_optional(
    tp: int, fp: int, fn: int
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else None
    rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else None
    denom = 2 * tp + fp + fn
    f1 = (2.0 * float(tp) / float(denom)) if denom > 0 else None
    return prec, rec, f1


def compute_planar_shadow_boundary_distance_grid(
    truth_thw: np.ndarray,
    valid_thw: np.ndarray,
    receiver_xyz_hw3: Optional[np.ndarray] = None,
    *,
    grid_spacing_m: float = 0.1953125,
    max_search_radius_m: float = 2.25,
    receiver_rc: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact 2D planar distance to nearest opposite-state shadow boundary on `(T, H, W)`.

    Strictly follows `distance_valid` semantics (`P1-6`):
    - If an opposite-state valid neighbor exists at distance `d_opp` and NO invalid/out-of-bounds
      cell exists at distance `<= min(d_opp, 2.0)`, then `distance_m = d_opp` and `distance_valid = True`.
    - If NO opposite-state valid neighbor exists within `max_search_radius_m` (`> 2.0m`) AND no
      invalid/out-of-bounds cell exists within `2.0m`, then the receiver is provably `> 2.0m` from
      any boundary (`distance_m = max_search_radius_m`, `distance_valid = True`, entering `gt_2p0m`).
    - Otherwise (missing/invalid or out-of-bounds cell could hide a nearer boundary `<= 2.0m`),
      `distance_valid = False` and `distance_m = NaN` — NEVER mixed into `< 0.5m`!

    If `receiver_rc` `(N, 2)` is passed, returns gathered `(T, N)` arrays; else `(T, H, W)`.
    """
    gt = np.asarray(truth_thw, dtype=np.int8)
    vm = np.asarray(valid_thw, dtype=bool) & (gt >= 0)
    if gt.ndim != 3 or vm.shape != gt.shape:
        raise ValueError(f"Expected (T, H, W) truth/valid grids, got {gt.shape} / {vm.shape}")
    t_steps, h, w = gt.shape

    if receiver_xyz_hw3 is not None:
        xyz = np.asarray(receiver_xyz_hw3, dtype=np.float64)
        if xyz.shape != (h, w, 3):
            raise ValueError(f"receiver_xyz_hw3 shape {xyz.shape} != ({h}, {w}, 3)")
        dx_step = float(np.median(np.abs(np.diff(xyz[:, :, 0], axis=1)))) if w > 1 else grid_spacing_m
        dz_step = float(np.median(np.abs(np.diff(xyz[:, :, 2], axis=0)))) if h > 1 else grid_spacing_m
        if dx_step <= 1e-6:
            dx_step = grid_spacing_m
        if dz_step <= 1e-6:
            dz_step = grid_spacing_m
    else:
        dx_step = float(grid_spacing_m)
        dz_step = float(grid_spacing_m)

    max_r = max(1, int(math.ceil(max_search_radius_m / min(dx_step, dz_step))))
    offsets: List[Tuple[int, int, float]] = []
    for dr in range(-max_r, max_r + 1):
        for dc in range(-max_r, max_r + 1):
            if dr == 0 and dc == 0:
                continue
            dist = math.hypot(dr * dz_step, dc * dx_step)
            if dist <= max_search_radius_m + 1e-9:
                offsets.append((dr, dc, dist))
    offsets.sort(key=lambda item: item[2])

    inf_val = float("inf")
    min_opp_dist = np.full((t_steps, h, w), inf_val, dtype=np.float32)
    min_unknown_dist = np.full((t_steps, h, w), inf_val, dtype=np.float32)

    # Distance to out-of-bounds grid border
    r_idx = np.arange(h, dtype=np.float32)[:, None]
    c_idx = np.arange(w, dtype=np.float32)[None, :]
    border_dist_hw = np.minimum(
        np.minimum(r_idx + 1.0, float(h) - r_idx) * float(dz_step),
        np.minimum(c_idx + 1.0, float(w) - c_idx) * float(dx_step),
    )
    min_unknown_dist[:] = border_dist_hw[None, :, :]

    for dr, dc, dist in offsets:
        d32 = np.float32(dist)
        r_src_0 = max(0, -dr)
        r_src_1 = min(h, h - dr)
        c_src_0 = max(0, -dc)
        c_src_1 = min(w, w - dc)
        r_nb_0 = r_src_0 + dr
        r_nb_1 = r_src_1 + dr
        c_nb_0 = c_src_0 + dc
        c_nb_1 = c_src_1 + dc

        nb_valid = vm[:, r_nb_0:r_nb_1, c_nb_0:c_nb_1]
        src_valid = vm[:, r_src_0:r_src_1, c_src_0:c_src_1]
        nb_gt = gt[:, r_nb_0:r_nb_1, c_nb_0:c_nb_1]
        src_gt = gt[:, r_src_0:r_src_1, c_src_0:c_src_1]

        unk_view = min_unknown_dist[:, r_src_0:r_src_1, c_src_0:c_src_1]
        need_unk = src_valid & (~nb_valid) & (d32 < unk_view)
        if np.any(need_unk):
            unk_view[need_unk] = d32

        opp_view = min_opp_dist[:, r_src_0:r_src_1, c_src_0:c_src_1]
        is_opp = src_valid & nb_valid & (nb_gt != src_gt) & (d32 < opp_view)
        if np.any(is_opp):
            opp_view[is_opp] = d32

    # Valid iff center is valid AND no unknown/border lies closer than min(min_opp_dist, 2.0m)
    cert_radius = np.minimum(min_opp_dist, np.float32(2.0001))
    dist_valid = vm & (min_unknown_dist > cert_radius)
    resolved_dist = np.where(
        np.isfinite(min_opp_dist),
        min_opp_dist,
        np.float32(max_search_radius_m),
    )
    resolved_dist = np.where(dist_valid, resolved_dist, np.float32(np.nan))

    if receiver_rc is not None:
        rc = np.asarray(receiver_rc, dtype=np.int64)
        rows = rc[:, 0]
        cols = rc[:, 1]
        return resolved_dist[:, rows, cols], dist_valid[:, rows, cols]
    return resolved_dist, dist_valid


@dataclasses.dataclass
class ConfusionAccumulator:
    """Exact integer confusion accumulator for state, hard-sun, and same-day transition events."""

    point_frames: int = 0
    rejected_or_unavailable_slots: int = 0
    data_unknown_slots: int = 0
    model_rejected_slots: int = 0
    state_tp: int = 0
    state_fp: int = 0
    state_tn: int = 0
    state_fn: int = 0
    brier_sum: float = 0.0
    bce_sum: float = 0.0

    hard_sun_point_frames: int = 0
    hard_sun_tp: int = 0
    hard_sun_fp: int = 0
    hard_sun_tn: int = 0
    hard_sun_fn: int = 0

    transition_pairs: int = 0
    event_tp: int = 0
    event_fp: int = 0
    event_tn: int = 0
    event_fn: int = 0

    sun_to_shadow_tp: int = 0
    sun_to_shadow_fp: int = 0
    sun_to_shadow_fn: int = 0

    shadow_to_sun_tp: int = 0
    shadow_to_sun_fp: int = 0
    shadow_to_sun_fn: int = 0

    nonfinite_count: int = 0
    nonfinite_window_count: int = 0
    max_latent_norm: float = 0.0

    def merge(self, other: "ConfusionAccumulator") -> None:
        self.point_frames += other.point_frames
        self.rejected_or_unavailable_slots += other.rejected_or_unavailable_slots
        self.data_unknown_slots += other.data_unknown_slots
        self.model_rejected_slots += other.model_rejected_slots
        self.state_tp += other.state_tp
        self.state_fp += other.state_fp
        self.state_tn += other.state_tn
        self.state_fn += other.state_fn
        self.brier_sum += other.brier_sum
        self.bce_sum += other.bce_sum

        self.hard_sun_point_frames += other.hard_sun_point_frames
        self.hard_sun_tp += other.hard_sun_tp
        self.hard_sun_fp += other.hard_sun_fp
        self.hard_sun_tn += other.hard_sun_tn
        self.hard_sun_fn += other.hard_sun_fn

        self.transition_pairs += other.transition_pairs
        self.event_tp += other.event_tp
        self.event_fp += other.event_fp
        self.event_tn += other.event_tn
        self.event_fn += other.event_fn

        self.sun_to_shadow_tp += other.sun_to_shadow_tp
        self.sun_to_shadow_fp += other.sun_to_shadow_fp
        self.sun_to_shadow_fn += other.sun_to_shadow_fn

        self.shadow_to_sun_tp += other.shadow_to_sun_tp
        self.shadow_to_sun_fp += other.shadow_to_sun_fp
        self.shadow_to_sun_fn += other.shadow_to_sun_fn

        self.nonfinite_count += other.nonfinite_count
        self.nonfinite_window_count += other.nonfinite_window_count
        self.max_latent_norm = max(self.max_latent_norm, other.max_latent_norm)

    def summarize(self) -> Dict[str, Any]:
        pf = self.point_frames
        # Separate dataset-level missing/unknown slots from genuine model abstentions/rejections (`P1` audit fix)
        data_unk = int(self.data_unknown_slots)
        model_rej = int(self.model_rejected_slots)
        legacy_unavail = int(self.rejected_or_unavailable_slots)
        if data_unk == 0 and model_rej == 0 and legacy_unavail > 0:
            data_unk = legacy_unavail

        valid_dataset_slots = pf + model_rej
        total_raw_slots = valid_dataset_slots + data_unk
        model_rej_rate = (
            float(model_rej) / float(valid_dataset_slots)
            if valid_dataset_slots > 0
            else None
        )
        data_unk_rate = (
            float(data_unk) / float(total_raw_slots)
            if total_raw_slots > 0
            else None
        )
        unavail_or_rej_rate = (
            float(legacy_unavail) / float(pf + legacy_unavail)
            if (pf + legacy_unavail) > 0
            else None
        )
        nonfinite_rate = (
            float(self.nonfinite_count) / float(valid_dataset_slots)
            if valid_dataset_slots > 0
            else None
        )

        if pf > 0:
            correct = self.state_tp + self.state_tn
            acc: Optional[float] = float(correct) / float(pf)
            err: Optional[float] = 1.0 - acc
            ci_tuple = wilson_interval_95(correct, pf)
            acc_ci: Optional[List[float]] = (
                [ci_tuple[0], ci_tuple[1]] if ci_tuple is not None else None
            )
            s_prec, s_rec, s_f1 = _safe_prf_optional(
                self.state_tp, self.state_fp, self.state_fn
            )
            sun_iou_denom = self.state_tp + self.state_fp + self.state_fn
            sun_iou: Optional[float] = (
                float(self.state_tp) / float(sun_iou_denom) if sun_iou_denom > 0 else None
            )
            sh_iou_denom = self.state_tn + self.state_fp + self.state_fn
            shadow_iou: Optional[float] = (
                float(self.state_tn) / float(sh_iou_denom) if sh_iou_denom > 0 else None
            )
            brier: Optional[float] = float(self.brier_sum / pf)
            bce: Optional[float] = float(self.bce_sum / pf)
        else:
            acc = None
            err = None
            acc_ci = None
            s_prec, s_rec, s_f1 = None, None, None
            sun_iou = None
            shadow_iou = None
            brier = None
            bce = None

        if self.hard_sun_point_frames > 0:
            hs_acc: Optional[float] = float(self.hard_sun_tp + self.hard_sun_tn) / float(
                self.hard_sun_point_frames
            )
            hs_p, hs_r, hs_f1 = _safe_prf_optional(
                self.hard_sun_tp, self.hard_sun_fp, self.hard_sun_fn
            )
            hs_iou_denom = self.hard_sun_tp + self.hard_sun_fp + self.hard_sun_fn
            hs_iou: Optional[float] = (
                float(self.hard_sun_tp) / float(hs_iou_denom) if hs_iou_denom > 0 else None
            )
        else:
            hs_acc = None
            hs_p, hs_r, hs_f1 = None, None, None
            hs_iou = None

        if self.transition_pairs > 0:
            e_prec, e_rec, e_f1 = _safe_prf_optional(
                self.event_tp, self.event_fp, self.event_fn
            )
            s2sh_p, s2sh_r, s2sh_f1 = _safe_prf_optional(
                self.sun_to_shadow_tp, self.sun_to_shadow_fp, self.sun_to_shadow_fn
            )
            sh2s_p, sh2s_r, sh2s_f1 = _safe_prf_optional(
                self.shadow_to_sun_tp, self.shadow_to_sun_fp, self.shadow_to_sun_fn
            )
        else:
            e_prec, e_rec, e_f1 = None, None, None
            s2sh_p, s2sh_r, s2sh_f1 = None, None, None
            sh2s_p, sh2s_r, sh2s_f1 = None, None, None

        return {
            "status": "OK" if pf > 0 else "NA_ZERO_SUPPORT",
            "point_frames": int(pf),
            "data_unknown_slots": int(data_unk),
            "data_unknown_rate": data_unk_rate,
            "model_rejected_slots": int(model_rej),
            "model_rejection_rate": model_rej_rate,
            "rejected_or_unavailable_slots": int(legacy_unavail),
            "unavailable_or_rejected_rate": unavail_or_rej_rate,
            "rejection_rate": model_rej_rate,
            "nonfinite_slots": int(self.nonfinite_count),
            "nonfinite_rate": nonfinite_rate,
            "state_confusion": {
                "tp": int(self.state_tp),
                "fp": int(self.state_fp),
                "tn": int(self.state_tn),
                "fn": int(self.state_fn),
            },
            "state_accuracy": acc,
            "state_error": err,
            "state_accuracy_wilson95": acc_ci,
            "state_precision": s_prec,
            "state_recall": s_rec,
            "state_f1": s_f1,
            "state_sun_iou": sun_iou,
            "state_shadow_iou": shadow_iou,
            "brier_score": brier,
            "bce_nll": bce,
            "hard_sun": {
                "point_frames": int(self.hard_sun_point_frames),
                "tp": int(self.hard_sun_tp),
                "fp": int(self.hard_sun_fp),
                "tn": int(self.hard_sun_tn),
                "fn": int(self.hard_sun_fn),
                "accuracy": hs_acc,
                "precision": hs_p,
                "recall": hs_r,
                "f1": hs_f1,
                "sun_iou": hs_iou,
            },
            "transition_pairs": int(self.transition_pairs),
            "transition_confusion": {
                "tp": int(self.event_tp),
                "fp": int(self.event_fp),
                "tn": int(self.event_tn),
                "fn": int(self.event_fn),
            },
            "transition_precision": e_prec,
            "transition_recall": e_rec,
            "transition_f1": e_f1,
            "directed_transitions": {
                "sun_to_shadow": {
                    "tp": int(self.sun_to_shadow_tp),
                    "fp": int(self.sun_to_shadow_fp),
                    "fn": int(self.sun_to_shadow_fn),
                    "precision": s2sh_p,
                    "recall": s2sh_r,
                    "f1": s2sh_f1,
                },
                "shadow_to_sun": {
                    "tp": int(self.shadow_to_sun_tp),
                    "fp": int(self.shadow_to_sun_fp),
                    "fn": int(self.shadow_to_sun_fn),
                    "precision": sh2s_p,
                    "recall": sh2s_r,
                    "f1": sh2s_f1,
                },
            },
            "nonfinite_count": int(self.nonfinite_count),
            "nonfinite_point_frame_count": int(self.nonfinite_count),
            "nonfinite_window_count": int(self.nonfinite_window_count),
            "max_latent_norm": float(self.max_latent_norm),
        }


def _accumulate_state_slice(
    target: ConfusionAccumulator,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    probs_slice: np.ndarray,
    logits_slice: np.ndarray,
    raw_logits_slice: np.ndarray,
    hard_mask_slice: Optional[np.ndarray] = None,
) -> None:
    pf = int(pred_mask.size)
    if pf == 0:
        return
    tp = int(np.sum(pred_mask & gt_mask))
    fp = int(np.sum(pred_mask & (~gt_mask)))
    tn = int(np.sum((~pred_mask) & (~gt_mask)))
    fn = int(np.sum((~pred_mask) & gt_mask))
    gf = gt_mask.astype(np.float64)
    brier_s = float(np.sum((probs_slice - gf) ** 2))
    bce_s = float(
        np.sum(
            np.maximum(logits_slice, 0.0)
            - logits_slice * gf
            + np.log1p(np.exp(-np.abs(logits_slice)))
        )
    )
    nf = int(np.sum(~np.isfinite(raw_logits_slice)))

    target.point_frames += pf
    target.state_tp += tp
    target.state_fp += fp
    target.state_tn += tn
    target.state_fn += fn
    target.brier_sum += brier_s
    target.bce_sum += bce_s
    target.nonfinite_count += nf

    if hard_mask_slice is not None and np.any(hard_mask_slice):
        hp = pred_mask[hard_mask_slice]
        hg = gt_mask[hard_mask_slice]
        target.hard_sun_point_frames += int(hp.size)
        target.hard_sun_tp += int(np.sum(hp & hg))
        target.hard_sun_fp += int(np.sum(hp & (~hg)))
        target.hard_sun_tn += int(np.sum((~hp) & (~hg)))
        target.hard_sun_fn += int(np.sum((~hp) & hg))


def evaluate_trajectory_predictions(
    logits: np.ndarray,
    truth: np.ndarray,
    valid_mask: np.ndarray,
    *,
    solar_regimes: Sequence[str],
    day_ids: np.ndarray,
    timestamps_unix_s: Optional[np.ndarray] = None,
    allow_missing_timestamps: bool = False,
    has_active_candidates: Optional[np.ndarray] = None,
    model_rejected_mask: Optional[np.ndarray] = None,
    boundary_distance_m: Optional[np.ndarray] = None,
    boundary_distance_valid: Optional[np.ndarray] = None,
    return_distance_bands: bool = False,
    decision_threshold: float = 0.5,
    temperature: float = 1.0,
    max_latent_norm: float = 0.0,
    extra_nonfinite_count: int = 0,
) -> (
    Tuple[ConfusionAccumulator, Dict[str, ConfusionAccumulator]]
    | Tuple[
        ConfusionAccumulator,
        Dict[str, ConfusionAccumulator],
        Dict[str, ConfusionAccumulator],
        int,
    ]
):
    """Evaluate a `(T, N)` trajectory on matched valid support, 5 solar regimes, and optional distance bands."""
    raw_logits = np.asarray(logits, dtype=np.float64)
    gt = np.asarray(truth, dtype=np.int8)
    data_valid = np.asarray(valid_mask, dtype=bool) & (gt >= 0)

    if raw_logits.shape != gt.shape or gt.shape != data_valid.shape:
        raise ValueError(
            f"Shape mismatch: logits={raw_logits.shape}, truth={gt.shape}, valid_mask={data_valid.shape}"
        )
    if model_rejected_mask is not None:
        m_rej = np.asarray(model_rejected_mask, dtype=bool)
        if m_rej.shape != data_valid.shape:
            raise ValueError(
                f"model_rejected_mask shape {m_rej.shape} != valid_mask {data_valid.shape}"
            )
    else:
        m_rej = np.zeros_like(data_valid, dtype=bool)

    bad = data_valid & ~m_rej & ~np.isfinite(raw_logits)
    m_rej = m_rej | bad
    vmask = data_valid & (~m_rej)
    t_steps = raw_logits.shape[0]
    if len(solar_regimes) != t_steps:
        raise ValueError(f"len(solar_regimes)={len(solar_regimes)} != T={t_steps}")

    hard_mask: Optional[np.ndarray] = None
    if has_active_candidates is not None:
        hard_mask = np.asarray(has_active_candidates, dtype=bool)
        if hard_mask.shape != vmask.shape:
            raise ValueError(
                f"has_active_candidates shape {hard_mask.shape} != valid_mask {vmask.shape}"
            )

    nonfinite_valid = int(bad.sum())
    has_nonfinite_window = 1 if (nonfinite_valid + int(extra_nonfinite_count)) > 0 else 0

    safe_temp = max(float(temperature), 1e-4)
    scaled = np.where(np.isfinite(raw_logits), np.clip(raw_logits / safe_temp, -60.0, 60.0), 0.0)
    probs = 1.0 / (1.0 + np.exp(-scaled))
    pred = probs >= float(decision_threshold)
    gt_bool = gt == 1

    data_unk_total = int(np.sum(~data_valid))
    model_rej_total = int(np.sum(data_valid & m_rej))
    overall = ConfusionAccumulator(
        rejected_or_unavailable_slots=data_unk_total + model_rej_total,
        data_unknown_slots=data_unk_total,
        model_rejected_slots=model_rej_total,
        nonfinite_count=nonfinite_valid+int(extra_nonfinite_count),
        nonfinite_window_count=has_nonfinite_window,
        max_latent_norm=float(max_latent_norm),
    )
    by_regime: Dict[str, ConfusionAccumulator] = {
        r: ConfusionAccumulator() for r in SOLAR_REGIME_ORDER
    }

    reg_arr = np.asarray(list(solar_regimes))
    unknown_regimes = sorted(set(str(x) for x in reg_arr.tolist()) - set(SOLAR_REGIME_ORDER))
    if unknown_regimes:
        raise ValueError(
            f"Unknown solar_regimes {unknown_regimes}; expected subset of {SOLAR_REGIME_ORDER}"
        )
    for r_name in SOLAR_REGIME_ORDER:
        t_sel = reg_arr == r_name
        if not np.any(t_sel):
            continue
        by_regime[r_name].nonfinite_count += int(bad[t_sel].sum())
        by_regime[r_name].nonfinite_window_count = int(bad[t_sel].any())
        sub_dv = data_valid[t_sel]
        sub_mr = m_rej[t_sel]
        sub_v = vmask[t_sel]
        du_cnt = int(np.sum(~sub_dv))
        mr_cnt = int(np.sum(sub_dv & sub_mr))
        by_regime[r_name].data_unknown_slots += du_cnt
        by_regime[r_name].model_rejected_slots += mr_cnt
        by_regime[r_name].rejected_or_unavailable_slots += du_cnt + mr_cnt
        pf = int(sub_v.sum())
        if pf == 0:
            continue
        p_b = pred[t_sel][sub_v]
        g_b = gt_bool[t_sel][sub_v]
        pr_b = probs[t_sel][sub_v]
        lg_b = scaled[t_sel][sub_v]
        rl_b = raw_logits[t_sel][sub_v]
        hm_b = hard_mask[t_sel][sub_v] if hard_mask is not None else None
        if np.any(~np.isfinite(rl_b)):
            by_regime[r_name].nonfinite_window_count = 1

        _accumulate_state_slice(by_regime[r_name], p_b, g_b, pr_b, lg_b, rl_b, hm_b)
        _accumulate_state_slice(overall, p_b, g_b, pr_b, lg_b, rl_b, hm_b)

    # Adjacent transition evaluation strictly within same day AND exact 120s cadence
    if t_steps >= 2:
        same_day_adj = build_same_day_adjacent_mask(
            day_ids,
            timestamps_unix_s,
            allow_missing_timestamps=allow_missing_timestamps,
        )
        common_valid = vmask[1:] & vmask[:-1] & same_day_adj[:, None]

        gt_prev = gt_bool[:-1]
        gt_curr = gt_bool[1:]
        pr_prev = pred[:-1]
        pr_curr = pred[1:]

        gt_switch = gt_curr != gt_prev
        pr_switch = pr_curr != pr_prev

        gt_s2sh = gt_prev & (~gt_curr)
        pr_s2sh = pr_prev & (~pr_curr)

        gt_sh2s = (~gt_prev) & gt_curr
        pr_sh2s = (~pr_prev) & pr_curr

        dest_regimes = reg_arr[1:]
        for r_name in SOLAR_REGIME_ORDER:
            r_rows = dest_regimes == r_name
            if not np.any(r_rows):
                continue
            cv = common_valid[r_rows]
            npairs = int(cv.sum())
            if npairs == 0:
                continue
            gs = gt_switch[r_rows][cv]
            ps = pr_switch[r_rows][cv]
            etp = int(np.sum(ps & gs))
            efp = int(np.sum(ps & (~gs)))
            etn = int(np.sum((~ps) & (~gs)))
            efn = int(np.sum((~ps) & gs))

            g_10 = gt_s2sh[r_rows][cv]
            p_10 = pr_s2sh[r_rows][cv]
            s2sh_tp = int(np.sum(p_10 & g_10))
            s2sh_fp = int(np.sum(p_10 & (~g_10)))
            s2sh_fn = int(np.sum((~p_10) & g_10))

            g_01 = gt_sh2s[r_rows][cv]
            p_01 = pr_sh2s[r_rows][cv]
            sh2s_tp = int(np.sum(p_01 & g_01))
            sh2s_fp = int(np.sum(p_01 & (~g_01)))
            sh2s_fn = int(np.sum((~p_01) & g_01))

            for target in (by_regime[r_name], overall):
                target.transition_pairs += npairs
                target.event_tp += etp
                target.event_fp += efp
                target.event_tn += etn
                target.event_fn += efn
                target.sun_to_shadow_tp += s2sh_tp
                target.sun_to_shadow_fp += s2sh_fp
                target.sun_to_shadow_fn += s2sh_fn
                target.shadow_to_sun_tp += sh2s_tp
                target.shadow_to_sun_fp += sh2s_fp
                target.shadow_to_sun_fn += sh2s_fn

    if not return_distance_bands:
        return overall, by_regime

    by_dist: Dict[str, ConfusionAccumulator] = {
        b: ConfusionAccumulator() for b in DISTANCE_BAND_ORDER
    }
    dist_invalid_excluded = 0
    if boundary_distance_m is not None and boundary_distance_valid is not None:
        d_arr = np.asarray(boundary_distance_m, dtype=np.float64)
        dv_arr = np.asarray(boundary_distance_valid, dtype=bool)
        if d_arr.shape != vmask.shape or dv_arr.shape != vmask.shape:
            raise ValueError(
                f"boundary_distance shape {d_arr.shape}/{dv_arr.shape} != valid_mask {vmask.shape}"
            )
        valid_and_dist = vmask & dv_arr & np.isfinite(d_arr)
        dist_invalid_excluded = int(np.sum(vmask & (~valid_and_dist)))

        band_masks = {
            "lt_0p5m": valid_and_dist & (d_arr < 0.5),
            "0p5m_to_2p0m": valid_and_dist & (d_arr >= 0.5) & (d_arr <= 2.0),
            "gt_2p0m": valid_and_dist & (d_arr > 2.0),
        }
        for b_name, b_sel in band_masks.items():
            if not np.any(b_sel):
                continue
            hm_b = hard_mask[b_sel] if hard_mask is not None else None
            _accumulate_state_slice(
                by_dist[b_name],
                pred[b_sel],
                gt_bool[b_sel],
                probs[b_sel],
                scaled[b_sel],
                raw_logits[b_sel],
                hm_b,
            )

    return overall, by_regime, by_dist, dist_invalid_excluded


def _cluster_sum_arrays(
    accumulators: Sequence[ConfusionAccumulator],
    cluster_ids: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    if cluster_ids is None:
        groups = [[i] for i in range(len(accumulators))]
    else:
        by_c: Dict[str, List[int]] = {}
        for idx, cid in enumerate(cluster_ids):
            by_c.setdefault(str(cid), []).append(idx)
        groups = [by_c[k] for k in sorted(by_c.keys())]

    k = len(groups)
    pf = np.zeros((k,), dtype=np.float64)
    corr = np.zeros((k,), dtype=np.float64)
    tp = np.zeros((k,), dtype=np.float64)
    fp = np.zeros((k,), dtype=np.float64)
    tn = np.zeros((k,), dtype=np.float64)
    fn = np.zeros((k,), dtype=np.float64)
    etp = np.zeros((k,), dtype=np.float64)
    efp = np.zeros((k,), dtype=np.float64)
    efn = np.zeros((k,), dtype=np.float64)

    for g_idx, indices in enumerate(groups):
        for i in indices:
            a = accumulators[i]
            pf[g_idx] += a.point_frames
            corr[g_idx] += a.state_tp + a.state_tn
            tp[g_idx] += a.state_tp
            fp[g_idx] += a.state_fp
            tn[g_idx] += a.state_tn
            fn[g_idx] += a.state_fn
            etp[g_idx] += a.event_tp
            efp[g_idx] += a.event_fp
            efn[g_idx] += a.event_fn

    return {
        "pf": pf,
        "corr": corr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "etp": etp,
        "efp": efp,
        "efn": efn,
    }


def bootstrap_tile_ci_95(
    per_tile_accumulators: Sequence[ConfusionAccumulator],
    *,
    spatial_group_ids: Optional[Sequence[str]] = None,
    num_resamples: int = 1000,
    seed: int = 20261111,
) -> Dict[str, Optional[List[float]] | int | str]:
    """Compute spatial-group (or tile) cluster bootstrap 95% CIs; returns `None` on zero support."""
    if not per_tile_accumulators or sum(a.point_frames for a in per_tile_accumulators) == 0:
        return {
            "cluster_unit": "none",
            "cluster_count": 0,
            "state_accuracy_ci95": None,
            "state_error_ci95": None,
            "state_sun_iou_ci95": None,
            "state_shadow_iou_ci95": None,
            "transition_f1_ci95": None,
        }

    use_groups = (
        spatial_group_ids is not None and len(set(str(g) for g in spatial_group_ids)) >= 2
    )
    arrs = _cluster_sum_arrays(
        per_tile_accumulators,
        cluster_ids=spatial_group_ids if use_groups else None,
    )
    k = int(arrs["pf"].size)
    unit_name = "spatial_group" if use_groups else "tile"

    if k == 1:
        merged = ConfusionAccumulator()
        for a in per_tile_accumulators:
            merged.merge(a)
        s = merged.summarize()
        return {
            "cluster_unit": unit_name,
            "cluster_count": 1,
            "state_accuracy_ci95": [s["state_accuracy"], s["state_accuracy"]]
            if s["state_accuracy"] is not None
            else None,
            "state_error_ci95": [s["state_error"], s["state_error"]]
            if s["state_error"] is not None
            else None,
            "state_sun_iou_ci95": [s["state_sun_iou"], s["state_sun_iou"]]
            if s["state_sun_iou"] is not None
            else None,
            "state_shadow_iou_ci95": [s["state_shadow_iou"], s["state_shadow_iou"]]
            if s["state_shadow_iou"] is not None
            else None,
            "transition_f1_ci95": [s["transition_f1"], s["transition_f1"]]
            if s["transition_f1"] is not None
            else None,
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(int(num_resamples), k))
    b_pf = arrs["pf"][idx].sum(axis=1)
    b_corr = arrs["corr"][idx].sum(axis=1)
    b_tp = arrs["tp"][idx].sum(axis=1)
    b_fp = arrs["fp"][idx].sum(axis=1)
    b_tn = arrs["tn"][idx].sum(axis=1)
    b_fn = arrs["fn"][idx].sum(axis=1)
    b_etp = arrs["etp"][idx].sum(axis=1)
    b_efp = arrs["efp"][idx].sum(axis=1)
    b_efn = arrs["efn"][idx].sum(axis=1)

    valid_pf = b_pf > 0
    acc_ci = (
        [
            float(np.percentile((b_corr[valid_pf] / b_pf[valid_pf]), 2.5)),
            float(np.percentile((b_corr[valid_pf] / b_pf[valid_pf]), 97.5)),
        ]
        if np.any(valid_pf)
        else None
    )
    err_ci = (
        [
            float(np.percentile(1.0 - (b_corr[valid_pf] / b_pf[valid_pf]), 2.5)),
            float(np.percentile(1.0 - (b_corr[valid_pf] / b_pf[valid_pf]), 97.5)),
        ]
        if np.any(valid_pf)
        else None
    )
    sun_d = b_tp + b_fp + b_fn
    sun_ci = (
        [
            float(np.percentile(b_tp[sun_d > 0] / sun_d[sun_d > 0], 2.5)),
            float(np.percentile(b_tp[sun_d > 0] / sun_d[sun_d > 0], 97.5)),
        ]
        if np.any(sun_d > 0)
        else None
    )
    sh_d = b_tn + b_fp + b_fn
    sh_ci = (
        [
            float(np.percentile(b_tn[sh_d > 0] / sh_d[sh_d > 0], 2.5)),
            float(np.percentile(b_tn[sh_d > 0] / sh_d[sh_d > 0], 97.5)),
        ]
        if np.any(sh_d > 0)
        else None
    )
    ev_d = 2.0 * b_etp + b_efp + b_efn
    ev_ci = (
        [
            float(np.percentile((2.0 * b_etp[ev_d > 0]) / ev_d[ev_d > 0], 2.5)),
            float(np.percentile((2.0 * b_etp[ev_d > 0]) / ev_d[ev_d > 0], 97.5)),
        ]
        if np.any(ev_d > 0)
        else None
    )

    return {
        "cluster_unit": unit_name,
        "cluster_count": k,
        "state_accuracy_ci95": acc_ci,
        "state_error_ci95": err_ci,
        "state_sun_iou_ci95": sun_ci,
        "state_shadow_iou_ci95": sh_ci,
        "transition_f1_ci95": ev_ci,
    }


def paired_bootstrap_model_comparison(
    model_a_by_tile: Mapping[str, ConfusionAccumulator],
    model_b_by_tile: Mapping[str, ConfusionAccumulator],
    *,
    tile_to_spatial_group: Optional[Mapping[str, str]] = None,
    num_resamples: int = 1000,
    seed: int = 20261111,
) -> Dict[str, Any]:
    """Compute paired `model_a - model_b` differences on identical `common_valid` tile/spatial clusters."""
    keys_a = sorted(model_a_by_tile.keys())
    keys_b = sorted(model_b_by_tile.keys())
    if keys_a != keys_b:
        raise ValueError(
            f"Paired comparison requires identical tile keys; diff={sorted(set(keys_a) ^ set(keys_b))}"
        )
    if not keys_a:
        return {"status": "NA_ZERO_SUPPORT", "paired_tile_count": 0, "deltas": None}

    for tid in keys_a:
        aa = model_a_by_tile[tid]
        bb = model_b_by_tile[tid]
        if aa.point_frames != bb.point_frames or aa.transition_pairs != bb.transition_pairs:
            raise ValueError(
                f"Unmatched denominator on tile {tid}: "
                f"model_a=(pf={aa.point_frames}, pairs={aa.transition_pairs}) vs "
                f"model_b=(pf={bb.point_frames}, pairs={bb.transition_pairs})"
            )

    seq_a = [model_a_by_tile[k] for k in keys_a]
    seq_b = [model_b_by_tile[k] for k in keys_a]
    sg_seq = [tile_to_spatial_group[k] for k in keys_a] if tile_to_spatial_group else None
    use_groups = sg_seq is not None and len(set(sg_seq)) >= 2

    arr_a = _cluster_sum_arrays(seq_a, cluster_ids=sg_seq if use_groups else None)
    arr_b = _cluster_sum_arrays(seq_b, cluster_ids=sg_seq if use_groups else None)
    k = int(arr_a["pf"].size)

    tot_a = ConfusionAccumulator()
    tot_b = ConfusionAccumulator()
    for a, b in zip(seq_a, seq_b):
        tot_a.merge(a)
        tot_b.merge(b)
    sum_a = tot_a.summarize()
    sum_b = tot_b.summarize()

    def _diff_opt(va: Optional[float], vb: Optional[float]) -> Optional[float]:
        if va is None or vb is None:
            return None
        return float(va - vb)

    point_deltas = {
        "delta_state_accuracy": _diff_opt(sum_a["state_accuracy"], sum_b["state_accuracy"]),
        "delta_state_error": _diff_opt(sum_a["state_error"], sum_b["state_error"]),
        "delta_state_sun_iou": _diff_opt(sum_a["state_sun_iou"], sum_b["state_sun_iou"]),
        "delta_state_shadow_iou": _diff_opt(sum_a["state_shadow_iou"], sum_b["state_shadow_iou"]),
        "delta_transition_f1": _diff_opt(sum_a["transition_f1"], sum_b["transition_f1"]),
    }

    if k <= 1 or tot_a.point_frames == 0:
        return {
            "status": "OK" if tot_a.point_frames > 0 else "NA_ZERO_SUPPORT",
            "paired_tile_count": len(keys_a),
            "cluster_unit": "spatial_group" if use_groups else "tile",
            "cluster_count": k,
            "matched_point_frames": int(tot_a.point_frames),
            "matched_transition_pairs": int(tot_a.transition_pairs),
            "deltas": point_deltas,
            "delta_ci95": {key: None for key in point_deltas},
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(int(num_resamples), k))

    pf = arr_a["pf"][idx].sum(axis=1)
    acc_a = np.where(pf > 0, arr_a["corr"][idx].sum(axis=1) / np.maximum(pf, 1.0), np.nan)
    acc_b = np.where(pf > 0, arr_b["corr"][idx].sum(axis=1) / np.maximum(pf, 1.0), np.nan)
    d_acc = acc_a - acc_b
    d_err = -d_acc

    sd_a = (arr_a["tp"][idx] + arr_a["fp"][idx] + arr_a["fn"][idx]).sum(axis=1)
    sd_b = (arr_b["tp"][idx] + arr_b["fp"][idx] + arr_b["fn"][idx]).sum(axis=1)
    siou_a = np.where(sd_a > 0, arr_a["tp"][idx].sum(axis=1) / np.maximum(sd_a, 1.0), np.nan)
    siou_b = np.where(sd_b > 0, arr_b["tp"][idx].sum(axis=1) / np.maximum(sd_b, 1.0), np.nan)
    d_siou = siou_a - siou_b

    shd_a = (arr_a["tn"][idx] + arr_a["fp"][idx] + arr_a["fn"][idx]).sum(axis=1)
    shd_b = (arr_b["tn"][idx] + arr_b["fp"][idx] + arr_b["fn"][idx]).sum(axis=1)
    shiou_a = np.where(shd_a > 0, arr_a["tn"][idx].sum(axis=1) / np.maximum(shd_a, 1.0), np.nan)
    shiou_b = np.where(shd_b > 0, arr_b["tn"][idx].sum(axis=1) / np.maximum(shd_b, 1.0), np.nan)
    d_shiou = shiou_a - shiou_b

    ed_a = (2.0 * arr_a["etp"][idx] + arr_a["efp"][idx] + arr_a["efn"][idx]).sum(axis=1)
    ed_b = (2.0 * arr_b["etp"][idx] + arr_b["efp"][idx] + arr_b["efn"][idx]).sum(axis=1)
    ef1_a = np.where(ed_a > 0, 2.0 * arr_a["etp"][idx].sum(axis=1) / np.maximum(ed_a, 1.0), np.nan)
    ef1_b = np.where(ed_b > 0, 2.0 * arr_b["etp"][idx].sum(axis=1) / np.maximum(ed_b, 1.0), np.nan)
    d_ef1 = ef1_a - ef1_b

    def _ci(vec: np.ndarray) -> Optional[List[float]]:
        fin = vec[np.isfinite(vec)]
        if fin.size == 0:
            return None
        return [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))]

    return {
        "status": "OK",
        "paired_tile_count": len(keys_a),
        "cluster_unit": "spatial_group" if use_groups else "tile",
        "cluster_count": k,
        "matched_point_frames": int(tot_a.point_frames),
        "matched_transition_pairs": int(tot_a.transition_pairs),
        "deltas": point_deltas,
        "delta_ci95": {
            "delta_state_accuracy": _ci(d_acc),
            "delta_state_error": _ci(d_err),
            "delta_state_sun_iou": _ci(d_siou),
            "delta_state_shadow_iou": _ci(d_shiou),
            "delta_transition_f1": _ci(d_ef1),
        },
    }


def aggregate_multiseed_reports(
    seed_reports: Sequence[Mapping[str, Any]],
    metric_keys: Sequence[str] = (
        "state_accuracy",
        "state_error",
        "state_sun_iou",
        "state_shadow_iou",
        "transition_f1",
        "brier_score",
        "bce_nll",
    ),
) -> Dict[str, Any]:
    """Aggregate multiple random-seed evaluation summaries into `Mean ± Std` (`NA` on zero support)."""
    n_seeds = len(seed_reports)
    if n_seeds == 0:
        return {"seed_count": 0, "metrics": {}}

    out_metrics: Dict[str, Any] = {}
    for key in metric_keys:
        vals: List[float] = []
        for rep in seed_reports:
            overall = rep.get("overall", rep)
            v = overall.get(key)
            if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)):
                vals.append(float(v))
        if not vals:
            out_metrics[key] = {
                "status": "NA",
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "formatted": "NA",
            }
        else:
            arr = np.asarray(vals, dtype=np.float64)
            mean_v = float(np.mean(arr))
            std_v = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
            out_metrics[key] = {
                "status": "OK",
                "n": int(arr.size),
                "mean": mean_v,
                "std": std_v,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "formatted": f"{mean_v:.4f} ± {std_v:.4f}",
            }
    return {
        "seed_count": n_seeds,
        "metrics": out_metrics,
    }


def build_receiver_sampling_disclosure(
    *,
    receiver_mode: str,
    eligible_receiver_count: int,
    evaluated_receiver_count: int,
    receiver_rc: np.ndarray,
) -> Dict[str, Any]:
    """Build explicit sampling-design disclosure metadata (`P1-6` / `P1-8`)."""
    rc_bytes = np.ascontiguousarray(receiver_rc, dtype=np.int32).tobytes()
    rc_sha = hashlib.sha256(rc_bytes).hexdigest()
    elig = int(eligible_receiver_count)
    eval_n = int(evaluated_receiver_count)
    incl_prob = float(eval_n) / float(elig) if elig > 0 else None
    is_full = receiver_mode == "all_valid" or (elig > 0 and eval_n == elig)
    return {
        "receiver_mode": str(receiver_mode),
        "estimand_scope": (
            "full_tile_all_eligible_receivers"
            if is_full
            else "stratified_spatial_receiver_sample_equal_tile_inclusion"
        ),
        "eligible_receiver_count": elig,
        "evaluated_receiver_count": eval_n,
        "sampling_inclusion_probability": incl_prob,
        "horvitz_thompson_tile_weight": (1.0 / incl_prob) if (incl_prob and incl_prob > 0) else None,
        "receiver_support_sha256": rc_sha,
    }


def _merge_blocker_reports(left, right):
    """Combine disjoint receiver sets. Transition pairs never cross receiver partitions."""
    def mean(a, b, key, weight):
        total = a[weight] + b[weight]
        return ((a[key] or 0.)*a[weight] + (b[key] or 0.)*b[weight])/total if total else None

    def merge_bucket(a, b):
        out = dict(a)
        for key, weight in (("state_accuracy", "point_frames"), ("brier_score", "point_frames"),
                            ("shadow_recall", "shadow_support"), ("false_sunlit_rate", "shadow_support")):
            out[key] = mean(a, b, key, weight)
        for key in ("point_frames", "shadow_support", "false_sunlit_count", "transition_pairs",
                    "attention_ray_support"):
            out[key] = a[key] + b[key]
        counts = {k: a["transition_confusion"][k]+b["transition_confusion"][k] for k in ("tp","fp","fn")}
        out["transition_confusion"] = counts
        out["transition_f1"] = _safe_prf_optional(**counts)[2]
        out["status"] = "OK" if out["point_frames"] else "NA_ZERO_SUPPORT"
        if any(q["candidate_recalled_ray_count"] is not None for q in (a,b)):
            out["candidate_recalled_ray_count"] = sum(q["candidate_recalled_ray_count"] or 0 for q in (a,b))
            out["candidate_first_blocker_recall"] = out["candidate_recalled_ray_count"]/out["point_frames"]
        cond = "conditional_on_candidate_recalled"
        out[cond] = {"attention_ray_support": a[cond]["attention_ray_support"]+b[cond]["attention_ray_support"]}
        for key in ("top1_attention_hit_rate","top2_attention_hit_rate","mean_true_blocker_attention_mass"):
            out[key] = mean(a, b, key, "attention_ray_support")
            out[cond][key] = mean(a[cond], b[cond], key, "attention_ray_support")
        return out

    result = dict(left)
    for group in ("buckets", "fine_buckets"):
        result[group] = {k: merge_bucket(v, right[group][k]) for k,v in left[group].items()}
    result["far_field_total"] = merge_bucket(left["far_field_total"], right["far_field_total"])
    a, b = left["unoccluded_open_sky"], right["unoccluded_open_sky"]
    result["unoccluded_open_sky"] = {"point_frames": a["point_frames"]+b["point_frames"],
                                     "state_accuracy": mean(a,b,"state_accuracy","point_frames")}
    return result


def compute_blocker_distance_stratified_metrics(
    logits: np.ndarray,
    truth: np.ndarray,
    valid_mask: np.ndarray,
    *,
    receivers_xyz: np.ndarray,
    sun_directions_xyz: np.ndarray,
    scene_triangles_xyz: np.ndarray,
    scene_opaque_mask: Optional[np.ndarray] = None,
    scene_source_triangle_ids: Optional[np.ndarray] = None,
    exclude_source_triangle_ids: Optional[np.ndarray] = None,
    day_ids: Optional[np.ndarray] = None,
    timestamps_unix_s: Optional[np.ndarray] = None,
    allow_missing_timestamps: bool = True,
    candidate_source_triangle_ids: Optional[np.ndarray] = None,
    candidate_mask: Optional[np.ndarray] = None,
    candidate_attn_weights: Optional[np.ndarray] = None,
    candidate_layout: str = "auto",
    max_ray_slots: int = 131072,
    _scene_bvh=None,
    decision_threshold: float = 0.5,
    temperature: float = 1.0,
    t_max_m: float = 4000.0,
) -> Dict[str, Any]:
    """Compute true 3D ray first-blocker distance stratification (`<50m / 50-300m / 300m-4km`, `P1`).

    Uses independent `ExactSceneBVH.query_first_hit` to determine the exact physical distance
    `d_hit in (1e-4, 4000]` and `source_triangle_id` of the nearest occluder for every `(t, r)` ray,
    and reports per-bucket state accuracy, shadow recall, transition `Event F1`, candidate recall,
    and `SphericalTangentCrossAttention` Top-1 / Top-2 blocker attention hit rate!
    """
    from .inverse_design import DEFAULT_T_MIN_M, ExactSceneBVH

    t_count, n_count = np.shape(logits)
    bvh = _scene_bvh if _scene_bvh is not None else ExactSceneBVH(
        scene_triangles_xyz, opaque_mask=scene_opaque_mask, source_triangle_ids=scene_source_triangle_ids)
    k_count = max(1, np.shape(candidate_source_triangle_ids)[-1]) if candidate_source_triangle_ids is not None else 1
    width = max(1, max_ray_slots // max(1, t_count*k_count))
    if n_count > width:
        layout = candidate_layout
        if layout == "auto" and candidate_source_triangle_ids is not None:
            layout = "time" if np.shape(candidate_source_triangle_ids)[:2] == (t_count,n_count) else "azimuth_bank"
        result = None
        for start in range(0,n_count,width):
            rec = slice(start, start+width)
            def candidates(array):
                if array is None:
                    return None
                return array[rec] if layout == "azimuth_bank" else array[:,rec]
            attn = candidate_attn_weights
            if attn is not None and np.ndim(attn) == 4:
                attn = attn[0]
            excluded = exclude_source_triangle_ids
            part = compute_blocker_distance_stratified_metrics(
                logits[:,rec], truth[:,rec], valid_mask[:,rec], receivers_xyz=receivers_xyz[rec],
                sun_directions_xyz=sun_directions_xyz, scene_triangles_xyz=scene_triangles_xyz,
                exclude_source_triangle_ids=excluded[rec] if excluded is not None and np.ndim(excluded) else excluded,
                day_ids=day_ids, timestamps_unix_s=timestamps_unix_s, allow_missing_timestamps=allow_missing_timestamps,
                candidate_source_triangle_ids=candidates(candidate_source_triangle_ids),candidate_mask=candidates(candidate_mask),
                candidate_attn_weights=attn[:,rec] if attn is not None else None,candidate_layout=layout,
                max_ray_slots=max_ray_slots, _scene_bvh=bvh, decision_threshold=decision_threshold,
                temperature=temperature,t_max_m=t_max_m)
            result = part if result is None else _merge_blocker_reports(result, part)
        return result

    raw_logits = np.asarray(logits, dtype=np.float64)
    gt = np.asarray(truth, dtype=np.int8)
    vmask = np.asarray(valid_mask, dtype=bool) & (gt >= 0)
    if np.any(vmask & ~np.isfinite(raw_logits)):
        raise ValueError("nonfinite blocker prediction on valid support")
    rec_np = np.asarray(receivers_xyz, dtype=np.float64)
    sun_np = np.asarray(sun_directions_xyz, dtype=np.float64)

    t_steps, n_pts = raw_logits.shape
    safe_temp = max(float(temperature), 1e-4)
    scaled = np.where(np.isfinite(raw_logits), np.clip(raw_logits / safe_temp, -60.0, 60.0), 0.0)
    probs = 1.0 / (1.0 + np.exp(-scaled))
    pred = probs >= float(decision_threshold)
    gt_bool = gt == 1

    s_norm = np.linalg.norm(sun_np, axis=-1)
    above_h = (s_norm > 1e-12) & ((sun_np[:, 1] / np.maximum(s_norm, 1e-12)) > 0.0)

    first_t_tn = np.full((t_steps, n_pts), np.inf, dtype=np.float64)
    first_sid_tn = np.zeros((t_steps, n_pts), dtype=np.uint64)
    occluded_tn = np.zeros((t_steps, n_pts), dtype=bool)

    valid_t_idx = np.flatnonzero(above_h)
    if valid_t_idx.size > 0 and n_pts > 0 and bvh.num_triangles > 0:
        excl_rec = (
            np.asarray(exclude_source_triangle_ids, dtype=np.uint64)
            if exclude_source_triangle_ids is not None
            else None
        )
        if excl_rec is not None and excl_rec.ndim == 0:
            excl_rec = np.broadcast_to(excl_rec[None], (n_pts,))
        frames_per_pass = max(1, 8192 // max(n_pts, 1))
        for s0 in range(0, valid_t_idx.size, frames_per_pass):
            tb = valid_t_idx[s0 : s0 + frames_per_pass]
            blen = tb.size
            o_rep = np.broadcast_to(rec_np[None, :, :], (blen, n_pts, 3)).reshape(blen * n_pts, 3)
            d_rep = np.broadcast_to(sun_np[tb, None, :], (blen, n_pts, 3)).reshape(blen * n_pts, 3)
            ex_rep = (
                np.broadcast_to(excl_rec[None, :], (blen, n_pts)).reshape(blen * n_pts)
                if excl_rec is not None
                else None
            )
            occ_flat, ft_flat, fsid_flat, _ = bvh.query_first_hit(
                o_rep,
                d_rep,
                t_min=DEFAULT_T_MIN_M,
                t_max=float(t_max_m),
                exclude_source_triangle_ids=ex_rep,
            )
            occluded_tn[tb] = occ_flat.reshape(blen, n_pts)
            first_t_tn[tb] = ft_flat.reshape(blen, n_pts)
            first_sid_tn[tb] = fsid_flat.reshape(blen, n_pts)

    # Resolve optional `(T, N, K)` candidate source triangle IDs
    cand_sids_tnk: Optional[np.ndarray] = None
    cand_mk_tnk: Optional[np.ndarray] = None
    if candidate_source_triangle_ids is not None:
        cs = np.asarray(candidate_source_triangle_ids, dtype=np.uint64)
        if candidate_layout not in {"auto", "time", "azimuth_bank"}:
            raise ValueError("unknown candidate_layout")
        if candidate_layout != "azimuth_bank" and cs.ndim == 3 and cs.shape[:2] == (t_steps, n_pts):
            cand_sids_tnk = cs
            cand_mk_tnk = (
                np.asarray(candidate_mask, dtype=bool)
                if candidate_mask is not None
                else np.ones_like(cs, dtype=bool)
            )
        elif candidate_layout != "time" and cs.ndim == 3 and cs.shape[0] == n_pts:
            from .fullday_data import sun_dirs_to_azimuth_bins

            bins = sun_dirs_to_azimuth_bins(sun_np, num_azimuth_bins=int(cs.shape[1]))
            cand_sids_tnk = np.transpose(cs, (1, 0, 2))[bins]
            if candidate_mask is not None:
                cm = np.asarray(candidate_mask, dtype=bool)
                cand_mk_tnk = (
                    np.transpose(cm, (1, 0, 2))[bins]
                    if cm.shape == cs.shape
                    else cm
                )
            else:
                cand_mk_tnk = np.ones_like(cand_sids_tnk, dtype=bool)

    attn_tnk: Optional[np.ndarray] = None
    if candidate_attn_weights is not None:
        aw = np.asarray(candidate_attn_weights)
        if aw.ndim == 4 and aw.shape[0] == 1:
            aw = aw[0]
        if aw.ndim == 3 and aw.shape[:2] == (t_steps, n_pts):
            attn_tnk = aw

    same_day_adj: Optional[np.ndarray] = None
    if t_steps >= 2:
        if day_ids is not None:
            same_day_adj = build_same_day_adjacent_mask(
                day_ids, timestamps_unix_s, allow_missing_timestamps=allow_missing_timestamps
            )
        else:
            same_day_adj = np.ones((t_steps - 1,), dtype=bool)

    bucket_defs = {
        "lt_50m": (0.0, 50.0),
        "50m_to_300m": (50.0, 300.0),
        "300m_to_4000m": (300.0, float(t_max_m) + 1e-6),
    }

    fine_edges = (0, 25, 100, 400, 800, 1600, 4000)
    fine_defs = {f"{lo}m_to_{hi}m": (lo, hi) for lo, hi in zip(fine_edges, fine_edges[1:])}
    far_total = {"400m_to_4000m": (400, 4000)}
    buckets_out: Dict[str, Any] = {}
    for b_name, (d_lo, d_hi) in {**bucket_defs, **fine_defs, **far_total}.items():
        def in_bucket(distance: np.ndarray) -> np.ndarray:
            if b_name in fine_defs or b_name in far_total:
                upper = distance <= d_hi if d_hi == fine_edges[-1] else distance < d_hi
                return (distance >= d_lo) & upper
            if b_name == "lt_50m":
                return (distance > 0) & (distance < d_hi)
            lower = distance >= d_lo if b_name == "50m_to_300m" else distance > d_lo
            return lower & (distance <= d_hi)

        b_ray_mask = vmask & occluded_tn & in_bucket(first_t_tn)
        pf = int(np.sum(b_ray_mask))
        if pf > 0:
            p_sub = pred[b_ray_mask]
            g_sub = gt_bool[b_ray_mask]
            pr_sub = probs[b_ray_mask]
            acc_b: Optional[float] = float(np.mean(p_sub == g_sub))
            sh_rec: Optional[float] = (
                float(np.mean(~p_sub[~g_sub])) if np.any(~g_sub) else None
            )
            brier_b: Optional[float] = float(np.mean((pr_sub - g_sub.astype(np.float64)) ** 2))
        else:
            acc_b = None
            sh_rec = None
            brier_b = None

        # Transition pairs whose nearest occluding endpoint falls into this distance bucket
        trans_pairs_b = 0
        etp, efp, efn = 0, 0, 0
        trans_f1_b: Optional[float] = None
        if same_day_adj is not None and t_steps >= 2:
            common_v = vmask[1:] & vmask[:-1] & same_day_adj[:, None]
            pair_min_d = np.minimum(first_t_tn[1:], first_t_tn[:-1])
            pair_has_occ = occluded_tn[1:] | occluded_tn[:-1]
            p_mask = common_v & pair_has_occ & in_bucket(pair_min_d)
            trans_pairs_b = int(np.sum(p_mask))
            if trans_pairs_b > 0:
                gs = (gt_bool[1:] != gt_bool[:-1])[p_mask]
                ps = (pred[1:] != pred[:-1])[p_mask]
                etp = int(np.sum(ps & gs))
                efp = int(np.sum(ps & (~gs)))
                efn = int(np.sum((~ps) & gs))
                _, _, trans_f1_b = _safe_prf_optional(etp, efp, efn)

        # Candidate recall & Attention Top-1/Top-2 hit rate on true first blocker
        recalled_count = 0
        cand_recall: Optional[float] = None
        top1_hit: Optional[float] = None
        top2_hit: Optional[float] = None
        mean_attn_mass: Optional[float] = None
        if pf > 0 and cand_sids_tnk is not None and cand_mk_tnk is not None:
            true_sid_sub = first_sid_tn[b_ray_mask][:, None]  # (P, 1)
            c_sid_sub = cand_sids_tnk[b_ray_mask]  # (P, K)
            c_mk_sub = cand_mk_tnk[b_ray_mask]  # (P, K)
            match_pk = c_mk_sub & (c_sid_sub == true_sid_sub)
            recalled_p = np.any(match_pk, axis=1)
            recalled_count = int(recalled_p.sum())
            cand_recall = recalled_count / pf
            if attn_tnk is not None and np.any(recalled_p):
                aw_sub = attn_tnk[b_ray_mask][recalled_p]  # (P_rec, K)
                m_sub = match_pk[recalled_p]
                c_mk_rec = c_mk_sub[recalled_p]
                masked_aw = np.where(c_mk_rec, aw_sub, -1.0)
                order_desc = np.argsort(-masked_aw, axis=1)
                top1_idx = order_desc[:, 0]
                top1_hit = float(np.mean(m_sub[np.arange(m_sub.shape[0]), top1_idx]))
                if aw_sub.shape[1] >= 2:
                    top2_idx = order_desc[:, :2]
                    top2_hit = float(
                        np.mean(
                            m_sub[np.arange(m_sub.shape[0]), top2_idx[:, 0]]
                            | m_sub[np.arange(m_sub.shape[0]), top2_idx[:, 1]]
                        )
                    )
                else:
                    top2_hit = top1_hit
                mean_attn_mass = float(np.mean(np.sum(aw_sub * m_sub, axis=1)))

        attention_support = pf if attn_tnk is not None and cand_recall is not None else 0
        def all_ray_mean(conditional):
            if not attention_support:
                return None
            return (conditional or 0.) * recalled_count / attention_support

        buckets_out[b_name] = {
            "status": "OK" if pf > 0 else "NA_ZERO_SUPPORT",
            "distance_range_m": [float(d_lo), float(min(d_hi, float(t_max_m)))],
            "point_frames": pf,
            "state_accuracy": acc_b,
            "shadow_recall": sh_rec,
            "shadow_support": int(np.sum(b_ray_mask & ~gt_bool)),
            "false_sunlit_count": int(np.sum(b_ray_mask & ~gt_bool & pred)),
            "false_sunlit_rate": None if sh_rec is None else 1.0 - sh_rec,
            "brier_score": brier_b,
            "transition_pairs": trans_pairs_b,
            "transition_confusion": {"tp": etp, "fp": efp, "fn": efn},
            "transition_f1": trans_f1_b,
            "candidate_first_blocker_recall": cand_recall,
            "attention_ray_support": attention_support,
            "candidate_recalled_ray_count": recalled_count if cand_recall is not None else None,
            "top1_attention_hit_rate": all_ray_mean(top1_hit),
            "top2_attention_hit_rate": all_ray_mean(top2_hit),
            "mean_true_blocker_attention_mass": all_ray_mean(mean_attn_mass),
            "conditional_on_candidate_recalled": {
                "attention_ray_support": recalled_count if attention_support else 0,
                "top1_attention_hit_rate": top1_hit,
                "top2_attention_hit_rate": top2_hit,
                "mean_true_blocker_attention_mass": mean_attn_mass,
            },
        }

    open_sky_mask = vmask & above_h[:, None] & (~occluded_tn)
    open_pf = int(np.sum(open_sky_mask))
    open_acc = float(np.mean(pred[open_sky_mask] == gt_bool[open_sky_mask])) if open_pf > 0 else None

    return {
        "attention_metric_schema": 2,
        "horizon_m": float(t_max_m),
        "buckets": {name: buckets_out[name] for name in bucket_defs},
        "far_field_total": buckets_out["400m_to_4000m"],
        "fine_buckets": {name: buckets_out[name] for name in fine_defs},
        "unoccluded_open_sky": {
            "point_frames": open_pf,
            "state_accuracy": open_acc,
        },
    }


def compute_shadow_contour_f1(
    logits, truth, valid_mask, receiver_rc, world_x_m, world_z_m, *,
    temperature=1., decision_threshold=.5,
):
    """Match shade-side contours of observed four-neighbor edges on a rectilinear raster."""
    rc = np.asarray(receiver_rc, dtype=np.int64)
    x = np.asarray(world_x_m, dtype=np.float64)
    z = np.asarray(world_z_m, dtype=np.float64)
    h, w = len(z), len(x)
    raw = np.asarray(logits)
    labels = np.asarray(truth)
    valid = np.asarray(valid_mask)
    if raw.shape != labels.shape or raw.shape != valid.shape or rc.shape != (raw.shape[1], 2):
        raise ValueError("contour arrays must describe the same receiver sequence")
    if np.any(valid.astype(bool) & (labels >= 0) & ~np.isfinite(raw)):
        raise ValueError("nonfinite contour prediction on valid support")
    r, c = rc.T
    if np.any((r < 0) | (r >= h) | (c < 0) | (c >= w)) or len(np.unique(r*w+c)) != len(r):
        raise ValueError("receiver coordinates must be unique and within the raster")
    for axis in (x, z):
        if len(axis) < 2 or not np.isfinite(axis).all() or not (np.all(np.diff(axis)>0) or np.all(np.diff(axis)<0)):
            raise ValueError("contour calculation requires finite monotone raster axes")
    dx, dz = np.abs(np.diff(x)), np.abs(np.diff(z))
    radii = {"one_pixel": None, "0.195m": .195, "0.5m": .5, "1.0m": 1.}
    counts = {key: [0, 0] for key in radii}
    n_pred = n_true = edges = 0

    def contour(state, known):
        shade = known & ~state
        sun = known & state
        border = np.zeros_like(known)
        border[:, 1:] |= shade[:, 1:] & sun[:, :-1]
        border[:, :-1] |= shade[:, :-1] & sun[:, 1:]
        border[1:] |= shade[1:] & sun[:-1]
        border[:-1] |= shade[:-1] & sun[1:]
        return border

    row_radius = min(h-1, max(1, int(1. / dz.min())))
    col_radius = min(w-1, max(1, int(1. / dx.min())))
    for t in range(raw.shape[0]):
        known = np.zeros((h, w), dtype=bool)
        known[r, c] = valid[t] & (labels[t] >= 0) & np.isfinite(raw[t])
        edges += int((known[1:] & known[:-1]).sum() + (known[:, 1:] & known[:, :-1]).sum())
        grid = np.zeros_like(known)
        grid[r, c] = labels[t] == 1
        actual = contour(grid, known)
        grid[r, c] = 1 / (1 + np.exp(-np.clip(raw[t].astype(np.float64)/temperature, -60, 60))) >= decision_threshold
        predicted = contour(grid, known)
        n_pred += int(predicted.sum())
        n_true += int(actual.sum())
        if not actual.any() or not predicted.any():
            continue
        # Search from the sparse contour pixels, with distances measured at actual world centers.
        for direction, (source, target) in enumerate(((predicted, actual), (actual, predicted))):
            sr, sc = np.nonzero(source)
            nearest = np.full(sr.shape, np.inf)
            pixel_match = np.zeros(sr.shape, dtype=bool)
            for dr in range(-row_radius, row_radius + 1):
                for dc in range(-col_radius, col_radius + 1):
                    tr, tc = sr + dr, sc + dc
                    inside = (tr >= 0) & (tr < h) & (tc >= 0) & (tc < w)
                    ids = np.flatnonzero(inside)
                    ids = ids[target[tr[ids], tc[ids]]]
                    if dr*dr + dc*dc <= 1:
                        pixel_match[ids] = True
                    dist = np.hypot(z[tr[ids]]-z[sr[ids]], x[tc[ids]]-x[sc[ids]])
                    nearest[ids] = np.minimum(nearest[ids], dist)
            for key, radius in radii.items():
                counts[key][direction] += int((pixel_match if radius is None else nearest <= radius).sum())
    scores = {}
    for key, (mp, mt) in counts.items():
        precision = mp/n_pred if n_pred else None
        recall = mt/n_true if n_true else None
        f1 = None if n_pred+n_true == 0 else 0.
        if precision and recall:
            f1 = 2*precision*recall/(precision+recall)
        scores[key] = dict(precision=precision, recall=recall, f1=f1, matched_pred=mp, matched_true=mt)
    return dict(observed_neighbor_pairs=edges, predicted_contour_points=n_pred,
                true_contour_points=n_true, scores=scores)

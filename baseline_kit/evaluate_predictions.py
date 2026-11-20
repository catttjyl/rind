#!/usr/bin/env python3
"""Unified standalone referee evaluator for RIND external baseline predictions.

Computes exact calibration (`calibration` 12 tiles, 181-grid `[0.05, 0.95]` threshold)
and 14-tile `val_spatial_clean` metrics (`state_acc`, `state_error`, `sun_iou`, `f1`,
`brier`, `insolation_bias`, `edge_lt0p5m_acc`, `shared_az0_hardsun_recall`,
`trans_f1_0s`, `event_recall_pm2min`, `median_lag_min`, `balanced_acc`, `ece` (15
equal-width bins), `changed_brier` / `stable_brier`, the `low_sun_0_10deg` subset bundle,
optional first-blocker distance buckets (`--blocker-buckets`), plus 4,000-resample paired
tile bootstrap 95% CI when `--ref-probs-dir` is supplied).

The reference model is calibrated exactly like the candidate: its own scalar temperature
(if logits) and its own 181-grid threshold, both fit on its `calibration` predictions
(`--ref-cal-preds-dir`).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

KIT_ROOT = Path(__file__).resolve().parents[1]

ECE_NUM_BINS = 15
LOW_SUN_MIN_DEG = 0.0
LOW_SUN_MAX_DEG = 10.0

# First-blocker distance buckets: same edges and inclusivity as
# `rind.metrics.compute_blocker_distance_stratified_metrics`.
BLOCKER_T_MAX_M = 4000.0
BLOCKER_BUCKETS: Tuple[str, ...] = ("lt_50m", "50m_to_300m", "300m_to_4000m")
BLOCKER_SCENE_KEYS: Tuple[str, ...] = (
    "receivers_xyz", "scene_triangles_xyz", "scene_opaque", "scene_source_ids",
    "receiver_source_triangle_ids",
)


def day_spans_from_ids(day_ids: np.ndarray, ts: np.ndarray) -> List[Tuple[int, int, int]]:
    """Split contiguous 120s-step day segments from `day_ids` and `timestamps_unix_s`."""
    spans: List[Tuple[int, int, int]] = []
    n = len(day_ids)
    if n == 0:
        return spans
    start = 0
    for i in range(1, n):
        if day_ids[i] != day_ids[i - 1] or (int(ts[i]) - int(ts[i - 1])) != 120:
            spans.append((int(day_ids[start]), start, i))
            start = i
    spans.append((int(day_ids[start]), start, n))
    return spans


def _sigmoid_temp(logits: np.ndarray, temp: float) -> np.ndarray:
    scaled = np.clip(logits.astype(np.float64) / max(float(temp), 1e-4), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def _fit_temperature_on_arrays(
    logit_vec: np.ndarray, gt_vec: np.ndarray
) -> Tuple[float, float]:
    """Coarse-to-fine scalar temperature calibration minimizing BCE NLL."""
    def _nll(temp: float) -> float:
        p = np.clip(_sigmoid_temp(logit_vec, temp), 1e-7, 1.0 - 1e-7)
        return float(-np.mean(gt_vec * np.log(p) + (1.0 - gt_vec) * np.log(1.0 - p)))

    coarse = np.linspace(0.25, 4.0, 81)
    best_t = float(min(coarse, key=_nll))
    fine = np.linspace(max(0.15, best_t - 0.15), min(4.5, best_t + 0.15), 61)
    best_t = float(min(fine, key=_nll))
    return best_t, _nll(best_t)


def calibrate_logits_or_probs(
    preds_list: Sequence[np.ndarray],
    truths_list: Sequence[np.ndarray],
    valid_list: Sequence[np.ndarray],
    *,
    is_logits: bool = False,
) -> Tuple[float, float, float]:
    """Calibrate scalar temperature (if logits) and `[0.05, 0.95]` threshold on `calibration`."""
    flat_pred: List[np.ndarray] = []
    flat_gt: List[np.ndarray] = []
    for arr, tr, vm in zip(preds_list, truths_list, valid_list):
        mask = np.asarray(vm, dtype=bool) & (np.asarray(tr, dtype=np.int8) >= 0)
        flat_pred.append(np.asarray(arr[mask], dtype=np.float64))
        flat_gt.append((np.asarray(tr[mask], dtype=np.int8) == 1).astype(np.float64))

    pred_vec = np.concatenate(flat_pred)
    gt_vec = np.concatenate(flat_gt)
    if pred_vec.size == 0:
        raise ValueError("Calibration split has zero valid point-frames.")

    if is_logits:
        best_t, best_nll = _fit_temperature_on_arrays(pred_vec, gt_vec)
        probs = _sigmoid_temp(pred_vec, best_t)
    else:
        best_t = 1.0
        probs = np.clip(pred_vec, 1e-7, 1.0 - 1e-7)
        best_nll = float(-np.mean(gt_vec * np.log(probs) + (1.0 - gt_vec) * np.log(1.0 - probs)))

    gt_b = gt_vec > 0.5
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
        score = 0.5 * (acc + f1)
        if score > best_score:
            best_score = score
            best_th = float(round(float(th), 4))
    return best_t, best_th, best_nll


def _compute_event_metrics(
    pr_b: np.ndarray,
    gt_b: np.ndarray,
    val: np.ndarray,
    day_ids: np.ndarray,
    ts: np.ndarray,
) -> Dict[str, Any]:
    """Compute strict `@0s` transition F1 and per-receiver `±1` frame (`±2 min`) event lag."""
    spans = day_spans_from_ids(day_ids, ts)
    tp0 = fp0 = fn0 = 0
    ev_tot = ev_hit_pm1 = 0
    lags_abs: List[int] = []

    for _, f0, f1 in spans:
        if f1 - f0 < 2:
            continue
        v_pair = val[f0 : f1 - 1] & val[f0 + 1 : f1]
        t_chg = (gt_b[f0 + 1 : f1] != gt_b[f0 : f1 - 1]) & v_pair
        p_chg = (pr_b[f0 + 1 : f1] != pr_b[f0 : f1 - 1]) & v_pair
        tp0 += int((t_chg & p_chg).sum())
        fp0 += int(((~t_chg) & p_chg & v_pair).sum())
        fn0 += int((t_chg & (~p_chg)).sum())

        t_dir = np.where(t_chg, np.where(gt_b[f0 + 1 : f1], 1, -1), 0)
        p_dir = np.where(p_chg, np.where(pr_b[f0 + 1 : f1], 1, -1), 0)
        for col in np.flatnonzero(np.any(t_dir != 0, axis=0)):
            for sign in (1, -1):
                te = np.flatnonzero(t_dir[:, col] == sign)
                if te.size == 0:
                    continue
                pe = np.flatnonzero(p_dir[:, col] == sign)
                ev_tot += int(te.size)
                for idx_t in te:
                    if pe.size > 0:
                        d_min = int(np.min(np.abs(pe - idx_t)))
                        if d_min <= 1:
                            ev_hit_pm1 += 1
                        lags_abs.append(d_min * 2)

    tf1_0s = (2 * tp0) / max(2 * tp0 + fp0 + fn0, 1)
    rec_pm2m = ev_hit_pm1 / max(ev_tot, 1) if ev_tot > 0 else None
    med_lag_min = float(np.median(lags_abs)) if lags_abs else None
    return {
        "trans_f1_0s": float(tf1_0s),
        "true_events": int(ev_tot),
        "event_hit_pm2min": int(ev_hit_pm1),
        "event_recall_pm2min": rec_pm2m,
        "median_lag_min": med_lag_min,
    }


def _ece_histogram(probs_v: np.ndarray, gt_v: np.ndarray, n_bins: int = ECE_NUM_BINS) -> Dict[str, List[float]]:
    """Per-bin point-frame count, probability sum, and positive count on equal-width `[0, 1]` bins."""
    p = np.asarray(probs_v, dtype=np.float64)
    idx = np.clip((p * n_bins).astype(np.int64), 0, n_bins - 1)
    return {
        "count": np.bincount(idx, minlength=n_bins).astype(np.int64).tolist(),
        "prob_sum": np.bincount(idx, weights=p, minlength=n_bins).tolist(),
        "pos_sum": np.bincount(idx, weights=gt_v.astype(np.float64), minlength=n_bins).tolist(),
    }


def _ece_from_histogram(hist: Dict[str, Sequence[float]]) -> Optional[float]:
    """ECE = sum over bins of `count / total * |mean prob - empirical sun rate|`."""
    cnt = np.asarray(hist["count"], dtype=np.float64)
    tot = float(cnt.sum())
    if tot <= 0:
        return None
    nz = cnt > 0
    gap = np.abs(np.asarray(hist["prob_sum"])[nz] - np.asarray(hist["pos_sum"])[nz]) / cnt[nz]
    return float(np.sum(cnt[nz] / tot * gap))


def _changed_stable_brier_sums(
    probs: np.ndarray,
    gt_b: np.ndarray,
    val: np.ndarray,
    day_ids: np.ndarray,
    ts: np.ndarray,
) -> Dict[str, Any]:
    """Split Brier over same-day adjacent valid frame pairs `(t-1, t)`, scored at frame `t`.

    A point-frame is `changed` if its truth differs from the previous frame, otherwise
    `stable`. Frames without a valid same-day predecessor belong to neither set.
    """
    ch_pf = st_pf = 0
    ch_sq = st_sq = 0.0
    for _, f0, f1 in day_spans_from_ids(day_ids, ts):
        if f1 - f0 < 2:
            continue
        v_pair = val[f0 : f1 - 1] & val[f0 + 1 : f1]
        chg = (gt_b[f0 + 1 : f1] != gt_b[f0 : f1 - 1]) & v_pair
        stb = v_pair & (~chg)
        sq = (probs[f0 + 1 : f1].astype(np.float64) - gt_b[f0 + 1 : f1].astype(np.float64)) ** 2
        ch_pf += int(chg.sum())
        st_pf += int(stb.sum())
        ch_sq += float(sq[chg].sum())
        st_sq += float(sq[stb].sum())
    return {
        "changed_pf": ch_pf,
        "changed_brier": (ch_sq / ch_pf) if ch_pf > 0 else None,
        "stable_pf": st_pf,
        "stable_brier": (st_sq / st_pf) if st_pf > 0 else None,
    }


def solar_altitude_from_gt(gt: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Per-frame solar altitude in degrees: shard `solar_altitude_deg`, else from EUN `sun_dirs`."""
    alt = gt.get("solar_altitude_deg")
    if alt is not None:
        return np.asarray(alt, dtype=np.float64).reshape(-1)
    sd = gt.get("sun_dirs")
    if sd is None:
        return None
    s = np.asarray(sd, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(s, axis=-1), 1e-12)
    return np.degrees(np.arcsin(np.clip(s[:, 1] / norms, -1.0, 1.0)))


def compute_metrics_bundle(
    probs: np.ndarray,
    truth: np.ndarray,
    valid_mask: np.ndarray,
    thresh: float,
    day_ids: np.ndarray,
    ts: np.ndarray,
    bdist: Optional[np.ndarray] = None,
    bval: Optional[np.ndarray] = None,
    shared_hard_mask: Optional[np.ndarray] = None,
    solar_altitude_deg: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute exact per-tile RIND evaluation metrics matching the master referee protocol.

    If `solar_altitude_deg` (shape `[T]`) is given, the same bundle is also computed on the
    `[0°, 10°]` low-sun frames and returned under `low_sun_0_10deg`.
    """
    val = np.asarray(valid_mask, dtype=bool) & (np.asarray(truth, dtype=np.int8) >= 0)
    pf = int(val.sum())
    if pf == 0:
        return {"point_frames": 0}
    gt_b = truth == 1
    pr_b = probs >= thresh

    tp = int((pr_b & gt_b & val).sum())
    fp = int((pr_b & (~gt_b) & val).sum())
    tn = int(((~pr_b) & (~gt_b) & val).sum())
    fn = int(((~pr_b) & gt_b & val).sum())
    acc = (tp + tn) / pf
    brier = float(np.mean((probs[val].astype(np.float64) - gt_b[val].astype(np.float64)) ** 2))

    edge_acc = None
    edge_pf = 0
    if bdist is not None and bval is not None:
        em = val & np.asarray(bval, dtype=bool) & (bdist < 0.5)
        edge_pf = int(em.sum())
        if edge_pf > 0:
            edge_acc = float((pr_b[em] == gt_b[em]).mean())

    hs_recall = None
    hs_acc = None
    hs_pf = 0
    hs_sun_pf = 0
    if shared_hard_mask is not None:
        hm_all = val & np.asarray(shared_hard_mask, dtype=bool)
        hs_pf = int(hm_all.sum())
        if hs_pf > 0:
            hs_acc = float((pr_b[hm_all] == gt_b[hm_all]).mean())
        hm_sun = hm_all & gt_b
        hs_sun_pf = int(hm_sun.sum())
        if hs_sun_pf > 0:
            hs_recall = float(pr_b[hm_sun].mean())

    ev = _compute_event_metrics(pr_b, gt_b, val, day_ids, ts)
    ece_hist = _ece_histogram(probs[val], gt_b[val])
    cs = _changed_stable_brier_sums(probs, gt_b, val, day_ids, ts)

    low_sun: Optional[Dict[str, Any]] = None
    if solar_altitude_deg is not None:
        alt = np.asarray(solar_altitude_deg, dtype=np.float64).reshape(-1)
        low_frames = (alt >= LOW_SUN_MIN_DEG) & (alt <= LOW_SUN_MAX_DEG)
        low_sun = compute_metrics_bundle(
            probs, truth, val & low_frames[:, None], thresh, day_ids, ts,
            bdist, bval, shared_hard_mask, solar_altitude_deg=None,
        )

    out: Dict[str, Any] = {
        "point_frames": pf,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "state_acc": float(acc),
        "state_error": float(1.0 - acc),
        "balanced_acc": _balanced_acc(tp, fp, tn, fn),
        "sun_iou": float(tp / max(tp + fp + fn, 1)),
        "f1": float((2 * tp) / max(2 * tp + fp + fn, 1)),
        "brier": brier,
        "ece": _ece_from_histogram(ece_hist),
        "ece_hist": ece_hist,
        **cs,
        "insolation_bias": float(abs(fp - fn) / pf),
        "edge_lt0p5m_pf": edge_pf,
        "edge_lt0p5m_acc": edge_acc,
        "shared_az0_hardsun_pf": hs_pf,
        "shared_az0_hardsun_sun_pf": hs_sun_pf,
        "shared_az0_hardsun_acc": hs_acc,
        "shared_az0_hardsun_recall": hs_recall,
        **ev,
    }
    if low_sun is not None:
        out["low_sun_0_10deg"] = low_sun
    return out


def trace_first_blockers(
    receivers_xyz: np.ndarray,
    sun_dirs: np.ndarray,
    scene_triangles_xyz: np.ndarray,
    *,
    scene_opaque: Optional[np.ndarray] = None,
    scene_source_ids: Optional[np.ndarray] = None,
    receiver_source_triangle_ids: Optional[np.ndarray] = None,
    t_max_m: float = BLOCKER_T_MAX_M,
    rays_per_pass: int = 8192,
) -> Dict[str, Any]:
    """Exact first-hit distance from every receiver toward the sun (geometry only, model-free).

    Mirrors the ray setup of `rind.metrics.compute_blocker_distance_stratified_metrics`:
    `ExactSceneBVH.query_first_hit` with `t_min = DEFAULT_T_MIN_M`, `t_max = t_max_m`, the
    receiver's own surface face excluded when `receiver_source_triangle_ids` is available,
    and frames with the sun at or below the horizon left untraced.
    """
    if str(KIT_ROOT) not in sys.path:
        sys.path.insert(0, str(KIT_ROOT))
    from rind.inverse_design import DEFAULT_T_MIN_M, ExactSceneBVH

    rec = np.asarray(receivers_xyz, dtype=np.float64)
    sun = np.asarray(sun_dirs, dtype=np.float64)
    t_steps, n_pts = sun.shape[0], rec.shape[0]
    s_norm = np.linalg.norm(sun, axis=-1)
    above = (s_norm > 1e-12) & ((sun[:, 1] / np.maximum(s_norm, 1e-12)) > 0.0)

    first_t = np.full((t_steps, n_pts), np.inf, dtype=np.float64)
    occluded = np.zeros((t_steps, n_pts), dtype=bool)
    bvh = ExactSceneBVH(scene_triangles_xyz, opaque_mask=scene_opaque, source_triangle_ids=scene_source_ids)
    excl = (np.asarray(receiver_source_triangle_ids, dtype=np.uint64).reshape(n_pts)
            if receiver_source_triangle_ids is not None else None)
    t_idx = np.flatnonzero(above)
    if t_idx.size and n_pts and bvh.num_triangles > 0:
        per_pass = max(1, int(rays_per_pass) // max(n_pts, 1))
        for s0 in range(0, t_idx.size, per_pass):
            tb = t_idx[s0 : s0 + per_pass]
            b = tb.size
            occ, ft, _, _ = bvh.query_first_hit(
                np.broadcast_to(rec[None], (b, n_pts, 3)).reshape(-1, 3),
                np.broadcast_to(sun[tb, None, :], (b, n_pts, 3)).reshape(-1, 3),
                t_min=DEFAULT_T_MIN_M,
                t_max=float(t_max_m),
                exclude_source_triangle_ids=(np.broadcast_to(excl[None], (b, n_pts)).reshape(-1)
                                             if excl is not None else None),
            )
            occluded[tb] = occ.reshape(b, n_pts)
            first_t[tb] = ft.reshape(b, n_pts)
    return {
        "first_hit_m": first_t,
        "occluded": occluded,
        "above_horizon": above,
        "self_face_excluded": excl is not None,
        "t_max_m": float(t_max_m),
    }


def load_or_trace_blockers(
    gt: Dict[str, np.ndarray], tile_id: str, cache_dir: Optional[Path]
) -> Dict[str, Any]:
    """Trace once per tile and reuse across models; the cache is keyed by receivers and sun."""
    missing = [k for k in ("receivers_xyz", "sun_dirs", "scene_triangles_xyz") if k not in gt]
    if missing:
        raise KeyError(f"{tile_id}: blocker buckets need {missing} (only `.tar.gz` shards carry the scene mesh)")
    cache_path = cache_dir / f"blocker_{tile_id}.npz" if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            if (np.array_equal(z["receivers_xyz"], gt["receivers_xyz"])
                    and np.array_equal(z["sun_dirs"], gt["sun_dirs"])
                    and float(z["t_max_m"]) == BLOCKER_T_MAX_M):
                return {"first_hit_m": z["first_hit_m"], "occluded": z["occluded"],
                        "above_horizon": z["above_horizon"],
                        "self_face_excluded": bool(z["self_face_excluded"]), "t_max_m": float(z["t_max_m"])}
    blk = trace_first_blockers(
        gt["receivers_xyz"], gt["sun_dirs"], gt["scene_triangles_xyz"],
        scene_opaque=gt.get("scene_opaque"), scene_source_ids=gt.get("scene_source_ids"),
        receiver_source_triangle_ids=gt.get("receiver_source_triangle_ids"),
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, receivers_xyz=gt["receivers_xyz"], sun_dirs=gt["sun_dirs"], **blk)
    return blk


def _blocker_bucket_mask(name: str, distance: np.ndarray) -> np.ndarray:
    """Bucket membership with the exact inclusivity used in `rind.metrics`."""
    if name == "lt_50m":
        return (distance > 0.0) & (distance < 50.0)
    if name == "50m_to_300m":
        return (distance >= 50.0) & (distance <= 300.0)
    if name == "300m_to_4000m":
        return (distance > 300.0) & (distance <= BLOCKER_T_MAX_M + 1e-6)
    raise KeyError(name)


def compute_blocker_bucket_counts(
    probs: np.ndarray,
    truth: np.ndarray,
    valid_mask: np.ndarray,
    thresh: float,
    blk: Dict[str, Any],
) -> Dict[str, Any]:
    """Additive per-tile counts for occluded buckets and the unoccluded open-sky set.

    Occluded buckets: valid point-frames whose sun ray hits an opaque triangle within the
    bucket's distance range. Open sky: sun above horizon and no hit within `t_max`; these
    have no blocker, so their error is reported as a separate false-shadow rate.
    """
    val = np.asarray(valid_mask, dtype=bool) & (np.asarray(truth, dtype=np.int8) >= 0)
    gt_b = truth == 1
    pr_b = probs >= thresh
    sq = (np.asarray(probs, dtype=np.float64) - gt_b.astype(np.float64)) ** 2
    occ = np.asarray(blk["occluded"], dtype=bool)
    dist = np.asarray(blk["first_hit_m"], dtype=np.float64)

    def counts(mask: np.ndarray) -> Dict[str, Any]:
        return {
            "point_frames": int(mask.sum()),
            "correct": int((mask & (pr_b == gt_b)).sum()),
            "sq_err_sum": float(sq[mask].sum()),
            "truth_shadow": int((mask & ~gt_b).sum()),
            "shadow_hit": int((mask & ~gt_b & ~pr_b).sum()),
            "truth_sun": int((mask & gt_b).sum()),
            "false_shadow": int((mask & gt_b & ~pr_b).sum()),
        }

    out = {name: counts(val & occ & _blocker_bucket_mask(name, dist)) for name in BLOCKER_BUCKETS}
    out["open_sky"] = counts(val & np.asarray(blk["above_horizon"], dtype=bool)[:, None] & ~occ)
    return out


def _finalize_blocker_counts(raw: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    def rate(num: int, den: int) -> Optional[float]:
        return (num / den) if den > 0 else None

    out: Dict[str, Any] = {}
    for name in BLOCKER_BUCKETS:
        c = raw[name]
        out[name] = {
            "point_frames": c["point_frames"],
            "shadow_support": c["truth_shadow"],
            "shadow_recall": rate(c["shadow_hit"], c["truth_shadow"]),
            "state_acc": rate(c["correct"], c["point_frames"]),
            "brier": (c["sq_err_sum"] / c["point_frames"]) if c["point_frames"] else None,
        }
    c = raw["open_sky"]
    out["open_sky"] = {
        "point_frames": c["point_frames"],
        "truth_sun_support": c["truth_sun"],
        "false_shadow_count": c["false_shadow"],
        "false_shadow_rate": rate(c["false_shadow"], c["truth_sun"]),
        "truth_shadow_count": c["truth_shadow"],
        "state_acc": rate(c["correct"], c["point_frames"]),
        "brier": (c["sq_err_sum"] / c["point_frames"]) if c["point_frames"] else None,
    }
    return out


def _sum_blocker_counts(rows: Sequence[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    keys = (*BLOCKER_BUCKETS, "open_sky")
    return {k: {f: sum(r[k][f] for r in rows) for f in rows[0][k]} for k in keys}


def _balanced_acc(tp: int, fp: int, tn: int, fn: int) -> Optional[float]:
    """`(TPR + TNR) / 2`; undefined when either class is absent."""
    if tp + fn == 0 or tn + fp == 0:
        return None
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def paired_tile_bootstrap_ci(
    rows_cand: Sequence[Dict[str, Any]],
    rows_ref: Sequence[Dict[str, Any]],
    *,
    metric: str = "state_acc",
    n_boot: int = 4000,
    seed: int = 20261115,
) -> Dict[str, Any]:
    """Compute 4,000-resample paired tile bootstrap 95% CI (candidate minus reference)."""
    if len(rows_cand) != len(rows_ref) or len(rows_cand) == 0:
        raise ValueError("Candidate and reference tile lists must have identical non-zero length.")
    deltas = np.array(
        [float(a[metric]) - float(b[metric]) for a, b in zip(rows_cand, rows_ref)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    n_tiles = len(deltas)
    idx = rng.integers(0, n_tiles, size=(int(n_boot), n_tiles))
    boot_means = deltas[idx].mean(axis=1)
    wins = int(np.sum(deltas > 0.0))
    return {
        "metric": metric,
        "n_tiles": n_tiles,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "tile_wins": wins,
        "mean_delta": float(np.mean(deltas)),
        "ci95_low": float(np.percentile(boot_means, 2.5)),
        "ci95_high": float(np.percentile(boot_means, 97.5)),
    }


def _load_tar_npz_dict(tar_path: Path, keys: Sequence[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    wanted = {f"{k}.npy": k for k in keys}
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            base = Path(member.name).name
            if base in wanted:
                fobj = tf.extractfile(member)
                if fobj is not None:
                    out[wanted[base]] = np.load(io.BytesIO(fobj.read()), allow_pickle=False)
    return out


def load_tile_ground_truth(
    data_dir: Path, tile_id: str, extra_keys: Sequence[str] = ()
) -> Dict[str, np.ndarray]:
    """Load ground truth arrays for `tile_id` from `.npz`, extracted dir, or `.tar.gz`.

    `extra_keys` are loaded when present (e.g. `BLOCKER_SCENE_KEYS` for blocker buckets).
    """
    req = ("truth", "valid_mask", "day_ids", "timestamps_unix_s")
    opt = ("boundary_distance_m", "boundary_distance_valid", "shared_az0_cand_nonempty", "cand_nonempty", "az_mask",
           "solar_altitude_deg", "sun_dirs", *extra_keys)

    npz_path = data_dir / f"{tile_id}.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files if k in (*req, *opt)}

    direct_dir = data_dir / tile_id
    if not direct_dir.exists():
        matches = list(data_dir.glob(f"*/{tile_id}"))
        if matches:
            direct_dir = matches[0]
    if direct_dir.exists():
        out = {k: np.load(direct_dir / f"{k}.npy", allow_pickle=False) for k in req}
        for k in opt:
            p = direct_dir / f"{k}.npy"
            if p.exists():
                out[k] = np.load(p, allow_pickle=False)
        return out

    tar_path = data_dir / f"{tile_id}.tar.gz"
    if tar_path.exists():
        return _load_tar_npz_dict(tar_path, (*req, *opt))

    raise FileNotFoundError(f"Could not locate shard for {tile_id} under {data_dir}")


def load_split_tile_ids(split_json: Path, manifest_json: Path) -> Dict[str, List[str]]:
    """Resolve `fit_67`, `calibration`, and `val_spatial_clean` tile IDs via `SplitContract`."""
    if str(KIT_ROOT) not in sys.path:
        sys.path.insert(0, str(KIT_ROOT))
    from rind.split_contract import SplitContract

    contract = SplitContract(split_path=split_json, manifest_path=manifest_json)
    return {
        "fit_67": contract.get_fit_tiles("fit_67"),
        "calibration": contract.get_calibration_tiles(),
        "val_spatial_clean": contract.get_eval_tiles("val_spatial_clean"),
    }


def _load_pred_array(pred_dir: Path, tile_id: str) -> np.ndarray:
    for pat in (f"pred_probs_{tile_id}.npy", f"pred_logits_{tile_id}.npy", f"{tile_id}.npy"):
        cand = pred_dir / pat
        if cand.exists():
            return np.load(cand, allow_pickle=False)
    raise FileNotFoundError(f"Missing prediction file for {tile_id} in {pred_dir}")


def _aggregate_val_metrics(
    per_tile: Sequence[Dict[str, Any]],
    *,
    include_per_tile: bool = True,
) -> Dict[str, Any]:
    """Aggregate confusion counts, edge/Hard-Sun slices, ECE, change/stable Brier, and event lag."""
    tp_s = sum(r["tp"] for r in per_tile)
    fp_s = sum(r["fp"] for r in per_tile)
    tn_s = sum(r["tn"] for r in per_tile)
    fn_s = sum(r["fn"] for r in per_tile)
    pf_s = sum(r["point_frames"] for r in per_tile)
    brier_sum = sum(r["brier"] * r["point_frames"] for r in per_tile)
    ed_pf = sum(r["edge_lt0p5m_pf"] for r in per_tile)
    ed_ok = sum(int(round((r["edge_lt0p5m_acc"] or 0.0) * r["edge_lt0p5m_pf"])) for r in per_tile)
    hs_sun = sum(r["shared_az0_hardsun_sun_pf"] for r in per_tile)
    hs_hit = sum(int(round((r["shared_az0_hardsun_recall"] or 0.0) * r["shared_az0_hardsun_sun_pf"])) for r in per_tile)
    ev_tot = sum(r["true_events"] for r in per_tile)
    ev_hit = sum(r["event_hit_pm2min"] for r in per_tile)
    lags_all = [r["median_lag_min"] for r in per_tile if r["median_lag_min"] is not None]
    overall_acc = (tp_s + tn_s) / max(pf_s, 1)
    ece_hist = {
        key: np.sum([np.asarray(r["ece_hist"][key], dtype=np.float64) for r in per_tile], axis=0).tolist()
        for key in ("count", "prob_sum", "pos_sum")
    }
    ch_pf = sum(r["changed_pf"] for r in per_tile)
    st_pf = sum(r["stable_pf"] for r in per_tile)
    ch_sq = sum((r["changed_brier"] or 0.0) * r["changed_pf"] for r in per_tile)
    st_sq = sum((r["stable_brier"] or 0.0) * r["stable_pf"] for r in per_tile)
    out: Dict[str, Any] = {
        "n_tiles": len(per_tile),
        "valid_point_frames": pf_s,
        "state_acc": overall_acc,
        "state_error": 1.0 - overall_acc,
        "balanced_acc": _balanced_acc(tp_s, fp_s, tn_s, fn_s),
        "state_f1": (2 * tp_s) / max(2 * tp_s + fp_s + fn_s, 1),
        "sun_iou": tp_s / max(tp_s + fp_s + fn_s, 1),
        "brier": brier_sum / max(pf_s, 1),
        "ece": _ece_from_histogram(ece_hist),
        "ece_num_bins": ECE_NUM_BINS,
        "changed_pf": ch_pf,
        "changed_brier": (ch_sq / ch_pf) if ch_pf > 0 else None,
        "stable_pf": st_pf,
        "stable_brier": (st_sq / st_pf) if st_pf > 0 else None,
        "mean_tile_insolation_bias": float(np.mean([r["insolation_bias"] for r in per_tile])),
        "edge_lt0p5m_pf": ed_pf,
        "edge_lt0p5m_acc": (ed_ok / ed_pf) if ed_pf > 0 else None,
        "shared_az0_hardsun_recall": (hs_hit / hs_sun) if hs_sun > 0 else None,
        "true_events": ev_tot,
        "event_recall_pm2min": (ev_hit / ev_tot) if ev_tot > 0 else None,
        "median_tile_lag_min": float(np.median(lags_all)) if lags_all else None,
    }
    low_rows = [r["low_sun_0_10deg"] for r in per_tile
                if r.get("low_sun_0_10deg", {}).get("point_frames", 0) > 0]
    if any("low_sun_0_10deg" in r for r in per_tile):
        out["low_sun_0_10deg"] = (
            _aggregate_val_metrics(low_rows, include_per_tile=False) if low_rows
            else {"n_tiles": 0, "valid_point_frames": 0}
        )
    blk_rows = [r["blocker_counts"] for r in per_tile if "blocker_counts" in r]
    if blk_rows:
        out["blocker_distance_buckets"] = _finalize_blocker_counts(_sum_blocker_counts(blk_rows))
    if include_per_tile:
        out["per_tile"] = list(per_tile)
    return out


def calibrate_prediction_dir(
    preds_dir: Path,
    tile_ids: Sequence[str],
    cal_truths: Sequence[np.ndarray],
    cal_valids: Sequence[np.ndarray],
    *,
    is_logits: bool,
) -> Dict[str, Any]:
    """Fit one model's temperature and threshold on its `calibration` predictions."""
    preds = [_load_pred_array(preds_dir, tid) for tid in tile_ids]
    temp, thresh, nll = calibrate_logits_or_probs(preds, cal_truths, cal_valids, is_logits=is_logits)
    return {"temperature": temp, "threshold": thresh, "calibration_bce_nll": nll, "is_logits": bool(is_logits)}


def _to_probs(raw: np.ndarray, cal: Dict[str, Any]) -> np.ndarray:
    if cal["is_logits"]:
        return _sigmoid_temp(raw, cal["temperature"]).astype(np.float32)
    return np.asarray(raw, dtype=np.float32)


PAIRED_BOOTSTRAP_METRICS: Tuple[str, ...] = ("state_acc", "brier", "trans_f1_0s")


def evaluate_directory(
    *,
    data_dir: Path,
    cal_preds_dir: Path,
    val_preds_dir: Path,
    split_json: Path,
    manifest_json: Path,
    is_logits: bool = False,
    ref_probs_dir: Optional[Path] = None,
    ref_cal_preds_dir: Optional[Path] = None,
    ref_is_logits: bool = False,
    blocker_buckets: bool = False,
    blocker_cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run calibration on `calibration` (12 tiles) and full metrics on `val_spatial_clean` (14 tiles).

    With `ref_probs_dir`, the reference is calibrated on its own `ref_cal_preds_dir` predictions
    with the same procedure as the candidate, then both are scored and paired-bootstrapped.
    """
    if ref_probs_dir is not None and ref_cal_preds_dir is None:
        raise ValueError(
            "--ref-probs-dir needs --ref-cal-preds-dir: the reference must get its own calibrated "
            "temperature/threshold instead of a fixed 0.5."
        )
    buckets = load_split_tile_ids(split_json, manifest_json)
    cal_truths, cal_valids = [], []
    for tid in buckets["calibration"]:
        gt = load_tile_ground_truth(data_dir, tid)
        cal_truths.append(gt["truth"])
        cal_valids.append(gt["valid_mask"])

    cand_cal = calibrate_prediction_dir(
        cal_preds_dir, buckets["calibration"], cal_truths, cal_valids, is_logits=is_logits)
    ref_cal = (calibrate_prediction_dir(
        ref_cal_preds_dir, buckets["calibration"], cal_truths, cal_valids, is_logits=ref_is_logits)
        if ref_probs_dir is not None else None)

    extra = BLOCKER_SCENE_KEYS if blocker_buckets else ()
    blocker_meta: Dict[str, Any] = {}
    per_tile: List[Dict[str, Any]] = []
    ref_per_tile: List[Dict[str, Any]] = []

    def score(probs: np.ndarray, thresh: float, gt: Dict[str, np.ndarray], hs_mask, alt_deg, blk) -> Dict[str, Any]:
        mb = compute_metrics_bundle(
            probs, gt["truth"], gt["valid_mask"], thresh,
            gt["day_ids"], gt["timestamps_unix_s"],
            gt.get("boundary_distance_m"), gt.get("boundary_distance_valid"), hs_mask,
            solar_altitude_deg=alt_deg,
        )
        if blk is not None:
            mb["blocker_counts"] = compute_blocker_bucket_counts(probs, gt["truth"], gt["valid_mask"], thresh, blk)
            mb["blocker_distance_buckets"] = _finalize_blocker_counts(mb["blocker_counts"])
        return mb

    for tid in buckets["val_spatial_clean"]:
        gt = load_tile_ground_truth(data_dir, tid, extra_keys=extra)
        hs_mask = gt.get("shared_az0_cand_nonempty")
        if hs_mask is None:
            hs_mask = gt.get("cand_nonempty")
        alt_deg = solar_altitude_from_gt(gt)
        blk = load_or_trace_blockers(gt, tid, blocker_cache_dir) if blocker_buckets else None
        if blk is not None:
            blocker_meta[tid] = {"self_face_excluded": bool(blk["self_face_excluded"])}

        probs = _to_probs(_load_pred_array(val_preds_dir, tid), cand_cal)
        per_tile.append({"tile_id": tid, **score(probs, cand_cal["threshold"], gt, hs_mask, alt_deg, blk)})
        if ref_cal is not None:
            ref_p = _to_probs(_load_pred_array(ref_probs_dir, tid), ref_cal)
            ref_per_tile.append({"tile_id": tid, **score(ref_p, ref_cal["threshold"], gt, hs_mask, alt_deg, blk)})

    summary: Dict[str, Any] = {
        "protocol": "MANHATTAN_93TILE_CAL12_VAL14_UNIFIED_REFEREE",
        "calibrated_temperature": cand_cal["temperature"],
        "calibrated_threshold": cand_cal["threshold"],
        "calibration_bce_nll": cand_cal["calibration_bce_nll"],
        "val_spatial_clean": _aggregate_val_metrics(per_tile),
    }
    if blocker_buckets:
        summary["blocker_trace"] = {
            "t_max_m": BLOCKER_T_MAX_M,
            "self_face_excluded_tiles": sum(m["self_face_excluded"] for m in blocker_meta.values()),
            "n_tiles": len(blocker_meta),
        }
    if ref_cal is not None:
        summary["reference_calibration"] = ref_cal
        summary["reference_val_spatial_clean"] = _aggregate_val_metrics(ref_per_tile, include_per_tile=False)
        summary["paired_bootstrap_vs_ref"] = {
            m: paired_tile_bootstrap_ci(per_tile, ref_per_tile, metric=m) for m in PAIRED_BOOTSTRAP_METRICS
        }
    return summary


def run_self_test() -> None:
    """Deterministic synthetic verification covering calibration, metrics, and mutation sensitivity."""
    t_steps, n_pts = 40, 16
    day_ids = np.zeros((t_steps,), dtype=np.int64)
    ts = np.arange(t_steps, dtype=np.int64) * 120
    rng = np.random.default_rng(20261118)
    truth = np.zeros((t_steps, n_pts), dtype=np.int8)
    truth[15:30, :8] = 1
    truth[20:35, 8:] = 1
    valid_mask = np.ones((t_steps, n_pts), dtype=bool)
    bdist = np.full((t_steps, n_pts), 1.5, dtype=np.float32)
    bdist[:, :4] = 0.25
    bval = np.ones((t_steps, n_pts), dtype=bool)
    hs_mask = np.zeros((t_steps, n_pts), dtype=bool)
    hs_mask[:, 4:10] = True

    good_probs = np.where(truth == 1, 0.92, 0.08).astype(np.float32)
    bad_probs = (1.0 - good_probs).astype(np.float32)
    noisy_logits = np.where(truth == 1, 2.2, -2.2) + rng.normal(0.0, 0.1, size=truth.shape)

    temp, th, nll = calibrate_logits_or_probs([noisy_logits], [truth], [valid_mask], is_logits=True)
    if not (0.1 < temp <= 4.5 and 0.05 <= th <= 0.95 and nll < 0.2):
        raise AssertionError(f"Calibration out of bounds: T={temp}, th={th}, nll={nll}")

    alt_deg = np.where(np.arange(t_steps) < 10, 5.0, 30.0)  # frames 0-9 are low sun
    m_good = compute_metrics_bundle(good_probs, truth, valid_mask, 0.5, day_ids, ts, bdist, bval, hs_mask, alt_deg)
    m_bad = compute_metrics_bundle(bad_probs, truth, valid_mask, 0.5, day_ids, ts, bdist, bval, hs_mask, alt_deg)
    if abs(m_good["state_acc"] - 1.0) > 1e-9 or m_bad["state_acc"] > 0.01:
        raise AssertionError(f"Mutation check failed: good={m_good['state_acc']} bad={m_bad['state_acc']}")
    if abs(m_good["edge_lt0p5m_acc"] - 1.0) > 1e-9 or abs(m_good["event_recall_pm2min"] - 1.0) > 1e-9:
        raise AssertionError(f"Edge/event metric mismatch: {m_good}")

    # Balanced accuracy, ECE (|0.92 - 1| = |0.08 - 0| = 0.08 in every occupied bin).
    if abs(m_good["balanced_acc"] - 1.0) > 1e-9 or m_bad["balanced_acc"] > 0.01:
        raise AssertionError(f"Balanced acc mismatch: good={m_good['balanced_acc']} bad={m_bad['balanced_acc']}")
    if abs(m_good["ece"] - 0.08) > 1e-6 or abs(m_bad["ece"] - 0.92) > 1e-6:
        raise AssertionError(f"ECE mismatch: good={m_good['ece']} bad={m_bad['ece']}")
    # Changed/stable split: 16 receivers x 2 truth switches = 32 changed point-frames;
    # 16 x 39 adjacent pairs - 32 = 592 stable point-frames.
    if m_good["changed_pf"] != 32 or m_good["stable_pf"] != 592:
        raise AssertionError(f"Changed/stable split mismatch: {m_good['changed_pf']}/{m_good['stable_pf']}")
    sharp = good_probs.copy()
    sharp[15, :8] = 0.5  # hurt only a changed point-frame set
    m_sharp = compute_metrics_bundle(sharp, truth, valid_mask, 0.5, day_ids, ts)
    if not (m_sharp["changed_brier"] > m_good["changed_brier"]
            and abs(m_sharp["stable_brier"] - m_good["stable_brier"]) < 1e-12):
        raise AssertionError(f"Changed/stable Brier not isolated: {m_sharp}")
    # Low-sun subset: frames 0-9 x 16 receivers.
    low = m_good["low_sun_0_10deg"]
    if low["point_frames"] != 160 or "low_sun_0_10deg" in low:
        raise AssertionError(f"Low-sun subset mismatch: {low['point_frames']}")
    # Pooled ECE is computed from summed histograms, not averaged per tile: both occupied bins
    # hold 640 point-frames with sun rate 240/640, so ECE = (|0.08-0.375| + |0.92-0.375|) / 2.
    agg = _aggregate_val_metrics([m_good, m_bad], include_per_tile=False)
    if agg["low_sun_0_10deg"]["valid_point_frames"] != 320 or abs(agg["ece"] - 0.42) > 1e-6:
        raise AssertionError(f"Aggregation mismatch: {agg}")

    boot = paired_tile_bootstrap_ci([m_good, m_good], [m_bad, m_bad], metric="state_acc", n_boot=200)
    if boot["ci95_low"] <= 0.9:
        raise AssertionError(f"Bootstrap CI failed: {boot}")

    # Reference calibration: a reference whose logits are scaled by 3 must recover T ~= 3
    # and the same threshold as the candidate instead of a fixed 0.5.
    overlap = np.where(truth == 1, 0.8, -0.8) + rng.normal(0.0, 1.0, size=truth.shape)
    c_t, c_th, _ = calibrate_logits_or_probs([overlap], [truth], [valid_mask], is_logits=True)
    ref_t, ref_th, _ = calibrate_logits_or_probs([overlap * 3.0], [truth], [valid_mask], is_logits=True)
    if abs(ref_t / c_t - 3.0) > 0.15 or abs(ref_th - c_th) > 0.05:
        raise AssertionError(f"Reference calibration mismatch: T={ref_t} vs {c_t}, th={ref_th} vs {c_th}")

    # Blocker buckets on a synthetic first-hit field (geometry tracing is tested against
    # `rind.metrics` separately because it needs `rind.inverse_design`).
    first_hit = np.full((t_steps, n_pts), np.inf)
    occ = np.zeros((t_steps, n_pts), dtype=bool)
    first_hit[:, 0:2], first_hit[:, 2:4], first_hit[:, 4:6] = 20.0, 50.0, 1200.0
    occ[:, 0:6] = True
    above = np.ones((t_steps,), dtype=bool)
    above[:5] = False  # below-horizon frames are never traced: no hit, not open sky
    occ[:5], first_hit[:5] = False, np.inf
    blk = {"first_hit_m": first_hit, "occluded": occ, "above_horizon": above}
    blk_truth = truth.copy()
    blk_truth[:, 0:6] = 0  # occluded receivers are in shadow
    blk_probs = np.where(blk_truth == 1, 0.9, 0.1).astype(np.float32)
    blk_probs[:, 0] = 0.9   # half of the <50 m bucket missed (predicted sunlit)
    blk_probs[5:, 6] = 0.1  # one open-sky receiver wrongly predicted shadow when truly sunlit
    bc = _finalize_blocker_counts(compute_blocker_bucket_counts(blk_probs, blk_truth, valid_mask, 0.5, blk))
    if (bc["lt_50m"]["point_frames"] != 70 or abs(bc["lt_50m"]["shadow_recall"] - 0.5) > 1e-9
            or bc["50m_to_300m"]["point_frames"] != 70 or bc["300m_to_4000m"]["shadow_recall"] != 1.0):
        raise AssertionError(f"Blocker bucket mismatch: {bc}")
    os_ = bc["open_sky"]
    sun_open = int(((blk_truth == 1) & above[:, None])[:, 6:].sum())
    false_open = int(((blk_truth == 1) & above[:, None])[5:, 6].sum())
    if os_["point_frames"] != 35 * 10 or os_["false_shadow_count"] != false_open or os_["truth_sun_support"] != sun_open:
        raise AssertionError(f"Open-sky mismatch: {os_}")
    print(f"[evaluate_predictions --self-test] PASS (T={temp:.3f}, th={th:.3f}, acc={m_good['state_acc']:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description="RIND unified baseline prediction evaluator.")
    ap.add_argument("--self-test", action="store_true", help="Run synthetic unit verification and exit.")
    ap.add_argument("--data-dir", type=Path, help="Directory holding 93-tile shards (.npz, extracted, or .tar.gz).")
    ap.add_argument("--cal-preds-dir", type=Path, help="Directory with pred_probs_{tile_id}.npy for calibration tiles.")
    ap.add_argument("--val-preds-dir", type=Path, help="Directory with pred_probs_{tile_id}.npy for val_spatial_clean.")
    ap.add_argument("--ref-probs-dir", type=Path, default=None,
                    help="Optional reference val_spatial_clean predictions for paired bootstrap CI.")
    ap.add_argument("--ref-cal-preds-dir", type=Path, default=None,
                    help="Reference predictions on calibration tiles; required with --ref-probs-dir.")
    ap.add_argument("--ref-logits", action="store_true",
                    help="Treat reference inputs as uncalibrated logits instead of probabilities.")
    ap.add_argument("--blocker-buckets", action="store_true",
                    help="Trace exact first blockers (needs scene mesh in .tar.gz shards) and report distance buckets.")
    ap.add_argument("--blocker-cache-dir", type=Path, default=None,
                    help="Cache traced first-blocker distances per tile so later models skip ray tracing.")
    ap.add_argument("--split-json", type=Path, default=KIT_ROOT / "meta" / "split.json")
    ap.add_argument("--manifest-json", type=Path, default=KIT_ROOT / "meta" / "manifest.json")
    ap.add_argument("--logits", action="store_true", help="Treat inputs as uncalibrated logits instead of probabilities.")
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not (args.data_dir and args.cal_preds_dir and args.val_preds_dir):
        ap.error("--data-dir, --cal-preds-dir, and --val-preds-dir are required unless --self-test is set.")

    summary = evaluate_directory(
        data_dir=args.data_dir,
        cal_preds_dir=args.cal_preds_dir,
        val_preds_dir=args.val_preds_dir,
        split_json=args.split_json,
        manifest_json=args.manifest_json,
        is_logits=bool(args.logits),
        ref_probs_dir=args.ref_probs_dir,
        ref_cal_preds_dir=args.ref_cal_preds_dir,
        ref_is_logits=bool(args.ref_logits),
        blocker_buckets=bool(args.blocker_buckets),
        blocker_cache_dir=args.blocker_cache_dir,
    )
    txt = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()

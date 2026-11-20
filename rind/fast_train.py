"""Accelerated Streaming Full-Day Training, Zero-Shrinkage ST-JEPA, Low-RAM Calibration & Speedup Engine.

Implements:
1. Strict `FitFirewallToken` verification before every normalizer update (including `ResponseTargetTeacher`)
   and gradient step, plus low-label budget firewall (`apply_label_budget_view`) and pure-supervised firewall.
2. Full zero-shrinkage multi-objective training (`RindJepaAuxiliaryModule` + serializable `ObjectiveConfig`
   supporting `jepa_mode`, `mask_mode`, `jepa_context_mode`, `ema_target_scope`, `transition_mode`, `reg_mode`,
   Kaifeng's `MetricBoundaryLoss`, Yushu's `HardSunlight` upweighting + contrastive margin,
   Angela's directionally balanced `FocalEventLoss` with Hard-Negative Mining, and Representation-Sufficiency
   bidirectional height interventions + Dominant-Occluder Dropout).
3. Streaming intra-day state threading (`carry_state_from_prev_chunk`), exact resumable training checkpoints
   (`10 steps + resume 10 steps == continuous 20 steps`), finite-gradient gate before `optimizer.step()`,
   cosine LR schedule, per-step `collapse_diagnostics.jsonl` logging, and real `HostResourceGuard` enforcement.
4. Constant-RAM (`< 64 MiB`) disk-backed streaming `calibrate_model_on_split` scaling to `all_valid`
   12-tile (`> 6B` point-frames) calibration without `logits_chunks` list accumulation.
5. Three-arm matched `benchmark_training_speed` (`baseline_raw_fp32`, `fast_shard_fp32`,
   `fast_shard_bf16_mixed`) verifying exact domain/candidate-ID parity and rejecting `timed_iters <= 0`.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import platform
import random
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import _BaseRindModel, compute_head_parameter_norms
from .objectives_jepa import (
    ObjectiveConfig,
    RindJepaAuxiliaryModule,
    append_collapse_diagnostics_jsonl,
    apply_label_budget_view,
    compute_candidate_face_perturbation_loss,
    compute_focal_event_hard_negative_loss,
    compute_hard_sunlight_and_contrastive_loss,
    compute_latent_health_diagnostics,
    compute_metric_boundary_loss,
)
from .split_contract import FitFirewallToken, SplitContract
from .system_guard import HostResourceGuard, sample_host_resources


def get_process_rss_mib() -> float:
    """Return current process resident memory in MiB via `system_guard`."""
    try:
        snap = sample_host_resources()
        return float(snap.process_working_set_bytes) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    pin_and_async: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if pin_and_async and device.type == "cuda" and v.device.type == "cpu":
                pinned = v if v.is_pinned() else v.pin_memory()
                out[k] = pinned.to(device, non_blocking=True)
            else:
                out[k] = v.to(device)
        else:
            out[k] = v
    return out


def compute_fullday_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    has_event_gate: bool = True,
    aux_bce_weight: float = 0.25,
    transition_loss_weight: float = 0.35,
    focal_gamma: float = 2.0,
    metric_boundary_weight: float = 0.20,
    hard_sunlight_weight: float = 0.20,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute masked full-day state + hard-sunlight + `<0.5m` metric boundary + hard-negative transition loss."""
    _ = focal_gamma
    logits = outputs["logits"].to(torch.float32)
    logits_bce = outputs["logits_bce"].to(torch.float32)
    truth = batch["truth"]
    valid_mask = batch["valid_mask"].to(torch.bool) & (truth >= 0)

    if logits.ndim == 3 and valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(0)
        truth = truth.unsqueeze(0)
    elif logits.ndim == 2 and valid_mask.ndim == 3 and valid_mask.shape[0] == 1:
        valid_mask = valid_mask.squeeze(0)
        truth = truth.squeeze(0)

    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        zero = logits.sum() * 0.0
        return zero, {
            "total_loss": 0.0,
            "state_bce_loss": 0.0,
            "aux_bce_head_loss": 0.0,
            "transition_focal_loss": 0.0,
            "metric_boundary_near05_loss": 0.0,
            "hard_sunlight_loss": 0.0,
            "hard_negative_stable_loss": 0.0,
            "valid_point_frames": 0.0,
            "same_day_transition_pairs": 0.0,
        }

    truth_f_full = (truth > 0).to(torch.float32)
    targets_f32 = truth_f_full[valid_mask]
    state_bce = F.binary_cross_entropy_with_logits(logits[valid_mask], targets_f32)
    eff_aux_w = float(aux_bce_weight) if bool(has_event_gate) else 0.0
    if eff_aux_w > 0.0:
        aux_bce = F.binary_cross_entropy_with_logits(logits_bce[valid_mask], targets_f32)
        combined_state = (state_bce + eff_aux_w * aux_bce) / (1.0 + eff_aux_w)
    else:
        aux_bce = state_bce.detach()
        combined_state = state_bce

    direct_bce = None
    if "logits_direct" in outputs:
        direct = outputs["logits_direct"].float().reshape_as(valid_mask)
        direct_bce = F.binary_cross_entropy_with_logits(direct[valid_mask], targets_f32)
        combined_state = (combined_state + direct_bce) * 0.5

    probs = torch.sigmoid(logits)
    same_day_adj = batch.get("same_day_adjacent_mask")
    carry_prev = bool(batch.get("carry_state_from_prev_chunk", False) and batch.get("prev_frame_same_day", False))

    trans_loss, trans_pairs_n, hn_diag = compute_focal_event_hard_negative_loss(
        probs,
        outputs.get("event_gate") if has_event_gate else None,
        truth_f_full,
        valid_mask,
        same_day_adj,
        prev_chunk_prob=batch.get("prev_chunk_last_prob"),
        prev_chunk_truth=batch.get("prev_chunk_last_truth"),
        prev_chunk_valid=batch.get("prev_chunk_last_valid"),
        carry_prev=carry_prev,
    )

    if has_event_gate and "logits_boundary" in outputs and float(metric_boundary_weight) > 0.0:
        mb_loss, mb_diag = compute_metric_boundary_loss(
            outputs["logits_boundary"],
            truth_f_full,
            valid_mask,
            batch,
            same_day_adj=same_day_adj,
        )
    else:
        mb_loss = logits.new_zeros(())
        mb_diag = {"metric_boundary_near05_loss": 0.0, "metric_boundary_grad_loss": 0.0}

    if float(hard_sunlight_weight) > 0.0 and "candidate_mask" in batch:
        hs_loss, hs_diag = compute_hard_sunlight_and_contrastive_loss(
            logits,
            truth_f_full,
            valid_mask,
            batch.get("candidate_mask"),
        )
    else:
        hs_loss = logits.new_zeros(())
        hs_diag = {"hard_sunlight_loss": 0.0, "hard_sunlight_fraction": 0.0}

    total_loss = (
        combined_state
        + float(transition_loss_weight) * trans_loss
        + float(metric_boundary_weight) * mb_loss
        + float(hard_sunlight_weight) * hs_loss
    )
    return total_loss, {
        "total_loss": float(total_loss.detach().item()),
        "state_bce_loss": float(state_bce.detach().item()),
        "direct_state_bce": float(direct_bce.detach()) if direct_bce is not None else 0.0,
        "aux_bce_head_loss": float(aux_bce.detach().item()),
        "transition_focal_loss": float(trans_loss.detach().item()),
        "metric_boundary_near05_loss": float(mb_diag["metric_boundary_near05_loss"]),
        "hard_sunlight_loss": float(hs_diag["hard_sunlight_loss"]),
        "hard_negative_stable_loss": float(hn_diag["hard_negative_stable_loss"]),
        "transition_pos_count": float(hn_diag.get("transition_pos_count", 0.0)),
        "transition_neg_count": float(hn_diag.get("transition_neg_count", 0.0)),
        "valid_point_frames": float(valid_count),
        "same_day_transition_pairs": float(trans_pairs_n),
    }


def directed_event_matches(truth, prediction, valid, stamps, cadence=120.):
    """Pair same-direction transitions within uninterrupted, jointly observed runs."""
    links = valid[:-1] & valid[1:] & (np.diff(stamps)==cadence)[:,None]
    gt = np.diff(truth.astype(np.int8),axis=0)
    pr = np.diff(prediction.astype(np.int8),axis=0)
    counts = dict(true_events=0,predicted_events=0,matched_events=0,missed_events=0,spurious_events=0)
    lags = []
    for col in np.flatnonzero(np.any(links & ((gt!=0)|(pr!=0)),axis=0)):
        cuts = np.flatnonzero(~links[:,col])
        for a,b in zip(np.r_[0,cuts+1],np.r_[cuts,len(links)]):
            for direction in (-1,1):
                actual = np.flatnonzero(gt[a:b,col]==direction)+a+1
                predicted = np.flatnonzero(pr[a:b,col]==direction)+a+1
                counts["true_events"] += len(actual)
                counts["predicted_events"] += len(predicted)
                if not (len(actual) and len(predicted)):
                    continue
                # Monotone maximum-cardinality, minimum absolute timing error.
                small,large = (actual,predicted) if len(actual)<=len(predicted) else (predicted,actual)
                dist = np.abs(stamps[small,None]-stamps[large])
                cost = np.full((len(small)+1,len(large)+1),np.inf)
                cost[0] = 0.
                take = np.zeros_like(cost,dtype=bool)
                for i in range(1,len(small)+1):
                    match = cost[i-1,:-1]+dist[i-1]
                    cost[i,1:] = np.minimum.accumulate(match)
                    take[i,1:] = match < cost[i,:-1]
                i,j = len(small),len(large)
                while i:
                    if take[i,j]:
                        offset = float(stamps[large[j-1]]) - float(stamps[small[i-1]])
                        lags.append(offset if small is actual else -offset)
                        i -= 1
                    j -= 1
    counts["matched_events"] = len(lags)
    counts["missed_events"] = counts["true_events"]-len(lags)
    counts["spurious_events"] = counts["predicted_events"]-len(lags)
    return counts,lags


def compute_long_day_lag_diagnostics(
    probs_tn: np.ndarray,
    truth_tn: np.ndarray,
    valid_tn: np.ndarray,
    day_ids_t: Sequence[Any],
    *,
    threshold: float = 0.5,
    cadence_seconds: float = 120.0,
    timestamps_unix_s=None,
) -> Dict[str, Optional[float]]:
    """Compute long-day (`T >= 400`, or longest day available) drift and transition-lag diagnostics."""
    probs = np.asarray(probs_tn, dtype=np.float64)
    truth = np.asarray(truth_tn, dtype=np.int8)
    valid = np.asarray(valid_tn, dtype=bool) & (truth >= 0)
    days = [str(d) for d in day_ids_t]
    t_total = int(probs.shape[0])
    stamps = np.asarray(timestamps_unix_s) if timestamps_unix_s is not None else np.arange(t_total)*cadence_seconds
    valid &= np.isfinite(probs)
    counts = dict(true_events=0,predicted_events=0,matched_events=0,missed_events=0,spurious_events=0)
    if t_total == 0 or len(days) != t_total:
        return {
            **counts,
            "signed_transition_lag_seconds_histogram": {},
            "long_day_count": 0,
            "max_day_frames": 0,
            "early_half_accuracy": None,
            "late_half_accuracy": None,
            "late_minus_early_accuracy_drift": None,
            "mean_transition_step_lag_frames": None,
            "mean_transition_lag_seconds": None,
        }

    spans: List[Tuple[int, int]] = []
    s0 = 0
    for i in range(1, t_total):
        if days[i] != days[s0]:
            spans.append((s0, i))
            s0 = i
    spans.append((s0, t_total))

    max_len = max((b - a) for a, b in spans)
    long_spans = [(a, b) for a, b in spans if (b - a) >= 400]
    if not long_spans:
        long_spans = [(a, b) for a, b in spans if (b - a) == max_len]

    early_correct = 0
    early_total = 0
    late_correct = 0
    late_total = 0
    lag_steps: List[float] = []
    pred_bin = probs >= float(threshold)
    gt_bin = truth == 1

    for a, b in long_spans:
        d_len = b - a
        if d_len < 2:
            continue
        mid = a + d_len // 2
        v_early = valid[a:mid]
        if np.any(v_early):
            early_correct += int(np.sum(pred_bin[a:mid][v_early] == gt_bin[a:mid][v_early]))
            early_total += int(np.sum(v_early))
        v_late = valid[mid:b]
        if np.any(v_late):
            late_correct += int(np.sum(pred_bin[mid:b][v_late] == gt_bin[mid:b][v_late]))
            late_total += int(np.sum(v_late))

        part, seconds = directed_event_matches(gt_bin[a:b],pred_bin[a:b],valid[a:b],stamps[a:b],cadence_seconds)
        for key,value in part.items():
            counts[key] += value
        lag_steps.extend(np.asarray(seconds)/cadence_seconds)

    early_acc = (float(early_correct) / float(early_total)) if early_total > 0 else None
    late_acc = (float(late_correct) / float(late_total)) if late_total > 0 else None
    drift = (late_acc - early_acc) if (early_acc is not None and late_acc is not None) else None
    mean_lag = float(np.mean(np.abs(lag_steps))) if lag_steps else None

    return {
        **counts,
        "signed_transition_lag_seconds_histogram": {str(int(k)): int(v) for k, v in zip(*np.unique(np.asarray(lag_steps)*cadence_seconds, return_counts=True))},
        "long_day_count": len(long_spans),
        "max_day_frames": int(max_len),
        "early_half_accuracy": early_acc,
        "late_half_accuracy": late_acc,
        "late_minus_early_accuracy_drift": drift,
        "mean_transition_step_lag_frames": mean_lag,
        "mean_transition_lag_seconds": (mean_lag * float(cadence_seconds)) if mean_lag is not None else None,
    }


def maybe_compile_model(model: nn.Module, enable_compile: bool = False) -> Tuple[nn.Module, bool]:
    if not enable_compile or not hasattr(torch, "compile"):
        return model, False
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        return compiled, True
    except Exception:
        return model, False


FORBIDDEN_LABEL_KEYS: Tuple[str, ...] = (
    "truth",
    "valid_mask",
    "boundary_distance_m",
    "boundary_distance_valid",
    "response_5x5_bitpack",
    "prev_frame_truth",
    "prev_frame_valid_mask",
    "prev_frame_valid",
    "prev_frame_boundary_distance_m",
    "prev_frame_boundary_distance_valid",
    "prev_frame_response_5x5_bitpack",
    "label_budget_mask",
)


class LabelReadForbiddenBatchView(Mapping[str, Any]):
    """Read-only batch firewall raising `RuntimeError` on any attempt to access label tensors."""

    def __init__(self, inner: Mapping[str, Any]) -> None:
        self._inner = inner

    def __getitem__(self, key: str) -> Any:
        if key in FORBIDDEN_LABEL_KEYS:
            raise RuntimeError(f"LabelReadForbiddenBatchView blocked forbidden label read: {key!r}")
        return self._inner[key]

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if key in FORBIDDEN_LABEL_KEYS:
            raise RuntimeError(f"LabelReadForbiddenBatchView blocked forbidden label `.get({key!r})`")
        return self._inner.get(key, default)

    def __contains__(self, key: object) -> bool:
        if key in FORBIDDEN_LABEL_KEYS:
            return False
        return key in self._inner

    def __iter__(self) -> Iterator[str]:
        for k in self._inner:
            if k not in FORBIDDEN_LABEL_KEYS:
                yield k

    def __len__(self) -> int:
        return sum(1 for k in self._inner if k not in FORBIDDEN_LABEL_KEYS)


def _assert_finite_gradients(*modules: nn.Module) -> None:
    for mod in modules:
        for name, p in mod.named_parameters():
            if p.grad is not None and not bool(torch.isfinite(p.grad).all().item()):
                raise RuntimeError(
                    f"Non-finite gradient detected on parameter `{name}` before optimizer.step()!"
                )


def _resolve_label_free_obs_bt_n(
    batch_view: Mapping[str, Any], logits_shape: Sequence[int], device: torch.device
) -> torch.Tensor:
    raw_obs = batch_view.get("receiver_obs_mask")
    if raw_obs is None:
        return torch.ones(tuple(int(x) for x in logits_shape), dtype=torch.bool, device=device)
    obs = raw_obs.to(device=device, dtype=torch.bool)
    if len(logits_shape) == 2:
        t_len, n_pts = int(logits_shape[0]), int(logits_shape[1])
        if obs.ndim == 1:
            return obs.unsqueeze(0).expand(t_len, n_pts)
        if obs.ndim == 2:
            return obs.expand(t_len, n_pts)
        if obs.ndim == 3:
            return obs[0].expand(t_len, n_pts)
    else:
        bsz, t_len, n_pts = int(logits_shape[0]), int(logits_shape[1]), int(logits_shape[2])
        if obs.ndim == 1:
            return obs.view(1, 1, n_pts).expand(bsz, t_len, n_pts)
        if obs.ndim == 2:
            if int(obs.shape[0]) == t_len and int(obs.shape[1]) == n_pts and bsz == 1:
                return obs.unsqueeze(0)
            return obs.unsqueeze(1).expand(bsz, t_len, n_pts)
        if obs.ndim == 3:
            return obs.expand(bsz, t_len, n_pts)
    return torch.ones(tuple(int(x) for x in logits_shape), dtype=torch.bool, device=device)


def _capture_portable_rng_states() -> Dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and CUDA RNG states in `weights_only=True`-safe primitives."""
    py_ver, py_internal, py_gauss = random.getstate()
    np_alg, np_keys, np_pos, np_has_gauss, np_cached_gauss = np.random.get_state()
    cuda_states: List[torch.Tensor] = []
    if torch.cuda.is_available():
        try:
            cuda_states = [s.detach().cpu() for s in torch.cuda.get_rng_state_all()]
        except Exception:
            cuda_states = []
    return {
        "python_rng": {
            "version": int(py_ver),
            "internal": [int(x) for x in py_internal],
            "gauss_next": float(py_gauss) if py_gauss is not None else None,
        },
        "numpy_rng": {
            "alg": str(np_alg),
            "keys": torch.from_numpy(np.asarray(np_keys, dtype=np.int64)).cpu(),
            "pos": int(np_pos),
            "has_gauss": int(np_has_gauss),
            "cached_gaussian": float(np_cached_gauss),
        },
        "torch_cpu_rng": torch.get_rng_state().detach().cpu(),
        "torch_cuda_rng_all": cuda_states,
    }


def _restore_portable_rng_states(rng_pack: Mapping[str, Any]) -> None:
    py_d = rng_pack.get("python_rng")
    if isinstance(py_d, Mapping):
        random.setstate(
            (
                int(py_d["version"]),
                tuple(int(x) for x in py_d["internal"]),
                py_d["gauss_next"],
            )
        )
    np_d = rng_pack.get("numpy_rng")
    if isinstance(np_d, Mapping):
        keys_u32 = np_d["keys"].detach().cpu().numpy().astype(np.uint32)
        np.random.set_state(
            (
                str(np_d["alg"]),
                keys_u32,
                int(np_d["pos"]),
                int(np_d["has_gauss"]),
                float(np_d["cached_gaussian"]),
            )
        )
    cpu_rng = rng_pack.get("torch_cpu_rng")
    if torch.is_tensor(cpu_rng):
        torch.set_rng_state(cpu_rng.detach().cpu().to(torch.uint8))
    cuda_all = rng_pack.get("torch_cuda_rng_all")
    if isinstance(cuda_all, list) and cuda_all and torch.cuda.is_available():
        try:
            cur_n = torch.cuda.device_count()
            if len(cuda_all) == cur_n:
                torch.cuda.set_rng_state_all([s.detach().cpu().to(torch.uint8) for s in cuda_all])
        except Exception:
            pass


def save_training_resume_state(
    path: Path | str,
    *,
    model: _BaseRindModel,
    aux_jepa: RindJepaAuxiliaryModule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch_idx: int,
    step_in_epoch: int,
    total_steps: int,
    total_valid_pf: int,
    max_latent_seen: float,
    nonfinite_seen: int,
    consumed_fit_tiles: Sequence[str],
    epoch_logs: Sequence[Mapping[str, Any]],
    ep_loss_sum: float,
    ep_jepa_sum: float,
    ep_reg_sum: float,
    ep_pf: int,
    prev_latent: Optional[torch.Tensor],
    prev_conv: Optional[torch.Tensor],
    prev_last_prob: Optional[torch.Tensor],
    prev_last_truth: Optional[torch.Tensor],
    prev_last_valid: Optional[torch.Tensor],
    prev_last_obs: Optional[torch.Tensor] = None,
    batch_schedule: Optional[Mapping[str, Any]] = None,
    last_latent_health: Mapping[str, Any],
    last_aux_stats: Mapping[str, Any],
    objective_config: ObjectiveConfig,
    expected_split_sha256: str,
    seed: int,
    cf_subgroup_size: int = 0,
    cf_every_n_steps: int = 2,
) -> Path:
    """Atomically write complete resumable training state (`P2-5`, `weights_only=True` safe)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    payload: Dict[str, Any] = {
        "schema": "rind.training.resume.v1",
        "horizon_history": {k:v.detach().cpu() for k,v in getattr(aux_jepa,"horizon_history",{}).items()},
        "transition_history": {k:v.cpu() for k,v in getattr(aux_jepa,"transition_history",{}).items()},
        "epoch_diagnostics": getattr(aux_jepa,"epoch_diagnostics",{}),
        "horizon_history_key": getattr(aux_jepa,"history_key",None),
        "batch_schedule": dict(batch_schedule) if batch_schedule is not None else None,
        "expected_split_sha256": str(expected_split_sha256),
        "seed": int(seed),
        "model_type": model.model_type,
        "model_config": model.get_model_config(),
        "objective_config": objective_config.to_dict(),
        "cf_config": {"cf_subgroup_size": int(cf_subgroup_size), "cf_every_n_steps": int(cf_every_n_steps)},
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "normalizer_tile_ids": list(model.consumed_normalizer_tile_ids),
        "aux_jepa_state_dict": {k: v.detach().cpu() for k, v in aux_jepa.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": grad_scaler.state_dict() if grad_scaler is not None else None,
        "epoch_idx": int(epoch_idx),
        "step_in_epoch": int(step_in_epoch),
        "total_steps": int(total_steps),
        "total_valid_pf": int(total_valid_pf),
        "max_latent_seen": float(max_latent_seen),
        "nonfinite_seen": int(nonfinite_seen),
        "consumed_fit_tiles": sorted({str(t) for t in consumed_fit_tiles}),
        "epoch_logs": [dict(x) for x in epoch_logs],
        "ep_loss_sum": float(ep_loss_sum),
        "ep_jepa_sum": float(ep_jepa_sum),
        "ep_reg_sum": float(ep_reg_sum),
        "ep_pf": int(ep_pf),
        "prev_latent": prev_latent.detach().cpu() if prev_latent is not None else None,
        "prev_conv": prev_conv.detach().cpu() if prev_conv is not None else None,
        "prev_last_prob": prev_last_prob.detach().cpu() if prev_last_prob is not None else None,
        "prev_last_truth": prev_last_truth.detach().cpu() if prev_last_truth is not None else None,
        "prev_last_obs": prev_last_obs.detach().cpu() if prev_last_obs is not None else None,
        "prev_last_valid": prev_last_valid.detach().cpu() if prev_last_valid is not None else None,
        "last_latent_health": dict(last_latent_health),
        "last_aux_stats": dict(last_aux_stats),
        "rng_states": _capture_portable_rng_states(),
    }
    torch.save(payload, tmp_path)
    tmp_path.replace(out_path)
    return out_path


def load_training_resume_state(
    path: Path | str,
    *,
    expected_split_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and verify resumable training state (`weights_only=True`)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Resume state checkpoint not found: {p}")
    payload = torch.load(p, map_location="cpu", weights_only=True)
    if payload.get("schema") != "rind.training.resume.v1":
        raise RuntimeError(f"Unsupported resume checkpoint schema: {payload.get('schema')!r}")
    if (
        expected_split_sha256 is not None
        and payload.get("expected_split_sha256") != expected_split_sha256
    ):
        raise RuntimeError(
            f"Resume split_sha256 mismatch: {payload.get('expected_split_sha256')} != {expected_split_sha256}"
        )
    return dict(payload)


def _validate_training_resume_configuration(
    saved_state: Mapping[str, Any],
    model: _BaseRindModel,
    *,
    cf_subgroup_size: int = 0,
    cf_every_n_steps: int = 2,
) -> None:
    """Reject semantic changes even when parameter tensor shapes still match."""
    defaults = {"film_beta_bound": 0.0, "attn_qk_norm": False, "cand_readout": "attn_only"}
    saved_model_config = dict(saved_state.get("model_config") or {})
    current_model_config = dict(model.get_model_config())
    # Legacy checkpoints predate these optional switches and mean the old defaults.
    for key, value in defaults.items():
        if key in current_model_config or key in saved_model_config:
            saved_model_config.setdefault(key, value)
            current_model_config.setdefault(key, value)
    if saved_model_config != current_model_config:
        changed = sorted(key for key in saved_model_config.keys() | current_model_config.keys()
                         if saved_model_config.get(key) != current_model_config.get(key))
        raise ValueError(f"resume model configuration changed: {', '.join(changed)}")
    saved_cf = dict(saved_state.get("cf_config") or {"cf_subgroup_size": 0, "cf_every_n_steps": 2})
    current_cf = {"cf_subgroup_size": int(cf_subgroup_size), "cf_every_n_steps": int(cf_every_n_steps)}
    if saved_cf != current_cf:
        raise ValueError("resume CF configuration changed")


def _resolve_objective_config_for_training(
    *,
    is_rind_cand: bool,
    objective_config: Optional[ObjectiveConfig],
    jepa_mode: Optional[str],
    reg_mode: Optional[str],
    mask_mode: Optional[str],
    mask_ratio: Optional[float],
    spatial_block_size: Optional[int],
    jepa_context_mode: Optional[str],
    transition_mode: Optional[str],
    ema_target_scope: Optional[str],
    pure_supervised: Optional[bool],
    label_budget_fraction: Optional[float],
    label_budget_seed: Optional[int],
    counterfactual_weight: Optional[float],
    dominant_occluder_dropout_prob: Optional[float],
) -> ObjectiveConfig:
    if objective_config is not None:
        base = ObjectiveConfig.from_dict(objective_config.to_dict())
    else:
        default_jepa = str(jepa_mode) if jepa_mode is not None else ("hybrid" if is_rind_cand else "none")
        default_reg = str(reg_mode) if reg_mode is not None else ("sigreg" if is_rind_cand else "none")
        base = ObjectiveConfig(
            jepa_mode=default_jepa,
            reg_mode=default_reg,
            counterfactual_weight=float(counterfactual_weight if counterfactual_weight is not None else 0.08),
        )

    kw = base.to_dict()
    if jepa_mode is not None:
        kw["jepa_mode"] = str(jepa_mode)
    if reg_mode is not None:
        kw["reg_mode"] = str(reg_mode)
    if mask_mode is not None:
        kw["mask_mode"] = str(mask_mode)
    if mask_ratio is not None:
        kw["mask_ratio"] = float(mask_ratio)
    if spatial_block_size is not None:
        kw["spatial_block_size"] = int(spatial_block_size)
    if jepa_context_mode is not None:
        kw["jepa_context_mode"] = str(jepa_context_mode)
    if transition_mode is not None:
        kw["transition_mode"] = str(transition_mode)
    if ema_target_scope is not None:
        kw["ema_target_scope"] = str(ema_target_scope)
    if pure_supervised is not None:
        kw["pure_supervised"] = bool(pure_supervised)
    if label_budget_fraction is not None:
        kw["label_budget_fraction"] = float(label_budget_fraction)
    if label_budget_seed is not None:
        kw["label_budget_seed"] = int(label_budget_seed)
    if counterfactual_weight is not None:
        kw["counterfactual_weight"] = float(counterfactual_weight)
    if dominant_occluder_dropout_prob is not None:
        kw["dominant_occluder_dropout_prob"] = float(dominant_occluder_dropout_prob)
    return ObjectiveConfig.from_dict(kw)


def fit_model_fullday(
    model: _BaseRindModel,
    batch_iterator_fn: Callable[[int], Iterable[Dict[str, Any]]],
    *,
    firewall_token: FitFirewallToken,
    expected_split_sha256: str,
    epochs: int = 3,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    device: str | torch.device = "cpu",
    use_bf16_amp: bool = True,
    pin_and_async: bool = True,
    prepare_gpu_tokens_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    resource_guard: Optional[HostResourceGuard] = None,
    seed: int = 20261111,
    objective_config: Optional[ObjectiveConfig] = None,
    jepa_mode: Optional[str] = None,
    reg_mode: Optional[str] = None,
    mask_mode: Optional[str] = None,
    mask_ratio: Optional[float] = None,
    spatial_block_size: Optional[int] = None,
    jepa_context_mode: Optional[str] = None,
    transition_mode: Optional[str] = None,
    ema_target_scope: Optional[str] = None,
    pure_supervised: Optional[bool] = None,
    label_budget_fraction: Optional[float] = None,
    label_budget_seed: Optional[int] = None,
    dominant_occluder_dropout_prob: Optional[float] = None,
    supervise_labels: bool = True,
    counterfactual_weight: Optional[float] = None,
    cf_every_n_steps: int = 2,
    cf_subgroup_size: int = 0,
    collapse_diagnostics_jsonl_path: Optional[Path | str] = None,
    diagnostics_every_steps: int = 8,
    save_every_steps: int = 0,
    resume_state_out: Optional[Path | str] = None,
    resume_from: Optional[Path | str] = None,
    batch_schedule: Optional[Mapping[str, Any]] = None,
    max_train_steps: int = 0,
) -> Dict[str, Any]:
    """Train any unified RIND model strictly on `fit` tiles with `Mask+X` ST-JEPA, firewalls & exact resume."""
    if int(cf_subgroup_size) < 0:
        raise ValueError("cf_subgroup_size must be non-negative")
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    dev = torch.device(device)
    guard = resource_guard or HostResourceGuard(device=dev)
    guard.check("fit_model_fullday:init")

    is_rind_cand = str(getattr(model, "model_type", "")).startswith("rind_")
    obj_cfg = _resolve_objective_config_for_training(
        is_rind_cand=is_rind_cand,
        objective_config=objective_config,
        jepa_mode=jepa_mode,
        reg_mode=reg_mode,
        mask_mode=mask_mode,
        mask_ratio=mask_ratio,
        spatial_block_size=spatial_block_size,
        jepa_context_mode=jepa_context_mode,
        transition_mode=transition_mode,
        ema_target_scope=ema_target_scope,
        pure_supervised=pure_supervised,
        label_budget_fraction=label_budget_fraction,
        label_budget_seed=label_budget_seed,
        counterfactual_weight=counterfactual_weight,
        dominant_occluder_dropout_prob=dominant_occluder_dropout_prob,
    )
    h_dim = int(model.get_model_config().get("hidden_dim", 96))

    model.to(dev)
    model.train()

    aux_jepa = RindJepaAuxiliaryModule(h_dim, objective_config=obj_cfg).to(dev)
    aux_jepa.bind_student_backbone(model)
    aux_jepa.to(dev)
    if not bool(supervise_labels) and aux_jepa.use_response_teacher:
        raise ValueError(
            f"supervise_labels=False requires a label-free jepa_mode ('ema_geom' or 'none'), got {obj_cfg.jepa_mode!r}."
        )
    aux_jepa.train()

    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)

    amp_enabled = bool(
        use_bf16_amp
        and dev.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    trainable_params = [
        p for p in [*model.parameters(), *aux_jepa.parameters()] if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=float(lr), weight_decay=float(weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(epochs), 1), eta_min=float(lr) * 0.1
    )

    epoch_logs: List[Dict[str, Any]] = []
    consumed_fit_tiles: set[str] = set()
    total_valid_pf = 0
    total_steps = 0
    max_latent_seen = 0.0
    nonfinite_seen = 0
    last_latent_health: Dict[str, Any] = {
        "sample_count": 0,
        "per_dim_std_min": None,
        "effective_rank": None,
        "singular_value_top1_ratio": None,
        "off_diag_cov_rms": None,
        "pairwise_latent_dist_mean": None,
    }
    last_aux_stats: Dict[str, float] = {
        "jepa_pred_loss": 0.0,
        "jepa_trans_vel_loss": 0.0,
        "dual_path_consistency_loss": 0.0,
        "anti_collapse_reg_loss": 0.0,
        "jepa_context_sensitivity_dx": 0.0,
        "jepa_mask_ratio_actual": 0.0,
        "candidate_face_perturbation_loss": 0.0,
    }

    resume_epoch_idx = 0
    resume_step_in_epoch = 0
    resume_rng_pending: Optional[Mapping[str, Any]] = None
    resume_ep_state: Optional[Dict[str, Any]] = None

    if resume_from is not None:
        r_state = load_training_resume_state(resume_from, expected_split_sha256=expected_split_sha256)
        _validate_training_resume_configuration(
            r_state, model, cf_subgroup_size=cf_subgroup_size, cf_every_n_steps=cf_every_n_steps
        )
        saved_schedule = r_state.get("batch_schedule")
        if saved_schedule is not None and saved_schedule != batch_schedule:
            raise ValueError("resume logical batch schedule changed")
        if saved_schedule is None and batch_schedule and batch_schedule.get("objective_window_frames"):
            raise ValueError("resume has no fixed logical batch schedule")
        model.load_state_dict(r_state["model_state_dict"], strict=True)
        model.record_normalizer_tiles(r_state.get("normalizer_tile_ids") or [])
        aux_jepa.load_state_dict(r_state["aux_jepa_state_dict"], strict=True)
        aux_jepa.horizon_history = {k:v.to(dev) for k,v in r_state.get("horizon_history",{}).items()}
        aux_jepa.transition_history = {k:v.to(dev) for k,v in r_state.get("transition_history",{}).items()}
        aux_jepa.epoch_diagnostics = r_state.get("epoch_diagnostics",{})
        aux_jepa.history_key = r_state.get("horizon_history_key")
        optimizer.load_state_dict(r_state["optimizer_state_dict"])
        scheduler.load_state_dict(r_state["scheduler_state_dict"])
        resume_epoch_idx = int(r_state["epoch_idx"])
        resume_step_in_epoch = int(r_state["step_in_epoch"])
        total_steps = int(r_state["total_steps"])
        total_valid_pf = int(r_state["total_valid_pf"])
        max_latent_seen = float(r_state["max_latent_seen"])
        nonfinite_seen = int(r_state["nonfinite_seen"])
        consumed_fit_tiles = {str(t) for t in r_state.get("consumed_fit_tiles", [])}
        epoch_logs = [dict(x) for x in r_state.get("epoch_logs", [])]
        last_latent_health = dict(r_state.get("last_latent_health", last_latent_health))
        last_aux_stats = dict(r_state.get("last_aux_stats", last_aux_stats))
        resume_rng_pending = r_state.get("rng_states")
        resume_ep_state = {
            "ep_loss_sum": float(r_state.get("ep_loss_sum", 0.0)),
            "ep_jepa_sum": float(r_state.get("ep_jepa_sum", 0.0)),
            "ep_reg_sum": float(r_state.get("ep_reg_sum", 0.0)),
            "ep_pf": int(r_state.get("ep_pf", 0)),
            "prev_latent": r_state["prev_latent"].to(dev) if r_state.get("prev_latent") is not None else None,
            "prev_conv": r_state["prev_conv"].to(dev) if r_state.get("prev_conv") is not None else None,
            "prev_last_prob": r_state["prev_last_prob"].to(dev) if r_state.get("prev_last_prob") is not None else None,
            "prev_last_truth": r_state["prev_last_truth"].to(dev) if r_state.get("prev_last_truth") is not None else None,
            "prev_last_obs": r_state["prev_last_obs"].to(dev) if r_state.get("prev_last_obs") is not None else None,
            "prev_last_valid": r_state["prev_last_valid"].to(dev) if r_state.get("prev_last_valid") is not None else None,
        }

    eff_cf_weight = 0.0 if obj_cfg.pure_supervised else float(obj_cfg.counterfactual_weight)
    diag_stride = max(1, int(diagnostics_every_steps))
    stopped_early = False
    t_start = time.perf_counter()

    for epoch_idx in range(int(epochs)):
        if epoch_idx < resume_epoch_idx:
            continue

        if not hasattr(aux_jepa,"epoch_diagnostics"):
            aux_jepa.epoch_diagnostics = {}
        epoch_diagnostics = aux_jepa.epoch_diagnostics.setdefault(
            str(epoch_idx + 1),
            {"leads": {}, "candidate_face_edits": {}, "cf_observability": {}, "aux_term_sums": {}},
        )
        cf_obs = epoch_diagnostics.setdefault("cf_observability", {})
        for _ck in (
            "scheduled_steps",
            "skipped_steps",
            "nonzero_steps",
            "edit_support_total",
            "flip_support_total",
            "unchanged_support_total",
            "dropout_bern_support_total",
            "secondary_hit_support_total",
        ):
            cf_obs.setdefault(_ck, 0)
        for _fk in (
            "raw_loss_sum",
            "weighted_loss_sum",
            "height_edit_raw_sum",
            "dropout_monotonicity_raw_sum",
            "secondary_occlusion_bce_raw_sum",
        ):
            cf_obs.setdefault(_fk, 0.0)
        aux_sums = epoch_diagnostics.setdefault("aux_term_sums", {})
        for _ak in (
            "sup_loss_sum",
            "multi_horizon_pred_raw_sum",
            "teacher_align_raw_sum",
            "trans_vel_scaled_sum",
            "dual_path_scaled_sum",
            "dx_sum",
            "dx_steps",
        ):
            aux_sums.setdefault(_ak, 0.0)
        if epoch_idx == resume_epoch_idx and resume_ep_state is not None:
            ep_loss_sum = float(resume_ep_state["ep_loss_sum"])
            ep_jepa_sum = float(resume_ep_state["ep_jepa_sum"])
            ep_reg_sum = float(resume_ep_state["ep_reg_sum"])
            ep_steps = int(resume_step_in_epoch)
            ep_pf = int(resume_ep_state["ep_pf"])
            prev_latent = resume_ep_state["prev_latent"]
            prev_conv = resume_ep_state["prev_conv"]
            prev_last_prob = resume_ep_state["prev_last_prob"]
            prev_last_truth = resume_ep_state["prev_last_truth"]
            prev_last_obs = resume_ep_state["prev_last_obs"]
            prev_last_valid = resume_ep_state["prev_last_valid"]
            resume_ep_state = None
        else:
            ep_loss_sum = 0.0
            ep_jepa_sum = 0.0
            ep_reg_sum = 0.0
            ep_steps = 0
            ep_pf = 0
            prev_latent = None
            prev_conv = None
            prev_last_prob = None
            prev_last_truth = None
            prev_last_valid = None
            prev_last_obs = None

        t_ep = time.perf_counter()

        for b_idx, raw_batch in enumerate(batch_iterator_fn(epoch_idx)):
            if epoch_idx == resume_epoch_idx and b_idx < resume_step_in_epoch:
                continue
            if resume_rng_pending is not None:
                _restore_portable_rng_states(resume_rng_pending)
                resume_rng_pending = None

            if int(max_train_steps) > 0 and total_steps >= int(max_train_steps):
                stopped_early = True
                break

            if "ray_candidate_count" in raw_batch:
                from .ray_candidates import record_budget
                record_budget(epoch_diagnostics.setdefault("ray_budget",{}),raw_batch["ray_candidate_count"],
                              raw_batch["az_mask"].shape[-1])
            tile_ids = raw_batch.get("tile_ids")
            if not tile_ids:
                raise RuntimeError("Training batch missing `tile_ids` provenance!")
            firewall_token.verify_tiles(tile_ids, expected_split_sha256)
            for tid in tile_ids:
                consumed_fit_tiles.add(str(tid))

            if ep_steps % 4 == 0:
                guard.check(f"fit_model_fullday:ep{epoch_idx}:step{ep_steps}")

            raw_to_move: Mapping[str, Any] = (
                raw_batch if bool(supervise_labels) else LabelReadForbiddenBatchView(raw_batch)
            )
            batch_dev = move_batch_to_device(raw_to_move, dev, pin_and_async=pin_and_async)
            if prepare_gpu_tokens_fn is not None:
                batch_dev = prepare_gpu_tokens_fn(batch_dev)

            # Enforce strict low-label budget firewall (`P0-2` / `P0-3`: hidden `1-frac` labels zeroed & stripped!)
            if bool(supervise_labels) and obj_cfg.label_budget_fraction < 1.0 - 1e-9:
                batch_dev, _ = apply_label_budget_view(
                    batch_dev,
                    label_budget_fraction=obj_cfg.label_budget_fraction,
                    label_budget_seed=obj_cfg.label_budget_seed,
                    device=dev,
                )

            if bool(batch_dev.get("carry_state_from_prev_chunk", False)) and prev_latent is not None:
                batch_dev["initial_latent_state"] = prev_latent
                if prev_last_obs is not None:
                    batch_dev["prev_chunk_obs_mask"] = prev_last_obs
                if prev_conv is not None:
                    batch_dev["initial_conv_state"] = prev_conv
                if bool(supervise_labels) and prev_last_prob is not None:
                    batch_dev["prev_chunk_last_prob"] = prev_last_prob
                    batch_dev["prev_chunk_last_truth"] = prev_last_truth
                    batch_dev["prev_chunk_last_valid"] = prev_last_valid
            else:
                prev_latent = None
                prev_conv = None
                prev_last_prob = None
                prev_last_truth = None
                prev_last_valid = None

            label_free_view = LabelReadForbiddenBatchView(batch_dev)
            jepa_batch_in: Mapping[str, Any] = (
                batch_dev if aux_jepa.use_response_teacher else label_free_view
            )

            if epoch_idx == 0:
                model.update_normalizers(
                    label_free_view,
                    firewall_token=firewall_token,
                    tile_ids=tile_ids,
                    expected_split_sha256=expected_split_sha256,
                )
                aux_jepa.update_normalizers(
                    jepa_batch_in,
                    firewall_token=firewall_token,
                    tile_ids=tile_ids,
                    expected_split_sha256=expected_split_sha256,
                    device=dev,
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=dev.type, dtype=torch.bfloat16, enabled=amp_enabled
            ):
                outputs = model(label_free_view, use_calibrated_temperature=False)

            if obj_cfg.pure_supervised:
                aux_loss = outputs["logits"].new_zeros((), dtype=torch.float32)
                aux_stats = {
                    "jepa_pred_loss": 0.0,
                    "jepa_trans_vel_loss": 0.0,
                    "dual_path_consistency_loss": 0.0,
                    "anti_collapse_reg_loss": 0.0,
                    "jepa_context_sensitivity_dx": 0.0,
                    "jepa_mask_ratio_actual": 0.0,
                }
            else:
                aux_loss, aux_stats = aux_jepa.forward_objectives(
                    outputs, jepa_batch_in, step_index=total_steps
                )

            if bool(supervise_labels):
                sup_loss, stats = compute_fullday_loss(
                    outputs,
                    batch_dev,
                    has_event_gate=bool(getattr(model, "has_event_gate", True)),
                    transition_loss_weight=float(obj_cfg.focal_event_weight),
                    metric_boundary_weight=float(obj_cfg.boundary_metric_weight),
                    hard_sunlight_weight=float(obj_cfg.hard_sun_weight),
                )
                step_valid_pf = int(stats["valid_point_frames"])
            else:
                sup_loss = aux_loss.new_zeros(())
                stats = {"valid_point_frames": 0.0}
                obs_mask_lf = _resolve_label_free_obs_bt_n(
                    label_free_view, outputs["logits"].shape, dev
                )
                step_valid_pf = int(obs_mask_lf.sum().item())

            cf_scheduled_now = bool(
                is_rind_cand
                and eff_cf_weight > 0.0
                and int(cf_every_n_steps) > 0
                and (total_steps % int(cf_every_n_steps) == 0)
            )
            cf_step_stats: Dict[str, float] = {}
            if cf_scheduled_now:
                with torch.amp.autocast(
                    device_type=dev.type, dtype=torch.bfloat16, enabled=amp_enabled
                ):
                    cf_raw = compute_candidate_face_perturbation_loss(
                        model,
                        label_free_view,
                        outputs["probs"],
                        dominant_occluder_dropout_prob=float(obj_cfg.dominant_occluder_dropout_prob),
                        step_seed=total_steps // int(cf_every_n_steps),
                        edit_counts=epoch_diagnostics["candidate_face_edits"],
                        cf_subgroup_size=cf_subgroup_size,
                        cf_step_stats=cf_step_stats,
                    )
                cf_raw_val = float(cf_raw.detach().to(torch.float32).item())
                cf_term = eff_cf_weight * cf_raw.to(torch.float32)
                cf_scalar = float(cf_term.detach().item())
                cf_obs["scheduled_steps"] = int(cf_obs["scheduled_steps"]) + 1
                if abs(cf_raw_val) > 0.0:
                    cf_obs["nonzero_steps"] = int(cf_obs["nonzero_steps"]) + 1
                cf_obs["raw_loss_sum"] = float(cf_obs["raw_loss_sum"]) + cf_raw_val
                cf_obs["weighted_loss_sum"] = float(cf_obs["weighted_loss_sum"]) + cf_scalar
                cf_obs["height_edit_raw_sum"] = float(cf_obs["height_edit_raw_sum"]) + float(
                    cf_step_stats.get("height_edit_raw_loss", 0.0)
                )
                cf_obs["dropout_monotonicity_raw_sum"] = float(
                    cf_obs["dropout_monotonicity_raw_sum"]
                ) + float(cf_step_stats.get("dropout_monotonicity_raw_loss", 0.0))
                cf_obs["secondary_occlusion_bce_raw_sum"] = float(
                    cf_obs["secondary_occlusion_bce_raw_sum"]
                ) + float(cf_step_stats.get("secondary_occlusion_bce_raw_loss", 0.0))
                cf_obs["edit_support_total"] = int(cf_obs["edit_support_total"]) + int(
                    round(cf_step_stats.get("edit_support", 0.0))
                )
                cf_obs["flip_support_total"] = int(cf_obs["flip_support_total"]) + int(
                    round(cf_step_stats.get("flip_support", 0.0))
                )
                cf_obs["unchanged_support_total"] = int(cf_obs["unchanged_support_total"]) + int(
                    round(cf_step_stats.get("unchanged_support", 0.0))
                )
                cf_obs["dropout_bern_support_total"] = int(
                    cf_obs["dropout_bern_support_total"]
                ) + int(round(cf_step_stats.get("dropout_bern_support", 0.0)))
                cf_obs["secondary_hit_support_total"] = int(
                    cf_obs["secondary_hit_support_total"]
                ) + int(round(cf_step_stats.get("secondary_hit_support", 0.0)))
            else:
                cf_raw_val = 0.0
                cf_term = sup_loss.new_zeros(())
                cf_scalar = 0.0
                cf_obs["skipped_steps"] = int(cf_obs["skipped_steps"]) + 1

            loss = sup_loss + aux_loss + cf_term
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"Non-finite training loss encountered: {stats} / {aux_stats}")
            if int(outputs["nonfinite_count"]) > 0:
                raise RuntimeError(
                    f"Non-finite model outputs encountered during training: {outputs['nonfinite_count']}"
                )

            loss.backward()
            _assert_finite_gradients(model, aux_jepa)
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(grad_clip_norm))
            optimizer.step()
            if not obj_cfg.pure_supervised:
                aux_jepa.step_ema()

            if "final_latent_state" in outputs and outputs["final_latent_state"] is not None:
                prev_latent = outputs["final_latent_state"].detach()
            if "final_conv_state" in outputs and outputs["final_conv_state"] is not None:
                prev_conv = outputs["final_conv_state"].detach()

            if bool(supervise_labels):
                p_det = outputs["probs"].detach()
                t_det = batch_dev["truth"].detach()
                v_det = (batch_dev["valid_mask"].to(torch.bool) & (t_det >= 0)).detach()
                prev_last_prob = p_det[:, -1] if p_det.ndim == 3 else p_det[-1]
                prev_last_truth = t_det[:, -1] if t_det.ndim == 3 else t_det[-1]
                prev_last_valid = v_det[:, -1] if v_det.ndim == 3 else v_det[-1]
            else:
                v_det = _resolve_label_free_obs_bt_n(
                    label_free_view, outputs["logits"].shape, dev
                ).detach()

            prev_last_obs = _resolve_label_free_obs_bt_n(label_free_view, outputs["logits"].shape, dev).detach()[..., -1, :]
            last_aux_stats = {
                **aux_stats,
                "candidate_face_perturbation_loss": cf_scalar,
                "candidate_face_perturbation_raw_loss": cf_raw_val,
                "cf_scheduled_on_this_step": bool(cf_scheduled_now),
            }
            aux_sums["sup_loss_sum"] = float(aux_sums["sup_loss_sum"]) + float(sup_loss.detach().item())
            aux_sums["multi_horizon_pred_raw_sum"] = float(aux_sums["multi_horizon_pred_raw_sum"]) + float(
                aux_stats.get("multi_horizon_prediction_loss", 0.0) or 0.0
            )
            aux_sums["teacher_align_raw_sum"] = float(aux_sums["teacher_align_raw_sum"]) + float(
                aux_stats.get("teacher_alignment_loss", 0.0) or 0.0
            )
            aux_sums["trans_vel_scaled_sum"] = float(aux_sums["trans_vel_scaled_sum"]) + float(
                aux_stats.get("jepa_trans_vel_loss", 0.0) or 0.0
            )
            aux_sums["dual_path_scaled_sum"] = float(aux_sums["dual_path_scaled_sum"]) + float(
                aux_stats.get("dual_path_consistency_loss", 0.0) or 0.0
            )
            aux_sums["dx_sum"] = float(aux_sums["dx_sum"]) + float(
                aux_stats.get("jepa_context_sensitivity_dx", 0.0) or 0.0
            )
            aux_sums["dx_steps"] = float(aux_sums["dx_steps"]) + 1.0
            for key,support in aux_stats.items():
                if key.startswith("lead_") and key.endswith("_support"):
                    lead = key.split("_")[1]
                    row = epoch_diagnostics["leads"].setdefault(lead,dict(support=0,loss_sum=0.,dx_sum=0.))
                    row["support"] += int(support)
                    for name in ("loss","dx"):
                        row[name+"_sum"] += aux_stats[f"lead_{lead}_{name}"] * support
                        row[name] = row[name+"_sum"]/row["support"] if row["support"] else None
            if ep_steps == 0 or (total_steps % diag_stride == 0):
                last_latent_health = compute_latent_health_diagnostics(
                    outputs.get("latent_direct", outputs["latent_roll"]),
                    v_det,
                )
                if collapse_diagnostics_jsonl_path is not None:
                    append_collapse_diagnostics_jsonl(
                        collapse_diagnostics_jsonl_path,
                        step=total_steps,
                        epoch=epoch_idx,
                        health=last_latent_health,
                        aux_stats=last_aux_stats,
                    )

            ep_loss_sum += float(loss.detach().item())
            ep_jepa_sum += float(aux_stats["jepa_pred_loss"])
            ep_reg_sum += float(aux_stats["anti_collapse_reg_loss"])
            ep_steps += 1
            ep_pf += step_valid_pf
            total_steps += 1
            total_valid_pf += step_valid_pf
            max_latent_seen = max(max_latent_seen, float(outputs["max_latent_norm"]))
            nonfinite_seen += int(outputs["nonfinite_count"])

            should_save_resume = resume_state_out is not None and (
                (int(save_every_steps) > 0 and (total_steps % int(save_every_steps) == 0))
                or (int(max_train_steps) > 0 and total_steps >= int(max_train_steps))
            )
            if should_save_resume:
                save_training_resume_state(
                    resume_state_out,
                    model=model,
                    aux_jepa=aux_jepa,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch_idx=epoch_idx,
                    step_in_epoch=b_idx + 1,
                    total_steps=total_steps,
                    total_valid_pf=total_valid_pf,
                    max_latent_seen=max_latent_seen,
                    nonfinite_seen=nonfinite_seen,
                    consumed_fit_tiles=sorted(consumed_fit_tiles),
                    epoch_logs=epoch_logs,
                    ep_loss_sum=ep_loss_sum,
                    ep_jepa_sum=ep_jepa_sum,
                    ep_reg_sum=ep_reg_sum,
                    ep_pf=ep_pf,
                    prev_latent=prev_latent,
                    prev_conv=prev_conv,
                    prev_last_prob=prev_last_prob,
                    prev_last_truth=prev_last_truth,
                    prev_last_valid=prev_last_valid,
                    prev_last_obs=prev_last_obs,
                    batch_schedule=batch_schedule,
                    last_latent_health=last_latent_health,
                    last_aux_stats=last_aux_stats,
                    objective_config=obj_cfg,
                    expected_split_sha256=expected_split_sha256,
                    seed=seed,
                    cf_subgroup_size=cf_subgroup_size,
                    cf_every_n_steps=cf_every_n_steps,
                )

            if int(max_train_steps) > 0 and total_steps >= int(max_train_steps):
                stopped_early = True
                break

        if stopped_early:
            break

        if resume_rng_pending is not None:
            _restore_portable_rng_states(resume_rng_pending)
            resume_rng_pending = None

        scheduler.step()
        guard.check(f"fit_model_fullday:ep{epoch_idx}:done")
        ep_sec = max(time.perf_counter() - t_ep, 1e-9)
        n_sched = max(int(cf_obs["scheduled_steps"]), 1)
        has_sched = int(cf_obs["scheduled_steps"]) > 0
        cf_obs["mean_raw_scheduled"] = float(cf_obs["raw_loss_sum"]) / n_sched if has_sched else 0.0
        cf_obs["mean_weighted_scheduled"] = (
            float(cf_obs["weighted_loss_sum"]) / n_sched if has_sched else 0.0
        )
        cf_obs["mean_weighted_all_steps"] = float(cf_obs["weighted_loss_sum"]) / max(ep_steps, 1)
        cf_obs["mean_height_edit_raw_scheduled"] = (
            float(cf_obs["height_edit_raw_sum"]) / n_sched if has_sched else 0.0
        )
        cf_obs["mean_dropout_monotonicity_raw_scheduled"] = (
            float(cf_obs["dropout_monotonicity_raw_sum"]) / n_sched if has_sched else 0.0
        )
        cf_obs["mean_secondary_occlusion_bce_raw_scheduled"] = (
            float(cf_obs["secondary_occlusion_bce_raw_sum"]) / n_sched if has_sched else 0.0
        )
        epoch_logs.append(
            {
                "epoch": epoch_idx + 1,
                "steps": ep_steps,
                "mean_loss": ep_loss_sum / max(ep_steps, 1),
                "mean_sup_loss": float(aux_sums["sup_loss_sum"]) / max(ep_steps, 1),
                "mean_jepa_loss": ep_jepa_sum / max(ep_steps, 1),
                "mean_multi_horizon_pred_raw_loss": float(aux_sums["multi_horizon_pred_raw_sum"]) / max(ep_steps, 1),
                "mean_teacher_align_raw_loss": float(aux_sums["teacher_align_raw_sum"]) / max(ep_steps, 1),
                "mean_trans_vel_scaled_loss": float(aux_sums["trans_vel_scaled_sum"]) / max(ep_steps, 1),
                "mean_dual_path_scaled_loss": float(aux_sums["dual_path_scaled_sum"]) / max(ep_steps, 1),
                "mean_anti_collapse_reg_loss": ep_reg_sum / max(ep_steps, 1),
                "mean_cf_weighted_all_steps": cf_obs["mean_weighted_all_steps"],
                "mean_cf_weighted_scheduled": cf_obs["mean_weighted_scheduled"],
                "mean_cf_raw_scheduled": cf_obs["mean_raw_scheduled"],
                "cf_scheduled_steps": int(cf_obs["scheduled_steps"]),
                "cf_skipped_steps": int(cf_obs["skipped_steps"]),
                "cf_nonzero_steps": int(cf_obs["nonzero_steps"]),
                "mean_context_sensitivity_dx": float(aux_sums["dx_sum"]) / max(float(aux_sums["dx_steps"]), 1.0),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "valid_point_frames": ep_pf,
                "valid_point_frames_per_sec": float(ep_pf) / ep_sec,
                "wall_seconds": ep_sec,
            }
        )

        if resume_state_out is not None and not stopped_early:
            save_training_resume_state(
                resume_state_out,
                model=model,
                aux_jepa=aux_jepa,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch_idx=epoch_idx + 1,
                step_in_epoch=0,
                total_steps=total_steps,
                total_valid_pf=total_valid_pf,
                max_latent_seen=max_latent_seen,
                nonfinite_seen=nonfinite_seen,
                consumed_fit_tiles=sorted(consumed_fit_tiles),
                epoch_logs=epoch_logs,
                ep_loss_sum=0.0,
                ep_jepa_sum=0.0,
                ep_reg_sum=0.0,
                ep_pf=0,
                prev_latent=None,
                prev_conv=None,
                prev_last_prob=None,
                prev_last_truth=None,
                prev_last_valid=None,
                batch_schedule=batch_schedule,
                last_latent_health=last_latent_health,
                last_aux_stats=last_aux_stats,
                objective_config=obj_cfg,
                expected_split_sha256=expected_split_sha256,
                seed=seed,
                cf_subgroup_size=cf_subgroup_size,
                cf_every_n_steps=cf_every_n_steps,
            )

    wall_total = max(time.perf_counter() - t_start, 1e-9)
    head_norms = compute_head_parameter_norms(model, aux_jepa_module=aux_jepa)
    return {
        "epochs": int(epochs),
        "jepa_mode": obj_cfg.jepa_mode,
        "reg_mode": obj_cfg.reg_mode,
        "objective_config": obj_cfg.to_dict(),
        "cf_config": {"cf_subgroup_size": int(cf_subgroup_size), "cf_every_n_steps": int(cf_every_n_steps)},
        "supervise_labels": bool(supervise_labels),
        "label_budget_fraction": float(obj_cfg.label_budget_fraction),
        "pure_supervised": bool(obj_cfg.pure_supervised),
        "label_reads_blocked_in_forward_and_ema_geom": True,
        "total_steps": total_steps,
        "total_valid_point_frames": total_valid_pf,
        "valid_point_frames_per_sec": float(total_valid_pf) / wall_total,
        "wall_seconds": wall_total,
        "bf16_amp_active": amp_enabled,
        "max_latent_norm": max_latent_seen,
        "nonfinite_count": nonfinite_seen,
        "consumed_fit_tile_ids": sorted(consumed_fit_tiles),
        "normalizer_tile_ids": model.consumed_normalizer_tile_ids,
        "head_parameter_norms": head_norms,
        "latent_health_diagnostics": last_latent_health,
        "jepa_and_aux_summary": last_aux_stats,
        "epoch_support_diagnostics": getattr(aux_jepa,"epoch_diagnostics",{}),
        "collapse_diagnostics_jsonl_path": str(collapse_diagnostics_jsonl_path) if collapse_diagnostics_jsonl_path else None,
        "resumed_from": str(resume_from) if resume_from else None,
        "resume_state_out": str(resume_state_out) if resume_state_out else None,
        "resource_telemetry": guard.summary(),
        "epoch_logs": epoch_logs,
    }


@torch.no_grad()
def calibrate_model_on_split(
    model: _BaseRindModel,
    calibration_batches: Iterable[Dict[str, Any]],
    *,
    contract: SplitContract,
    device: str | torch.device = "cpu",
    prepare_gpu_tokens_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    resource_guard: Optional[HostResourceGuard] = None,
) -> Dict[str, Any]:
    """Constant-RAM (`< 64 MiB`) disk-backed streaming calibration scaling to 12 `all_valid` tiles."""
    dev = torch.device(device)
    guard = resource_guard or HostResourceGuard(device=dev)
    guard.check("calibrate_model_on_split:init")

    model.to(dev)
    model.eval()

    seen_tiles: set[str] = set()
    prev_latent: Optional[torch.Tensor] = None
    prev_conv: Optional[torch.Tensor] = None
    step_idx = 0
    total_valid_slots = 0

    with tempfile.TemporaryDirectory(prefix="rind_cal_stream_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        logits_bin_path = tmp_dir / "cal_logits_f32.bin"
        truth_bin_path = tmp_dir / "cal_truth_u8.bin"

        with logits_bin_path.open("wb") as f_lg, truth_bin_path.open("wb") as f_gt:
            for raw_batch in calibration_batches:
                tids = list(raw_batch.get("tile_ids", []))
                contract.assert_calibration_allowed(tids)
                for tid in tids:
                    seen_tiles.add(str(tid))

                if step_idx % 8 == 0:
                    guard.check(f"calibrate_model_on_split:step{step_idx}")
                step_idx += 1

                b_dev = move_batch_to_device(raw_batch, dev, pin_and_async=False)
                if prepare_gpu_tokens_fn is not None:
                    b_dev = prepare_gpu_tokens_fn(b_dev)

                if bool(b_dev.get("carry_state_from_prev_chunk", False)) and prev_latent is not None:
                    b_dev["initial_latent_state"] = prev_latent
                    if prev_conv is not None:
                        b_dev["initial_conv_state"] = prev_conv
                else:
                    prev_latent = None
                    prev_conv = None

                out = model(LabelReadForbiddenBatchView(b_dev), use_calibrated_temperature=False)
                if "final_latent_state" in out and out["final_latent_state"] is not None:
                    prev_latent = out["final_latent_state"].detach()
                if "final_conv_state" in out and out["final_conv_state"] is not None:
                    prev_conv = out["final_conv_state"].detach()

                lg = out["logits"].detach().cpu().numpy().astype(np.float32)
                gt = b_dev["truth"].detach().cpu().numpy().astype(np.int8)
                vm = b_dev["valid_mask"].detach().cpu().numpy().astype(bool) & (gt >= 0)
                if lg.ndim == 3 and vm.ndim == 2:
                    lg = lg.squeeze(0)
                if np.any(vm):
                    lg_valid = np.ascontiguousarray(lg[vm], dtype=np.float32)
                    gt_valid = np.ascontiguousarray(gt[vm], dtype=np.uint8)
                    f_lg.write(lg_valid.tobytes())
                    f_gt.write(gt_valid.tobytes())
                    total_valid_slots += int(lg_valid.size)

        guard.check("calibrate_model_on_split:post_collect")
        if total_valid_slots <= 0:
            raise RuntimeError("No valid `calibration` point-frames collected for scalar calibration!")

        all_logits = np.memmap(logits_bin_path, dtype=np.float32, mode="r", shape=(total_valid_slots,))
        all_truth = np.memmap(truth_bin_path, dtype=np.uint8, mode="r", shape=(total_valid_slots,))
        try:
            def nll_at_temp(temp: float) -> float:
                total_nll = 0.0
                chunk = 1_000_000
                inv_t = 1.0 / float(temp)
                for s in range(0, total_valid_slots, chunk):
                    e = min(total_valid_slots, s + chunk)
                    x = np.clip(np.asarray(all_logits[s:e], dtype=np.float64) * inv_t, -60.0, 60.0)
                    y = np.asarray(all_truth[s:e], dtype=np.float64)
                    total_nll += float(
                        np.sum(np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x))))
                    )
                return total_nll / max(float(total_valid_slots), 1.0)

            pre_nll = nll_at_temp(1.0)
            coarse_temps = np.exp(np.linspace(math.log(0.25), math.log(4.0), 81))
            best_temp = float(min(coarse_temps, key=nll_at_temp))
            fine_temps = np.linspace(max(0.1, best_temp * 0.8), best_temp * 1.25, 61)
            best_temp = float(min(fine_temps, key=nll_at_temp))
            post_nll = nll_at_temp(best_temp)

            thresholds = np.linspace(0.30, 0.70, 81)
            tp_arr = np.zeros(len(thresholds), dtype=np.float64)
            fp_arr = np.zeros(len(thresholds), dtype=np.float64)
            fn_arr = np.zeros(len(thresholds), dtype=np.float64)
            tn_arr = np.zeros(len(thresholds), dtype=np.float64)

            inv_best_t = 1.0 / best_temp
            for s in range(0, total_valid_slots, 500_000):
                e = min(total_valid_slots, s + 500_000)
                probs_c = 1.0 / (
                    1.0
                    + np.exp(
                        -np.clip(np.asarray(all_logits[s:e], dtype=np.float64) * inv_best_t, -60.0, 60.0)
                    )
                )
                gt_c = np.asarray(all_truth[s:e]) > 0
                for idx, th in enumerate(thresholds):
                    pr_c = probs_c >= float(th)
                    tp_arr[idx] += float(np.sum(pr_c & gt_c))
                    fp_arr[idx] += float(np.sum(pr_c & (~gt_c)))
                    fn_arr[idx] += float(np.sum((~pr_c) & gt_c))
                    tn_arr[idx] += float(np.sum((~pr_c) & (~gt_c)))
        finally:
            raw_mm_l = getattr(all_logits, "_mmap", None)
            if raw_mm_l is not None:
                raw_mm_l.close()
            raw_mm_t = getattr(all_truth, "_mmap", None)
            if raw_mm_t is not None:
                raw_mm_t.close()
            del all_logits, all_truth

    best_thresh = 0.5
    best_score = -1.0
    for idx, th in enumerate(thresholds):
        tp, fp, fn, tn = tp_arr[idx], fp_arr[idx], fn_arr[idx], tn_arr[idx]
        acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
        f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0
        score = 0.5 * acc + 0.5 * f1
        if score > best_score:
            best_score = score
            best_thresh = float(th)

    model.set_calibration(best_temp, best_thresh)
    guard.check("calibrate_model_on_split:done")
    return {
        "calibration_tile_ids": sorted(seen_tiles),
        "calibration_point_frames": int(total_valid_slots),
        "valid_point_frames": int(total_valid_slots),
        "pre_calibration_bce_nll": pre_nll,
        "post_calibration_bce_nll": post_nll,
        "calibrated_temperature": best_temp,
        "calibrated_threshold": best_thresh,
        "resource_telemetry": guard.summary(),
    }


def _flatten_grads(*modules: nn.Module) -> torch.Tensor:
    parts = [
        p.grad.detach().to(torch.float64).reshape(-1)
        for mod in modules
        for p in mod.parameters()
        if p.grad is not None
    ]
    if not parts:
        return torch.zeros((1,), dtype=torch.float64)
    return torch.cat(parts, dim=0)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = (torch.linalg.norm(a) * torch.linalg.norm(b)).clamp_min(1e-12)
    return float(torch.dot(a, b).item() / denom.item())


def _as_numpy_arr(val: Any) -> np.ndarray:
    if torch.is_tensor(val):
        return val.detach().cpu().numpy()
    return np.asarray(val)


def _verify_matched_domain(b_raw: Mapping[str, Any], a_dev: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify step-0 domain, timestamps, receivers, truth, valid_mask, and candidate_source_ids."""
    if list(b_raw.get("tile_ids", [])) != list(a_dev.get("tile_ids", [])):
        raise RuntimeError(
            f"Tile ID mismatch between baseline ({b_raw.get('tile_ids')}) and shard ({a_dev.get('tile_ids')})!"
        )
    if "timestamps_unix_s" in b_raw and "timestamps_unix_s" in a_dev:
        ts_b = _as_numpy_arr(b_raw["timestamps_unix_s"])
        ts_a = _as_numpy_arr(a_dev["timestamps_unix_s"])
        if not np.array_equal(ts_b, ts_a):
            raise RuntimeError("Timestamp mismatch between raw baseline and shard arm!")

    if "receiver_rc" in b_raw and "receiver_rc" in a_dev:
        rc_b = _as_numpy_arr(b_raw["receiver_rc"])
        rc_a = _as_numpy_arr(a_dev["receiver_rc"])
        if not np.array_equal(rc_b, rc_a):
            raise RuntimeError("receiver_rc mismatch between raw baseline and shard arm!")

    vm_b = _as_numpy_arr(b_raw["valid_mask"]).astype(bool)
    vm_a = _as_numpy_arr(a_dev["valid_mask"]).astype(bool)
    if not np.array_equal(vm_b, vm_a):
        raise RuntimeError("valid_mask mismatch between raw baseline and shard arm!")

    gt_b = _as_numpy_arr(b_raw["truth"])
    gt_a = _as_numpy_arr(a_dev["truth"])
    if not np.array_equal(gt_b[vm_b], gt_a[vm_a]):
        raise RuntimeError("Ground-truth label mismatch on valid points between raw and shard arms!")

    cand_id_mismatch = 0
    if "candidate_source_ids" in b_raw and "candidate_source_ids" in a_dev:
        cid_b = _as_numpy_arr(b_raw["candidate_source_ids"])
        cid_a = _as_numpy_arr(a_dev["candidate_source_ids"])
        cm_b = _as_numpy_arr(b_raw["candidate_mask"]).astype(bool)
        cm_a = _as_numpy_arr(a_dev["candidate_mask"]).astype(bool)
        if not np.array_equal(cm_b, cm_a):
            raise RuntimeError("candidate_mask mismatch between raw CPU and GPU shard arms!")
        cand_id_mismatch = int(np.sum(cid_b[cm_b] != cid_a[cm_a]))
        if cand_id_mismatch != 0:
            raise RuntimeError(
                f"Candidate triangle source_id mismatch ({cand_id_mismatch} slots) between raw CPU and GPU shard!"
            )

    return {
        "tile_ids": list(b_raw.get("tile_ids", [])),
        "valid_point_frames_per_step": int(np.sum(vm_b & (gt_b >= 0))),
        "candidate_id_mismatch_count": cand_id_mismatch,
        "timestamps_verified": ("timestamps_unix_s" in b_raw and "timestamps_unix_s" in a_dev),
        "receiver_rc_verified": ("receiver_rc" in b_raw and "receiver_rc" in a_dev),
    }


def benchmark_training_speed(
    model_factory: Callable[[], _BaseRindModel],
    baseline_loader_fn: Callable[[], Dict[str, Any]],
    accelerated_loader_fn: Callable[[], Dict[str, Any]],
    *,
    firewall_token: FitFirewallToken,
    expected_split_sha256: str,
    prepare_gpu_tokens_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    device: str | torch.device = "cpu",
    warmup_iters: int = 2,
    timed_iters: int = 5,
    grad_clip_norm: float = 1.0,
    lr: float = 1e-3,
    enable_compile: bool = False,
    shard_build_wall_seconds: Optional[float] = None,
    enforce_parity_gates: bool = True,
    resource_guard: Optional[HostResourceGuard] = None,
    jepa_mode: Optional[str] = None,
    reg_mode: Optional[str] = None,
    counterfactual_weight: float = 0.08,
    cf_every_n_steps: int = 2,
    cf_subgroup_size: int = 0,
) -> Dict[str, Any]:
    """Run 3-arm matched training throughput & numerical parity benchmark (`G4` / `P1-5`) with full ST-JEPA."""
    if int(cf_subgroup_size) < 0:
        raise ValueError("cf_subgroup_size must be non-negative")
    if int(timed_iters) <= 0:
        raise ValueError(f"benchmark_training_speed requires `timed_iters >= 1`, got {timed_iters}")
    if int(warmup_iters) < 0:
        raise ValueError(f"benchmark_training_speed requires `warmup_iters >= 0`, got {warmup_iters}")

    dev = torch.device(device)
    guard = resource_guard or HostResourceGuard(device=dev)
    guard.check("benchmark_training_speed:init")

    init_ref = model_factory().to(dev)
    is_rind_cand = str(getattr(init_ref, "model_type", "")).startswith("rind_")
    eff_jepa = str(jepa_mode) if jepa_mode is not None else ("hybrid" if is_rind_cand else "none")
    eff_reg = str(reg_mode) if reg_mode is not None else ("sigreg" if is_rind_cand else "none")
    h_dim = int(init_ref.get_model_config().get("hidden_dim", 96))
    init_aux = RindJepaAuxiliaryModule(h_dim, jepa_mode=eff_jepa, reg_mode=eff_reg).to(dev)

    m_raw_fp32 = copy.deepcopy(init_ref).to(dev)
    m_shard_fp32 = copy.deepcopy(init_ref).to(dev)
    m_shard_bf16 = copy.deepcopy(init_ref).to(dev)
    aux_raw_fp32 = copy.deepcopy(init_aux).to(dev)
    aux_shard_fp32 = copy.deepcopy(init_aux).to(dev)
    aux_shard_bf16 = copy.deepcopy(init_aux).to(dev)
    aux_raw_fp32.bind_student_backbone(m_raw_fp32)
    aux_shard_fp32.bind_student_backbone(m_shard_fp32)
    aux_shard_bf16.bind_student_backbone(m_shard_bf16)

    has_eg = bool(getattr(init_ref, "has_event_gate", True))
    amp_ok = bool(
        dev.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    b_raw = baseline_loader_fn()
    firewall_token.verify_tiles(b_raw["tile_ids"], expected_split_sha256)
    b_dev = move_batch_to_device(b_raw, dev, pin_and_async=False)

    a_raw = accelerated_loader_fn()
    firewall_token.verify_tiles(a_raw["tile_ids"], expected_split_sha256)
    a_dev = move_batch_to_device(a_raw, dev, pin_and_async=(dev.type == "cuda"))
    a_dev = prepare_gpu_tokens_fn(a_dev)

    domain_parity = _verify_matched_domain(b_dev, a_dev)

    def _single_parity_step(
        mod: _BaseRindModel,
        aux_mod: RindJepaAuxiliaryModule,
        batch_in: Dict[str, Any],
        use_amp: bool,
    ) -> Tuple[float, torch.Tensor]:
        mod.train()
        aux_mod.train()
        lf_view = LabelReadForbiddenBatchView(batch_in)
        jepa_in: Mapping[str, Any] = batch_in if aux_mod.use_response_teacher else lf_view
        mod.update_normalizers(
            lf_view,
            firewall_token=firewall_token,
            tile_ids=batch_in["tile_ids"],
            expected_split_sha256=expected_split_sha256,
        )
        aux_mod.update_normalizers(
            jepa_in,
            firewall_token=firewall_token,
            tile_ids=batch_in["tile_ids"],
            expected_split_sha256=expected_split_sha256,
            device=dev,
        )
        mod.zero_grad(set_to_none=True)
        aux_mod.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            out = mod(lf_view, use_calibrated_temperature=False)
        sup_l, _ = compute_fullday_loss(out, batch_in, has_event_gate=has_eg)
        aux_l, _ = aux_mod.forward_objectives(out, jepa_in, step_index=0)
        if is_rind_cand and float(counterfactual_weight) > 0.0 and int(cf_every_n_steps) > 0:
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                cf_l = float(counterfactual_weight) * compute_candidate_face_perturbation_loss(
                    mod, lf_view, out["probs"], step_seed=0, cf_subgroup_size=cf_subgroup_size
                ).to(torch.float32)
        else:
            cf_l = sup_l.new_zeros(())
        loss = sup_l + aux_l + cf_l
        loss.backward()
        _assert_finite_gradients(mod, aux_mod)
        all_p = [p for p in [*mod.parameters(), *aux_mod.parameters()] if p.requires_grad]
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(all_p, float(grad_clip_norm))
        return float(loss.detach().item()), _flatten_grads(mod, aux_mod).cpu()

    loss_raw32, grad_raw32 = _single_parity_step(m_raw_fp32, aux_raw_fp32, b_dev, False)
    loss_shd32, grad_shd32 = _single_parity_step(m_shard_fp32, aux_shard_fp32, a_dev, False)
    loss_shd16, grad_shd16 = _single_parity_step(m_shard_bf16, aux_shard_bf16, a_dev, amp_ok)

    fp32_loss_diff = abs(loss_raw32 - loss_shd32)
    fp32_grad_cos = _cosine_sim(grad_raw32, grad_shd32)
    bf16_vs_shd32_loss_diff = abs(loss_shd32 - loss_shd16)
    bf16_vs_shd32_grad_cos = _cosine_sim(grad_shd32, grad_shd16)
    bf16_vs_raw32_loss_diff = abs(loss_raw32 - loss_shd16)
    bf16_vs_raw32_grad_cos = _cosine_sim(grad_raw32, grad_shd16)

    if enforce_parity_gates:
        if fp32_loss_diff > 1e-4 or fp32_grad_cos < 0.9999:
            raise RuntimeError(
                f"FP32 shard parity gate failed: loss_abs_diff={fp32_loss_diff:.3e}, grad_cos={fp32_grad_cos:.6f}"
            )
        if amp_ok and (bf16_vs_shd32_loss_diff > 1e-2 or bf16_vs_shd32_grad_cos < 0.99):
            raise RuntimeError(
                f"BF16 mixed parity gate failed: loss_abs_diff={bf16_vs_shd32_loss_diff:.3e}, "
                f"grad_cos={bf16_vs_shd32_grad_cos:.6f}"
            )

    def _time_arm(
        arm_name: str,
        fetch_fn: Callable[[], Dict[str, Any]],
        *,
        use_shard_gpu_tokens: bool,
        use_bf16: bool,
    ) -> Dict[str, Any]:
        arm_mod = copy.deepcopy(init_ref).to(dev)
        arm_aux = copy.deepcopy(init_aux).to(dev)
        arm_aux.bind_student_backbone(arm_mod)
        arm_mod.train()
        arm_aux.train()
        run_mod, compiled_active = (
            maybe_compile_model(arm_mod, enable_compile=enable_compile)
            if use_shard_gpu_tokens
            else (arm_mod, False)
        )
        arm_params = [p for p in [*arm_mod.parameters(), *arm_aux.parameters()] if p.requires_grad]
        opt = torch.optim.AdamW(arm_params, lr=float(lr))

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)

        def _exec_step(bd: Dict[str, Any], step_k: int, do_norm_init: bool) -> Tuple[Dict[str, Any], Dict[str, float]]:
            lf_view = LabelReadForbiddenBatchView(bd)
            jepa_in: Mapping[str, Any] = bd if arm_aux.use_response_teacher else lf_view
            if do_norm_init:
                arm_mod.update_normalizers(
                    lf_view,
                    firewall_token=firewall_token,
                    tile_ids=bd["tile_ids"],
                    expected_split_sha256=expected_split_sha256,
                )
                arm_aux.update_normalizers(
                    jepa_in,
                    firewall_token=firewall_token,
                    tile_ids=bd["tile_ids"],
                    expected_split_sha256=expected_split_sha256,
                    device=dev,
                )
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=dev.type, dtype=torch.bfloat16, enabled=(use_bf16 and amp_ok)
            ):
                o = run_mod(lf_view)
            sup_l, st = compute_fullday_loss(o, bd, has_event_gate=has_eg)
            aux_l, _ = arm_aux.forward_objectives(o, jepa_in, step_index=step_k)
            if (
                is_rind_cand
                and float(counterfactual_weight) > 0.0
                and int(cf_every_n_steps) > 0
                and (step_k % int(cf_every_n_steps) == 0)
            ):
                with torch.amp.autocast(
                    device_type=dev.type, dtype=torch.bfloat16, enabled=(use_bf16 and amp_ok)
                ):
                    cf_l = float(counterfactual_weight) * compute_candidate_face_perturbation_loss(
                        arm_mod, lf_view, o["probs"], step_seed=step_k, cf_subgroup_size=cf_subgroup_size
                    ).to(torch.float32)
            else:
                cf_l = sup_l.new_zeros(())
            total_l = sup_l + aux_l + cf_l
            total_l.backward()
            _assert_finite_gradients(arm_mod, arm_aux)
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(arm_params, float(grad_clip_norm))
            opt.step()
            arm_aux.step_ema()
            return o, st

        warmup_secs: List[float] = []
        for w_i in range(int(warmup_iters)):
            tw0 = time.perf_counter()
            rb = fetch_fn()
            bd = move_batch_to_device(
                rb, dev, pin_and_async=(use_shard_gpu_tokens and dev.type == "cuda")
            )
            if use_shard_gpu_tokens:
                bd = prepare_gpu_tokens_fn(bd)
            _exec_step(bd, w_i, do_norm_init=(w_i == 0))
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            warmup_secs.append(time.perf_counter() - tw0)

        guard.check(f"benchmark_training_speed:{arm_name}:post_warmup")
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)

        total_tile_frames = 0
        total_valid_pf = 0
        nonfinite_n = 0
        step_secs: List[float] = []
        t0 = time.perf_counter()
        for t_i in range(int(timed_iters)):
            ts0 = time.perf_counter()
            rb = fetch_fn()
            bd = move_batch_to_device(
                rb, dev, pin_and_async=(use_shard_gpu_tokens and dev.type == "cuda")
            )
            if use_shard_gpu_tokens:
                bd = prepare_gpu_tokens_fn(bd)
            o, st = _exec_step(bd, int(warmup_iters) + t_i, do_norm_init=False)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            step_secs.append(time.perf_counter() - ts0)

            suns = bd["sun_dirs"]
            b_tiles = int(suns.shape[0]) if suns.ndim == 3 else 1
            t_len = int(suns.shape[-2])
            total_tile_frames += b_tiles * t_len
            total_valid_pf += int(st["valid_point_frames"])
            nonfinite_n += int(o["nonfinite_count"])

        elapsed = max(time.perf_counter() - t0, 1e-9)
        snap = guard.check(f"benchmark_training_speed:{arm_name}:done")
        step_ms = [1000.0 * s for s in step_secs]
        return {
            "arm": arm_name,
            "jepa_mode": eff_jepa,
            "reg_mode": eff_reg,
            "counterfactual_weight": float(counterfactual_weight),
            "warmup_iters": int(warmup_iters),
            "warmup_seconds_list": warmup_secs,
            "timed_iters": int(timed_iters),
            "wall_seconds": elapsed,
            "step_ms_list": step_ms,
            "ms_per_step": 1000.0 * elapsed / max(int(timed_iters), 1),
            "median_ms_per_step": float(np.median(step_ms)) if step_ms else 0.0,
            "p90_ms_per_step": float(np.percentile(step_ms, 90)) if step_ms else 0.0,
            "tile_frames": total_tile_frames,
            "valid_point_frames": total_valid_pf,
            "tile_frames_per_sec": float(total_tile_frames) / elapsed,
            "valid_point_frames_per_sec": float(total_valid_pf) / elapsed,
            "bf16_amp_active": bool(use_bf16 and amp_ok),
            "torch_compile_active": compiled_active,
            "peak_cuda_vram_mib": snap.cuda_max_allocated_bytes / (1024.0 * 1024.0),
            "peak_cuda_reserved_mib": snap.cuda_max_reserved_bytes / (1024.0 * 1024.0),
            "host_rss_mib": snap.process_working_set_bytes / (1024.0 * 1024.0),
            "host_physical_used_pct": snap.physical_used_percent,
            "nonfinite_count": nonfinite_n,
        }

    arm_raw_fp32 = _time_arm(
        "baseline_raw_fp32", baseline_loader_fn, use_shard_gpu_tokens=False, use_bf16=False
    )
    arm_shard_fp32 = _time_arm(
        "fast_shard_fp32", accelerated_loader_fn, use_shard_gpu_tokens=True, use_bf16=False
    )
    arm_shard_bf16 = _time_arm(
        "fast_shard_bf16_mixed", accelerated_loader_fn, use_shard_gpu_tokens=True, use_bf16=True
    )

    speedup_shard_fp32 = arm_shard_fp32["valid_point_frames_per_sec"] / max(
        arm_raw_fp32["valid_point_frames_per_sec"], 1e-9
    )
    speedup_bf16_over_shard_fp32 = arm_shard_bf16["valid_point_frames_per_sec"] / max(
        arm_shard_fp32["valid_point_frames_per_sec"], 1e-9
    )
    speedup_total_bf16 = arm_shard_bf16["valid_point_frames_per_sec"] / max(
        arm_raw_fp32["valid_point_frames_per_sec"], 1e-9
    )

    saved_sec_per_step = max(
        (arm_raw_fp32["ms_per_step"] - arm_shard_bf16["ms_per_step"]) / 1000.0, 0.0
    )
    breakeven_steps: Optional[float] = None
    if shard_build_wall_seconds is not None:
        if saved_sec_per_step > 1e-9:
            breakeven_steps = float(shard_build_wall_seconds) / saved_sec_per_step

    hw_meta = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(dev) if (dev.type == "cuda" and torch.cuda.is_available()) else None,
    }

    return {
        "scope_note": (
            "TRAINING_PIPELINE_ONLY: measures neural training step acceleration on existing "
            "annual 18,808-frame tiles with full ST-JEPA + SIGReg + EMA + counterfactual objectives active."
        ),
        "device": str(dev),
        "jepa_mode": eff_jepa,
        "reg_mode": eff_reg,
        "counterfactual_weight": float(counterfactual_weight),
        "hardware_and_runtime": hw_meta,
        "matched_domain": domain_parity,
        "matched_valid_point_frames_per_step": domain_parity["valid_point_frames_per_step"],
        "matched_domain_verified": True,
        "shard_build_amortization": {
            "shard_build_wall_seconds": shard_build_wall_seconds,
            "time_saved_per_step_seconds": saved_sec_per_step,
            "amortized_breakeven_steps": breakeven_steps,
        },
        "numerical_parity": {
            "baseline_loss": loss_raw32,
            "shard_fp32_loss": loss_shd32,
            "accelerated_loss": loss_shd16,
            "fp32_vs_fp32_loss_abs_diff": fp32_loss_diff,
            "fp32_vs_fp32_gradient_cosine": fp32_grad_cos,
            "bf16_vs_shard_fp32_loss_abs_diff": bf16_vs_shd32_loss_diff,
            "bf16_vs_shard_fp32_gradient_cosine": bf16_vs_shd32_grad_cos,
            "loss_abs_diff": bf16_vs_raw32_loss_diff,
            "gradient_cosine_similarity": bf16_vs_raw32_grad_cos,
        },
        "baseline_raw_fp32": arm_raw_fp32,
        "fast_shard_fp32": arm_shard_fp32,
        "fast_shard_bf16_mixed": arm_shard_bf16,
        "baseline_python_fp32": arm_raw_fp32,
        "accelerated_mmap_bf16": arm_shard_bf16,
        "speedup_shard_fp32_vs_raw_fp32": speedup_shard_fp32,
        "speedup_bf16_vs_shard_fp32": speedup_bf16_over_shard_fp32,
        "speedup_factor": speedup_total_bf16,
        "resource_telemetry": guard.summary(),
    }

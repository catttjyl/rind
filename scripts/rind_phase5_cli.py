#!/usr/bin/env python3
"""Standalone Phase 5 Pure-3D-Mesh De-Cheated ST-JEPA CLI Wrapper (`rind_phase5_cli.py`).

Zero modifications to `portable_pkg/rind/` or `phase1..phase4` scripts.
Inherits the exact OpenMP `query_rays` C kernel (`RIND_OMP_SO`), `az24` sector caching,
canonical deterministic 256-receiver stratified view (`compute_canonical_stratified_indices`,
producing the exact `5,119,233` point-frames on `val_spatial_clean`), `[0.05, 0.95, 181]` wide
calibration grid, and strict IEEE FP32 (`allow_tf32=False`) from `rind_omp_cli.py`, while adding:
  1. Dynamic `ObjectiveConfig` overrides (`RIND_BOUNDARY_METRIC_WEIGHT`, `RIND_REG_WEIGHT`,
     `RIND_DUAL_PATH_WEIGHT`, `RIND_JEPA_WEIGHT`, `RIND_TEACHER_ALIGN_WEIGHT`,
     `RIND_TRANSITION_WEIGHT`, `RIND_GUARD_BUFFER_M`), plus batch-level stripping of continuous
     ray-traced `boundary_distance_m` / `boundary_distance_valid` whenever `boundary_metric_weight == 0.0`.
  2. Calibration-split same-budget firewall (`RIND_CAL_LABEL_BUDGET_FRACTION` &
     `RIND_CAL_LABEL_BUDGET_SEED=20261112`): applies `apply_label_budget_view` to every batch in
     `calibrate_model_on_split` before truth/valid_mask collection (`0.0` hidden-label sensitivity).
  3. Automatic compact `aux_jepa.pt` export alongside `raw.pt` (when `not pure_supervised`) for Track B.
  4. Optional live step-metrics stream (`RIND_STEP_METRICS_JSONL`) for `:14320/panel#/experiments`.
"""
from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

from rind import cli, fast_data, fast_train, fullday_data, objectives_jepa, ray_candidates

_ORIG_BVH_CANDIDATE_PROVIDER = ray_candidates.BVHCandidateProvider
_ORIG_RAY_SCENE_QUERY = ray_candidates.RayCandidateScene.query


def _npz_arrays_sha256(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        c = np.ascontiguousarray(a)
        h.update(f"{c.dtype.str}:{c.shape}:".encode("ascii"))
        if c.nbytes:
            h.update(memoryview(c).cast("B"))
    return h.hexdigest()


def _validate_bvh_cache_arrays(
    raw_c: np.ndarray,
    raw_s: np.ndarray,
    saved_sha: str,
    exp_counts_shape: tuple[int, int],
    capacity: int,
    n_tris_total: int,
) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if type(raw_c) is not np.ndarray or type(raw_s) is not np.ndarray or type(saved_sha) is not str:
        return False, None, None, None
    if raw_c.dtype != np.int32 or raw_s.dtype != np.int32:
        return False, None, None, None
    if saved_sha != _npz_arrays_sha256(raw_c, raw_s):
        return False, None, None, None
    counts = raw_c.astype(np.int64, copy=False)
    selected = raw_s.astype(np.int64, copy=False)
    if counts.shape != exp_counts_shape or not bool(np.all(counts >= 0)):
        return False, None, None, None
    mask = np.arange(int(capacity)) < counts[..., None]
    exp_flat_len = int(mask.sum())
    if selected.ndim != 1 or int(selected.size) != exp_flat_len:
        return False, None, None, None
    if exp_flat_len > 0 and (int(selected.min()) < 0 or int(selected.max()) >= int(n_tris_total)):
        return False, None, None, None
    return True, counts, mask, selected


def _install_omp_and_lazy_cache(omp_so_path: str, ray_cache_dir: Optional[str]) -> None:
    if not omp_so_path:
        ray_candidates.RayCandidateScene.query = _ORIG_RAY_SCENE_QUERY
        if hasattr(ray_candidates.RayCandidateScene, "_omp_query_raw"):
            delattr(ray_candidates.RayCandidateScene, "_omp_query_raw")
    if omp_so_path:
        kernel = ctypes.CDLL(str(omp_so_path)).query_rays
        ptr = ctypes.c_void_p
        kernel.argtypes = [ctypes.c_int64] * 4 + [ctypes.c_double] + [ptr] * 14
        kernel.restype = ctypes.c_int

        def _omp_query_raw(
            self: ray_candidates.RayCandidateScene,
            xyz: np.ndarray,
            sun: np.ndarray,
            horizon: float,
            capacity: int,
            own: Optional[np.ndarray],
        ):
            index = self.index
            shape = (len(xyz), len(sun), capacity)
            ids = np.zeros(shape, np.int64)
            counts = np.zeros(shape[:2], np.int64)
            scratch = np.empty(capacity, np.float64)
            arrays = (
                index._low,
                index._high,
                index._links,
                index._order,
                index._face_low,
                index._face_high,
                index._reasons,
                index._refs,
            )
            expected = (
                np.float64,
                np.float64,
                np.int64,
                np.int64,
                np.float64,
                np.float64,
                np.uint8,
                np.uint64,
            )
            if any(not a.flags.c_contiguous or a.dtype != dt for a, dt in zip(arrays, expected)):
                raise ValueError("Unexpected c47 scene index layout")
            status = kernel(
                len(xyz),
                len(sun),
                capacity,
                index.root,
                float(horizon),
                xyz.ctypes.data,
                sun.ctypes.data,
                own.ctypes.data if own is not None else None,
                *(a.ctypes.data for a in arrays),
                ids.ctypes.data,
                counts.ctypes.data,
                scratch.ctypes.data,
            )
            if status == 1:
                raise ValueError("Unresolved near-degenerate source face on ray")
            if status:
                raise ValueError("Scene index traversal stack exhausted")
            if ray_candidates.require_complete.get() and np.any(counts > capacity):
                raise ValueError(f"requires_more_candidates: capacity={capacity}, required={counts.max()}")
            mask = np.arange(capacity) < counts[..., None]
            selected = ids[mask].astype(np.int32, copy=False)
            return counts, mask, selected

        def _omp_query(
            self: ray_candidates.RayCandidateScene,
            receivers: Any,
            suns: Any,
            horizon: Any,
            capacity: Any,
            receiver_sources: Any = None,
        ):
            xyz = np.ascontiguousarray(receivers, np.float64)
            sun = np.ascontiguousarray(suns, np.float64)
            if xyz.ndim != 2 or sun.ndim != 2 or xyz.shape[1] != 3 or sun.shape[1] != 3:
                raise ValueError("3D receiver coordinates and sun directions required")
            norms = np.linalg.norm(sun, axis=1)
            if (
                not np.isfinite(xyz).all()
                or not np.isfinite(sun).all()
                or (abs(norms - 1.0) > 1e-3).any()
                or not np.isfinite(horizon)
                or horizon <= 0
                or capacity < 1
            ):
                raise ValueError("Finite origins, unit directions and positive budget/horizon required")
            sun = np.ascontiguousarray(sun / norms[:, None])
            own = None
            if receiver_sources is not None:
                own = np.ascontiguousarray(receiver_sources, np.uint64)
                if own.shape != (len(xyz),):
                    raise ValueError("One original source ID per receiver required")
            counts, mask, selected = _omp_query_raw(self, xyz, sun, float(horizon), int(capacity), own)
            shape = (len(xyz), len(sun), int(capacity))
            relative = np.zeros((*shape, 3, 3), np.float32)
            opaque = np.zeros(shape, bool)
            refs = np.full(shape, -1, np.int64)
            rx = np.nonzero(mask)[0]
            index = self.index
            relative[mask] = index._triangles[selected] - xyz[rx, None, :]
            opaque[mask] = index._opaque[selected]
            refs[mask] = index._refs[selected].view(np.int64)
            return relative, opaque, mask, refs, counts

        ray_candidates.RayCandidateScene._omp_query_raw = _omp_query_raw  # type: ignore[attr-defined]
        ray_candidates.RayCandidateScene.query = _omp_query  # type: ignore[assignment]

    so_sha16 = (
        hashlib.sha256(Path(omp_so_path).read_bytes()).hexdigest()[:16]
        if (omp_so_path and Path(omp_so_path).exists())
        else "pure_py_scene"
    )
    cli_sha16 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    impl_version_tag = f"lazy_bvh_v4|cli={cli_sha16}|so={so_sha16}"

    if ray_cache_dir:
        cdir = Path(ray_cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)

        class LazyCachedBVHProvider:
            def __init__(self, triangles: Any, opaque: Any, source_ids: Any):
                self.geometry_identity = None
                self._scene = None
                self._triangles = None
                self._opaque = None
                self._refs = None
                self.update_geometry(triangles, opaque, source_ids)

            def update_geometry(self, triangles: Any, opaque: Any, source_ids: Any):
                tri_ro = np.array(triangles, dtype=np.float64, copy=True)
                op_ro = np.array(opaque, dtype=bool, copy=True)
                ref_ro = np.array(source_ids, dtype=np.uint64, copy=True)
                tri_ro.setflags(write=False)
                op_ro.setflags(write=False)
                ref_ro.setflags(write=False)
                identity = ray_candidates._geometry_digest(tri_ro, op_ro, ref_ro)
                if identity != self.geometry_identity:
                    self._scene = None
                    self._triangles = tri_ro
                    self._opaque = op_ro
                    self._refs = ref_ro
                    self.geometry_identity = identity
                return self

            def _ensure_scene(self):
                if self._scene is None:
                    self._scene = ray_candidates.RayCandidateScene(self._triangles, self._opaque, self._refs)
                return self._scene

            @property
            def scene(self):
                return self._ensure_scene()

            @property
            def index(self):
                return self._ensure_scene().index

            def query(self, receivers: Any, suns: Any, horizon: Any, capacity: Any, receiver_sources: Any = None):
                xyz = np.ascontiguousarray(receivers, np.float64)
                sun = np.ascontiguousarray(suns, np.float64)
                if xyz.ndim != 2 or sun.ndim != 2 or xyz.shape[1] != 3 or sun.shape[1] != 3:
                    raise ValueError("3D receiver coordinates and sun directions required")
                norms = np.linalg.norm(sun, axis=1)
                if (
                    not np.isfinite(xyz).all()
                    or not np.isfinite(sun).all()
                    or (abs(norms - 1.0) > 1e-3).any()
                    or not np.isfinite(horizon)
                    or horizon <= 0
                    or capacity < 1
                ):
                    raise ValueError("Finite origins, unit directions and positive budget/horizon required")
                sun = np.ascontiguousarray(sun / norms[:, None])
                own = None
                if receiver_sources is not None:
                    own = np.ascontiguousarray(receiver_sources, np.uint64)
                    if own.shape != (len(xyz),):
                        raise ValueError("One original source ID per receiver required")

                req_complete = bool(ray_candidates.require_complete.get())
                hor_hex = float(horizon).hex()
                h = hashlib.sha256()
                h.update(f"{impl_version_tag}|".encode())
                h.update(str(self.geometry_identity).encode())
                h.update(f"|{hor_hex}|{int(capacity)}|{int(req_complete)}|{xyz.shape}|{sun.shape}|".encode())
                h.update(memoryview(xyz).cast("B"))
                h.update(memoryview(sun).cast("B"))
                if own is not None:
                    h.update(b"|own|")
                    h.update(memoryview(own).cast("B"))
                qkey = h.hexdigest()
                cfile = cdir / f"{qkey}.npz"

                exp_counts_shape = (len(xyz), len(sun))
                n_tris_total = len(self._triangles)
                cache_loaded = False
                if cfile.exists():
                    try:
                        with np.load(cfile, allow_pickle=False) as z:
                            if "counts" in z and "selected" in z and "sha256" in z:
                                ok_c, counts_v, mask_v, sel_v = _validate_bvh_cache_arrays(
                                    z["counts"],
                                    z["selected"],
                                    str(z["sha256"].item()),
                                    exp_counts_shape,
                                    int(capacity),
                                    n_tris_total,
                                )
                                if ok_c:
                                    counts, mask, selected = counts_v, mask_v, sel_v
                                    cache_loaded = True
                    except Exception:
                        cache_loaded = False
                if cache_loaded:
                    if req_complete and np.any(counts > capacity):
                        raise ValueError(f"requires_more_candidates: capacity={capacity}, required={counts.max()}")
                else:
                    sc = self._ensure_scene()
                    if hasattr(sc, "_omp_query_raw"):
                        counts, mask, selected = sc._omp_query_raw(xyz, sun, float(horizon), int(capacity), own)
                    else:
                        rel, op, mask, rf, counts = sc.query(xyz, sun, float(horizon), int(capacity), own)
                        ref_u64 = self._refs
                        order_u64 = np.argsort(ref_u64)
                        sorted_u64 = ref_u64[order_u64]
                        flat_rf = rf[mask].astype(np.uint64, copy=False)
                        pos = np.searchsorted(sorted_u64, flat_rf)
                        selected = order_u64[pos].astype(np.int32, copy=False)
                    c_i32 = np.ascontiguousarray(counts, dtype=np.int32)
                    s_i32 = np.ascontiguousarray(selected, dtype=np.int32)
                    chk = np.array(_npz_arrays_sha256(c_i32, s_i32))
                    tmp = cdir / f".tmp_{os.getpid()}_{qkey}.npz"
                    np.savez(tmp, counts=c_i32, selected=s_i32, sha256=chk)
                    os.replace(tmp, cfile)

                shape = (len(xyz), len(sun), int(capacity))
                relative = np.zeros((*shape, 3, 3), np.float32)
                opaque = np.zeros(shape, bool)
                refs = np.full(shape, -1, np.int64)
                rx = np.nonzero(mask)[0]
                if len(selected):
                    tri_f64 = np.asarray(self._triangles[selected], dtype=np.float64)
                    relative[mask] = (tri_f64 - xyz[rx, None, :]).astype(np.float32)
                    opaque[mask] = self._opaque[selected]
                    refs[mask] = self._refs[selected].view(np.int64)
                return relative, opaque, mask, refs, counts

        ray_candidates.BVHCandidateProvider = LazyCachedBVHProvider  # type: ignore[assignment]
        ray_candidates.load_shared_scene.cache_clear()
    else:
        ray_candidates.BVHCandidateProvider = _ORIG_BVH_CANDIDATE_PROVIDER  # type: ignore[assignment]
        ray_candidates.load_shared_scene.cache_clear()


REC_AXIS0 = frozenset({
    "receivers_xyz",
    "receiver_rc",
    "local_features",
    "az_rel_triangles",
    "az_opaque",
    "az_mask",
    "az_source_ids",
    "receiver_source_triangle_ids",
})
REC_AXIS1 = frozenset({
    "truth",
    "valid_mask",
    "boundary_distance_m",
    "boundary_distance_valid",
    "response_5x5_bitpack",
})


def compute_canonical_stratified_indices(
    tile_id: str, total_elig: int, max_receivers: int, receiver_seed: int = 20261111
) -> np.ndarray:
    """Exact deterministic spatial stratification from `rind/fullday_data.py:2061-2074`."""
    if max_receivers <= 0 or total_elig <= int(max_receivers):
        return np.arange(total_elig, dtype=np.int64)
    tile_hash = int.from_bytes(
        hashlib.sha256(f"{tile_id}::{receiver_seed}".encode("utf-8")).digest()[:4], "little"
    )
    stride = total_elig / float(max_receivers)
    offset = (tile_hash % 997) / 997.0
    pick_idx = np.floor((np.arange(int(max_receivers)) + offset) * stride).astype(np.int64)
    pick_idx = np.clip(pick_idx, 0, total_elig - 1)
    pick_idx = np.unique(pick_idx)
    if pick_idx.size < int(max_receivers):
        rem = np.setdiff1d(np.arange(total_elig, dtype=np.int64), pick_idx, assume_unique=True)
        need = int(max_receivers) - int(pick_idx.size)
        pick_idx = np.sort(np.concatenate([pick_idx, rem[:need]]))
    return pick_idx


def _install_portable_view(max_r: int, seed: int, az_cache_dir: Optional[str]) -> None:
    orig_open = fast_data.build_or_open_single_tile_shard
    orig_slice = fast_data.ContiguousTileMmapShard._mmap_slice_copy
    az_cdir = Path(az_cache_dir) if az_cache_dir else None
    if az_cdir is not None:
        az_cdir.mkdir(parents=True, exist_ok=True)

    def _patched_slice(self: fast_data.ContiguousTileMmapShard, name: str, sl: Any = slice(None)) -> np.ndarray:
        az_ovr = getattr(self, "_az_override_arrays", None)
        if az_ovr is not None and name in az_ovr:
            return np.array(az_ovr[name][sl], copy=True)
        pick = getattr(self, "_strat_pick_idx", None)
        if pick is None or (name not in REC_AXIS0 and name not in REC_AXIS1):
            return orig_slice(self, name, sl)
        cache = self._strat_array_cache
        if name not in cache:
            full_arr = orig_slice(self, name, slice(None))
            if name in REC_AXIS0:
                cache[name] = np.ascontiguousarray(full_arr[pick])
            else:
                cache[name] = np.ascontiguousarray(full_arr[:, pick])
        return np.array(cache[name][sl], copy=True)

    def _patched_open(*args: Any, **kwargs: Any) -> fast_data.ContiguousTileMmapShard:
        req_az = int(kwargs.get("num_azimuth_bins", 0) or 0)
        open_kw = dict(kwargs)
        reader = next((a for a in args if hasattr(a, "portable_index")), kwargs.get("reader"))
        has_portable = getattr(reader, "portable_index", None) is not None
        if req_az > 0 and has_portable:
            open_kw["num_azimuth_bins"] = 0
        shard = orig_open(*args, **open_kw)
        if max_r > 0 and getattr(shard, "_strat_pick_idx", None) is None and shard.num_receivers > max_r:
            pick = compute_canonical_stratified_indices(shard.tile_id, shard.num_receivers, max_r, seed)
            shard._strat_pick_idx = pick
            shard._strat_array_cache = {}
            shard.num_receivers = int(len(pick))
            shard.receiver_mode = "sampled"
            shard._global_knn_cache = None
        if req_az > 0 and has_portable and getattr(shard, "_az_override_arrays", None) is None:
            rec_xyz = _patched_slice(shard, "receivers_xyz", slice(None))
            suns = orig_slice(shard, "sun_dirs", slice(None))
            max_k = int(kwargs.get("max_candidates", shard.provenance.get("max_candidates", 32)))
            hor_m = float(kwargs.get("bbox_margin_m", shard.bbox_margin_m))
            tri_sha = str(shard.provenance.get("array_sha256", {}).get("scene_triangles_xyz", ""))
            op_sha = str(shard.provenance.get("array_sha256", {}).get("scene_opaque", ""))
            sid_sha = str(shard.provenance.get("array_sha256", {}).get("scene_source_ids", ""))
            if any(len(s) != 64 or any(c not in "0123456789abcdef" for c in s) for s in (tri_sha, op_sha, sid_sha)):
                raise RuntimeError(f"Missing or malformed geometry SHA-256 in shard provenance for {shard.tile_id}")
            cli_sha16 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
            own_preview = _patched_slice(shard, "receiver_source_triangle_ids", slice(None))
            hk = hashlib.sha256()
            hk.update(
                f"az_sector_v3|cli={cli_sha16}|{shard.tile_id}|{tri_sha}|{op_sha}|{sid_sha}|{req_az}|{max_k}|{hor_m.hex()}|".encode()
            )
            hk.update(memoryview(np.ascontiguousarray(rec_xyz, dtype=np.float32)).cast("B"))
            hk.update(memoryview(np.ascontiguousarray(own_preview, dtype=np.int64)).cast("B"))
            cfile = (az_cdir / f"{shard.tile_id}_{hk.hexdigest()[:24]}.npz") if az_cdir is not None else None
            exp_az_3d = (len(rec_xyz), req_az, max_k)
            az_loaded = False
            if cfile is not None and cfile.exists():
                try:
                    with np.load(cfile, allow_pickle=False) as z:
                        if all(k in z for k in ("az_rel_triangles", "az_opaque", "az_mask", "az_source_ids", "sha256")):
                            az_t = z["az_rel_triangles"]
                            az_o = z["az_opaque"]
                            az_m = z["az_mask"]
                            az_sids = z["az_source_ids"]
                            saved_az_sha = str(z["sha256"].item())
                            if (
                                az_t.dtype == np.float32
                                and az_o.dtype == np.bool_
                                and az_m.dtype == np.bool_
                                and az_sids.dtype == np.int64
                                and az_t.shape == (*exp_az_3d, 3, 3)
                                and az_o.shape == exp_az_3d
                                and az_m.shape == exp_az_3d
                                and az_sids.shape == exp_az_3d
                                and bool(np.isfinite(az_t).all())
                                and saved_az_sha == _npz_arrays_sha256(az_t, az_o, az_m, az_sids)
                            ):
                                az_loaded = True
                except Exception:
                    az_loaded = False
            if not az_loaded:
                tris = orig_slice(shard, "scene_triangles_xyz", slice(None))
                s_op = orig_slice(shard, "scene_opaque", slice(None))
                s_ids = orig_slice(shard, "scene_source_ids", slice(None))
                own_ids = _patched_slice(shard, "receiver_source_triangle_ids", slice(None))
                az_t, az_o, az_m, az_sids = fullday_data.select_azimuth_sector_candidates(
                    rec_xyz,
                    tris,
                    scene_opaque=s_op,
                    scene_source_triangle_ids=s_ids,
                    exclude_source_triangle_ids=own_ids,
                    num_azimuth_bins=req_az,
                    max_candidates=max_k,
                    horizon_m=hor_m,
                    return_source_ids=True,
                )
                az_t = np.ascontiguousarray(az_t, dtype=np.float32)
                az_o = np.ascontiguousarray(az_o, dtype=bool)
                az_m = np.ascontiguousarray(az_m, dtype=bool)
                az_sids = np.ascontiguousarray(az_sids, dtype=np.int64)
                if cfile is not None:
                    az_chk = np.array(_npz_arrays_sha256(az_t, az_o, az_m, az_sids))
                    tmp = cfile.with_name(f".tmp_{os.getpid()}_{cfile.name}")
                    np.savez(
                        tmp,
                        az_rel_triangles=az_t,
                        az_opaque=az_o,
                        az_mask=az_m,
                        az_source_ids=az_sids,
                        sha256=az_chk,
                    )
                    os.replace(tmp, cfile)
            az_bins = np.ascontiguousarray(
                fullday_data.sun_dirs_to_azimuth_bins(suns, num_azimuth_bins=req_az).astype(np.int16)
            )
            shard._az_override_arrays = {
                "az_rel_triangles": az_t,
                "az_opaque": az_o,
                "az_mask": az_m,
                "az_source_ids": az_sids,
                "azimuth_bin_idx": az_bins,
            }
            shard.array_paths = {k: v for k, v in shard.array_paths.items() if k != "scene_triangles_xyz"}
            shard.provenance = {**dict(shard.provenance), "num_azimuth_bins": req_az}
        return shard

    fast_data.ContiguousTileMmapShard._mmap_slice_copy = _patched_slice  # type: ignore[assignment]
    fast_data.build_or_open_single_tile_shard = _patched_open  # type: ignore[assignment]
    cli.build_or_open_single_tile_shard = _patched_open  # type: ignore[assignment]


def resolve_dynamic_env_objective_overrides(cfg: objectives_jepa.ObjectiveConfig) -> objectives_jepa.ObjectiveConfig:
    """Apply dynamic Phase 5 `RIND_*` float overrides onto `ObjectiveConfig`."""
    mapping = {
        "boundary_metric_weight": "RIND_BOUNDARY_METRIC_WEIGHT",
        "reg_weight": "RIND_REG_WEIGHT",
        "jepa_weight": "RIND_JEPA_WEIGHT",
        "teacher_align_weight": "RIND_TEACHER_ALIGN_WEIGHT",
        "transition_weight": "RIND_TRANSITION_WEIGHT",
        "dual_path_weight": "RIND_DUAL_PATH_WEIGHT",
        "guard_buffer_m": "RIND_GUARD_BUFFER_M",
    }
    upd: Dict[str, float] = {}
    for field_name, env_key in mapping.items():
        val = os.environ.get(env_key, "").strip()
        if val != "":
            upd[field_name] = float(val)
    return dataclasses.replace(cfg, **upd) if upd else cfg


def sanitize_batch_zero_continuous_boundary(raw_batch: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip precomputed ray-traced continuous `boundary_distance_m` / `boundary_distance_valid`
    when `boundary_metric_weight == 0.0`, leaving strictly binary `0/1` visibility labels `truth`.
    """
    out = dict(raw_batch)
    for k in (
        "boundary_distance_m",
        "boundary_distance_valid",
        "prev_frame_boundary_distance_m",
        "prev_frame_boundary_distance_valid",
    ):
        out.pop(k, None)
    return out


def wrap_calibration_batches_with_label_budget(
    calibration_batches: Iterable[Mapping[str, Any]],
    *,
    cal_label_budget_fraction: float,
    cal_label_budget_seed: int = 20261112,
    strip_continuous_boundary: bool = False,
) -> Iterable[Dict[str, Any]]:
    """Same-budget calibration firewall: masks hidden `1 - cal_frac` truth & valid_mask BEFORE
    `calibrate_model_on_split` ever sees the batch, so `(T*, tau*)` has 0.0 sensitivity to hidden labels!
    """
    frac = float(cal_label_budget_fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"RIND_CAL_LABEL_BUDGET_FRACTION must be in (0, 1], got {frac}")
    for raw_b in calibration_batches:
        b = sanitize_batch_zero_continuous_boundary(raw_b) if strip_continuous_boundary else dict(raw_b)
        if frac < 1.0 - 1e-9:
            b, _ = objectives_jepa.apply_label_budget_view(
                b,
                label_budget_fraction=frac,
                label_budget_seed=int(cal_label_budget_seed),
                device=torch.device("cpu"),
            )
        yield b


def _extract_cli_path_arg(argv: list[str], flag: str) -> Optional[Path]:
    for idx, tok in enumerate(argv):
        if tok == flag and idx + 1 < len(argv):
            return Path(argv[idx + 1])
        if tok.startswith(f"{flag}="):
            return Path(tok.split("=", 1)[1])
    return None


def _install_phase5_hooks() -> None:
    orig_build_obj = cli._build_train_objective_config
    orig_cal = fast_train.calibrate_model_on_split
    orig_fit = fast_train.fit_model_fullday
    orig_head_norms = fast_train.compute_head_parameter_norms
    orig_sup_loss = fast_train.compute_fullday_loss

    def _patched_build_obj(args: Any) -> objectives_jepa.ObjectiveConfig:
        cfg = orig_build_obj(args)
        return resolve_dynamic_env_objective_overrides(cfg)

    def _patched_calibrate(model: Any, calibration_batches: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        cal_frac_str = os.environ.get("RIND_CAL_LABEL_BUDGET_FRACTION", "").strip()
        cal_frac = float(cal_frac_str) if cal_frac_str != "" else 1.0
        cal_seed = int(os.environ.get("RIND_CAL_LABEL_BUDGET_SEED", "20261112") or "20261112")
        bnd_w_str = os.environ.get("RIND_BOUNDARY_METRIC_WEIGHT", "").strip()
        strip_bnd = (bnd_w_str != "" and float(bnd_w_str) == 0.0)

        wrapped_batches = wrap_calibration_batches_with_label_budget(
            calibration_batches,
            cal_label_budget_fraction=cal_frac,
            cal_label_budget_seed=cal_seed,
            strip_continuous_boundary=strip_bnd,
        )

        use_wide = os.environ.get("RIND_WIDE_CALIBRATION", "1") == "1"
        orig_ls = np.linspace
        invoked = {"hit": False}

        def _patched_ls(start: Any, stop: Any, num: int = 50, *a: Any, **kw: Any):
            if use_wide and num == 81 and abs(float(start) - 0.30) < 1e-9 and abs(float(stop) - 0.70) < 1e-9:
                invoked["hit"] = True
                return orig_ls(0.05, 0.95, 181, *a, **kw)
            return orig_ls(start, stop, num, *a, **kw)

        np.linspace = _patched_ls
        try:
            res = orig_cal(model, wrapped_batches, *args, **kwargs)
        finally:
            np.linspace = orig_ls
        if use_wide:
            if not invoked["hit"]:
                raise RuntimeError("Wide calibration hook did not intercept [0.30, 0.70, 81] threshold grid!")
            res["threshold_grid_range"] = [0.05, 0.95, 181]
        res["calibration_label_budget_fraction"] = cal_frac
        res["calibration_label_budget_seed"] = cal_seed
        return res

    def _patched_fit_model_fullday(
        model: Any,
        epoch_batches_fn: Callable[[int], Iterable[Mapping[str, Any]]],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        is_rind_cand = str(getattr(model, "model_type", "")).startswith("rind_")
        resolved_cfg = fast_train._resolve_objective_config_for_training(
            is_rind_cand=is_rind_cand,
            objective_config=kwargs.get("objective_config"),
            jepa_mode=kwargs.get("jepa_mode"),
            reg_mode=kwargs.get("reg_mode"),
            mask_mode=kwargs.get("mask_mode"),
            mask_ratio=kwargs.get("mask_ratio"),
            spatial_block_size=kwargs.get("spatial_block_size"),
            jepa_context_mode=kwargs.get("jepa_context_mode"),
            transition_mode=kwargs.get("transition_mode"),
            ema_target_scope=kwargs.get("ema_target_scope"),
            pure_supervised=kwargs.get("pure_supervised"),
            label_budget_fraction=kwargs.get("label_budget_fraction"),
            label_budget_seed=kwargs.get("label_budget_seed"),
            counterfactual_weight=kwargs.get("counterfactual_weight"),
            dominant_occluder_dropout_prob=kwargs.get("dominant_occluder_dropout_prob"),
        )
        eff_cfg = resolve_dynamic_env_objective_overrides(resolved_cfg)
        kwargs["objective_config"] = eff_cfg

        strip_continuous_bnd = float(eff_cfg.boundary_metric_weight) == 0.0
        if strip_continuous_bnd:
            orig_epoch_fn = epoch_batches_fn

            def _sanitized_epoch_batches(ep_idx: int):
                for raw_b in orig_epoch_fn(ep_idx):
                    yield sanitize_batch_zero_continuous_boundary(raw_b)

            eff_epoch_fn = _sanitized_epoch_batches
        else:
            eff_epoch_fn = epoch_batches_fn

        captured_aux: Dict[str, Any] = {"module": None, "step": 0}
        step_jsonl_env = os.environ.get("RIND_STEP_METRICS_JSONL", "").strip()

        def _capturing_head_norms(m: Any, aux_jepa_module: Any = None):
            captured_aux["module"] = aux_jepa_module
            return orig_head_norms(m, aux_jepa_module=aux_jepa_module)

        def _logging_sup_loss(*s_args: Any, **s_kwargs: Any):
            loss_t, st_d = orig_sup_loss(*s_args, **s_kwargs)
            captured_aux["step"] += 1
            cur_s = int(captured_aux["step"])
            if step_jsonl_env and (cur_s == 1 or cur_s % 8 == 0 or cur_s == 804):
                try:
                    sp = Path(step_jsonl_env)
                    sp.parent.mkdir(parents=True, exist_ok=True)
                    rec = {"step": cur_s, "sup_loss": float(loss_t.detach().to(torch.float32).item())}
                    with sp.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                except Exception:
                    pass
            return loss_t, st_d

        fast_train.compute_head_parameter_norms = _capturing_head_norms
        if step_jsonl_env:
            fast_train.compute_fullday_loss = _logging_sup_loss
        try:
            train_stats = orig_fit(model, eff_epoch_fn, *args, **kwargs)
        finally:
            fast_train.compute_head_parameter_norms = orig_head_norms
            if step_jsonl_env:
                fast_train.compute_fullday_loss = orig_sup_loss

        aux_mod = captured_aux["module"]
        aux_out_env = os.environ.get("RIND_AUX_JEPA_OUT", "").strip()
        if aux_out_env == "":
            ckpt_arg = _extract_cli_path_arg(sys.argv[1:], "--checkpoint-out")
            if ckpt_arg is not None and not bool(eff_cfg.pure_supervised):
                aux_out_env = str(ckpt_arg.parent / "aux_jepa.pt")
        if aux_out_env and aux_mod is not None and not bool(eff_cfg.pure_supervised):
            aux_p = Path(aux_out_env)
            aux_p.parent.mkdir(parents=True, exist_ok=True)
            tmp_aux = aux_p.with_name(f".tmp_{os.getpid()}_{aux_p.name}")
            aux_payload = {
                "schema": "rind.aux_jepa.v1",
                "hidden_dim": int(getattr(model, "hidden_dim", 96)),
                "model_type": str(getattr(model, "model_type", "")),
                "objective_config": eff_cfg.to_dict(),
                "aux_jepa_state_dict": {k: v.detach().cpu() for k, v in aux_mod.state_dict().items()},
            }
            torch.save(aux_payload, tmp_aux)
            os.replace(tmp_aux, aux_p)
            train_stats["aux_jepa_checkpoint_path"] = str(aux_p.resolve())
            train_stats["aux_jepa_checkpoint_sha256"] = hashlib.sha256(aux_p.read_bytes()).hexdigest()

        train_stats["continuous_boundary_stripped_at_batch_boundary"] = bool(strip_continuous_bnd)
        return train_stats

    cli._build_train_objective_config = _patched_build_obj  # type: ignore[assignment]
    fast_train.calibrate_model_on_split = _patched_calibrate  # type: ignore[assignment]
    cli.calibrate_model_on_split = _patched_calibrate  # type: ignore[assignment]
    fast_train.fit_model_fullday = _patched_fit_model_fullday  # type: ignore[assignment]
    cli.fit_model_fullday = _patched_fit_model_fullday  # type: ignore[assignment]


_install_omp_and_lazy_cache(
    os.environ.get("RIND_OMP_SO", "").strip(),
    os.environ.get("RIND_RAY_CACHE_DIR", "").strip() or None,
)

_strat_r = int(os.environ.get("RIND_STRATIFIED_RECEIVERS", "0") or "0")
_strat_seed = int(os.environ.get("RIND_RECEIVER_SEED", "20261111") or "20261111")
_az_cdir_env = os.environ.get("RIND_AZ_CACHE_DIR", "").strip() or None
if _strat_r > 0 or _az_cdir_env is not None:
    _install_portable_view(_strat_r, _strat_seed, _az_cdir_env)

_install_phase5_hooks()


def build_runtime_precision_and_kernel_audit() -> Dict[str, Any]:
    omp_env = os.environ.get("RIND_OMP_SO", "").strip()
    omp_p = Path(omp_env) if omp_env else None
    omp_sha = hashlib.sha256(omp_p.read_bytes()).hexdigest() if (omp_p is not None and omp_p.exists()) else None
    cal_frac_str = os.environ.get("RIND_CAL_LABEL_BUDGET_FRACTION", "").strip()
    bnd_w_str = os.environ.get("RIND_BOUNDARY_METRIC_WEIGHT", "").strip()
    audit = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "default_dtype": str(torch.get_default_dtype()),
        "omp_kernel_installed": bool(hasattr(ray_candidates.RayCandidateScene, "_omp_query_raw")),
        "omp_so_path": omp_env or None,
        "omp_so_sha256": omp_sha,
        "rind_phase5_cli_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "env_boundary_metric_weight": float(bnd_w_str) if bnd_w_str != "" else None,
        "env_cal_label_budget_fraction": float(cal_frac_str) if cal_frac_str != "" else None,
    }
    if audit["cuda_matmul_allow_tf32"] or audit["cudnn_allow_tf32"] or audit["float32_matmul_precision"] != "highest":
        raise RuntimeError(f"Strict IEEE FP32 subprocess violation: {audit}")
    return audit


def _stamp_output_json_with_runtime_audit(argv: list[str], input_ckpt_sha: Optional[str] = None) -> None:
    out_path = _extract_cli_path_arg(argv, "--output")
    audit = build_runtime_precision_and_kernel_audit()
    if out_path is not None and out_path.exists():
        doc = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(doc, dict):
            doc["runtime_precision_and_kernel_audit"] = audit
            if input_ckpt_sha is not None:
                doc["input_checkpoint_sha256"] = input_ckpt_sha
            tmp = out_path.with_name(f".tmp_{os.getpid()}_{out_path.name}")
            tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, out_path)


if __name__ == "__main__":
    build_runtime_precision_and_kernel_audit()
    _in_ckpt = _extract_cli_path_arg(sys.argv[1:], "--checkpoint")
    _in_ckpt_sha = (
        hashlib.sha256(_in_ckpt.read_bytes()).hexdigest()
        if (_in_ckpt is not None and _in_ckpt.exists())
        else None
    )
    rc = cli.main()
    if rc == 0:
        _stamp_output_json_with_runtime_audit(sys.argv[1:], input_ckpt_sha=_in_ckpt_sha)
    raise SystemExit(rc)

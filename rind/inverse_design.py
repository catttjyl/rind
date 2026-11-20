"""Full-Scene 4km Exact Double-Precision BVH Oracle & Volume-Conserving Full-Day Inverse Design.

Fixes Astra P0-5, P0-6, P0-7, and P1-9:
1. Exact double-precision (`float64`) forward ray-triangle Möller–Trumbore intersection
   with unified `DEFAULT_T_MIN_M = 1e-4`, `|det| > 1e-12`, `u >= 0`, `v >= 0`, `u + v <= 1.0`.
   Strictly rejects behind-origin (`t <= 0`) and sub-threshold (`t <= 1e-4`) intersections while
   accepting valid grazing forward hits (`t = 2e-4`).
2. `ExactSceneBVH`: conservative `float64` AABB bounding-volume hierarchy over the full `4 km`
   opaque scene mesh (`source_triangle_ids`, optional per-ray `exclude_triangle_ids` to prevent
   self-intersection while preserving occlusion from all other faces of the same building),
   verified bit-for-bit identical to brute-force `float64` Möller–Trumbore across low-sun `1.2 km+`
   blockers.
3. 3D shared-edge `assign_building_ids_from_triangles` with ground exclusion (`building_id = -1`)
   and optional `source_instance_ids` (never merging stacked XZ surfaces separated in `Y`).
4. Exact `Δh = 0` identity (`max |V' - V| == 0.0` for `0.75 m` low walls, sloped roofs, canopies,
   and fixed ground) + per-building wall/cap logic (never disabling standalone canopy roofs via a
   global scene flag).
5. Area-weighted volume conservation `sum_b A_b * delta_h_b = 0` (`delta_h_b in [-15m, +15m]`)
   verified against closed-mesh signed 3D volume, plus 3 distinct optimization arms
   (`rind_checkpoint_grad`, `analytic_diff_surrogate`, `blackbox_zero_order`) evaluated on the
   exact `4 km` `ExactSceneBVH` physical oracle with unclipped signed deltas.
"""

from __future__ import annotations

import dataclasses
import math
import time
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

DEFAULT_T_MIN_M: float = 1e-4
DEFAULT_T_MAX_M: float = 4000.0
DEFAULT_DET_EPS: float = 1e-12


def moller_trumbore_intersect_f64(
    origins_xyz: np.ndarray,
    directions_xyz: np.ndarray,
    triangles_xyz: np.ndarray,
    *,
    t_min: float = DEFAULT_T_MIN_M,
    t_max: float = DEFAULT_T_MAX_M,
    det_eps: float = DEFAULT_DET_EPS,
    _already_normalized: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact double-precision Möller–Trumbore ray-triangle intersection.

    Args:
        origins_xyz: `(R, 3)` ray origins in world meters (+X East, +Y Up, +Z North).
        directions_xyz: `(R, 3)` ray directions (normalized internally in float64 once).
        triangles_xyz: `(M, 3, 3)` triangle vertices `(v0, v1, v2)` in world meters.
        t_min: Minimum strictly positive ray distance `t > t_min` (`1e-4` m default).
        t_max: Maximum scene horizon ray distance `t <= t_max` (`4000.0` m default).
        det_eps: Minimum `|det|` for non-parallel rays (`1e-12`).
        _already_normalized: Internal flag set by `ExactSceneBVH` when `directions_xyz`
            has already been normalized once in `float64`, avoiding double-normalization ULP drift.

    Returns:
        hit_matrix: `(R, M)` boolean array (`True` iff forward ray intersects triangle within `(t_min, t_max]`).
        t_matrix: `(R, M)` float64 ray parameter (`np.inf` where `hit_matrix` is `False`).
    """
    orig = np.asarray(origins_xyz, dtype=np.float64)
    dirs = np.asarray(directions_xyz, dtype=np.float64)
    tris = np.asarray(triangles_xyz, dtype=np.float64)

    if orig.ndim == 1:
        orig = orig[None, :]
    if dirs.ndim == 1:
        dirs = dirs[None, :]
    if orig.shape != dirs.shape or orig.shape[-1] != 3:
        raise ValueError(f"origins {orig.shape} and directions {dirs.shape} must be (R, 3)")
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError(f"triangles_xyz must have shape (M, 3, 3), got {tris.shape}")

    r_count = orig.shape[0]
    m_count = tris.shape[0]
    if r_count == 0 or m_count == 0:
        return (
            np.zeros((r_count, m_count), dtype=bool),
            np.full((r_count, m_count), np.inf, dtype=np.float64),
        )

    if _already_normalized:
        valid_dir = np.any(np.abs(dirs) > 0.0, axis=-1)
        safe_dirs = dirs
    else:
        dir_norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
        valid_dir = dir_norm[..., 0] > 1e-12
        safe_dirs = np.where(dir_norm > 1e-12, dirs / np.maximum(dir_norm, 1e-12), 0.0)

    v0 = tris[:, 0, :][None, :, :]
    v1 = tris[:, 1, :][None, :, :]
    v2 = tris[:, 2, :][None, :, :]
    e1 = v1 - v0
    e2 = v2 - v0

    d = safe_dirs[:, None, :]  # (R, 1, 3)
    pvec = np.cross(d, e2)  # (R, M, 3)
    det = np.sum(e1 * pvec, axis=-1)  # (R, M)
    non_parallel = (np.abs(det) > float(det_eps)) & valid_dir[:, None]
    inv_det = np.where(non_parallel, 1.0 / np.where(non_parallel, det, 1.0), 0.0)

    tvec = orig[:, None, :] - v0  # (R, M, 3)
    u = np.sum(tvec * pvec, axis=-1) * inv_det  # (R, M)

    qvec = np.cross(tvec, e1)  # (R, M, 3)
    v = np.sum(d * qvec, axis=-1) * inv_det  # (R, M)
    t = np.sum(e2 * qvec, axis=-1) * inv_det  # (R, M)

    hit = (
        non_parallel
        & (t > float(t_min))
        & (t <= float(t_max))
        & (u >= 0.0)
        & (v >= 0.0)
        & ((u + v) <= 1.0)
    )
    t_out = np.where(hit, t, np.inf)
    return hit, t_out


@dataclasses.dataclass
class _BVHNode:
    bmin: np.ndarray  # (3,) float64
    bmax: np.ndarray  # (3,) float64
    tri_indices: Optional[np.ndarray] = None  # leaf indices into filtered triangles
    left: Optional["_BVHNode"] = None
    right: Optional["_BVHNode"] = None


class ExactSceneBVH:
    """Exact conservative `float64` AABB BVH over opaque 3D scene triangles (`4 km` horizon).

    Guarantees 100% bit-for-bit agreement with brute-force `moller_trumbore_intersect_f64`
    while accelerating multi-thousand-triangle scenes by 10x–100x via vectorized slab culling
    and early occlusion termination.
    """

    def __init__(
        self,
        triangles_xyz: np.ndarray,
        *,
        opaque_mask: Optional[np.ndarray] = None,
        source_triangle_ids: Optional[np.ndarray] = None,
        leaf_size: int = 32,
        aabb_pad_m: float = 1e-7,
    ) -> None:
        raw_tris = np.asarray(triangles_xyz, dtype=np.float64)
        if raw_tris.ndim != 3 or raw_tris.shape[1:] != (3, 3):
            raise ValueError(f"triangles_xyz must be (M, 3, 3), got {raw_tris.shape}")
        m_raw = raw_tris.shape[0]

        if opaque_mask is not None:
            op = np.asarray(opaque_mask, dtype=bool)
            if op.shape != (m_raw,):
                raise ValueError(f"opaque_mask shape {op.shape} != ({m_raw},)")
        else:
            op = np.ones((m_raw,), dtype=bool)

        if source_triangle_ids is not None:
            s_ids = np.asarray(source_triangle_ids, dtype=np.uint64)
            if s_ids.shape != (m_raw,):
                raise ValueError(f"source_triangle_ids shape {s_ids.shape} != ({m_raw},)")
        else:
            s_ids = np.arange(m_raw, dtype=np.uint64)

        self.triangles_xyz = np.ascontiguousarray(raw_tris[op], dtype=np.float64)
        self.source_triangle_ids = np.ascontiguousarray(s_ids[op], dtype=np.uint64)
        self.leaf_size = max(4, int(leaf_size))
        self.aabb_pad_m = float(aabb_pad_m)
        self.num_triangles = int(self.triangles_xyz.shape[0])

        if self.num_triangles > 0:
            self._tri_min = self.triangles_xyz.min(axis=1) - self.aabb_pad_m
            self._tri_max = self.triangles_xyz.max(axis=1) + self.aabb_pad_m
            self._tri_cent = 0.5 * (self._tri_min + self._tri_max)
            all_idx = np.arange(self.num_triangles, dtype=np.int64)
            self._root: Optional[_BVHNode] = self._build_node(all_idx)
        else:
            self._tri_min = np.zeros((0, 3), dtype=np.float64)
            self._tri_max = np.zeros((0, 3), dtype=np.float64)
            self._tri_cent = np.zeros((0, 3), dtype=np.float64)
            self._root = None

    def _build_node(self, indices: np.ndarray) -> _BVHNode:
        bmin = self._tri_min[indices].min(axis=0)
        bmax = self._tri_max[indices].max(axis=0)
        if indices.size <= self.leaf_size:
            return _BVHNode(bmin=bmin, bmax=bmax, tri_indices=indices)
        c_sub = self._tri_cent[indices]
        ext = c_sub.max(axis=0) - c_sub.min(axis=0)
        axis = int(np.argmax(ext))
        if ext[axis] <= 1e-12:
            return _BVHNode(bmin=bmin, bmax=bmax, tri_indices=indices)
        mid = int(indices.size // 2)
        part = np.argpartition(c_sub[:, axis], mid)
        left_idx = indices[part[:mid]]
        right_idx = indices[part[mid:]]
        return _BVHNode(
            bmin=bmin,
            bmax=bmax,
            left=self._build_node(left_idx),
            right=self._build_node(right_idx),
        )

    @staticmethod
    def _rays_hit_aabb(
        orig: np.ndarray,
        safe_d: np.ndarray,
        bmin: np.ndarray,
        bmax: np.ndarray,
        t_min: float,
        t_max: float = DEFAULT_T_MAX_M,
    ) -> np.ndarray:
        """Conservative IEEE-754 slab test `(R_sub,)` for forward rays `t_min <= t <= t_max`."""
        t_enter = np.full((orig.shape[0],), float(t_min), dtype=np.float64)
        t_exit = np.full((orig.shape[0],), float(t_max), dtype=np.float64)
        alive = np.ones((orig.shape[0],), dtype=bool)
        for ax in range(3):
            da = safe_d[:, ax]
            oa = orig[:, ax]
            parallel = np.abs(da) <= 1e-15
            if np.any(parallel):
                in_slab = (oa >= (bmin[ax] - 1e-11)) & (oa <= (bmax[ax] + 1e-11))
                alive &= (~parallel) | in_slab
            non_p = ~parallel
            if np.any(non_p):
                safe_da = np.where(non_p, da, 1.0)
                t0 = (bmin[ax] - oa) / safe_da
                t1 = (bmax[ax] - oa) / safe_da
                near = np.minimum(t0, t1)
                far = np.maximum(t0, t1)
                pad_t = 1e-11 * (1.0 + np.maximum(np.abs(near), np.abs(far)))
                t_enter = np.where(non_p, np.maximum(t_enter, near - pad_t), t_enter)
                t_exit = np.where(non_p, np.minimum(t_exit, far + pad_t), t_exit)
        return alive & (t_exit >= t_enter)

    def query_occluded(
        self,
        origins_xyz: np.ndarray,
        directions_xyz: np.ndarray,
        *,
        t_min: float = DEFAULT_T_MIN_M,
        t_max: float = DEFAULT_T_MAX_M,
        det_eps: float = DEFAULT_DET_EPS,
        exclude_source_triangle_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return `(R,)` bool (`True` = occluded by at least one non-excluded opaque triangle)."""
        orig = np.asarray(origins_xyz, dtype=np.float64)
        dirs = np.asarray(directions_xyz, dtype=np.float64)
        squeeze = False
        if orig.ndim == 1:
            orig = orig[None, :]
            squeeze = True
        if dirs.ndim == 1:
            dirs = np.broadcast_to(dirs[None, :], orig.shape)
        if orig.shape != dirs.shape or orig.shape[-1] != 3:
            raise ValueError(f"origins {orig.shape} and directions {dirs.shape} must be (R, 3)")

        r_total = orig.shape[0]
        occluded = np.zeros((r_total,), dtype=bool)
        if r_total == 0 or self._root is None or self.num_triangles == 0:
            return occluded[0] if squeeze else occluded

        excl_ids: Optional[np.ndarray] = None
        if exclude_source_triangle_ids is not None:
            excl_ids = np.asarray(exclude_source_triangle_ids, dtype=np.uint64)
            if excl_ids.ndim == 0:
                excl_ids = np.broadcast_to(excl_ids[None], (r_total,))
            elif excl_ids.shape != (r_total,):
                raise ValueError(
                    f"exclude_source_triangle_ids shape {excl_ids.shape} != ({r_total},)"
                )

        d_norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
        valid_d = d_norm[:, 0] > 1e-12
        safe_d = np.where(d_norm > 1e-12, dirs / np.maximum(d_norm, 1e-12), 0.0)

        active = np.flatnonzero(valid_d)
        if active.size == 0:
            return occluded[0] if squeeze else occluded

        stack: List[Tuple[_BVHNode, np.ndarray]] = [(self._root, active)]
        while stack:
            node, cand_rays = stack.pop()
            # Filter out rays already proven occluded by an earlier leaf!
            cand_rays = cand_rays[~occluded[cand_rays]]
            if cand_rays.size == 0:
                continue
            hit_box = self._rays_hit_aabb(
                orig[cand_rays],
                safe_d[cand_rays],
                node.bmin,
                node.bmax,
                float(t_min),
                float(t_max),
            )
            if not np.any(hit_box):
                continue
            sub_rays = cand_rays[hit_box]
            if node.tri_indices is not None:
                leaf_tris = self.triangles_xyz[node.tri_indices]
                hit_mat, _ = moller_trumbore_intersect_f64(
                    orig[sub_rays],
                    safe_d[sub_rays],
                    leaf_tris,
                    t_min=t_min,
                    t_max=t_max,
                    det_eps=det_eps,
                    _already_normalized=True,
                )
                if excl_ids is not None:
                    leaf_sids = self.source_triangle_ids[node.tri_indices][None, :]
                    ray_excl = excl_ids[sub_rays][:, None]
                    hit_mat = hit_mat & (leaf_sids != ray_excl)
                occluded[sub_rays] |= np.any(hit_mat, axis=1)
            else:
                if node.right is not None:
                    stack.append((node.right, sub_rays))
                if node.left is not None:
                    stack.append((node.left, sub_rays))

        return occluded[0] if squeeze else occluded

    def query_first_hit(
        self,
        origins_xyz: np.ndarray,
        directions_xyz: np.ndarray,
        *,
        t_min: float = DEFAULT_T_MIN_M,
        t_max: float = DEFAULT_T_MAX_M,
        det_eps: float = DEFAULT_DET_EPS,
        exclude_source_triangle_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return exact nearest forward intersection `(occluded, first_t_m, first_source_tri_id, first_tri_idx)`.

        Used for physical first-blocker distance stratification (`<50m`, `50-300m`, `300m-4km`)
        and attention hit-rate auditing without early-termination approximation.
        """
        orig = np.asarray(origins_xyz, dtype=np.float64)
        dirs = np.asarray(directions_xyz, dtype=np.float64)
        if orig.ndim == 1:
            orig = orig[None, :]
        if dirs.ndim == 1:
            dirs = np.broadcast_to(dirs[None, :], orig.shape)
        if orig.shape != dirs.shape or orig.shape[-1] != 3:
            raise ValueError(f"origins {orig.shape} and directions {dirs.shape} must be (R, 3)")

        r_total = orig.shape[0]
        best_t = np.full((r_total,), np.inf, dtype=np.float64)
        best_sid = np.zeros((r_total,), dtype=np.uint64)
        best_idx = np.full((r_total,), -1, dtype=np.int64)
        if r_total == 0 or self._root is None or self.num_triangles == 0:
            return np.zeros((r_total,), dtype=bool), best_t, best_sid, best_idx

        excl_ids: Optional[np.ndarray] = None
        if exclude_source_triangle_ids is not None:
            excl_ids = np.asarray(exclude_source_triangle_ids, dtype=np.uint64)
            if excl_ids.ndim == 0:
                excl_ids = np.broadcast_to(excl_ids[None], (r_total,))
            elif excl_ids.shape != (r_total,):
                raise ValueError(f"exclude_source_triangle_ids shape {excl_ids.shape} != ({r_total},)")

        d_norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
        valid_d = d_norm[:, 0] > 1e-12
        safe_d = np.where(d_norm > 1e-12, dirs / np.maximum(d_norm, 1e-12), 0.0)
        active = np.flatnonzero(valid_d)
        if active.size == 0:
            return np.zeros((r_total,), dtype=bool), best_t, best_sid, best_idx

        stack: List[Tuple[_BVHNode, np.ndarray]] = [(self._root, active)]
        while stack:
            node, cand_rays = stack.pop()
            if cand_rays.size == 0:
                continue
            hit_box = self._rays_hit_aabb(
                orig[cand_rays],
                safe_d[cand_rays],
                node.bmin,
                node.bmax,
                float(t_min),
                float(t_max),
            )
            if not np.any(hit_box):
                continue
            sub_rays = cand_rays[hit_box]
            if node.tri_indices is not None:
                leaf_tris = self.triangles_xyz[node.tri_indices]
                hit_mat, t_mat = moller_trumbore_intersect_f64(
                    orig[sub_rays],
                    safe_d[sub_rays],
                    leaf_tris,
                    t_min=t_min,
                    t_max=t_max,
                    det_eps=det_eps,
                    _already_normalized=True,
                )
                if excl_ids is not None:
                    leaf_sids = self.source_triangle_ids[node.tri_indices][None, :]
                    ray_excl = excl_ids[sub_rays][:, None]
                    valid_hit = hit_mat & (leaf_sids != ray_excl)
                    t_mat = np.where(valid_hit, t_mat, np.inf)
                leaf_min_pos = np.argmin(t_mat, axis=1)
                leaf_min_t = t_mat[np.arange(sub_rays.size), leaf_min_pos]
                improved = leaf_min_t < best_t[sub_rays]
                if np.any(improved):
                    imp_rays = sub_rays[improved]
                    imp_tri_idx = node.tri_indices[leaf_min_pos[improved]]
                    best_t[imp_rays] = leaf_min_t[improved]
                    best_idx[imp_rays] = imp_tri_idx
                    best_sid[imp_rays] = self.source_triangle_ids[imp_tri_idx]
            else:
                if node.right is not None:
                    stack.append((node.right, sub_rays))
                if node.left is not None:
                    stack.append((node.left, sub_rays))

        occluded = np.isfinite(best_t)
        return occluded, best_t, best_sid, best_idx

    def query_trajectory_sunlit(
        self,
        receivers_xyz: np.ndarray,
        sun_directions_xyz: np.ndarray,
        *,
        t_min: float = DEFAULT_T_MIN_M,
        t_max: float = DEFAULT_T_MAX_M,
        det_eps: float = DEFAULT_DET_EPS,
        exclude_source_triangle_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute exact `(T, N)` sunlit indicator (`1` = sunlit, `0` = shadow) using the BVH."""
        rec = np.asarray(receivers_xyz, dtype=np.float64)
        suns = np.asarray(sun_directions_xyz, dtype=np.float64)
        t_steps = suns.shape[0]
        n_pts = rec.shape[0]
        sunlit = np.zeros((t_steps, n_pts), dtype=np.int8)
        if n_pts == 0 or t_steps == 0:
            return sunlit

        s_norm = np.linalg.norm(suns, axis=-1)
        valid_sun = (s_norm > 1e-12) & ((suns[:, 1] / np.maximum(s_norm, 1e-12)) > 0.0)
        if not np.any(valid_sun):
            return sunlit
        if self.num_triangles == 0:
            sunlit[valid_sun, :] = 1
            return sunlit

        excl_rec: Optional[np.ndarray] = None
        if exclude_source_triangle_ids is not None:
            excl_rec = np.asarray(exclude_source_triangle_ids, dtype=np.uint64)
            if excl_rec.ndim == 0:
                excl_rec = np.broadcast_to(excl_rec[None], (n_pts,))

        valid_t_idx = np.flatnonzero(valid_sun)
        # Batch rays across multiple time steps (up to 16,384 rays per BVH traversal pass)
        frames_per_pass = max(1, 16384 // max(n_pts, 1))
        for start in range(0, valid_t_idx.size, frames_per_pass):
            t_batch = valid_t_idx[start : start + frames_per_pass]
            b_len = t_batch.size
            raw_s = suns[t_batch]
            o_rep = np.broadcast_to(rec[None, :, :], (b_len, n_pts, 3)).reshape(b_len * n_pts, 3)
            d_rep = np.broadcast_to(raw_s[:, None, :], (b_len, n_pts, 3)).reshape(b_len * n_pts, 3)
            excl_rep = (
                np.broadcast_to(excl_rec[None, :], (b_len, n_pts)).reshape(b_len * n_pts)
                if excl_rec is not None
                else None
            )
            occ = self.query_occluded(
                o_rep,
                d_rep,
                t_min=t_min,
                t_max=t_max,
                det_eps=det_eps,
                exclude_source_triangle_ids=excl_rep,
            ).reshape(b_len, n_pts)
            sunlit[t_batch, :] = np.where(occ, 0, 1).astype(np.int8)
        return sunlit


def exact_moller_trumbore_occluded_f64(
    origins_xyz: np.ndarray,
    sun_directions_xyz: np.ndarray,
    triangles_xyz: np.ndarray,
    *,
    opaque_mask: Optional[np.ndarray] = None,
    source_triangle_ids: Optional[np.ndarray] = None,
    exclude_source_triangle_ids: Optional[np.ndarray] = None,
    t_min: float = DEFAULT_T_MIN_M,
    t_max: float = DEFAULT_T_MAX_M,
    det_eps: float = DEFAULT_DET_EPS,
    ray_chunk_size: int = 512,
    use_aabb_prefilter: bool = True,
) -> np.ndarray:
    """Evaluate exact forward occlusion `(R,)` against full scene triangles in `float64`."""
    tris = np.asarray(triangles_xyz, dtype=np.float64)
    if use_aabb_prefilter and tris.shape[0] > 64:
        bvh = ExactSceneBVH(
            tris,
            opaque_mask=opaque_mask,
            source_triangle_ids=source_triangle_ids,
        )
        return bvh.query_occluded(
            origins_xyz,
            sun_directions_xyz,
            t_min=t_min,
            t_max=t_max,
            det_eps=det_eps,
            exclude_source_triangle_ids=exclude_source_triangle_ids,
        )

    orig = np.asarray(origins_xyz, dtype=np.float64)
    suns = np.asarray(sun_directions_xyz, dtype=np.float64)
    squeeze = False
    if orig.ndim == 1:
        orig = orig[None, :]
        squeeze = True
    if suns.ndim == 1:
        suns = np.broadcast_to(suns[None, :], orig.shape)

    if opaque_mask is not None:
        op = np.asarray(opaque_mask, dtype=bool)
        tris_active = tris[op]
        sids_active = (
            np.asarray(source_triangle_ids, dtype=np.uint64)[op]
            if source_triangle_ids is not None
            else np.arange(tris.shape[0], dtype=np.uint64)[op]
        )
    else:
        tris_active = tris
        sids_active = (
            np.asarray(source_triangle_ids, dtype=np.uint64)
            if source_triangle_ids is not None
            else np.arange(tris.shape[0], dtype=np.uint64)
        )

    r_total = orig.shape[0]
    if r_total == 0:
        return np.zeros((0,), dtype=bool)
    if tris_active.shape[0] == 0:
        out = np.zeros((r_total,), dtype=bool)
        return out[0] if squeeze else out

    excl = (
        np.asarray(exclude_source_triangle_ids, dtype=np.uint64)
        if exclude_source_triangle_ids is not None
        else None
    )
    if excl is not None and excl.ndim == 0:
        excl = np.broadcast_to(excl[None], (r_total,))

    occluded = np.zeros((r_total,), dtype=bool)
    for start in range(0, r_total, ray_chunk_size):
        end = min(r_total, start + ray_chunk_size)
        hit_mat, _ = moller_trumbore_intersect_f64(
            orig[start:end],
            suns[start:end],
            tris_active,
            t_min=t_min,
            t_max=t_max,
            det_eps=det_eps,
        )
        if excl is not None:
            hit_mat = hit_mat & (sids_active[None, :] != excl[start:end, None])
        occluded[start:end] = np.any(hit_mat, axis=1)
    return occluded[0] if squeeze else occluded


def exact_moller_trumbore_trajectory_sunlit_f64(
    receivers_xyz: np.ndarray,
    sun_directions_xyz: np.ndarray,
    triangles_xyz: np.ndarray,
    *,
    opaque_mask: Optional[np.ndarray] = None,
    source_triangle_ids: Optional[np.ndarray] = None,
    exclude_source_triangle_ids: Optional[np.ndarray] = None,
    t_min: float = DEFAULT_T_MIN_M,
    t_max: float = DEFAULT_T_MAX_M,
) -> np.ndarray:
    """Compute exact `(T, N)` sunlit indicator (`1` = sunlit, `0` = shadow) across a trajectory."""
    bvh = ExactSceneBVH(
        triangles_xyz,
        opaque_mask=opaque_mask,
        source_triangle_ids=source_triangle_ids,
    )
    return bvh.query_trajectory_sunlit(
        receivers_xyz,
        sun_directions_xyz,
        t_min=t_min,
        t_max=t_max,
        exclude_source_triangle_ids=exclude_source_triangle_ids,
    )


def assign_building_ids_from_triangles(
    triangles_xyz: np.ndarray,
    *,
    source_instance_ids: Optional[np.ndarray] = None,
    ground_mask: Optional[np.ndarray] = None,
    ground_y_m: Optional[float] = None,
    exclude_ground: bool = True,
    shared_edge_tol_m: float = 1e-3,
) -> np.ndarray:
    """Group 3D building triangles `(M, 3, 3)` into connected `building_id` components (`-1` = fixed ground).

    Fixes Astra P0-6:
    - Uses `source_instance_ids` directly when available.
    - Otherwise builds **3D shared-edge connected components** (quantized at `1 mm` in full 3D
      `(X, Y, Z)`), NEVER 2D `(X, Z)` single-vertex projection! Thus two collocated `(X, Z)`
      surfaces separated by `20 m` in `Y` get distinct `building_id`s, while walls and roof of
      an extruded building sharing 3D roof edges get a single `building_id`.
    - Excludes flat ground-plane triangles (`building_id = -1`) so ground never glues buildings
      together and never deforms.
    """
    tris = np.asarray(triangles_xyz, dtype=np.float64)
    m_count = tris.shape[0]
    if m_count == 0:
        return np.zeros((0,), dtype=np.int64)

    # Identify ground triangles
    e1 = tris[:, 1, :] - tris[:, 0, :]
    e2 = tris[:, 2, :] - tris[:, 0, :]
    fn = np.cross(e1, e2)
    fn_norm = np.linalg.norm(fn, axis=-1)
    abs_ny = np.where(fn_norm > 1e-12, np.abs(fn[:, 1]) / np.maximum(fn_norm, 1e-12), 0.0)
    tri_ymax = tris[:, :, 1].max(axis=1)
    tri_ymin = tris[:, :, 1].min(axis=1)

    if ground_mask is not None:
        is_ground = np.asarray(ground_mask, dtype=bool)
        if is_ground.shape != (m_count,):
            raise ValueError(f"ground_mask shape {is_ground.shape} != ({m_count},)")
    elif exclude_ground:
        scene_ymin = float(tri_ymin.min())
        scene_ymax = float(tri_ymax.max())
        ref_ground_y = float(ground_y_m) if ground_y_m is not None else scene_ymin
        # Only auto-classify bottom horizontal triangles as ground when elevated structure exists above them
        if ground_y_m is not None or (scene_ymax - ref_ground_y) > 0.20:
            is_ground = (abs_ny >= 0.98) & (tri_ymax <= ref_ground_y + 0.02)
        else:
            is_ground = np.zeros((m_count,), dtype=bool)
    else:
        is_ground = np.zeros((m_count,), dtype=bool)

    if source_instance_ids is not None:
        raw_inst = np.asarray(source_instance_ids, dtype=np.int64)
        if raw_inst.shape != (m_count,):
            raise ValueError(f"source_instance_ids shape {raw_inst.shape} != ({m_count},)")
        out = np.full((m_count,), -1, dtype=np.int64)
        inst_map: Dict[int, int] = {}
        for idx in range(m_count):
            if is_ground[idx] or raw_inst[idx] < 0:
                continue
            key = int(raw_inst[idx])
            cid = inst_map.get(key)
            if cid is None:
                cid = len(inst_map)
                inst_map[key] = cid
            out[idx] = cid
        return out

    parent = np.arange(m_count, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    inv_tol = 1.0 / max(float(shared_edge_tol_m), 1e-5)
    q_verts = np.round(tris * inv_tol).astype(np.int64)  # (M, 3, 3)

    # 3D undirected edge -> triangle index
    edge_to_tri: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], int] = {}
    for idx in range(m_count):
        if is_ground[idx]:
            continue
        p0 = (int(q_verts[idx, 0, 0]), int(q_verts[idx, 0, 1]), int(q_verts[idx, 0, 2]))
        p1 = (int(q_verts[idx, 1, 0]), int(q_verts[idx, 1, 1]), int(q_verts[idx, 1, 2]))
        p2 = (int(q_verts[idx, 2, 0]), int(q_verts[idx, 2, 1]), int(q_verts[idx, 2, 2]))
        for va, vb in ((p0, p1), (p1, p2), (p2, p0)):
            if va == vb:
                continue
            ekey = (va, vb) if va <= vb else (vb, va)
            prev = edge_to_tri.get(ekey)
            if prev is None:
                edge_to_tri[ekey] = idx
            else:
                union(idx, prev)

    root_to_compact: Dict[int, int] = {}
    building_ids = np.full((m_count,), -1, dtype=np.int64)
    for idx in range(m_count):
        if is_ground[idx]:
            continue
        r = find(idx)
        cid = root_to_compact.get(r)
        if cid is None:
            cid = len(root_to_compact)
            root_to_compact[r] = cid
        building_ids[idx] = cid
    return building_ids


def compute_building_vertical_anchors(
    triangles_xyz: np.ndarray,
    building_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute true per-building base `y_min_b` and height `H_b = y_max_b - y_min_b` (`b >= 0`)."""
    tris = np.asarray(triangles_xyz, dtype=np.float64)
    b_ids = np.asarray(building_ids, dtype=np.int64)
    active = b_ids >= 0
    if tris.shape[0] == 0 or not np.any(active):
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64), 0

    num_buildings = int(b_ids[active].max()) + 1
    y_min = np.full((num_buildings,), np.inf, dtype=np.float64)
    y_max = np.full((num_buildings,), -np.inf, dtype=np.float64)
    tri_ymin = tris[active, :, 1].min(axis=1)
    tri_ymax = tris[active, :, 1].max(axis=1)
    np.minimum.at(y_min, b_ids[active], tri_ymin)
    np.maximum.at(y_max, b_ids[active], tri_ymax)
    present = np.isfinite(y_min) & np.isfinite(y_max)
    y_min = np.where(present, y_min, 0.0)
    height = np.where(present, np.maximum(y_max - y_min, 0.0), 0.0)
    return y_min, height, num_buildings


def compute_closed_mesh_signed_volume_f64(
    triangles_xyz: np.ndarray,
    *,
    reference_y_m: Optional[float] = None,
) -> float:
    """Compute signed 3D volume `1/6 * sum_t (v0 - ref) . ((v1 - ref) x (v2 - ref))`.

    When `reference_y_m` is set to the flat base `y_min` of an open-bottom extruded building,
    any missing bottom cap on `y = y_min` contributes literally `0`, yielding the exact enclosed
    prism volume!
    """
    tris = np.asarray(triangles_xyz, dtype=np.float64)
    if tris.shape[0] == 0:
        return 0.0
    if reference_y_m is not None:
        ref = np.array([0.0, float(reference_y_m), 0.0], dtype=np.float64)
        v = tris - ref[None, None, :]
    else:
        v = tris
    v0 = v[:, 0, :]
    v1 = v[:, 1, :]
    v2 = v[:, 2, :]
    tet6 = np.sum(v0 * np.cross(v1, v2), axis=-1)
    return float(np.sum(tet6) / 6.0)


def compute_building_effective_areas_f64(
    triangles_xyz: np.ndarray,
    building_ids: np.ndarray,
) -> np.ndarray:
    """Compute effective planform area weight `A_b > 0` (`m^2`) for each building `b in 0..B-1`.

    Uses exact enclosed building volume divided by `H_b` when closed/capped (`V_{0,b} / H_b`),
    so `V_b(H_b + delta_h_b) - V_b(H_b) == A_b * delta_h_b` holds to machine precision!
    Falls back to horizontal roof projected area or XZ footprint area for uncapped walls/canopies.
    """
    tris = np.asarray(triangles_xyz, dtype=np.float64)
    b_ids = np.asarray(building_ids, dtype=np.int64)
    y_min, height, num_b = compute_building_vertical_anchors(tris, b_ids)
    if num_b == 0:
        return np.zeros((0,), dtype=np.float64)

    areas = np.zeros((num_b,), dtype=np.float64)
    for b in range(num_b):
        sel = b_ids == b
        if not np.any(sel):
            areas[b] = 1.0
            continue
        sub = tris[sel]
        hb = float(height[b])
        if hb > 1e-6:
            vol_abs = abs(compute_closed_mesh_signed_volume_f64(sub, reference_y_m=float(y_min[b])))
            if vol_abs > 1e-6:
                areas[b] = vol_abs / hb
                continue
        # Fallback: horizontal projection of upper/roof faces or XZ bounding/wall length
        cross_y = np.abs(
            np.cross(sub[:, 1, :] - sub[:, 0, :], sub[:, 2, :] - sub[:, 0, :])[:, 1]
        )
        horiz_proj = 0.5 * float(np.sum(cross_y))
        if horiz_proj > 1e-6:
            areas[b] = horiz_proj
        else:
            dx = float(sub[:, :, 0].max() - sub[:, :, 0].min())
            dz = float(sub[:, :, 2].max() - sub[:, :, 2].min())
            areas[b] = max(dx * dz, math.hypot(dx, dz) * 0.25, 1.0)
    return areas


def verify_extruded_building_manifold_closure(
    triangles_xyz: np.ndarray,
    building_ids: np.ndarray,
) -> Dict[str, Any]:
    """Verify per-building base-resting solid closure & exact Jacobian consistency (`dV/dh == A_b`).

    Checks:
    1. Every undirected 3D edge `(va, vb)` above the flat base plane `y_min_b + 1e-3` is incident to an
       even number of faces of building `b` (zero open cracks or unclosed single-triangle surfaces;
       both 12-triangle watertight prisms and 10-triangle ground-resting extruded buildings pass).
    2. Finite-difference enclosed volume Jacobian matches `compute_building_effective_areas_f64`:
       `|(V_b(+1m) - V_b(0m)) - A_b| <= 1e-4 * max(1.0, A_b)`.
    """
    tris = np.asarray(triangles_xyz, dtype=np.float64)
    b_ids = np.asarray(building_ids, dtype=np.int64)
    y_min, height, num_b = compute_building_vertical_anchors(tris, b_ids)
    if num_b == 0 or tris.shape[0] == 0:
        return {
            "all_closed_manifold": True,
            "num_buildings": int(num_b),
            "nonclosed_building_ids": [],
            "max_jacobian_rel_error": 0.0,
        }

    areas = compute_building_effective_areas_f64(tris, b_ids)
    inv_tol = 1000.0  # 1 mm vertex quantization
    q_verts = np.round(tris * inv_tol).astype(np.int64)
    nonclosed_ids: List[int] = []
    max_jac_rel_err = 0.0

    for b in range(num_b):
        sel = b_ids == b
        if not np.any(sel):
            continue
        sub_tris = tris[sel]
        sub_q = q_verts[sel]
        hb = float(height[b])
        yb = float(y_min[b])
        if hb <= 1e-5 or sub_tris.shape[0] < 4:
            nonclosed_ids.append(int(b))
            continue

        edge_counts: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], int] = {}
        base_q_thresh = int(math.floor((yb + 1e-3) * inv_tol))
        for ti in range(sub_q.shape[0]):
            p0 = (int(sub_q[ti, 0, 0]), int(sub_q[ti, 0, 1]), int(sub_q[ti, 0, 2]))
            p1 = (int(sub_q[ti, 1, 0]), int(sub_q[ti, 1, 1]), int(sub_q[ti, 1, 2]))
            p2 = (int(sub_q[ti, 2, 0]), int(sub_q[ti, 2, 1]), int(sub_q[ti, 2, 2]))
            for va, vb in ((p0, p1), (p1, p2), (p2, p0)):
                if va == vb:
                    continue
                # Edges lying strictly on the ground base plane y == y_min_b may belong to an open-bottom extrusion
                if va[1] <= base_q_thresh and vb[1] <= base_q_thresh:
                    continue
                ekey = (va, vb) if va <= vb else (vb, va)
                edge_counts[ekey] = edge_counts.get(ekey, 0) + 1

        has_open_boundary = any((cnt % 2) != 0 for cnt in edge_counts.values())
        v0 = abs(compute_closed_mesh_signed_volume_f64(sub_tris, reference_y_m=yb))
        dh_test = np.zeros((num_b,), dtype=np.float64)
        dh_test[b] = 1.0
        pert_sub = perturb_buildings_consistent_f64(
            sub_tris, np.zeros((sub_tris.shape[0],), dtype=np.int64), np.array([1.0], dtype=np.float64)
        )
        v1 = abs(compute_closed_mesh_signed_volume_f64(pert_sub, reference_y_m=yb))
        dv = v1 - v0
        jac_rel_err = abs(dv - float(areas[b])) / max(1.0, abs(float(areas[b])))
        max_jac_rel_err = max(max_jac_rel_err, float(jac_rel_err))
        if has_open_boundary or v0 <= 1e-6 or jac_rel_err > 1e-4:
            nonclosed_ids.append(int(b))

    return {
        "all_closed_manifold": len(nonclosed_ids) == 0,
        "num_buildings": int(num_b),
        "nonclosed_building_ids": nonclosed_ids,
        "max_jacobian_rel_error": float(max_jac_rel_err),
    }


def project_area_weighted_volume_conservation_f64(
    raw_delta_h: np.ndarray,
    areas_m2: np.ndarray,
    heights_m: np.ndarray,
    *,
    max_delta_h_m: float = 15.0,
    min_building_height_m: float = 0.05,
    _return_lam_star: bool = False,
) -> Any:
    """Exact $O(B \\log B)$ piecewise-linear breakpoint KKT Euclidean projection onto
    `{L_b <= d_b <= U_b, sum_b A_b * d_b == 0}` (compensated IEEE-754 volume residual).
    """
    v = np.asarray(raw_delta_h, dtype=np.float64)
    a = np.asarray(areas_m2, dtype=np.float64)
    h = np.asarray(heights_m, dtype=np.float64)
    if v.size == 0:
        return (v.copy(), 0.0) if _return_lam_star else v.copy()
    if np.any(a <= 0.0):
        raise ValueError("All building effective areas `A_b` must be strictly positive.")

    bound = float(max_delta_h_m)
    min_drop = np.where(
        h > 1e-9,
        np.minimum(0.0, float(min_building_height_m) - h),
        np.minimum(0.0, -bound),
    )
    lo = np.maximum(-bound, min_drop)
    hi = np.full_like(lo, bound)

    # Note 0 is always in [lo, hi], so feasible set is always non-empty; if B == 1, d_0 = 0.0.
    if v.size == 1:
        z_out = np.zeros_like(v)
        lam_single = float(v[0] / a[0])
        return (z_out, lam_single) if _return_lam_star else z_out

    # Exact piecewise-linear monotone decreasing root-finding on:
    #   f(lam) = sum_b a_b * clip(v_b - lam * a_b, lo_b, hi_b) == 0.
    # Each term transitions from hi_b -> (v_b - lam * a_b) at lam_lo_b = (v_b - hi_b) / a_b,
    # and from (v_b - lam * a_b) -> lo_b at lam_hi_b = (v_b - lo_b) / a_b >= lam_lo_b.
    lam_lo = (v - hi) / a
    lam_hi = (v - lo) / a
    breakpoints = np.unique(np.concatenate([lam_lo, lam_hi], axis=0))

    if breakpoints.size == 1:
        lam0 = float(breakpoints[0])
        res0 = np.clip(v - lam0 * a, lo, hi)
        return (res0, lam0) if _return_lam_star else res0

    # Binary search over exact sorted breakpoints in O(B log B) total time (<= ceil(log2(2B)) evals)
    idx_lo = 0
    idx_hi = int(breakpoints.size) - 1
    f_lo = float(np.dot(a, np.clip(v - breakpoints[idx_lo] * a, lo, hi)))
    f_hi = float(np.dot(a, np.clip(v - breakpoints[idx_hi] * a, lo, hi)))

    if f_lo <= 0.0:
        lam0 = float(breakpoints[idx_lo])
        res0 = np.clip(v - lam0 * a, lo, hi)
        return (res0, lam0) if _return_lam_star else res0
    if f_hi >= 0.0:
        lam0 = float(breakpoints[idx_hi])
        res0 = np.clip(v - lam0 * a, lo, hi)
        return (res0, lam0) if _return_lam_star else res0

    while idx_hi - idx_lo > 1:
        mid = (idx_lo + idx_hi) // 2
        f_mid = float(np.dot(a, np.clip(v - breakpoints[mid] * a, lo, hi)))
        if f_mid == 0.0:
            idx_lo = idx_hi = mid
            break
        if f_mid > 0.0:
            idx_lo = mid
        else:
            idx_hi = mid

    tau_l = float(breakpoints[idx_lo])
    tau_r = float(breakpoints[idx_hi])
    tau_mid = 0.5 * (tau_l + tau_r)

    # On the open interval (tau_l, tau_r), the active set is strictly constant and f(lam) is EXACTLY affine!
    active = (lam_lo < tau_mid) & (tau_mid < lam_hi)
    if np.any(active):
        clamped_upper = tau_mid <= lam_lo
        clamped_lower = tau_mid >= lam_hi
        fixed_vol = (
            math.fsum((a[clamped_upper] * hi[clamped_upper]).tolist())
            + math.fsum((a[clamped_lower] * lo[clamped_lower]).tolist())
            + math.fsum((a[active] * v[active]).tolist())
        )
        slope = math.fsum(((a[active] ** 2)).tolist())
        lam_star = fixed_vol / max(slope, 1e-30)
        lam_star = min(max(lam_star, tau_l), tau_r)
        # 2-step compensated float64 Newton polish on the exact affine segment
        for _ in range(2):
            d_cur = np.clip(v - lam_star * a, lo, hi)
            res = math.fsum((a * d_cur).tolist())
            free_now = (d_cur > lo) & (d_cur < hi)
            s_now = math.fsum(((a[free_now] ** 2)).tolist()) if np.any(free_now) else slope
            if s_now <= 1e-30:
                break
            lam_star += res / s_now
    else:
        lam_star = tau_mid

    out = np.clip(v - lam_star * a, lo, hi)
    # Cascaded descending-area compensated fsum polish + 2-variable IEEE-754 ULP lattice refinement
    unclipped_now = v - lam_star * a
    free_idx = np.flatnonzero((unclipped_now > lo + 1e-13) & (unclipped_now < hi - 1e-13))
    if free_idx.size > 0:
        order_desc = free_idx[np.argsort(-a[free_idx])]
        for j_raw in order_desc:
            rem = math.fsum((a * out).tolist())
            if abs(rem) == 0.0:
                break
            j = int(j_raw)
            cand = float(np.clip(out[j] - rem / float(a[j]), lo[j], hi[j]))
            out[j] = cand

        rem = math.fsum((a * out).tolist())
        if abs(rem) > 1e-15:
            j0 = int(order_desc[-1])
            base_prods = (a * out).tolist()
            if order_desc.size >= 2:
                j1 = int(order_desc[-2])
                base_0 = float(out[j0])
                base_1 = float(out[j1])
                a0 = float(a[j0])
                a1 = float(a[j1])
                lo0, hi0 = float(lo[j0]), float(hi[j0])
                lo1, hi1 = float(lo[j1]), float(hi[j1])
                other_prods = [
                    base_prods[i] for i in range(len(base_prods)) if i != j0 and i != j1
                ]
                other_sum = math.fsum(other_prods)
                best_abs = abs(rem)
                best_pair = (base_0, base_1)

                cur_up = base_0
                cur_dn = base_0
                cands_0 = [base_0]
                for _ in range(128):
                    cur_up = math.nextafter(cur_up, math.inf)
                    if cur_up <= hi0:
                        cands_0.append(cur_up)
                    cur_dn = math.nextafter(cur_dn, -math.inf)
                    if cur_dn >= lo0:
                        cands_0.append(cur_dn)

                for x0 in cands_0:
                    p0 = a0 * x0
                    target_p1 = -(other_sum + p0)
                    x1_est = float(np.clip(target_p1 / a1, lo1, hi1))
                    x1_up = math.nextafter(x1_est, math.inf)
                    x1_dn = math.nextafter(x1_est, -math.inf)
                    for x1 in (x1_est, x1_up, x1_dn):
                        if lo1 <= x1 <= hi1:
                            r_try = abs(math.fsum([*other_prods, p0, a1 * x1]))
                            if r_try < best_abs:
                                best_abs = r_try
                                best_pair = (x0, x1)
                                if best_abs == 0.0:
                                    break
                    if best_abs == 0.0:
                        break
                out[j0], out[j1] = best_pair
            else:
                a0 = float(a[j0])
                lo0, hi0 = float(lo[j0]), float(hi[j0])
                other_prods = [base_prods[i] for i in range(len(base_prods)) if i != j0]
                x0_est = float(np.clip(-math.fsum(other_prods) / a0, lo0, hi0))
                best_x0 = float(out[j0])
                best_abs = abs(rem)
                for x0 in (
                    x0_est,
                    math.nextafter(x0_est, math.inf),
                    math.nextafter(x0_est, -math.inf),
                ):
                    if lo0 <= x0 <= hi0:
                        r_try = abs(math.fsum([*other_prods, a0 * x0]))
                        if r_try < best_abs:
                            best_abs = r_try
                            best_x0 = x0
                out[j0] = best_x0
    return (out, float(lam_star)) if _return_lam_star else out


class DifferentiableAreaWeightedVolumeKKT(torch.autograd.Function):
    """Differentiable exact KKT projection onto `{L_b <= d_b <= U_b, sum_b A_b * d_b == 0}`.

    Forward solves exact $O(B \\log B)$ breakpoint KKT in `float64` and records the exact
    solver active set via `unclipped = v - lam_star * a` (fixing Astra `P1` near-boundary interior
    misclassification at `1e-10` from box bounds); backward applies the exact implicit-KKT tangent
    hyperplane projection on the free support `F = {b : L_b < v_b - lam* a_b < U_b}`.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        raw_delta_h: torch.Tensor,
        areas_m2: torch.Tensor,
        heights_m: torch.Tensor,
        max_delta_h_m: float = 15.0,
        min_building_height_m: float = 0.05,
    ) -> torch.Tensor:
        v_np = raw_delta_h.detach().cpu().numpy().astype(np.float64)
        a_np = areas_m2.detach().cpu().numpy().astype(np.float64)
        h_np = heights_m.detach().cpu().numpy().astype(np.float64)
        proj_np, lam_star = project_area_weighted_volume_conservation_f64(
            v_np,
            a_np,
            h_np,
            max_delta_h_m=float(max_delta_h_m),
            min_building_height_m=float(min_building_height_m),
            _return_lam_star=True,
        )
        bound = float(max_delta_h_m)
        min_drop = np.where(
            h_np > 1e-9,
            np.minimum(0.0, float(min_building_height_m) - h_np),
            np.minimum(0.0, -bound),
        )
        lo_np = np.maximum(-bound, min_drop)
        hi_np = np.full_like(lo_np, bound)
        unclipped_np = v_np - float(lam_star) * a_np
        eps_kkt = 1e-14 * np.maximum(1.0, np.maximum(np.abs(lo_np), np.abs(hi_np)))
        free_np = (unclipped_np > (lo_np + eps_kkt)) & (unclipped_np < (hi_np - eps_kkt))

        proj_t = torch.as_tensor(proj_np, dtype=raw_delta_h.dtype, device=raw_delta_h.device)
        free_t = torch.as_tensor(free_np, dtype=torch.bool, device=raw_delta_h.device)
        areas_t = areas_m2.to(dtype=raw_delta_h.dtype, device=raw_delta_h.device)
        ctx.save_for_backward(free_t, areas_t)
        return proj_t

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore[override]
        free_t, areas_t = ctx.saved_tensors
        if grad_output.numel() <= 1 or not bool(free_t.any().item()):
            return torch.zeros_like(grad_output), None, None, None, None
        free_f = free_t.to(dtype=grad_output.dtype)
        a_free = areas_t * free_f
        denom = (a_free * a_free).sum()
        g_free = grad_output * free_f
        if float(denom.item()) <= 1e-20:
            return g_free, None, None, None, None
        proj_coeff = (a_free * g_free).sum() / denom
        grad_in = free_f * (grad_output - proj_coeff * areas_t)
        return grad_in, None, None, None, None


def perturb_buildings_consistent_f64(
    triangles_xyz: np.ndarray,
    building_ids: np.ndarray,
    delta_h_per_building: np.ndarray,
    *,
    min_building_height_m: float = 0.05,
) -> np.ndarray:
    """Apply manifold-consistent vertical perturbation `delta_h_b` grouped strictly by `building_id`.

    Fixes Astra P0-6:
    - Fixed ground (`building_id < 0`) is never modified.
    - At `delta_h_b == 0.0`, `eff_delta_b == 0.0`, so `y' = y + 0.0 == y` **bit-for-bit identical**
      even for `0.75 m` low walls, canopies, and sloped roofs!
    - Shared 3D edges within each building transform via a single affine vertical map, preventing
      cracks or self-intersections on sloped and flat roofs alike.
    """
    tris = np.asarray(triangles_xyz, dtype=np.float64).copy()
    if tris.shape[0] == 0:
        return tris
    b_ids = np.asarray(building_ids, dtype=np.int64)
    delta_h = np.asarray(delta_h_per_building, dtype=np.float64)
    y_min, height, num_b = compute_building_vertical_anchors(tris, b_ids)
    if delta_h.shape != (num_b,):
        raise ValueError(f"delta_h_per_building shape {delta_h.shape} != ({num_b},)")
    if num_b == 0:
        return tris

    active = b_ids >= 0
    if not np.any(active):
        return tris

    b_act = b_ids[active]
    h_act = height[b_act][:, None]
    ymin_act = y_min[b_act][:, None]
    dh_act = delta_h[b_act][:, None]

    min_drop = np.where(
        h_act > 1e-9,
        np.minimum(0.0, float(min_building_height_m) - h_act),
        np.minimum(0.0, float(min_building_height_m) - ymin_act),
    )
    eff_dh = np.maximum(dh_act, min_drop)

    y_orig = tris[active, :, 1]
    rel = np.where(h_act > 1e-9, (y_orig - ymin_act) / np.where(h_act > 1e-9, h_act, 1.0), 1.0)
    tris[active, :, 1] = y_orig + eff_dh * rel
    return tris


def perturb_buildings_consistent_torch(
    triangles_xyz: torch.Tensor,
    building_ids: torch.Tensor,
    delta_h_per_building: torch.Tensor,
    y_min_per_building: torch.Tensor,
    height_per_building: torch.Tensor,
    *,
    min_building_height_m: float = 0.05,
) -> torch.Tensor:
    """Differentiable PyTorch twin of `perturb_buildings_consistent_f64` (exact identity at `Δh = 0`)."""
    if triangles_xyz.shape[0] == 0 or delta_h_per_building.numel() == 0:
        return triangles_xyz

    active = building_ids >= 0
    safe_idx = building_ids.clamp_min(0)
    h_b = height_per_building[safe_idx].unsqueeze(-1)
    ymin_b = y_min_per_building[safe_idx].unsqueeze(-1)
    dh_b = delta_h_per_building[safe_idx].unsqueeze(-1)

    zero = torch.zeros_like(h_b)
    min_drop = torch.where(
        h_b > 1e-9,
        torch.minimum(zero, float(min_building_height_m) - h_b),
        torch.minimum(zero, float(min_building_height_m) - ymin_b),
    )
    eff_dh = torch.maximum(dh_b, min_drop)

    y_orig = triangles_xyz[:, :, 1]
    safe_h = torch.where(h_b > 1e-9, h_b, torch.ones_like(h_b))
    rel = torch.where(h_b > 1e-9, (y_orig - ymin_b) / safe_h, torch.ones_like(y_orig))
    y_pert = y_orig + eff_dh * rel
    y_new = torch.where(active.unsqueeze(-1), y_pert, y_orig)
    return torch.stack([triangles_xyz[:, :, 0], y_new, triangles_xyz[:, :, 2]], dim=-1)


def _differentiable_soft_moller_trumbore_sunlit(
    receivers_xyz: torch.Tensor,
    sun_directions_xyz: torch.Tensor,
    triangles_xyz: torch.Tensor,
    *,
    building_ids: Optional[torch.Tensor] = None,
    source_triangle_ids: Optional[torch.Tensor] = None,
    exclude_source_triangle_ids: Optional[torch.Tensor] = None,
    temperature_m: float = 0.5,
    t_min: float = DEFAULT_T_MIN_M,
    t_max: float = DEFAULT_T_MAX_M,
) -> torch.Tensor:
    """Smooth differentiable full-day sunlight surrogate aligned with exact Möller–Trumbore (`P0-8`).

    Key exact-alignment guarantees:
    1. Evaluates ALL triangles (vertical, sloped, AND horizontal) via exact 3D ray-orthogonal
       projected signed edge margins `(m_0, m_1, m_2)` — NEVER expanding a single right-angled
       vertical triangle into its bounding rectangle!
    2. Enforces exact forward ray-interval admissibility `(t_plane > t_min) & (t_plane <= t_max)`
       and optional per-receiver `exclude_source_triangle_ids` self-face exclusion (preventing
       `~0.5` self-occlusion at `t <= t_min` or `t = 0`).
    3. Groups co-planar faces (`plane_inv_idx`) before smooth cross-plane union so shared
       tessellation diagonals satisfy `sigmoid(d/tau) + sigmoid(-d/tau) == 1.0` without seam leaks,
       while retaining the Jordan-Curve Solid-Prism filter for horizontal caps of walled prisms.
    """
    t_steps = sun_directions_xyz.shape[0]
    n_pts = receivers_xyz.shape[0]
    m_tris = triangles_xyz.shape[0]
    if m_tris == 0:
        above = (sun_directions_xyz[:, 1:2] > 0.0).to(receivers_xyz.dtype)
        return above.expand(t_steps, n_pts)

    s_norm = torch.linalg.norm(sun_directions_xyz, dim=-1, keepdim=True).clamp_min(1e-12)
    s_unit = sun_directions_xyz / s_norm
    above_horizon = s_unit[:, 1] > 0.0

    v0 = triangles_xyz[:, 0, :]
    v1 = triangles_xyz[:, 1, :]
    v2 = triangles_xyz[:, 2, :]

    raw_fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fn_len = torch.linalg.norm(raw_fn, dim=-1, keepdim=True).clamp_min(1e-12)
    unit_fn = raw_fn / fn_len
    abs_ny = unit_fn[:, 1].abs()
    is_vert = abs_ny <= 1e-3

    # Canonicalize plane normal hemisphere across ALL triangles so co-planar wall/roof pairs share a plane key
    fn_det = unit_fn.detach()
    flip = (
        (fn_det[:, 1] < -1e-5)
        | ((fn_det[:, 1].abs() <= 1e-5) & (fn_det[:, 0] < -1e-5))
        | ((fn_det[:, 1].abs() <= 1e-5) & (fn_det[:, 0].abs() <= 1e-5) & (fn_det[:, 2] < 0.0))
    )
    fn_canon = torch.where(flip.unsqueeze(-1), -fn_det, fn_det)
    d_off = (fn_canon * v0.detach()).sum(dim=-1, keepdim=True)
    if building_ids is not None and building_ids.numel() == m_tris:
        b_col = building_ids.detach().to(fn_canon.dtype).unsqueeze(-1)
        plane_feat = torch.cat([b_col, torch.round(fn_canon * 200.0), torch.round(d_off * 50.0)], dim=-1)
    else:
        plane_feat = torch.cat([torch.round(fn_canon * 200.0), torch.round(d_off * 50.0)], dim=-1)
    _, plane_inv_idx = torch.unique(plane_feat.to(torch.int64), dim=0, return_inverse=True)
    num_planes = int(plane_inv_idx.max().item()) + 1 if plane_inv_idx.numel() > 0 else 0

    # Per-(receiver, triangle) admissibility mask: self-face exclusion + Jordan-Curve walled horizontal cap filter
    tri_rec_active_nm = torch.ones(
        (n_pts, m_tris), dtype=receivers_xyz.dtype, device=receivers_xyz.device
    )
    if exclude_source_triangle_ids is not None:
        if source_triangle_ids is not None:
            sids_t = source_triangle_ids.to(device=receivers_xyz.device, dtype=torch.int64)
        else:
            sids_t = torch.arange(m_tris, device=receivers_xyz.device, dtype=torch.int64)
        excl_t = exclude_source_triangle_ids.to(device=receivers_xyz.device, dtype=torch.int64)
        if excl_t.ndim == 0:
            excl_t = excl_t.view(1).expand(n_pts)
        not_self_nm = (sids_t.unsqueeze(0) != excl_t.unsqueeze(1)).to(receivers_xyz.dtype)
        tri_rec_active_nm = tri_rec_active_nm * not_self_nm

    is_horiz_all = abs_ny.detach() > 0.999
    if building_ids is not None and building_ids.numel() == m_tris and bool(is_horiz_all.any().item()):
        b_all = building_ids.detach()
        max_bid = int(b_all.max().item()) if b_all.numel() > 0 else -1
        if max_bid >= 0:
            b_has_vert = torch.zeros((max_bid + 1,), dtype=torch.bool, device=receivers_xyz.device)
            vert_bids = b_all[is_vert.detach() & (b_all >= 0)]
            if vert_bids.numel() > 0:
                b_has_vert[vert_bids] = True
            b_ymin = torch.full(
                (max_bid + 1,), 1e9, dtype=receivers_xyz.dtype, device=receivers_xyz.device
            )
            valid_b_mask = b_all >= 0
            if bool(valid_b_mask.any().item()):
                tri_ymin_det = triangles_xyz[:, :, 1].detach().min(dim=-1).values
                b_ymin.scatter_reduce_(
                    0, b_all[valid_b_mask], tri_ymin_det[valid_b_mask], reduce="amin", include_self=True
                )

            horiz_idx = torch.nonzero(is_horiz_all, as_tuple=False).squeeze(-1)
            b_h = b_all[horiz_idx]
            safe_bh = b_h.clamp_min(0)
            walled_cap_h = (b_h >= 0) & b_has_vert[safe_bh]
            if bool(walled_cap_h.any().item()):
                p_xz = receivers_xyz[:, [0, 2]].detach()[:, None, :]
                t0_xz = triangles_xyz[horiz_idx, 0][:, [0, 2]].detach()[None, :, :]
                t1_xz = triangles_xyz[horiz_idx, 1][:, [0, 2]].detach()[None, :, :]
                t2_xz = triangles_xyz[horiz_idx, 2][:, [0, 2]].detach()[None, :, :]
                area2_sign = torch.where(
                    ((t1_xz - t0_xz)[..., 0] * (t2_xz - t0_xz)[..., 1]
                     - (t1_xz - t0_xz)[..., 1] * (t2_xz - t0_xz)[..., 0]) >= 0.0,
                    torch.ones((1, int(horiz_idx.numel())), dtype=receivers_xyz.dtype, device=receivers_xyz.device),
                    -torch.ones((1, int(horiz_idx.numel())), dtype=receivers_xyz.dtype, device=receivers_xyz.device),
                )
                e_dists = []
                for pa, pb in ((t0_xz, t1_xz), (t1_xz, t2_xz), (t2_xz, t0_xz)):
                    ed = pb - pa
                    elen = torch.linalg.norm(ed, dim=-1).clamp_min(1e-6)
                    rel_p = p_xz - pa
                    cross2 = (ed[..., 0] * rel_p[..., 1] - ed[..., 1] * rel_p[..., 0]) * area2_sign
                    e_dists.append(cross2 / elen)
                in_fp_tri_nh = torch.minimum(torch.minimum(e_dists[0], e_dists[1]), e_dists[2]) >= -0.05
                h_plane_inv = plane_inv_idx[horiz_idx]
                if num_planes > 0:
                    plane_in_fp = torch.zeros(
                        (n_pts, num_planes), dtype=torch.int32, device=receivers_xyz.device
                    )
                    plane_in_fp.scatter_reduce_(
                        1,
                        h_plane_inv.unsqueeze(0).expand(n_pts, -1),
                        in_fp_tri_nh.to(torch.int32),
                        reduce="amax",
                        include_self=True,
                    )
                    in_fp_nh = torch.gather(
                        plane_in_fp > 0, 1, h_plane_inv.unsqueeze(0).expand(n_pts, -1)
                    )
                else:
                    in_fp_nh = in_fp_tri_nh

                cap_ymax_h = triangles_xyz[horiz_idx, :, 1].detach().max(dim=-1).values
                ymin_h = b_ymin[safe_bh]
                rec_y_col = receivers_xyz[:, 1:2].detach()
                floating_nh = ymin_h[None, :] > (rec_y_col + 0.1)
                above_rec_nh = cap_ymax_h[None, :] > (rec_y_col + 1e-3)
                allow_walled_nh = above_rec_nh & (floating_nh | in_fp_nh)
                horiz_factor = torch.where(
                    walled_cap_h[None, :],
                    allow_walled_nh.to(receivers_xyz.dtype),
                    torch.ones_like(allow_walled_nh, dtype=receivers_xyz.dtype),
                )
                tri_rec_active_nm[:, horiz_idx] = tri_rec_active_nm[:, horiz_idx] * horiz_factor

    tau = max(float(temperature_m), 1e-3)
    tmin_f = float(t_min)
    tmax_f = float(t_max)
    beta = 0.08

    r0 = v0[None, :, :] - receivers_xyz[:, None, :]  # (N, M, 3)
    r1 = v1[None, :, :] - receivers_xyz[:, None, :]
    r2 = v2[None, :, :] - receivers_xyz[:, None, :]
    e01 = r1 - r0
    e12 = r2 - r1
    e20 = r0 - r2
    fn_nm = raw_fn[None, :, :]  # (1, M, 3)
    r0_dot_fn = (r0 * fn_nm).sum(dim=-1)  # (N, M)

    sunlit_rows = []
    for t_idx in range(t_steps):
        if not bool(above_horizon[t_idx].item()):
            sunlit_rows.append(
                torch.zeros((n_pts,), dtype=receivers_xyz.dtype, device=receivers_xyz.device)
            )
            continue
        d_view = s_unit[t_idx].view(1, 1, 3)
        orient = (fn_nm * d_view).sum(dim=-1)  # (1, M)
        valid_proj = orient.abs() > 1e-9
        safe_orient = torch.where(valid_proj, orient, torch.ones_like(orient))
        t_plane = r0_dot_fn / safe_orient  # (N, M)
        in_ray_interval = (
            valid_proj
            & (t_plane.detach() > tmin_f)
            & (t_plane.detach() <= tmax_f)
        )
        orient_sign = torch.where(
            orient >= 0.0, torch.ones_like(orient), -torch.ones_like(orient)
        )

        margins = []
        for ra, edge in ((r0, e01), (r1, e12), (r2, e20)):
            edge_perp = edge - (edge * d_view).sum(dim=-1, keepdim=True) * d_view
            perp_len = torch.linalg.norm(edge_perp, dim=-1).clamp_min(1e-5)
            num = (torch.cross(edge, -ra, dim=-1) * d_view).sum(dim=-1) * orient_sign
            margins.append(num / perp_len)
        min_margin_m = torch.minimum(torch.minimum(margins[0], margins[1]), margins[2])

        occ_tri = (
            in_ray_interval.to(receivers_xyz.dtype)
            * tri_rec_active_nm
            * torch.sigmoid(min_margin_m / tau)
        )
        if num_planes > 0:
            occ_planes = torch.zeros(
                (n_pts, num_planes), dtype=receivers_xyz.dtype, device=receivers_xyz.device
            )
            occ_planes.scatter_add_(
                1, plane_inv_idx.unsqueeze(0).expand(n_pts, -1), occ_tri
            )
            occ_all = occ_planes.clamp(0.0, 1.0)
        else:
            occ_all = occ_tri.clamp(0.0, 1.0)

        # Zero-dilution C^inf Boltzmann max: planes with p == 0 contribute 0 * exp(0) == 0!
        scaled_w = occ_all * torch.exp(occ_all / beta)
        occ_ray = ((scaled_w * occ_all).sum(dim=-1) / (scaled_w.sum(dim=-1) + 1e-12)).clamp(0.0, 1.0)
        sunlit_rows.append(1.0 - occ_ray)
    return torch.stack(sunlit_rows, dim=0)


def _apply_editable_height_slack(
    tris: np.ndarray,
    *,
    building_ids: Optional[np.ndarray] = None,
    ground_mask: Optional[np.ndarray] = None,
    height_slack_m: float = 15.0,
) -> np.ndarray:
    """Elevate editable building triangles (`building_ids >= 0`) by `height_slack_m`."""
    slack = max(0.0, float(height_slack_m))
    if slack <= 0.0:
        return tris
    b_ids_arr = (
        np.asarray(building_ids, dtype=np.int64)
        if building_ids is not None
        else assign_building_ids_from_triangles(tris, ground_mask=ground_mask)
    )
    editable_mask = b_ids_arr >= 0
    tris_reach = tris.copy()
    if np.any(editable_mask):
        tris_reach[editable_mask, :, 1] += slack
    return tris_reach


def _resolve_model_azimuth_bins(
    rind_model: Optional[torch.nn.Module],
    num_azimuth_bins: Optional[int] = None,
) -> int:
    """Infer `num_azimuth_bins` from explicit arg or model metadata, warning on fallback."""
    if num_azimuth_bins is not None:
        az_bins = int(num_azimuth_bins)
    elif rind_model is not None:
        if hasattr(rind_model, "num_azimuth_bins"):
            az_bins = int(getattr(rind_model, "num_azimuth_bins"))
        else:
            warnings.warn(
                "rind_model has no `num_azimuth_bins` attribute (missing candidate_protocol metadata); "
                "falling back to num_azimuth_bins=24.",
                UserWarning,
                stacklevel=3,
            )
            az_bins = 24
    else:
        az_bins = 24
    if az_bins < 0:
        raise ValueError(f"num_azimuth_bins must be >= 0, got {az_bins}")
    return az_bins


def _preselect_candidate_triangle_indices(
    receivers_xyz: np.ndarray,
    triangles_xyz: np.ndarray,
    *,
    opaque_mask: Optional[np.ndarray] = None,
    source_triangle_ids: Optional[np.ndarray] = None,
    building_ids: Optional[np.ndarray] = None,
    ground_mask: Optional[np.ndarray] = None,
    num_azimuth_bins: int = 24,
    max_candidates: int = 8,
    horizon_m: float = DEFAULT_T_MAX_M,
    height_slack_m: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(az_tri_idx, az_mask)` of shapes `(N, AzBins, K)` aligned with `select_azimuth_sector_candidates`.

    Uses `fullday_data.select_azimuth_sector_candidates` with synthetic unique indices `0..M-1`
    and an optional upward `height_slack_m` applied **strictly to editable building triangles
    (`building_ids >= 0`)** — NEVER elevating ground/background (`building_ids < 0`)!
    """
    from .fullday_data import select_azimuth_sector_candidates

    rec = np.asarray(receivers_xyz, dtype=np.float64)
    tris = np.asarray(triangles_xyz, dtype=np.float64)
    n_pts = rec.shape[0]
    m_tris = tris.shape[0]
    az_bins = int(num_azimuth_bins)
    k_max = int(max_candidates)
    if n_pts == 0 or m_tris == 0:
        return (
            np.zeros((n_pts, az_bins, k_max), dtype=np.int64),
            np.zeros((n_pts, az_bins, k_max), dtype=bool),
        )

    slack = max(0.0, float(height_slack_m))
    if slack > 0.0:
        tris_reach = _apply_editable_height_slack(
            tris, building_ids=building_ids, ground_mask=ground_mask, height_slack_m=slack
        )
    else:
        tris_reach = tris

    synth_idx_ids = np.arange(m_tris, dtype=np.uint64)
    _, _, az_mask, az_idx_u64 = select_azimuth_sector_candidates(
        rec,
        tris_reach,
        scene_opaque=opaque_mask,
        scene_source_triangle_ids=synth_idx_ids,
        num_azimuth_bins=az_bins,
        max_candidates=k_max,
        horizon_m=float(horizon_m),
        return_source_ids=True,
    )
    _ = source_triangle_ids
    return az_idx_u64.astype(np.int64), az_mask


def _points_in_2d_triangles_xz(pts_xz: np.ndarray, tris_xz: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Exact 2D `(x, z)` barycentric containment test for `pts_xz` `(Q, 2)` in `tris_xz` `(M_h, 3, 2)`."""
    q_cnt = int(pts_xz.shape[0])
    if q_cnt == 0 or int(tris_xz.shape[0]) == 0:
        return np.zeros((q_cnt,), dtype=bool)
    a = tris_xz[:, 0, :]  # (M_h, 2)
    b = tris_xz[:, 1, :]
    c = tris_xz[:, 2, :]
    v0 = b - a  # (M_h, 2)
    v1 = c - a  # (M_h, 2)
    det = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]  # (M_h,)
    nondeg = np.abs(det) > 1e-9
    if not np.any(nondeg):
        return np.zeros((q_cnt,), dtype=bool)
    a = a[nondeg]
    v0 = v0[nondeg]
    v1 = v1[nondeg]
    inv_det = 1.0 / det[nondeg]

    inside = np.zeros((q_cnt,), dtype=bool)
    step = 4096
    for s0 in range(0, q_cnt, step):
        s1 = min(q_cnt, s0 + step)
        vp = pts_xz[s0:s1, None, :] - a[None, :, :]  # (Q_s, M_h, 2)
        u = (vp[:, :, 0] * v1[None, :, 1] - v1[None, :, 0] * vp[:, :, 1]) * inv_det[None, :]
        v = (v0[None, :, 0] * vp[:, :, 1] - vp[:, :, 0] * v0[None, :, 1]) * inv_det[None, :]
        hit = (u >= -eps) & (v >= -eps) & ((u + v) <= 1.0 + eps)
        inside[s0:s1] = np.any(hit, axis=1)
    return inside


def _barycentric_rel_height_in_2d_triangles_xz(
    pts_xz: np.ndarray,
    tris_xyz: np.ndarray,
    y_min_b: float,
    height_b: float,
    eps: float = 1e-7,
) -> np.ndarray:
    """Return exact 2D `(x, z)` barycentric relative-height weight `r(x, z) in [0, 1]` (`0` outside).

    Under `perturb_buildings_consistent_f64`, each vertex `y_i` shifts by `eff_dh * ((y_i - y_min_b) / H_b)`.
    For flat roofs `r == 1.0`; for sloped roofs `r(x, z) = (1 - u - v)*r_0 + u*r_1 + v*r_2`, matching
    the exact rasterized heightmap shift on `(x, z)`!
    """
    q_cnt = int(pts_xz.shape[0])
    if q_cnt == 0 or int(tris_xyz.shape[0]) == 0:
        return np.zeros((q_cnt,), dtype=np.float64)
    tris_xz = tris_xyz[:, :, [0, 2]]
    a = tris_xz[:, 0, :]
    b = tris_xz[:, 1, :]
    c = tris_xz[:, 2, :]
    v0 = b - a
    v1 = c - a
    det = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    nondeg = np.abs(det) > 1e-9
    if not np.any(nondeg):
        return np.zeros((q_cnt,), dtype=np.float64)

    a = a[nondeg]
    v0 = v0[nondeg]
    v1 = v1[nondeg]
    inv_det = 1.0 / det[nondeg]
    y_verts = tris_xyz[nondeg, :, 1]
    if height_b > 1e-9:
        rel_v = np.clip((y_verts - float(y_min_b)) / float(height_b), 0.0, 1.0)
    else:
        rel_v = np.ones_like(y_verts)

    out_rel = np.zeros((q_cnt,), dtype=np.float64)
    step = 4096
    for s0 in range(0, q_cnt, step):
        s1 = min(q_cnt, s0 + step)
        vp = pts_xz[s0:s1, None, :] - a[None, :, :]
        u = (vp[:, :, 0] * v1[None, :, 1] - v1[None, :, 0] * vp[:, :, 1]) * inv_det[None, :]
        v = (v0[None, :, 0] * vp[:, :, 1] - vp[:, :, 0] * v0[None, :, 1]) * inv_det[None, :]
        hit = (u >= -eps) & (v >= -eps) & ((u + v) <= 1.0 + eps)
        if np.any(hit):
            w0 = 1.0 - u - v
            r_interp = (
                w0 * rel_v[None, :, 0]
                + u * rel_v[None, :, 1]
                + v * rel_v[None, :, 2]
            )
            # Take maximum upper-surface relative height across overlapping cap/roof faces of building b
            out_rel[s0:s1] = np.max(np.where(hit, np.clip(r_interp, 0.0, 1.0), 0.0), axis=1)
    return out_rel


DEFAULT_TILE_METERS_PER_PIXEL: float = 100.0 / 512.0  # 0.1953125 m/px


def _build_local_feature_height_couplings(
    receivers_xyz: np.ndarray,
    triangles_xyz: np.ndarray,
    building_ids: np.ndarray,
    num_buildings: int,
    *,
    meters_per_pixel: float = DEFAULT_TILE_METERS_PER_PIXEL,
    stencil_radius_px: int = 4,
    grid_step_xz: Optional[Tuple[float, float]] = None,
    grid_bounds_xz: Optional[Tuple[float, float, float, float]] = None,
    neighborhood_radius_m: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact raster-aligned `(N, B)` center-pixel and `9x9` (`81`-pixel) box-filter couplings.

    Fixes Astra `P0-7`:
    - Uses exact tile grid resolution `meters_per_pixel = 100.0 / 512.0 = 0.1953125` (instead of `0.5`).
    - Evaluates exact 2D `(x, z)` barycentric relative-height weights `(1-u-v)*r0 + u*r1 + v*r2` on
      upper roof/sloped faces (excluding bottom base caps `y <= y_min_b + 1e-3`), matching both flat
      and sloped roof deformations in `perturb_buildings_consistent_f64`!
    """
    _ = neighborhood_radius_m
    n_pts = int(receivers_xyz.shape[0])
    if n_pts == 0 or num_buildings <= 0 or triangles_xyz.shape[0] == 0:
        z = np.zeros((n_pts, max(num_buildings, 0)), dtype=np.float32)
        return z, z.copy()

    rec_xz = np.asarray(receivers_xyz[:, [0, 2]], dtype=np.float64)
    tris_xyz = np.asarray(triangles_xyz, dtype=np.float64)
    tris_xz = tris_xyz[:, :, [0, 2]]
    b_ids = np.asarray(building_ids, dtype=np.int64)
    y_min_arr, height_arr, _ = compute_building_vertical_anchors(tris_xyz, b_ids)

    pad_r = max(int(stencil_radius_px), 0)
    mpp = max(float(meters_per_pixel), 1e-4)
    dx, dz = grid_step_xz or (mpp, mpp)
    offs = np.arange(-pad_r, pad_r + 1, dtype=np.float64)
    dc_grid, dr_grid = np.meshgrid(offs * dx, offs * dz, indexing="xy")
    offsets_k2 = np.stack([dc_grid.reshape(-1), dr_grid.reshape(-1)], axis=-1)  # (81, 2)
    k_stencil = int(offsets_k2.shape[0])

    stencil_xz = rec_xz[:, None, :] + offsets_k2[None, :, :]  # (N, 81, 2)
    if grid_bounds_xz is not None:
        xmin_g, xmax_g, zmin_g, zmax_g = (float(x) for x in grid_bounds_xz)
        stencil_xz[:, :, 0] = np.clip(stencil_xz[:, :, 0], xmin_g, xmax_g)
        stencil_xz[:, :, 1] = np.clip(stencil_xz[:, :, 1], zmin_g, zmax_g)

    exact_w = np.zeros((n_pts, num_buildings), dtype=np.float32)
    neigh_w = np.zeros((n_pts, num_buildings), dtype=np.float32)
    margin_m = pad_r * max(abs(dx), abs(dz)) + 1e-4

    for b in range(num_buildings):
        sel = b_ids == b
        if not np.any(sel):
            continue
        sub_xyz = tris_xyz[sel]
        sub_xz = tris_xz[sel]
        bmin_xz = sub_xz.min(axis=(0, 1))
        bmax_xz = sub_xz.max(axis=(0, 1))
        yb = float(y_min_arr[b]) if b < y_min_arr.size else 0.0
        hb = float(height_arr[b]) if b < height_arr.size else 0.0

        cross_y = np.abs(
            (sub_xz[:, 1, 0] - sub_xz[:, 0, 0]) * (sub_xz[:, 2, 1] - sub_xz[:, 0, 1])
            - (sub_xz[:, 2, 0] - sub_xz[:, 0, 0]) * (sub_xz[:, 1, 1] - sub_xz[:, 0, 1])
        )
        above_base = sub_xyz[:, :, 1].max(axis=1) > (yb + 1e-3) if hb > 1e-5 else np.ones((sub_xyz.shape[0],), dtype=bool)
        horiz_sel = (cross_y > 1e-7) & above_base
        roof_tris_xyz = sub_xyz[horiz_sel] if np.any(horiz_sel) else None

        # 1. Exact center barycentric relative-height coupling `(N,)`
        cand_c = np.flatnonzero(
            (rec_xz[:, 0] >= bmin_xz[0] - 1e-6)
            & (rec_xz[:, 0] <= bmax_xz[0] + 1e-6)
            & (rec_xz[:, 1] >= bmin_xz[1] - 1e-6)
            & (rec_xz[:, 1] <= bmax_xz[1] + 1e-6)
        )
        if cand_c.size > 0:
            if roof_tris_xyz is not None:
                rel_c = _barycentric_rel_height_in_2d_triangles_xz(rec_xz[cand_c], roof_tris_xyz, yb, hb)
                exact_w[cand_c, b] = rel_c.astype(np.float32)
            else:
                exact_w[cand_c, b] = 1.0

        # 2. Exact `9x9` (`81`-point) raster box-filter stencil coupling `sum_k r_k / 81.0`
        cand_n = np.flatnonzero(
            (rec_xz[:, 0] >= bmin_xz[0] - margin_m)
            & (rec_xz[:, 0] <= bmax_xz[0] + margin_m)
            & (rec_xz[:, 1] >= bmin_xz[1] - margin_m)
            & (rec_xz[:, 1] <= bmax_xz[1] + margin_m)
        )
        if cand_n.size > 0:
            pts_sub = stencil_xz[cand_n].reshape(-1, 2)
            in_b_box = (
                (pts_sub[:, 0] >= bmin_xz[0] - 1e-6)
                & (pts_sub[:, 0] <= bmax_xz[0] + 1e-6)
                & (pts_sub[:, 1] >= bmin_xz[1] - 1e-6)
                & (pts_sub[:, 1] <= bmax_xz[1] + 1e-6)
            )
            rel_all = np.zeros((pts_sub.shape[0],), dtype=np.float64)
            box_idx = np.flatnonzero(in_b_box)
            if box_idx.size > 0:
                if roof_tris_xyz is not None:
                    rel_all[box_idx] = _barycentric_rel_height_in_2d_triangles_xz(
                        pts_sub[box_idx], roof_tris_xyz, yb, hb
                    )
                else:
                    rel_all[box_idx] = 1.0
            sums = rel_all.reshape(cand_n.size, k_stencil).sum(axis=1)
            neigh_w[cand_n, b] = (sums / float(max(k_stencil, 1))).astype(np.float32)

    return exact_w, neigh_w


def _build_spatial_crop_height_couplings(
    receivers_xyz: np.ndarray,
    triangles_xyz: np.ndarray,
    building_ids: np.ndarray,
    num_buildings: int,
    *,
    crop_h: int = 21,
    crop_w: int = 21,
    meters_per_pixel: float = DEFAULT_TILE_METERS_PER_PIXEL,
    grid_step_xz: Optional[Tuple[float, float]] = None,
    grid_bounds_xz: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """Compute `(N, H_c, W_c, B)` spatial patch height couplings for `receiver_crops` / `spatial_patch_crops`."""
    n_pts = int(receivers_xyz.shape[0])
    hc = max(1, int(crop_h))
    wc = max(1, int(crop_w))
    if n_pts == 0 or num_buildings <= 0 or triangles_xyz.shape[0] == 0:
        return np.zeros((n_pts, hc, wc, max(num_buildings, 0)), dtype=np.float32)

    rec_xz = np.asarray(receivers_xyz[:, [0, 2]], dtype=np.float64)
    tris_xyz = np.asarray(triangles_xyz, dtype=np.float64)
    tris_xz = tris_xyz[:, :, [0, 2]]
    b_ids = np.asarray(building_ids, dtype=np.int64)
    y_min_arr, height_arr, _ = compute_building_vertical_anchors(tris_xyz, b_ids)

    mpp = max(float(meters_per_pixel), 1e-4)
    dx, dz = grid_step_xz or (mpp, mpp)
    offs_c = (np.arange(wc, dtype=np.float64) - wc // 2) * dx
    offs_r = (np.arange(hc, dtype=np.float64) - hc // 2) * dz
    dc_grid, dr_grid = np.meshgrid(offs_c, offs_r, indexing="xy")
    patch_xz = rec_xz[:, None, None, :] + np.stack([dc_grid, dr_grid], axis=-1)
    in_grid = None
    if grid_bounds_xz is not None:
        xmin, xmax, zmin, zmax = grid_bounds_xz
        x, z = patch_xz[..., 0], patch_xz[..., 1]
        in_grid = (x >= xmin) & (x <= xmax) & (z >= zmin) & (z <= zmax)

    crop_w_out = np.zeros((n_pts, hc, wc, num_buildings), dtype=np.float32)
    margin_m = 0.5 * max(hc * abs(dz), wc * abs(dx)) + 1e-4
    k_pix = hc * wc

    for b in range(num_buildings):
        sel = b_ids == b
        if not np.any(sel):
            continue
        sub_xyz = tris_xyz[sel]
        sub_xz = tris_xz[sel]
        bmin_xz = sub_xz.min(axis=(0, 1))
        bmax_xz = sub_xz.max(axis=(0, 1))
        yb = float(y_min_arr[b]) if b < y_min_arr.size else 0.0
        hb = float(height_arr[b]) if b < height_arr.size else 0.0

        cross_y = np.abs(
            (sub_xz[:, 1, 0] - sub_xz[:, 0, 0]) * (sub_xz[:, 2, 1] - sub_xz[:, 0, 1])
            - (sub_xz[:, 2, 0] - sub_xz[:, 0, 0]) * (sub_xz[:, 1, 1] - sub_xz[:, 0, 1])
        )
        above_base = sub_xyz[:, :, 1].max(axis=1) > (yb + 1e-3) if hb > 1e-5 else np.ones((sub_xyz.shape[0],), dtype=bool)
        horiz_sel = (cross_y > 1e-7) & above_base
        roof_tris_xyz = sub_xyz[horiz_sel] if np.any(horiz_sel) else None

        cand_n = np.flatnonzero(
            (rec_xz[:, 0] >= bmin_xz[0] - margin_m)
            & (rec_xz[:, 0] <= bmax_xz[0] + margin_m)
            & (rec_xz[:, 1] >= bmin_xz[1] - margin_m)
            & (rec_xz[:, 1] <= bmax_xz[1] + margin_m)
        )
        if cand_n.size == 0:
            continue
        pts_sub = patch_xz[cand_n].reshape(-1, 2)
        in_b_box = (
            (pts_sub[:, 0] >= bmin_xz[0] - 1e-6)
            & (pts_sub[:, 0] <= bmax_xz[0] + 1e-6)
            & (pts_sub[:, 1] >= bmin_xz[1] - 1e-6)
            & (pts_sub[:, 1] <= bmax_xz[1] + 1e-6)
        )
        rel_all = np.zeros((pts_sub.shape[0],), dtype=np.float64)
        box_idx = np.flatnonzero(in_b_box)
        if box_idx.size > 0:
            if roof_tris_xyz is not None:
                rel_all[box_idx] = _barycentric_rel_height_in_2d_triangles_xz(
                    pts_sub[box_idx], roof_tris_xyz, yb, hb
                )
            else:
                rel_all[box_idx] = 1.0
        crop_w_out[cand_n, :, :, b] = rel_all.reshape(cand_n.size, hc, wc).astype(np.float32)

    if in_grid is not None:
        crop_w_out *= in_grid[..., None]
    return crop_w_out


def _differentiably_shift_local_features(
    base_lf_t: torch.Tensor,
    d_vec: torch.Tensor,
    exact_w_t: torch.Tensor,
    neigh_w_t: torch.Tensor,
) -> torch.Tensor:
    """Differentiably propagate building `Δh` (`m`) into `local_features` height channels (`m / 100`)."""
    if base_lf_t.shape[-1] < 16 or d_vec.numel() == 0:
        return base_lf_t
    exact_dh_scaled = (exact_w_t @ d_vec) / 100.0  # (N,)
    neigh_dh_scaled = (neigh_w_t @ d_vec) / 100.0  # (N,)
    cols = [base_lf_t[:, j] for j in range(base_lf_t.shape[-1])]
    cols[1] = cols[1] + exact_dh_scaled
    cols[2] = cols[2] + exact_dh_scaled
    cols[9] = cols[9] + neigh_dh_scaled
    cols[10] = cols[10] + neigh_dh_scaled
    return torch.stack(cols, dim=-1)


def _differentiably_shift_spatial_crops(
    base_crops_t: torch.Tensor,
    d_vec: torch.Tensor,
    crop_w_t: torch.Tensor,
) -> torch.Tensor:
    """Differentiably propagate building `Δh` (`m`) into `(N, 4, H_c, W_c)` spatial patch height channels (`1, 2`)."""
    if base_crops_t.ndim != 4 or base_crops_t.shape[1] < 3 or d_vec.numel() == 0:
        return base_crops_t
    dh_patch = torch.tensordot(crop_w_t, d_vec, dims=([-1], [0])) / 100.0  # (N, H_c, W_c)
    chans = [base_crops_t[:, c] for c in range(base_crops_t.shape[1])]
    chans[1] = chans[1] + dh_patch
    chans[2] = chans[2] + dh_patch
    return torch.stack(chans, dim=1)


def _compute_spatial_building_laplacian_loss(
    d_vec: torch.Tensor,
    building_centroids_xz_t: torch.Tensor,
    *,
    k_neighbors: int = 4,
    sigma_m: float = 60.0,
) -> torch.Tensor:
    """Spatial smoothness prior penalizing sharp adjacent height jumps across nearby buildings."""
    b_cnt = int(d_vec.numel())
    if b_cnt <= 1:
        return d_vec.new_zeros(())
    k_eff = min(max(1, int(k_neighbors)), b_cnt - 1)
    diff_xz = building_centroids_xz_t[:, None, :] - building_centroids_xz_t[None, :, :]
    dist_sq = (diff_xz * diff_xz).sum(dim=-1)
    eye_inf = torch.eye(b_cnt, dtype=dist_sq.dtype, device=dist_sq.device) * 1e12
    knn_d2, knn_idx = torch.topk(dist_sq + eye_inf, k=k_eff, dim=-1, largest=False)
    weights = torch.exp(-knn_d2 / max(2.0 * float(sigma_m) ** 2, 1.0))
    d_knn = d_vec[knn_idx]
    dh_diff = d_vec[:, None] - d_knn
    return (weights * dh_diff.square()).mean()


class AmortizedHeightSolver(torch.nn.Module):
    """Lightweight amortized height proposal network warm-starting volume-conserving optimization.

    Maps per-building geometric descriptors `(centroid_xz, height, log_area)` + global solar/receiver
    context summary into unconstrained height proposals projected via `DifferentiableAreaWeightedVolumeKKT`.
    """

    def __init__(self, hidden_dim: int = 64, max_delta_h_m: float = 15.0) -> None:
        super().__init__()
        self.max_delta_h_m = float(max_delta_h_m)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(9, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        torch.nn.init.zeros_(self.encoder[-1].weight)
        torch.nn.init.zeros_(self.encoder[-1].bias)

    def forward(
        self,
        building_centroids_xz: torch.Tensor,
        heights_m: torch.Tensor,
        areas_m2: torch.Tensor,
        mean_receiver_xyz: torch.Tensor,
        mean_sun_dir_xyz: torch.Tensor,
        *,
        enforce_volume_conservation: bool = True,
    ) -> torch.Tensor:
        b_cnt = int(heights_m.numel())
        if b_cnt == 0:
            return heights_m.clone()
        rel_xz = (building_centroids_xz - mean_receiver_xyz[[0, 2]].unsqueeze(0)) / 200.0
        h_norm = (heights_m / 100.0).unsqueeze(-1)
        a_norm = (torch.log1p(areas_m2.clamp_min(0.0)) / 10.0).unsqueeze(-1)
        rec_y = (mean_receiver_xyz[1:2] / 100.0).expand(b_cnt, 1)
        sun_ctx = mean_sun_dir_xyz.unsqueeze(0).expand(b_cnt, 3)
        dist_norm = torch.linalg.norm(rel_xz, dim=-1, keepdim=True)
        feats = torch.cat([rel_xz, dist_norm, h_norm, a_norm, rec_y, sun_ctx], dim=-1)
        raw = self.max_delta_h_m * torch.tanh(self.encoder(feats).squeeze(-1))
        if enforce_volume_conservation:
            return DifferentiableAreaWeightedVolumeKKT.apply(
                raw, areas_m2, heights_m, self.max_delta_h_m, 0.05
            )
        return raw.clamp(-self.max_delta_h_m, self.max_delta_h_m)


def select_active_disagreement_queries(
    probs_or_ensemble: np.ndarray | torch.Tensor,
    *,
    candidate_mask: Optional[np.ndarray | torch.Tensor] = None,
    receivers_xyz: Optional[np.ndarray | torch.Tensor] = None,
    valid_mask: Optional[np.ndarray | torch.Tensor] = None,
    top_k: int = 32,
    spatial_suppression_radius_m: float = 5.0,
    hypothesis_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Select active ray-tracing oracle queries maximizing uncertainty, boundary proximity & diversity.

    Supports both ensemble probability stacks `(M, T, N)` (disagreement variance + predictive entropy)
    and single-model calibrated trajectories `(T, N)` (Bernoulli boundary entropy + temporal flip score),
    weighted by active geometric blocker presence and greedy spatial diversity suppression.
    """
    p_arr = (
        probs_or_ensemble.detach().cpu().numpy()
        if isinstance(probs_or_ensemble, torch.Tensor)
        else np.asarray(probs_or_ensemble)
    ).astype(np.float64)

    if p_arr.ndim == 3:
        p_mean = np.clip(np.average(p_arr,axis=0,weights=hypothesis_weights), 1e-6, 1.0 - 1e-6)
        disagreement = np.average((p_arr-p_mean)**2,axis=0,weights=hypothesis_weights)
    elif p_arr.ndim == 2:
        p_mean = np.clip(p_arr, 1e-6, 1.0 - 1e-6)
        disagreement = p_mean * (1.0 - p_mean)
    else:
        raise ValueError(f"probs_or_ensemble must have shape (M, T, N) or (T, N), got {p_arr.shape}")

    t_steps, n_pts = p_mean.shape
    entropy = -(p_mean * np.log(p_mean) + (1.0 - p_mean) * np.log(1.0 - p_mean)) / math.log(2.0)

    # Temporal transition sensitivity |p_t - p_{t-1}|
    temp_jump = np.zeros_like(p_mean)
    if t_steps >= 2:
        jump = np.abs(p_mean[1:] - p_mean[:-1])
        temp_jump[1:] = np.maximum(temp_jump[1:], jump)
        temp_jump[:-1] = np.maximum(temp_jump[:-1], jump)

    score_tn = entropy * (1.0 + 2.0 * disagreement + 0.5 * temp_jump)

    if candidate_mask is not None:
        cm = (
            candidate_mask.detach().cpu().numpy()
            if isinstance(candidate_mask, torch.Tensor)
            else np.asarray(candidate_mask)
        ).astype(bool)
        if cm.ndim == 3:
            has_cand = np.any(cm, axis=-1)
        else:
            has_cand = cm
        score_tn = score_tn * np.where(has_cand, 1.5, 0.25)

    if valid_mask is not None:
        vm = (
            valid_mask.detach().cpu().numpy()
            if isinstance(valid_mask, torch.Tensor)
            else np.asarray(valid_mask)
        ).astype(bool)
        score_tn = np.where(vm, score_tn, -1.0)

    flat_total = t_steps * n_pts
    k_sel = min(max(0, int(top_k)), flat_total)
    if k_sel == 0:
        return {"selected_frame_receiver_pairs": [], "scores": [], "mean_acquisition_score": 0.0}

    rec_xz: Optional[np.ndarray] = None
    if receivers_xyz is not None:
        r_np = (
            receivers_xyz.detach().cpu().numpy()
            if isinstance(receivers_xyz, torch.Tensor)
            else np.asarray(receivers_xyz)
        ).astype(np.float64)
        rec_xz = r_np[:, [0, 2]]

    # Greedy diverse selection over top candidate pool
    pool_size = min(flat_total, max(k_sel * 16, 128))
    flat_scores = score_tn.reshape(-1)
    cand_flat = np.argpartition(-flat_scores, pool_size - 1)[:pool_size]
    cand_flat = cand_flat[np.argsort(-flat_scores[cand_flat])]

    selected_pairs: List[Tuple[int, int]] = []
    selected_scores: List[float] = []
    r_sup2 = max(float(spatial_suppression_radius_m), 0.0) ** 2

    work_scores = flat_scores[cand_flat].copy()
    chosen_mask = np.zeros((cand_flat.size,), dtype=bool)
    cand_t = cand_flat // max(n_pts, 1)
    cand_r = cand_flat % max(n_pts, 1)

    for _ in range(k_sel):
        avail = np.where(~chosen_mask, work_scores, -np.inf)
        best_local = int(np.argmax(avail))
        best_sc = float(avail[best_local])
        if not math.isfinite(best_sc) or best_sc < 0.0:
            break
        chosen_mask[best_local] = True
        ti = int(cand_t[best_local])
        ri = int(cand_r[best_local])
        selected_pairs.append((ti, ri))
        selected_scores.append(best_sc)

        if rec_xz is not None and r_sup2 > 0.0:
            same_or_adj_t = np.abs(cand_t - ti) <= 1
            d_xz = rec_xz[cand_r] - rec_xz[ri][None, :]
            close_sp = (d_xz[:, 0] ** 2 + d_xz[:, 1] ** 2) <= r_sup2
            suppress = same_or_adj_t & close_sp & (~chosen_mask)
            work_scores[suppress] *= 0.35

    return {
        "selected_frame_receiver_pairs": selected_pairs,
        "scores": selected_scores,
        "mean_acquisition_score": float(np.mean(selected_scores)) if selected_scores else 0.0,
    }


@dataclasses.dataclass(frozen=True)
class FullDayInverseDesignResult:
    optimizer_arm: str
    checkpoint_sha256: Optional[str]
    num_receivers: int
    num_frames: int
    num_triangles: int
    num_buildings: int
    max_abs_delta_h_m: float
    enforce_volume_conservation: bool
    enforce_zero_sum_height: bool
    building_effective_areas_m2: List[float]
    delta_h_per_building_m: List[float]
    area_weighted_volume_residual_m3: float
    mesh_signed_volume_delta_m3: float
    initial_exact_sunlit_fraction: Optional[float]
    optimized_exact_sunlit_fraction: Optional[float]
    signed_net_sunlight_delta_fraction: Optional[float]
    signed_net_sunlight_delta_point_frames: Optional[int]
    improved_receiver_count: int
    degraded_receiver_count: int
    unchanged_receiver_count: int
    regime_signed_deltas: Dict[str, Dict[str, float]]
    surrogate_initial_sunlit_fraction: Optional[float]
    surrogate_optimized_sunlit_fraction: Optional[float]
    oracle_queries: int
    model_queries: int
    wall_seconds: float
    building_consistency_verified: bool
    zero_perturbation_identity_verified: bool
    instance_provenance_tier: str = "shared_edge_3d_component_proxy"
    closed_mesh_volume_delta_m3: float = 0.0
    volume_conservation_residual_m3: float = 0.0
    relative_volume_residual: float = 0.0
    closed_solid_manifold_verified: bool = True
    valid_point_frames: int = 0
    invalid_receiver_count: int = 0
    empty_support_reason: Optional[str] = None
    max_oracle_queries: Optional[int] = None
    max_wall_clock_s: Optional[float] = None
    budget_stopped_early: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def optimize_fullday_building_heights(
    receivers_xyz: np.ndarray,
    sun_directions_xyz: np.ndarray,
    triangles_xyz: np.ndarray,
    *,
    reference_trajectory=None,
    opaque_mask: Optional[np.ndarray] = None,
    source_triangle_ids: Optional[np.ndarray] = None,
    exclude_source_triangle_ids: Optional[np.ndarray] = None,
    source_instance_ids: Optional[np.ndarray] = None,
    ground_mask: Optional[np.ndarray] = None,
    building_ids: Optional[np.ndarray] = None,
    solar_regimes: Optional[Sequence[str]] = None,
    valid_mask: Optional[np.ndarray] = None,
    same_day_adjacent_mask: Optional[np.ndarray] = None,
    local_features: Optional[np.ndarray] = None,
    receiver_crops: Optional[np.ndarray] = None,
    spatial_patch_crops: Optional[np.ndarray] = None,
    meters_per_pixel: float = DEFAULT_TILE_METERS_PER_PIXEL,
    grid_step_xz: Optional[Tuple[float, float]] = None,
    grid_bounds_xz: Optional[Tuple[float, float, float, float]] = None,
    optimizer_arm: str = "analytic_diff_surrogate",
    rind_model: Optional[torch.nn.Module] = None,
    checkpoint_sha256: Optional[str] = None,
    token_builder_torch: Optional[Callable[..., torch.Tensor]] = None,
    amortized_solver: Optional[AmortizedHeightSolver] = None,
    max_candidates: int = 8,
    editable_radius_m: Optional[float] = None,
    max_delta_h_m: float = 15.0,
    t_max_m: float = DEFAULT_T_MAX_M,
    steps: int = 40,
    lr: float = 1.25,
    enforce_volume_conservation: bool = True,
    enforce_zero_sum_height: bool = False,
    allow_nonclosed_surface_proxy: bool = True,
    require_full_4km_horizon: bool = False,
    scene_bbox_margin_m: Optional[float] = None,
    max_oracle_queries: Optional[int] = None,
    max_wall_clock_s: Optional[float] = None,
    regularization_l2: float = 1e-4,
    smoothness_weight: float = 0.0,
    seed: int = 20261111,
    device: str = "cpu",
) -> FullDayInverseDesignResult:
    """Optimize building height perturbations `delta_h_b in [-15m, +15m]` across a full trajectory."""
    valid_arms = ("rind_checkpoint_grad", "analytic_diff_surrogate", "blackbox_zero_order")
    if optimizer_arm not in valid_arms:
        raise ValueError(f"Unknown optimizer_arm {optimizer_arm!r}; valid={valid_arms}")
    if optimizer_arm == "rind_checkpoint_grad" and rind_model is None:
        raise ValueError("optimizer_arm='rind_checkpoint_grad' requires `rind_model` checkpoint.")
    if require_full_4km_horizon:
        if float(t_max_m) < 4000.0 - 1e-3:
            raise RuntimeError(
                f"Formal 4km inverse design requires `t_max_m >= 4000.0`, got {t_max_m}."
            )
        if scene_bbox_margin_m is not None and float(scene_bbox_margin_m) < 4000.0 - 1e-3:
            raise RuntimeError(
                f"Formal 4km inverse design forbids truncated upstream `--bbox-margin-m` "
                f"({scene_bbox_margin_m} < 4000.0 m)."
            )

    t_wall_start = time.perf_counter()
    rec_np = np.asarray(receivers_xyz, dtype=np.float64)
    sun_np = np.asarray(sun_directions_xyz, dtype=np.float64)
    tri_np = np.asarray(triangles_xyz, dtype=np.float64)
    t_steps = sun_np.shape[0]
    n_pts = rec_np.shape[0]
    m_tris = tri_np.shape[0]

    if valid_mask is None:
        val_np = np.ones((t_steps, n_pts), dtype=bool)
    else:
        val_np = np.asarray(valid_mask, dtype=bool)
        if val_np.shape != (t_steps, n_pts):
            raise ValueError(f"valid_mask shape {val_np.shape} != ({t_steps}, {n_pts})")

    g_mask_np = np.asarray(ground_mask, dtype=bool) if ground_mask is not None else None

    if building_ids is not None:
        b_ids_np = np.asarray(building_ids, dtype=np.int64)
        provenance_tier = "explicit_instance_ids"
    elif source_instance_ids is not None:
        b_ids_np = assign_building_ids_from_triangles(
            tri_np, source_instance_ids=source_instance_ids, ground_mask=g_mask_np
        )
        provenance_tier = "explicit_instance_ids"
        if editable_radius_m is not None and m_tris > 0 and n_pts > 0 and np.any(b_ids_np >= 0):
            rec_c_xz = rec_np[:, [0, 2]].mean(axis=0)
            rec_rad = float(np.max(np.linalg.norm(rec_np[:, [0, 2]] - rec_c_xz[None, :], axis=-1)))
            thresh_r = float(editable_radius_m) + rec_rad
            tri_min_xz = np.min(
                np.linalg.norm(tri_np[:, :, [0, 2]] - rec_c_xz[None, None, :], axis=-1), axis=1
            )
            n_full_b = int(b_ids_np.max()) + 1
            filtered_bids = np.full((m_tris,), -1, dtype=np.int64)
            next_id = 0
            for bid in range(n_full_b):
                cmask = b_ids_np == bid
                if np.any(cmask) and float(np.min(tri_min_xz[cmask])) <= thresh_r:
                    filtered_bids[cmask] = next_id
                    next_id += 1
            b_ids_np = filtered_bids
    elif editable_radius_m is not None and m_tris > 0 and n_pts > 0:
        provenance_tier = "shared_edge_3d_component_proxy"
        rec_c_xz = rec_np[:, [0, 2]].mean(axis=0)
        rec_rad = float(np.max(np.linalg.norm(rec_np[:, [0, 2]] - rec_c_xz[None, :], axis=-1)))
        outer_r = float(editable_radius_m) + rec_rad + 250.0
        tri_min_xz = np.min(np.linalg.norm(tri_np[:, :, [0, 2]] - rec_c_xz[None, None, :], axis=-1), axis=1)
        near_idx = np.flatnonzero(tri_min_xz <= outer_r)
        b_ids_np = np.full((m_tris,), -1, dtype=np.int64)
        if near_idx.size > 0:
            local_gmask = g_mask_np[near_idx] if g_mask_np is not None else None
            local_bids = assign_building_ids_from_triangles(tri_np[near_idx], ground_mask=local_gmask)
            n_loc_b = int(local_bids.max()) + 1 if np.any(local_bids >= 0) else 0
            next_id = 0
            thresh_r = float(editable_radius_m) + rec_rad
            boundary_r = outer_r - 10.0
            for lb in range(n_loc_b):
                comp_mask = local_bids == lb
                comp_global = near_idx[comp_mask]
                comp_max_r = float(
                    np.max(np.linalg.norm(tri_np[comp_global][:, :, [0, 2]] - rec_c_xz[None, None, :], axis=-1))
                )
                comp_min_r = float(np.min(tri_min_xz[comp_global]))
                if comp_min_r <= thresh_r and comp_max_r < boundary_r:
                    b_ids_np[comp_global] = next_id
                    next_id += 1
    else:
        b_ids_np = assign_building_ids_from_triangles(tri_np, ground_mask=g_mask_np)
        provenance_tier = "shared_edge_3d_component_proxy"
    if b_ids_np.shape != (m_tris,):
        raise ValueError(f"building_ids shape {b_ids_np.shape} != ({m_tris},)")

    manifold_report = verify_extruded_building_manifold_closure(tri_np, b_ids_np)
    closed_manifold_ok = bool(manifold_report["all_closed_manifold"])
    if enforce_volume_conservation and (not allow_nonclosed_surface_proxy) and (not closed_manifold_ok):
        raise RuntimeError(
            "Formal volume-conserving inverse design rejected non-closed surface proxy buildings: "
            f"nonclosed_building_ids={manifold_report['nonclosed_building_ids']}, "
            f"max_jacobian_rel_error={manifold_report['max_jacobian_rel_error']:.3e}."
        )

    y_min_np, height_np, num_b = compute_building_vertical_anchors(tri_np, b_ids_np)
    areas_np = (
        np.ones((num_b,), dtype=np.float64)
        if (enforce_zero_sum_height and not enforce_volume_conservation)
        else compute_building_effective_areas_f64(tri_np, b_ids_np)
    )
    use_vol_proj = bool(enforce_volume_conservation or enforce_zero_sum_height)

    b_cent_xz_np = np.zeros((num_b, 2), dtype=np.float32)
    for b in range(num_b):
        sel_b = b_ids_np == b
        if np.any(sel_b):
            b_cent_xz_np[b] = tri_np[sel_b][:, :, [0, 2]].mean(axis=(0, 1)).astype(np.float32)

    zero_pert = perturb_buildings_consistent_f64(
        tri_np, b_ids_np, np.zeros((num_b,), dtype=np.float64)
    )
    zero_identity_ok = bool(np.max(np.abs(zero_pert - tri_np)) == 0.0) if m_tris > 0 else True

    oracle_queries = 0
    model_queries = 0
    budget_stopped_early = False

    def _wall_elapsed() -> float:
        return float(time.perf_counter() - t_wall_start)

    def _budget_exhausted(*, reserve_final_oracle: int = 0) -> bool:
        if max_wall_clock_s is not None and _wall_elapsed() >= float(max_wall_clock_s):
            return True
        if max_oracle_queries is not None and (oracle_queries + reserve_final_oracle) >= int(max_oracle_queries):
            return True
        return False

    def _eval_exact_oracle(scene_tris_f64: np.ndarray) -> np.ndarray:
        nonlocal oracle_queries
        oracle_queries += 1
        bvh = ExactSceneBVH(
            scene_tris_f64,
            opaque_mask=opaque_mask,
            source_triangle_ids=source_triangle_ids,
        )
        return bvh.query_trajectory_sunlit(
            rec_np,
            sun_np,
            t_min=DEFAULT_T_MIN_M,
            t_max=float(t_max_m),
            exclude_source_triangle_ids=exclude_source_triangle_ids,
        )

    def _project_delta_np(raw_d: np.ndarray) -> np.ndarray:
        if num_b == 0:
            return raw_d.copy()
        if use_vol_proj:
            weights = (
                np.ones((num_b,), dtype=np.float64)
                if (enforce_zero_sum_height and not enforce_volume_conservation)
                else areas_np
            )
            return project_area_weighted_volume_conservation_f64(
                raw_d, weights, height_np, max_delta_h_m=max_delta_h_m
            )
        min_drop = np.where(height_np > 1e-9, np.minimum(0.0, 0.05 - height_np), -float(max_delta_h_m))
        return np.clip(raw_d, np.maximum(-float(max_delta_h_m), min_drop), float(max_delta_h_m))

    valid_count = int(val_np.sum())
    if valid_count == 0:
        wall_s = _wall_elapsed()
        return FullDayInverseDesignResult(
            optimizer_arm=optimizer_arm,
            checkpoint_sha256=checkpoint_sha256,
            num_receivers=n_pts,
            num_frames=t_steps,
            num_triangles=m_tris,
            num_buildings=num_b,
            max_abs_delta_h_m=float(max_delta_h_m),
            enforce_volume_conservation=bool(enforce_volume_conservation),
            enforce_zero_sum_height=bool(enforce_zero_sum_height),
            building_effective_areas_m2=[float(x) for x in areas_np.tolist()],
            delta_h_per_building_m=[0.0 for _ in range(num_b)],
            area_weighted_volume_residual_m3=0.0,
            mesh_signed_volume_delta_m3=0.0,
            initial_exact_sunlit_fraction=None,
            optimized_exact_sunlit_fraction=None,
            signed_net_sunlight_delta_fraction=None,
            signed_net_sunlight_delta_point_frames=None,
            improved_receiver_count=0,
            degraded_receiver_count=0,
            unchanged_receiver_count=0,
            regime_signed_deltas={},
            surrogate_initial_sunlit_fraction=None,
            surrogate_optimized_sunlit_fraction=None,
            oracle_queries=0,
            model_queries=0,
            wall_seconds=wall_s,
            building_consistency_verified=bool(zero_identity_ok),
            zero_perturbation_identity_verified=bool(zero_identity_ok),
            instance_provenance_tier=provenance_tier,
            closed_mesh_volume_delta_m3=0.0,
            volume_conservation_residual_m3=0.0,
            relative_volume_residual=0.0,
            closed_solid_manifold_verified=closed_manifold_ok,
            valid_point_frames=0,
            invalid_receiver_count=int(n_pts),
            empty_support_reason="no_valid_receiver_frame_pairs",
            max_oracle_queries=max_oracle_queries,
            max_wall_clock_s=max_wall_clock_s,
            budget_stopped_early=False,
        )

    # 1. Exact float64 BVH baseline across full trajectory
    init_exact = _eval_exact_oracle(tri_np)
    init_frac = float(init_exact[val_np].sum() / valid_count)

    if num_b == 0 or m_tris == 0 or steps <= 0 or (use_vol_proj and num_b <= 1):
        delta_h_opt = np.zeros((num_b,), dtype=np.float64)
        surr_init = init_frac
        surr_opt = init_frac
    elif optimizer_arm == "blackbox_zero_order":
        rng = np.random.default_rng(seed)
        best_d = np.zeros((num_b,), dtype=np.float64)
        best_score = init_frac
        surr_init = init_frac
        step_sigma = min(4.0, 0.35 * float(max_delta_h_m))
        for step_idx in range(int(steps)):
            if _budget_exhausted(reserve_final_oracle=1):
                budget_stopped_early = True
                break
            decay = 1.0 - 0.65 * (float(step_idx) / float(max(int(steps) - 1, 1)))
            proposal = _project_delta_np(best_d + rng.normal(0.0, step_sigma * decay, size=(num_b,)))
            cand_tris = perturb_buildings_consistent_f64(tri_np, b_ids_np, proposal)
            cand_sun = _eval_exact_oracle(cand_tris)
            cand_score = float(cand_sun[val_np].sum() / valid_count)
            if cand_score > best_score:
                best_score = cand_score
                best_d = proposal
        delta_h_opt = best_d
        surr_opt = best_score
    elif optimizer_arm == "rind_checkpoint_grad":
        assert rind_model is not None
        if token_builder_torch is None:
            from .fast_data import build_canonical37_torch as token_builder_torch  # local import avoids cycle
        from .fullday_data import sun_dirs_to_azimuth_bins

        dev = torch.device(device)
        rind_model = rind_model.to(dev)
        rind_model.eval()
        for p in rind_model.parameters():
            p.requires_grad_(False)

        if reference_trajectory is None:
            az_bins = _resolve_model_azimuth_bins(rind_model)
            op_arr = (
                np.asarray(opaque_mask, dtype=bool)
                if opaque_mask is not None
                else np.ones((m_tris,), dtype=bool)
            )
            if az_bins == 0:
                k_max = int(max_candidates)
                if n_pts == 0 or m_tris == 0 or k_max <= 0:
                    gather_idx_tnk = np.zeros((t_steps, n_pts, max(k_max, 0)), dtype=np.int64)
                    gather_mk_tnk = np.zeros((t_steps, n_pts, max(k_max, 0)), dtype=bool)
                else:
                    from .ray_candidates import RayCandidateScene

                    tris_reach = _apply_editable_height_slack(
                        tri_np,
                        building_ids=b_ids_np,
                        ground_mask=g_mask_np,
                        height_slack_m=float(max_delta_h_m),
                    )
                    ray_scene = RayCandidateScene(tris_reach, op_arr, np.arange(m_tris, dtype=np.uint64))
                    _, _, ray_mk, ray_refs, _ = ray_scene.query(
                        rec_np, sun_np, float(t_max_m), k_max
                    )
                    gather_idx_tnk = np.where(ray_mk, ray_refs, 0).swapaxes(0, 1).astype(np.int64)
                    gather_mk_tnk = ray_mk.swapaxes(0, 1)
            else:
                az_idx_np, az_mk_np = _preselect_candidate_triangle_indices(
                    rec_np,
                    tri_np,
                    opaque_mask=opaque_mask,
                    source_triangle_ids=source_triangle_ids,
                    building_ids=b_ids_np,
                    ground_mask=g_mask_np,
                    num_azimuth_bins=az_bins,
                    max_candidates=int(max_candidates),
                    horizon_m=float(t_max_m),
                    height_slack_m=float(max_delta_h_m),
                )
                bin_idx_np = sun_dirs_to_azimuth_bins(sun_np, num_azimuth_bins=az_bins)
                gather_idx_tnk = np.transpose(az_idx_np, (1, 0, 2))[bin_idx_np]
                gather_mk_tnk = np.transpose(az_mk_np, (1, 0, 2))[bin_idx_np]
            gather_op_tnk = op_arr[gather_idx_tnk] & gather_mk_tnk
            exact_w_np, neigh_w_np = _build_local_feature_height_couplings(
                rec_np,
                tri_np,
                b_ids_np,
                num_b,
                meters_per_pixel=float(meters_per_pixel),
                grid_step_xz=grid_step_xz,
                grid_bounds_xz=grid_bounds_xz,
            )

            raw_crops_np = receiver_crops if receiver_crops is not None else spatial_patch_crops
            crops_t: Optional[torch.Tensor] = None
            crop_w_t: Optional[torch.Tensor] = None
            if raw_crops_np is not None:
                c_arr = np.asarray(raw_crops_np, dtype=np.float32)
                if c_arr.ndim == 5 and c_arr.shape[0] == 1:
                    c_arr = c_arr[0]
                if c_arr.ndim == 4:
                    crops_t = torch.as_tensor(c_arr, dtype=torch.float32, device=dev)
                    crop_w_np = _build_spatial_crop_height_couplings(
                        rec_np,
                        tri_np,
                        b_ids_np,
                        num_b,
                        crop_h=int(c_arr.shape[2]),
                        crop_w=int(c_arr.shape[3]),
                        meters_per_pixel=float(meters_per_pixel),
                        grid_step_xz=grid_step_xz,
                        grid_bounds_xz=grid_bounds_xz,
                    )
                    crop_w_t = torch.as_tensor(crop_w_np, dtype=torch.float32, device=dev)

        rec_t = torch.as_tensor(rec_np, dtype=torch.float32, device=dev)
        sun_t = torch.as_tensor(sun_np, dtype=torch.float32, device=dev)
        tri_t = torch.as_tensor(tri_np, dtype=torch.float32, device=dev)
        val_t = torch.as_tensor(val_np, dtype=torch.float32, device=dev)
        b_ids_t = torch.as_tensor(b_ids_np, dtype=torch.long, device=dev)
        ymin_t = torch.as_tensor(y_min_np, dtype=torch.float32, device=dev)
        h_t = torch.as_tensor(height_np, dtype=torch.float32, device=dev)
        if reference_trajectory is None:
            idx_tnk_t = torch.as_tensor(gather_idx_tnk, dtype=torch.long, device=dev)
            mk_tnk_t = torch.as_tensor(gather_mk_tnk, dtype=torch.bool, device=dev)
            op_tnk_t = torch.as_tensor(gather_op_tnk, dtype=torch.bool, device=dev)
            exact_w_t = torch.as_tensor(exact_w_np, dtype=torch.float32, device=dev)
            neigh_w_t = torch.as_tensor(neigh_w_np, dtype=torch.float32, device=dev)
        b_cent_t = torch.as_tensor(b_cent_xz_np, dtype=torch.float32, device=dev)
        adj_t = (
            torch.as_tensor(same_day_adjacent_mask, dtype=torch.bool, device=dev)
            if same_day_adjacent_mask is not None
            else torch.ones((max(t_steps - 1, 0),), dtype=torch.bool, device=dev)
        )
        lf_t = (
            torch.as_tensor(local_features, dtype=torch.float32, device=dev)
            if local_features is not None
            else torch.zeros((n_pts, 16), dtype=torch.float32, device=dev)
        )
        denom_t = val_t.sum().clamp_min(1.0)

        proj_weights_np = (
            np.ones((num_b,), dtype=np.float64)
            if (enforce_zero_sum_height and not enforce_volume_conservation)
            else areas_np
        )
        proj_w_t = torch.as_tensor(proj_weights_np, dtype=torch.float32, device=dev)
        min_drop_t = torch.where(
            h_t > 1e-9,
            torch.minimum(torch.zeros_like(h_t), 0.05 - h_t),
            torch.full_like(h_t, -float(max_delta_h_m)),
        )
        lo_box_t = torch.maximum(torch.full_like(h_t, -float(max_delta_h_m)), min_drop_t)

        def _project_delta_torch(raw_vec: torch.Tensor) -> torch.Tensor:
            if use_vol_proj:
                return DifferentiableAreaWeightedVolumeKKT.apply(
                    raw_vec, proj_w_t, h_t, float(max_delta_h_m), 0.05
                )
            return torch.minimum(torch.maximum(raw_vec, lo_box_t), torch.full_like(raw_vec, float(max_delta_h_m)))

        def _forward_rind_mean_sun(d_vec: torch.Tensor) -> torch.Tensor:
            nonlocal model_queries
            model_queries += 1
            pert_tris = perturb_buildings_consistent_torch(tri_t, b_ids_t, d_vec, ymin_t, h_t)
            if reference_trajectory is not None:
                from .native_scene import native_scene_batch
                edited = perturb_buildings_consistent_f64(tri_np, b_ids_np, d_vec.detach().cpu().double().numpy())
                batch = native_scene_batch(reference_trajectory, rind_model, edited, device=dev,
                    differentiable_triangles=pert_tris)
                probs = rind_model(batch,use_calibrated_temperature=True)["probs"].reshape_as(val_t)
                return (probs * val_t).sum() / denom_t
            cand_abs = pert_tris[idx_tnk_t]  # (T, N, K, 3, 3)
            rel_tris = cand_abs - rec_t[None, :, None, None, :]
            tokens = token_builder_torch(rel_tris, op_tnk_t, mk_tnk_t, sun_t)
            if hasattr(rind_model, "sanitize_tokens"):
                tokens = rind_model.sanitize_tokens(tokens)  # type: ignore[operator]
            lf_pert = _differentiably_shift_local_features(lf_t, d_vec, exact_w_t, neigh_w_t)
            batch_in: Dict[str, Any] = {
                "candidate_tokens": tokens.unsqueeze(0),
                "candidate_mask": mk_tnk_t.unsqueeze(0),
                "local_features": lf_pert.unsqueeze(0),
                "receivers_xyz": rec_t.unsqueeze(0),
                "sun_dirs": sun_t.unsqueeze(0),
                "same_day_adjacent_mask": adj_t.unsqueeze(0),
            }
            if crops_t is not None and crop_w_t is not None:
                crops_pert = _differentiably_shift_spatial_crops(crops_t, d_vec, crop_w_t).unsqueeze(0)
                batch_in["receiver_crops"] = crops_pert
                batch_in["spatial_patch_crops"] = crops_pert
            if hasattr(rind_model, "calibrated_temperature"):
                out = rind_model(batch_in, use_calibrated_temperature=True)
            else:
                out = rind_model(batch_in)
            probs = out["probs"].squeeze(0)
            return (probs * val_t).sum() / denom_t

        init_raw = torch.zeros((num_b,), dtype=torch.float32, device=dev)
        if amortized_solver is not None:
            with torch.no_grad():
                amortized_solver = amortized_solver.to(dev)
                amortized_solver.eval()
                warm_d = amortized_solver(
                    b_cent_t,
                    h_t,
                    proj_w_t,
                    rec_t.mean(dim=0),
                    sun_t.mean(dim=0),
                    enforce_volume_conservation=use_vol_proj,
                )
                if float(_forward_rind_mean_sun(warm_d).item()) > float(_forward_rind_mean_sun(init_raw).item()):
                    init_raw = warm_d.detach().clone()

        raw_param = init_raw.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([raw_param], lr=float(lr))

        with torch.no_grad():
            zero_d_t = torch.zeros((num_b,), dtype=torch.float32, device=dev)
            surr_init = float(_forward_rind_mean_sun(zero_d_t).item())
        best_rind_obj = surr_init
        best_rind_sun = surr_init
        best_rind_d = np.zeros((num_b,), dtype=np.float64)

        for _ in range(int(steps)):
            if max_wall_clock_s is not None and _wall_elapsed() >= float(max_wall_clock_s):
                budget_stopped_early = True
                break
            opt.zero_grad()
            d_feas = _project_delta_torch(raw_param)
            mean_sun = _forward_rind_mean_sun(d_feas)
            reg = float(regularization_l2) * d_feas.square().mean()
            if float(smoothness_weight) > 0.0:
                reg = reg + float(smoothness_weight) * _compute_spatial_building_laplacian_loss(d_feas, b_cent_t)
            cur_obj = float((mean_sun - reg).item())
            if cur_obj > best_rind_obj:
                best_rind_obj = cur_obj
                best_rind_sun = float(mean_sun.item())
                best_rind_d = _project_delta_np(d_feas.detach().cpu().numpy())
            loss = -mean_sun + reg
            loss.backward()
            opt.step()
            with torch.no_grad():
                raw_param.clamp_(-3.0 * float(max_delta_h_m), 3.0 * float(max_delta_h_m))

        with torch.no_grad():
            d_fin = _project_delta_torch(raw_param)
            fin_sun = float(_forward_rind_mean_sun(d_fin).item())
            fin_reg = float(regularization_l2) * float(d_fin.square().mean().item())
            if float(smoothness_weight) > 0.0:
                fin_reg += float(smoothness_weight) * float(
                    _compute_spatial_building_laplacian_loss(d_fin, b_cent_t).item()
                )
            fin_obj = fin_sun - fin_reg
            if fin_obj >= best_rind_obj:
                surr_opt = fin_sun
                delta_h_opt = _project_delta_np(d_fin.detach().cpu().numpy())
            else:
                surr_opt = best_rind_sun
                delta_h_opt = best_rind_d
    else:
        dev = torch.device(device)
        edit_sel = b_ids_np >= 0
        static_sel = ~edit_sel
        op_full = (
            np.asarray(opaque_mask, dtype=bool)
            if opaque_mask is not None
            else np.ones((m_tris,), dtype=bool)
        )
        if np.any(static_sel):
            static_bvh = ExactSceneBVH(
                tri_np[static_sel],
                opaque_mask=op_full[static_sel],
                source_triangle_ids=source_triangle_ids[static_sel] if source_triangle_ids is not None else None,
            )
            static_sun_np = static_bvh.query_trajectory_sunlit(
                rec_np,
                sun_np,
                t_min=DEFAULT_T_MIN_M,
                t_max=float(t_max_m),
                exclude_source_triangle_ids=exclude_source_triangle_ids,
            ).astype(np.float32)
        else:
            static_sun_np = np.ones((t_steps, n_pts), dtype=np.float32)

        edit_op_sel = edit_sel & op_full
        rec_t = torch.as_tensor(rec_np, dtype=torch.float32, device=dev)
        sun_t = torch.as_tensor(sun_np, dtype=torch.float32, device=dev)
        tri_edit_t = torch.as_tensor(tri_np[edit_op_sel], dtype=torch.float32, device=dev)
        b_edit_t = torch.as_tensor(b_ids_np[edit_op_sel], dtype=torch.long, device=dev)
        sids_edit_t = (
            torch.as_tensor(np.asarray(source_triangle_ids, dtype=np.int64)[edit_op_sel], dtype=torch.int64, device=dev)
            if source_triangle_ids is not None
            else None
        )
        excl_rec_t = (
            torch.as_tensor(np.asarray(exclude_source_triangle_ids, dtype=np.int64), dtype=torch.int64, device=dev)
            if exclude_source_triangle_ids is not None
            else None
        )
        static_sun_t = torch.as_tensor(static_sun_np, dtype=torch.float32, device=dev)
        val_t = torch.as_tensor(val_np, dtype=torch.float32, device=dev)
        ymin_t = torch.as_tensor(y_min_np, dtype=torch.float32, device=dev)
        h_t = torch.as_tensor(height_np, dtype=torch.float32, device=dev)
        b_cent_t = torch.as_tensor(b_cent_xz_np, dtype=torch.float32, device=dev)

        denom_t = val_t.sum().clamp_min(1.0)
        proj_weights_np = (
            np.ones((num_b,), dtype=np.float64)
            if (enforce_zero_sum_height and not enforce_volume_conservation)
            else areas_np
        )
        proj_w_t = torch.as_tensor(proj_weights_np, dtype=torch.float32, device=dev)
        min_drop_t = torch.where(
            h_t > 1e-9,
            torch.minimum(torch.zeros_like(h_t), 0.05 - h_t),
            torch.full_like(h_t, -float(max_delta_h_m)),
        )
        lo_box_t = torch.maximum(torch.full_like(h_t, -float(max_delta_h_m)), min_drop_t)

        def _project_delta_surr(raw_vec: torch.Tensor) -> torch.Tensor:
            if use_vol_proj:
                return DifferentiableAreaWeightedVolumeKKT.apply(
                    raw_vec, proj_w_t, h_t, float(max_delta_h_m), 0.05
                )
            return torch.minimum(torch.maximum(raw_vec, lo_box_t), torch.full_like(raw_vec, float(max_delta_h_m)))

        def _eval_ref_surrogate(d_vec: torch.Tensor) -> float:
            with torch.no_grad():
                pt = perturb_buildings_consistent_torch(tri_edit_t, b_edit_t, d_vec, ymin_t, h_t)
                sf = static_sun_t * _differentiable_soft_moller_trumbore_sunlit(
                    rec_t,
                    sun_t,
                    pt,
                    building_ids=b_edit_t,
                    source_triangle_ids=sids_edit_t,
                    exclude_source_triangle_ids=excl_rec_t,
                    temperature_m=1.2,
                    t_min=DEFAULT_T_MIN_M,
                    t_max=float(t_max_m),
                )
                return float(((sf * val_t).sum() / denom_t).item())

        raw_param = torch.zeros((num_b,), dtype=torch.float32, device=dev, requires_grad=True)
        opt = torch.optim.Adam([raw_param], lr=float(lr))

        surr_init = _eval_ref_surrogate(raw_param)
        best_surr_sun = surr_init
        best_surr_obj = surr_init
        best_surr_d = np.zeros((num_b,), dtype=np.float64)

        n_steps = int(steps)
        for step_idx in range(n_steps):
            if max_wall_clock_s is not None and _wall_elapsed() >= float(max_wall_clock_s):
                budget_stopped_early = True
                break
            opt.zero_grad()
            prog = float(step_idx) / float(max(n_steps - 1, 1))
            tau_m = 5.0 * ((1.2 / 5.0) ** prog)
            d_feas = _project_delta_surr(raw_param)
            pert_tris = perturb_buildings_consistent_torch(
                tri_edit_t, b_edit_t, d_feas, ymin_t, h_t
            )
            soft_sun = static_sun_t * _differentiable_soft_moller_trumbore_sunlit(
                rec_t,
                sun_t,
                pert_tris,
                building_ids=b_edit_t,
                source_triangle_ids=sids_edit_t,
                exclude_source_triangle_ids=excl_rec_t,
                temperature_m=tau_m,
                t_min=DEFAULT_T_MIN_M,
                t_max=float(t_max_m),
            )
            mean_sun = (soft_sun * val_t).sum() / denom_t
            reg = float(regularization_l2) * d_feas.square().mean()
            if float(smoothness_weight) > 0.0:
                reg = reg + float(smoothness_weight) * _compute_spatial_building_laplacian_loss(d_feas, b_cent_t)
            loss = -mean_sun + reg
            loss.backward()
            opt.step()
            with torch.no_grad():
                raw_param.clamp_(-3.0 * float(max_delta_h_m), 3.0 * float(max_delta_h_m))
                proj_np = _project_delta_np(raw_param.detach().cpu().numpy())
                proj_t = torch.as_tensor(proj_np, dtype=torch.float32, device=dev)
                cand_ref_sun = _eval_ref_surrogate(proj_t)
                cand_ref_obj = cand_ref_sun - float(regularization_l2) * float(np.mean(proj_np ** 2))
                if cand_ref_obj > best_surr_obj:
                    best_surr_obj = cand_ref_obj
                    best_surr_sun = cand_ref_sun
                    best_surr_d = proj_np.copy()

        surr_opt = best_surr_sun
        delta_h_opt = best_surr_d

    # 2. Exact float64 full-scene BVH physical re-verification
    opt_tris_np = perturb_buildings_consistent_f64(tri_np, b_ids_np, delta_h_opt)
    opt_exact = _eval_exact_oracle(opt_tris_np)

    consistency_ok = bool(zero_identity_ok)
    vol_residual = float(math.fsum((areas_np * delta_h_opt).tolist())) if num_b > 0 else 0.0
    total_scale_vol = float(math.fsum((areas_np * np.maximum(height_np, 1.0)).tolist())) if num_b > 0 else 1.0
    rel_vol_residual = abs(vol_residual) / max(total_scale_vol, 1.0)
    mesh_vol_delta = 0.0
    has_nonvolumetric_fallback = False
    if num_b > 0 and m_tris > 0:
        _, opt_height_np, _ = compute_building_vertical_anchors(opt_tris_np, b_ids_np)
        min_drop = np.where(height_np > 1e-9, np.minimum(0.0, 0.05 - height_np), 0.0)
        eff_dh = np.maximum(delta_h_opt, min_drop)
        expected_h = np.where(height_np > 1e-9, height_np + eff_dh, 0.0)
        if not np.allclose(opt_height_np, expected_h, atol=1e-6):
            consistency_ok = False
        for b in range(num_b):
            sel = b_ids_np == b
            if np.any(sel) and height_np[b] > 1e-6:
                v_i = abs(
                    compute_closed_mesh_signed_volume_f64(
                        tri_np[sel], reference_y_m=float(y_min_np[b])
                    )
                )
                v_o = abs(
                    compute_closed_mesh_signed_volume_f64(
                        opt_tris_np[sel], reference_y_m=float(y_min_np[b])
                    )
                )
                if v_i <= 1e-6:
                    has_nonvolumetric_fallback = True
                mesh_vol_delta += v_o - v_i
        if enforce_volume_conservation:
            if abs(vol_residual) > 1e-9 or rel_vol_residual > 1e-12:
                consistency_ok = False
            if (not has_nonvolumetric_fallback) and abs(mesh_vol_delta) > 1e-5:
                consistency_ok = False
            if not consistency_ok:
                raise RuntimeError(
                    "Volume/manifold conservation gate failed in optimize_fullday_building_heights: "
                    f"vol_residual={vol_residual:.3e} m^3, rel_vol_residual={rel_vol_residual:.3e}, "
                    f"mesh_vol_delta={mesh_vol_delta:.3e} m^3"
                )

    unified_vol_res = (
        max(abs(vol_residual), abs(mesh_vol_delta))
        if (not has_nonvolumetric_fallback and closed_manifold_ok)
        else abs(vol_residual)
    )

    init_sum = int(init_exact[val_np].sum())
    opt_sum = int(opt_exact[val_np].sum())
    opt_frac = float(opt_sum / valid_count)
    signed_delta_pf = int(opt_sum - init_sum)
    signed_delta_frac = float((opt_sum - init_sum) / valid_count)

    rec_has_valid = np.any(val_np, axis=0)
    per_rec_init = (init_exact * val_np).sum(axis=0)
    per_rec_opt = (opt_exact * val_np).sum(axis=0)
    per_rec_diff = per_rec_opt - per_rec_init
    improved_n = int(np.sum(rec_has_valid & (per_rec_diff > 0)))
    degraded_n = int(np.sum(rec_has_valid & (per_rec_diff < 0)))
    unchanged_n = int(np.sum(rec_has_valid & (per_rec_diff == 0)))
    invalid_rec_n = int(np.sum(~rec_has_valid))

    regime_deltas: Dict[str, Dict[str, float]] = {}
    if solar_regimes is not None and len(solar_regimes) == t_steps:
        reg_arr = np.asarray(list(solar_regimes))
        for r_name in sorted(set(reg_arr.tolist())):
            rmask = (reg_arr == r_name)[:, None] & val_np
            rcount = int(rmask.sum())
            if rcount > 0:
                ri = float(init_exact[rmask].sum() / rcount)
                ro = float(opt_exact[rmask].sum() / rcount)
                regime_deltas[str(r_name)] = {
                    "point_frames": float(rcount),
                    "initial_sunlit_fraction": ri,
                    "optimized_sunlit_fraction": ro,
                    "signed_delta_fraction": float(ro - ri),
                }

    wall_s = _wall_elapsed()
    return FullDayInverseDesignResult(
        optimizer_arm=optimizer_arm,
        checkpoint_sha256=checkpoint_sha256,
        num_receivers=n_pts,
        num_frames=t_steps,
        num_triangles=m_tris,
        num_buildings=num_b,
        max_abs_delta_h_m=float(max_delta_h_m),
        enforce_volume_conservation=bool(enforce_volume_conservation),
        enforce_zero_sum_height=bool(enforce_zero_sum_height),
        building_effective_areas_m2=[float(x) for x in areas_np.tolist()],
        delta_h_per_building_m=[float(x) for x in delta_h_opt.tolist()],
        area_weighted_volume_residual_m3=vol_residual,
        mesh_signed_volume_delta_m3=mesh_vol_delta,
        initial_exact_sunlit_fraction=init_frac,
        optimized_exact_sunlit_fraction=opt_frac,
        signed_net_sunlight_delta_fraction=signed_delta_frac,
        signed_net_sunlight_delta_point_frames=signed_delta_pf,
        improved_receiver_count=improved_n,
        degraded_receiver_count=degraded_n,
        unchanged_receiver_count=unchanged_n,
        regime_signed_deltas=regime_deltas,
        surrogate_initial_sunlit_fraction=surr_init,
        surrogate_optimized_sunlit_fraction=surr_opt,
        oracle_queries=oracle_queries,
        model_queries=model_queries,
        wall_seconds=wall_s,
        building_consistency_verified=consistency_ok,
        zero_perturbation_identity_verified=zero_identity_ok,
        instance_provenance_tier=provenance_tier,
        closed_mesh_volume_delta_m3=mesh_vol_delta,
        volume_conservation_residual_m3=unified_vol_res,
        relative_volume_residual=rel_vol_residual,
        closed_solid_manifold_verified=closed_manifold_ok,
        valid_point_frames=valid_count,
        invalid_receiver_count=invalid_rec_n,
        empty_support_reason=None,
        max_oracle_queries=max_oracle_queries,
        max_wall_clock_s=max_wall_clock_s,
        budget_stopped_early=budget_stopped_early,
    )


def aggregate_multi_tile_inverse_design(
    results: Sequence[FullDayInverseDesignResult | Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate multi-tile inverse-design receipts, excluding empty valid supports from denominators."""
    dicts = [r.to_dict() if isinstance(r, FullDayInverseDesignResult) else dict(r) for r in results]
    valid_items = [
        d for d in dicts
        if d.get("signed_net_sunlight_delta_fraction") is not None and int(d.get("valid_point_frames", 1)) > 0
    ]
    excluded_empty = len(dicts) - len(valid_items)
    if not valid_items:
        return {
            "total_tiles": len(dicts),
            "evaluated_nonempty_tiles": 0,
            "excluded_empty_support_tiles": excluded_empty,
            "mean_signed_net_sunlight_delta_fraction": None,
            "median_signed_net_sunlight_delta_fraction": None,
            "negative_gain_count": 0,
            "negative_gain_fraction": None,
            "mean_oracle_queries": 0.0,
            "mean_wall_seconds": 0.0,
        }
    deltas = np.array([float(d["signed_net_sunlight_delta_fraction"]) for d in valid_items], dtype=np.float64)
    neg_cnt = int(np.sum(deltas < 0.0))
    return {
        "total_tiles": len(dicts),
        "evaluated_nonempty_tiles": len(valid_items),
        "excluded_empty_support_tiles": excluded_empty,
        "mean_signed_net_sunlight_delta_fraction": float(np.mean(deltas)),
        "median_signed_net_sunlight_delta_fraction": float(np.median(deltas)),
        "std_signed_net_sunlight_delta_fraction": float(np.std(deltas)),
        "negative_gain_count": neg_cnt,
        "negative_gain_fraction": float(neg_cnt / len(valid_items)),
        "mean_oracle_queries": float(np.mean([float(d.get("oracle_queries", 0)) for d in valid_items])),
        "mean_model_queries": float(np.mean([float(d.get("model_queries", 0)) for d in valid_items])),
        "mean_wall_seconds": float(np.mean([float(d.get("wall_seconds", 0.0)) for d in valid_items])),
        "max_volume_conservation_residual_m3": float(
            np.max([abs(float(d.get("volume_conservation_residual_m3", 0.0))) for d in valid_items])
        ),
    }


def _reconstruct_and_predict_scene_trajectory(
    receivers_xyz: np.ndarray,
    sun_directions_xyz: np.ndarray,
    scene_triangles_xyz: np.ndarray,
    *,
    reference_trajectory=None,
    blind_faces=None,
    exclude_receivers=None,
    rind_model: Optional[torch.nn.Module] = None,
    opaque_mask: Optional[np.ndarray] = None,
    source_triangle_ids: Optional[np.ndarray] = None,
    exclude_source_triangle_ids: Optional[np.ndarray] = None,
    same_day_adjacent_mask: Optional[np.ndarray] = None,
    max_candidates: int = 8,
    num_azimuth_bins: Optional[int] = None,
    first_k_candidates_only: Optional[int] = None,
    building_ids_for_bottleneck: Optional[np.ndarray] = None,
    dominant_building_per_receiver: Optional[np.ndarray] = None,
    crop_size_px: int = 21,
    meters_per_pixel: float = DEFAULT_TILE_METERS_PER_PIXEL,
    t_max_m: float = DEFAULT_T_MAX_M,
    device: str = "cpu",
) -> np.ndarray:
    """Anti-leakage forward evaluation `(T, N)` reconstructing ALL geometric inputs from `scene_triangles_xyz`.

    Guarantees ZERO leakage of any unedited or hidden building heights:
    - Re-extracts `local_features` and `receiver_crops` from `scene_triangles_xyz`.
    - Re-selects azimuth-sector candidates and rebuilds `canonical37` tokens from `scene_triangles_xyz`.
    - Optionally restricts structural memory to `dominant_building_per_receiver` (anchored first-occluder
      bottleneck that retains ONLY the dominant occluder explaining `t_0` and drops rear structures `B`)
      or `first_k_candidates_only`.
    """
    rec_np = np.asarray(receivers_xyz, dtype=np.float64)
    sun_np = np.asarray(sun_directions_xyz, dtype=np.float64)
    tri_np = np.asarray(scene_triangles_xyz, dtype=np.float64)
    t_steps = int(sun_np.shape[0])
    n_pts = int(rec_np.shape[0])
    m_tris = int(tri_np.shape[0])
    if t_steps == 0 or n_pts == 0:
        return np.zeros((t_steps, n_pts), dtype=np.float64)
    az_bins = _resolve_model_azimuth_bins(rind_model, num_azimuth_bins)

    if reference_trajectory is not None and rind_model is not None:
        from .native_scene import native_scene_batch
        if not np.array_equal(reference_trajectory.sun_dirs, sun_np):
            raise ValueError("native trajectory and requested solar sequence differ")
        rind_model.to(device).eval()
        def predict(faces=None):
            batch = native_scene_batch(reference_trajectory, rind_model, tri_np,
                receivers=rec_np, first_k=first_k_candidates_only, face_mask=faces,
                blind_faces=blind_faces, exclude_receivers=exclude_receivers, device=device)
            with torch.no_grad():
                probs = rind_model(batch,use_calibrated_temperature=True)["probs"].reshape(t_steps,n_pts)
            if not torch.isfinite(probs).all():
                raise ValueError("nonfinite native scene prediction")
            return probs.cpu().numpy().astype(np.float64)
        if dominant_building_per_receiver is None:
            return predict()
        result = np.empty((t_steps,n_pts), dtype=np.float64)
        ids = np.asarray(building_ids_for_bottleneck)
        for bid in np.unique(dominant_building_per_receiver):
            rows = dominant_building_per_receiver == bid
            result[:,rows] = predict((ids == bid) | (ids < 0))[:,rows]
        return result

    if rind_model is not None and dominant_building_per_receiver is not None:
        # Rebuild every input, including local height/crops, from the oracle-selected scene.
        if building_ids_for_bottleneck is None:
            raise ValueError("oracle building selection requires scene building ids")
        output = np.empty((t_steps, n_pts), dtype=np.float64)
        for building in np.unique(dominant_building_per_receiver):
            rows = np.asarray(dominant_building_per_receiver) == building
            ids = np.asarray(building_ids_for_bottleneck)
            faces = (ids == building) | (ids < 0)
            output[:, rows] = _reconstruct_and_predict_scene_trajectory(
                rec_np[rows], sun_np, tri_np[faces], rind_model=rind_model,
                opaque_mask=None if opaque_mask is None else np.asarray(opaque_mask)[faces],
                source_triangle_ids=None if source_triangle_ids is None else np.asarray(source_triangle_ids)[faces],
                exclude_source_triangle_ids=None if exclude_source_triangle_ids is None else np.asarray(exclude_source_triangle_ids)[rows],
                same_day_adjacent_mask=same_day_adjacent_mask, max_candidates=max_candidates,
                num_azimuth_bins=az_bins, crop_size_px=crop_size_px,
                meters_per_pixel=meters_per_pixel, t_max_m=t_max_m, device=device,
            )
        return output

    if rind_model is None:
        dev = torch.device(device)
        op_arr = (
            np.asarray(opaque_mask, dtype=bool)
            if opaque_mask is not None
            else np.ones((m_tris,), dtype=bool)
        )
        if (
            dominant_building_per_receiver is not None
            and building_ids_for_bottleneck is not None
            and m_tris > 0
        ):
            dom_r = np.asarray(dominant_building_per_receiver, dtype=np.int64)
            bids_m = np.asarray(building_ids_for_bottleneck, dtype=np.int64)
            excl_arr = (
                np.asarray(exclude_source_triangle_ids, dtype=np.int64)
                if exclude_source_triangle_ids is not None
                else None
            )
            out_probs = np.ones((t_steps, n_pts), dtype=np.float64)
            for ni in range(n_pts):
                b_dom = int(dom_r[ni])
                if b_dom < 0:
                    sel_m = op_arr & (bids_m < 0)
                else:
                    sel_m = op_arr & ((bids_m == b_dom) | (bids_m < 0))
                if not np.any(sel_m):
                    out_probs[:, ni] = np.where(sun_np[:, 1] > 0.0, 1.0, 0.0)
                    continue
                sids_sub = (
                    torch.as_tensor(np.asarray(source_triangle_ids, dtype=np.int64)[sel_m], dtype=torch.int64, device=dev)
                    if source_triangle_ids is not None
                    else None
                )
                excl_sub = (
                    torch.as_tensor(excl_arr[ni : ni + 1], dtype=torch.int64, device=dev)
                    if excl_arr is not None
                    else None
                )
                with torch.no_grad():
                    p_col = _differentiable_soft_moller_trumbore_sunlit(
                        torch.as_tensor(rec_np[ni : ni + 1], dtype=torch.float32, device=dev),
                        torch.as_tensor(sun_np, dtype=torch.float32, device=dev),
                        torch.as_tensor(tri_np[sel_m], dtype=torch.float32, device=dev),
                        source_triangle_ids=sids_sub,
                        exclude_source_triangle_ids=excl_sub,
                        temperature_m=0.35,
                        t_min=DEFAULT_T_MIN_M,
                        t_max=float(t_max_m),
                    )
                out_probs[:, ni] = p_col[:, 0].detach().cpu().numpy().astype(np.float64)
            return out_probs
        tris_act = tri_np[op_arr] if m_tris > 0 else tri_np
        if first_k_candidates_only is not None and int(first_k_candidates_only) >= 1 and tris_act.shape[0] > 0:
            k_cap = int(first_k_candidates_only)
            if az_bins == 0:
                from .ray_candidates import RayCandidateScene

                sids_arr = (
                    np.asarray(source_triangle_ids, dtype=np.uint64)
                    if source_triangle_ids is not None
                    else np.arange(m_tris, dtype=np.uint64)
                )
                ray_scene = RayCandidateScene(tri_np, op_arr, sids_arr)
                ray_rel, _, ray_mk, _, _ = ray_scene.query(
                    rec_np, sun_np, float(t_max_m), k_cap,
                    receiver_sources=exclude_source_triangle_ids,
                )
                rel_tnk = ray_rel.swapaxes(0, 1)
                mk_tnk = ray_mk.swapaxes(0, 1)
            else:
                from .fullday_data import select_azimuth_sector_candidates, sun_dirs_to_azimuth_bins

                relative, _, az_mk, _ = select_azimuth_sector_candidates(
                    rec_np, tri_np, scene_opaque=op_arr, num_azimuth_bins=az_bins, max_candidates=k_cap,
                    horizon_m=float(t_max_m), scene_source_triangle_ids=source_triangle_ids,
                    exclude_source_triangle_ids=exclude_source_triangle_ids, return_source_ids=True,
                )
                bins = sun_dirs_to_azimuth_bins(sun_np, num_azimuth_bins=az_bins)
                rel_tnk = relative.swapaxes(0, 1)[bins]
                mk_tnk = az_mk.swapaxes(0, 1)[bins]

            out_probs = np.ones((t_steps, n_pts), dtype=np.float64)
            for ti in range(t_steps):
                if sun_np[ti, 1] <= 0.0:
                    out_probs[ti] = 0.0
                    continue
                for ni in range(n_pts):
                    mk_row = mk_tnk[ti, ni, :k_cap]
                    if not np.any(mk_row):
                        out_probs[ti, ni] = 1.0
                        continue
                    sub_t = rec_np[ni] + rel_tnk[ti, ni, :k_cap][mk_row]
                    with torch.no_grad():
                        p_val = _differentiable_soft_moller_trumbore_sunlit(
                            torch.as_tensor(rec_np[ni : ni + 1], dtype=torch.float32, device=dev),
                            torch.as_tensor(sun_np[ti : ti + 1], dtype=torch.float32, device=dev),
                            torch.as_tensor(sub_t, dtype=torch.float32, device=dev),
                            temperature_m=0.35,
                            t_min=DEFAULT_T_MIN_M,
                            t_max=float(t_max_m),
                        )
                    out_probs[ti, ni] = float(p_val.item())
            return out_probs

        with torch.no_grad():
            sids_act = (
                torch.as_tensor(np.asarray(source_triangle_ids, dtype=np.int64)[op_arr], dtype=torch.int64, device=dev)
                if source_triangle_ids is not None and m_tris > 0
                else None
            )
            excl_t = (
                torch.as_tensor(np.asarray(exclude_source_triangle_ids, dtype=np.int64), dtype=torch.int64, device=dev)
                if exclude_source_triangle_ids is not None
                else None
            )
            probs_t = _differentiable_soft_moller_trumbore_sunlit(
                torch.as_tensor(rec_np, dtype=torch.float32, device=dev),
                torch.as_tensor(sun_np, dtype=torch.float32, device=dev),
                torch.as_tensor(tris_act, dtype=torch.float32, device=dev),
                source_triangle_ids=sids_act,
                exclude_source_triangle_ids=excl_t,
                temperature_m=0.35,
                t_min=DEFAULT_T_MIN_M,
                t_max=float(t_max_m),
            )
        return probs_t.detach().cpu().numpy().astype(np.float64)

    from .fast_data import build_canonical37_torch
    from .fullday_data import (
        select_azimuth_sector_candidates,
        sun_dirs_to_azimuth_bins,
    )

    dev = torch.device(device)
    rind_model = rind_model.to(dev)
    rind_model.eval()
    k_model = int(getattr(rind_model, "max_candidates", max_candidates))
    c_side = int(getattr(rind_model, "crop_size", crop_size_px))

    b_ids_rec = assign_building_ids_from_triangles(tri_np)
    _, h_rec_arr, num_b_rec = compute_building_vertical_anchors(tri_np, b_ids_rec)
    exact_w_r, neigh_w_r = _build_local_feature_height_couplings(
        rec_np,
        tri_np,
        b_ids_rec,
        num_b_rec,
        meters_per_pixel=float(meters_per_pixel),
    )
    crop_w_r = _build_spatial_crop_height_couplings(
        rec_np,
        tri_np,
        b_ids_rec,
        num_b_rec,
        crop_h=c_side,
        crop_w=c_side,
        meters_per_pixel=float(meters_per_pixel),
    )
    h_vec_f32 = h_rec_arr.astype(np.float32) if num_b_rec > 0 else np.zeros((0,), dtype=np.float32)
    h_center_scaled = (exact_w_r @ h_vec_f32) / 100.0 if num_b_rec > 0 else np.zeros((n_pts,), dtype=np.float32)
    h_neigh_scaled = (neigh_w_r @ h_vec_f32) / 100.0 if num_b_rec > 0 else np.zeros((n_pts,), dtype=np.float32)
    h_patch_scaled = (
        np.tensordot(crop_w_r, h_vec_f32, axes=([-1], [0])).astype(np.float32) / 100.0
        if num_b_rec > 0
        else np.zeros((n_pts, c_side, c_side), dtype=np.float32)
    )

    lf_np = np.zeros((n_pts, 16), dtype=np.float32)
    rec_y_scaled = (rec_np[:, 1] / 100.0).astype(np.float32)
    lf_np[:, 0] = rec_y_scaled
    lf_np[:, 1] = h_center_scaled
    lf_np[:, 2] = h_center_scaled
    lf_np[:, 3] = 1.0
    lf_np[:, 8] = rec_y_scaled
    lf_np[:, 9] = h_neigh_scaled
    lf_np[:, 10] = h_neigh_scaled
    lf_np[:, 11] = 1.0

    crops_np = np.zeros((n_pts, 9, c_side, c_side), dtype=np.float32)
    crops_np[:, 0, :, :] = rec_y_scaled[:, None, None]
    crops_np[:, 1, :, :] = h_patch_scaled
    crops_np[:, 2, :, :] = h_patch_scaled
    crops_np[:, 3, :, :] = 1.0
    crops_np[:, 8, :, :] = 1.0
    if az_bins == 0:
        if m_tris == 0 or k_model <= 0:
            rel_np = np.zeros((t_steps, n_pts, max(k_model, 0), 3, 3), dtype=np.float32)
            op_np = np.zeros((t_steps, n_pts, max(k_model, 0)), dtype=bool)
            mk_np = np.zeros((t_steps, n_pts, max(k_model, 0)), dtype=bool)
        else:
            from .ray_candidates import RayCandidateScene

            op_arr = (
                np.asarray(opaque_mask, dtype=bool)
                if opaque_mask is not None
                else np.ones((m_tris,), dtype=bool)
            )
            sids_arr = (
                np.asarray(source_triangle_ids, dtype=np.uint64)
                if source_triangle_ids is not None
                else np.arange(m_tris, dtype=np.uint64)
            )
            ray_scene = RayCandidateScene(tri_np, op_arr, sids_arr)
            ray_rel, ray_op, ray_mk, _, _ = ray_scene.query(
                rec_np,
                sun_np,
                float(t_max_m),
                k_model,
                receiver_sources=exclude_source_triangle_ids,
            )
            if first_k_candidates_only is not None:
                ray_mk[..., int(first_k_candidates_only):] = False
            rel_np = ray_rel.swapaxes(0, 1)
            op_np = ray_op.swapaxes(0, 1)
            mk_np = ray_mk.swapaxes(0, 1)
    else:
        az_rel, az_op, az_mk, _ = select_azimuth_sector_candidates(
            rec_np, tri_np, scene_opaque=opaque_mask, scene_source_triangle_ids=source_triangle_ids,
            exclude_source_triangle_ids=exclude_source_triangle_ids, num_azimuth_bins=az_bins,
            max_candidates=k_model, horizon_m=float(t_max_m), return_source_ids=True,
        )
        bins = sun_dirs_to_azimuth_bins(sun_np, num_azimuth_bins=az_bins)
        if first_k_candidates_only is not None:
            az_mk[..., int(first_k_candidates_only):] = False
        rel_np = az_rel.swapaxes(0, 1)[bins]
        op_np = az_op.swapaxes(0, 1)[bins]
        mk_np = az_mk.swapaxes(0, 1)[bins]
    with torch.no_grad():
        rec_t = torch.as_tensor(rec_np, dtype=torch.float32, device=dev)
        sun_t = torch.as_tensor(sun_np, dtype=torch.float32, device=dev)
        mk_t = torch.as_tensor(mk_np, device=dev)
        op_t = torch.as_tensor(op_np, device=dev) & mk_t
        rel_tris = torch.as_tensor(rel_np, device=dev)
        tokens = build_canonical37_torch(rel_tris, op_t, mk_t, sun_t)
        if hasattr(rind_model, "sanitize_tokens"):
            tokens = rind_model.sanitize_tokens(tokens)  # type: ignore[operator]
        adj_t = (
            torch.as_tensor(same_day_adjacent_mask, dtype=torch.bool, device=dev)
            if same_day_adjacent_mask is not None
            else torch.ones((max(t_steps - 1, 0),), dtype=torch.bool, device=dev)
        )
        crops_t = torch.as_tensor(crops_np, dtype=torch.float32, device=dev).unsqueeze(0)
        batch_in: Dict[str, Any] = {
            "candidate_tokens": tokens.unsqueeze(0),
            "candidate_mask": mk_t.unsqueeze(0),
            "local_features": torch.as_tensor(lf_np, dtype=torch.float32, device=dev).unsqueeze(0),
            "receiver_crops": crops_t,
            "spatial_patch_crops": crops_t,
            "receivers_xyz": rec_t.unsqueeze(0),
            "sun_dirs": sun_t.unsqueeze(0),
            "same_day_adjacent_mask": adj_t.unsqueeze(0),
        }
        if hasattr(rind_model, "calibrated_temperature"):
            out = rind_model(batch_in, use_calibrated_temperature=True)
        else:
            out = rind_model(batch_in)
        return out["probs"].squeeze(0).detach().cpu().numpy().astype(np.float64)


def infer_unknown_building_heights_bayesian(
    receivers_xyz: np.ndarray,
    sun_directions_xyz: np.ndarray,
    triangles_xyz: np.ndarray,
    *,
    reference_trajectory=None,
    receiver_elevation_public: bool = False,
    heldout_frame_indices: Sequence[int] = (),
    max_joint_hypotheses: int = 1024,
    unknown_building_ids: Optional[Sequence[int]] = None,
    building_ids: Optional[np.ndarray] = None,
    source_instance_ids: Optional[np.ndarray] = None,
    ground_mask: Optional[np.ndarray] = None,
    opaque_mask: Optional[np.ndarray] = None,
    source_triangle_ids: Optional[np.ndarray] = None,
    exclude_source_triangle_ids: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    same_day_adjacent_mask: Optional[np.ndarray] = None,
    rind_model: Optional[torch.nn.Module] = None,
    max_unknown_buildings: int = 2,
    height_grid_min_m: float = 2.0,
    height_grid_max_m: float = 50.0,
    num_grid_hypotheses: int = 25,
    neutral_reference_height_m: float = 15.0,
    sensor_noise_flip_prob: float = 0.05,
    initial_queries: int = 6,
    active_rounds: int = 4,
    queries_per_round: int = 6,
    t_max_m: float = DEFAULT_T_MAX_M,
    seed: int = 20261112,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Bayesian unknown building height inference under noisy/sparse observations (`P1` Highlight §5 Task 1).

    Strict anti-leakage protocol:
    1. Uses true 3D scene ONLY inside `ExactSceneBVH` + Bernoulli flip noise `sensor_noise_flip_prob`
       to simulate sparse field sensor readings on queried `(frame, receiver)` slots, and at the very end
       to score posterior MAE and `95%` credible-interval coverage.
    2. Immediately scrubs every unknown building in `scrubbed_triangles_xyz` to a fixed neutral reference
       height (`neutral_reference_height_m`), then reconstructs `local_features`, `receiver_crops`, and
       `canonical37` tokens from scratch at each candidate height hypothesis `h_g`!
    3. Evaluates active disagreement query selection (`select_active_disagreement_queries`) vs matched-budget
       random query selection across `0 .. active_rounds`, reporting MAE, `95%` credible set width shrinkage,
       and true-height coverage.
    """
    rec_np = np.asarray(receivers_xyz, dtype=np.float64)
    sun_np = np.asarray(sun_directions_xyz, dtype=np.float64)
    tri_true_np = np.asarray(triangles_xyz, dtype=np.float64)
    t_steps = int(sun_np.shape[0])
    n_pts = int(rec_np.shape[0])
    m_tris = int(tri_true_np.shape[0])

    val_np = (
        np.asarray(valid_mask, dtype=bool)
        if valid_mask is not None
        else np.ones((t_steps, n_pts), dtype=bool)
    )
    g_mask_np = np.asarray(ground_mask, dtype=bool) if ground_mask is not None else None
    if building_ids is not None:
        b_ids_np = np.asarray(building_ids, dtype=np.int64)
    else:
        b_ids_np = assign_building_ids_from_triangles(
            tri_true_np, source_instance_ids=source_instance_ids, ground_mask=g_mask_np
        )
    _, true_heights_all, num_b = compute_building_vertical_anchors(tri_true_np, b_ids_np)
    if num_b == 0 or m_tris == 0 or int(val_np.sum()) == 0:
        return {
            "status": "empty_scene_or_support",
            "anti_leakage_verified": True,
            "unknown_building_ids": [],
            "final_posterior_mae_m": None,
            "credible_95_coverage": None,
            "active_query_curve": [],
            "random_query_curve": [],
        }

    # Exact oracle truth + noisy sensor observation table (sampled only when queried!)
    true_bvh = ExactSceneBVH(
        tri_true_np, opaque_mask=opaque_mask, source_triangle_ids=source_triangle_ids
    )
    exact_sun_tn = true_bvh.query_trajectory_sunlit(
        rec_np,
        sun_np,
        t_min=DEFAULT_T_MIN_M,
        t_max=float(t_max_m),
        exclude_source_triangle_ids=exclude_source_triangle_ids,
    ).astype(np.float64)

    rng = np.random.default_rng(seed)
    eta = float(np.clip(sensor_noise_flip_prob, 0.0, 0.45))
    flips = rng.uniform(0.0, 1.0, size=(t_steps, n_pts)) < eta
    noisy_obs_pool_tn = np.where(flips, 1.0 - exact_sun_tn, exact_sun_tn)

    public_ids = sorted(set(b_ids_np[b_ids_np >= 0].tolist()))
    unk_bids = (list(map(int,unknown_building_ids)) if unknown_building_ids is not None
                else public_ids[:max(0,int(max_unknown_buildings))])
    if len(set(unk_bids)) != len(unk_bids) or not set(unk_bids).issubset(public_ids):
        raise ValueError("unknown building ids must be distinct published component ids")
    if not unk_bids:
        return {
            "status": "no_editable_buildings",
            "anti_leakage_verified": True,
            "unknown_building_ids": [],
            "final_posterior_mae_m": None,
            "credible_95_coverage": None,
            "active_query_curve": [],
            "random_query_curve": [],
        }

    for bid in unk_bids:
        faces = tri_true_np[b_ids_np == bid]
        levels = faces[:,:,1]
        base_y = levels.min()
        normals = np.cross(faces[:,1]-faces[:,0], faces[:,2]-faces[:,0])
        has_base_face = ((levels.max(axis=1)-base_y <= 1e-6) & (abs(normals[:,1]) > 1e-8)).any()
        has_wall_base = (((levels-base_y <= 1e-6).sum(axis=1)>=2) &
                         (levels.max(axis=1)>base_y+1e-6) &
                         (np.abs(normals[:,1]) <= 1e-6*np.linalg.norm(normals,axis=1)) &
                         (np.linalg.norm(normals,axis=1)>1e-8)).any()
        if true_heights_all[bid] <= 1e-6 or not (has_base_face or has_wall_base):
            raise ValueError(f"Unknown building {bid} lacks a supported public base and vertical span")
    roof_excluded = np.zeros(n_pts,dtype=bool)
    unknown_source = np.ones(n_pts,dtype=bool)
    if not receiver_elevation_public:
        face_ids = (np.arange(m_tris,dtype=np.uint64) if source_triangle_ids is None
                    else np.asarray(source_triangle_ids,dtype=np.uint64))
        if exclude_source_triangle_ids is not None:
            sensor_ids = np.asarray(exclude_source_triangle_ids,dtype=np.uint64)
            unknown_source = ~np.isin(sensor_ids,face_ids) | (sensor_ids == np.iinfo(np.uint64).max)
            roof_excluded = ~unknown_source & np.isin(sensor_ids,face_ids[np.isin(b_ids_np,unk_bids)])
        # Unidentified receivers inside a public horizontal building bounding box
        # cannot be certified as ground sensors. This rule never examines their y.
        for bid in unk_bids:
            footprint = tri_true_np[b_ids_np==bid][:,:,[0,2]]
            lo,hi = footprint.min(axis=(0,1)),footprint.max(axis=(0,1))
            in_box = ((rec_np[:,[0,2]]>=lo-1e-6)&(rec_np[:,[0,2]]<=hi+1e-6)).all(axis=1)
            roof_excluded |= unknown_source & in_box
        val_np = val_np & ~roof_excluded[None,:]
        rec_np = rec_np.copy()
        rec_np[roof_excluded,1] = 0.
    true_unk_heights = np.array([float(true_heights_all[b]) for b in unk_bids], dtype=np.float64)

    # Scrub ALL unknown buildings in `scrubbed_tris` to `neutral_reference_height_m` so zero true height remains!
    scrub_dh = np.zeros((num_b,), dtype=np.float64)
    for b in unk_bids:
        scrub_dh[b] = float(neutral_reference_height_m) - float(true_heights_all[b])
    scrubbed_tris = perturb_buildings_consistent_f64(tri_true_np, b_ids_np, scrub_dh)
    _, scrubbed_heights_all, _ = compute_building_vertical_anchors(scrubbed_tris, b_ids_np)
    anti_leakage_ok = bool(
        np.allclose(
            [scrubbed_heights_all[b] for b in unk_bids],
            float(neutral_reference_height_m),
            atol=1e-5,
        )
    )

    if not anti_leakage_ok:
        raise ValueError("Neutralization did not remove all unknown heights; blind inference unavailable")

    if not (np.isfinite(height_grid_min_m) and np.isfinite(height_grid_max_m)
            and 0 <= height_grid_min_m < height_grid_max_m):
        raise ValueError("height grid requires finite 0 <= min < max")
    h_grid = np.linspace(
        float(height_grid_min_m),
        float(height_grid_max_m),
        max(5, int(num_grid_hypotheses)),
        dtype=np.float64,
    )
    g_cnt = int(h_grid.size)
    u_cnt = len(unk_bids)

    import itertools
    total_states = g_cnt ** u_cnt
    limit = int(max_joint_hypotheses)
    if limit < 1:
        raise ValueError("max_joint_hypotheses must be positive")
    if total_states <= limit:
        tuples = np.asarray(list(itertools.product(range(g_cnt),repeat=u_cnt)),dtype=np.int64)
    else:
        sampled = set()
        proposal = np.random.default_rng(seed)
        while len(sampled)<limit:
            sampled.add(tuple(proposal.integers(g_cnt,size=u_cnt).tolist()))
        tuples = np.asarray(sorted(sampled),dtype=np.int64)
    hyp_probs = np.empty((len(tuples),t_steps,n_pts),dtype=np.float64)
    for index, indices in enumerate(tuples):
        dh_hyp = np.zeros(num_b)
        dh_hyp[unk_bids] = h_grid[indices] - scrubbed_heights_all[unk_bids]
        cand_tris = perturb_buildings_consistent_f64(scrubbed_tris,b_ids_np,dh_hyp)
        hyp_probs[index] = _reconstruct_and_predict_scene_trajectory(
            rec_np,sun_np,cand_tris,reference_trajectory=reference_trajectory,
            blind_faces=np.isin(b_ids_np,unk_bids),exclude_receivers=roof_excluded,rind_model=rind_model,opaque_mask=opaque_mask,
            source_triangle_ids=source_triangle_ids,exclude_source_triangle_ids=exclude_source_triangle_ids,
            same_day_adjacent_mask=same_day_adjacent_mask,t_max_m=t_max_m,device=device)
    eps_lik = max(eta,.03)
    def normalize(log_post):
        p = np.exp(log_post-log_post.max())
        return p/p.sum()

    def _summarize_posterior(log_post_ug: np.ndarray, num_q: int) -> Dict[str, Any]:
        joint = normalize(log_post_ug)
        post_ug = np.stack([np.bincount(indices,weights=joint,minlength=g_cnt) for indices in tuples.T])
        mean_h = (post_ug * h_grid[None, :]).sum(axis=1)
        map_h = h_grid[tuples[np.argmax(joint)]]
        marginal_map = h_grid[np.argmax(post_ug,axis=1)]
        cdf = np.cumsum(post_ug, axis=1)
        ci_lo = np.array([float(h_grid[min(g_cnt-1,np.searchsorted(cdf[u], 0.025, side="left"))]) for u in range(u_cnt)])
        ci_hi = np.array([float(h_grid[min(g_cnt - 1, int(np.searchsorted(cdf[u], 0.975, side="left")))]) for u in range(u_cnt)])
        covered = (true_unk_heights >= (ci_lo - 1e-6)) & (true_unk_heights <= (ci_hi + 1e-6))
        ent = -np.sum(post_ug * np.log2(np.clip(post_ug, 1e-15, 1.0)), axis=1)
        return {
            "num_queries": int(num_q),
            "mae_m": float(np.mean(np.abs(mean_h - true_unk_heights))),
            "map_mae_m": float(np.mean(np.abs(map_h - true_unk_heights))),
            "rmse_m": float(np.sqrt(np.mean((mean_h - true_unk_heights) ** 2))),
            "credible_95_coverage": float(np.mean(covered)),
            "mean_credible_95_width_m": float(np.mean(ci_hi - ci_lo)),
            "mean_posterior_entropy_bits": float(np.mean(ent)),
            "posterior_mean_heights_m": [float(x) for x in mean_h.tolist()],
            "posterior_map_heights_m": map_h.tolist(),
            "marginal_map_heights_m": marginal_map.tolist(),
            "credible_interval_95_m": [[float(ci_lo[u]), float(ci_hi[u])] for u in range(u_cnt)],
        }

    def _update_log_post(log_post_ug: np.ndarray, new_pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
        out_lp = log_post_ug.copy()
        for ti, ri in new_pairs:
            y_val = float(noisy_obs_pool_tn[ti, ri])
            p_ug = hyp_probs[:,ti,ri]
            q_ug = np.clip((1.0 - eps_lik) * p_ug + eps_lik * (1.0 - p_ug), 1e-6, 1.0 - 1e-6)
            out_lp += y_val * np.log(q_ug) + (1.0 - y_val) * np.log(1.0 - q_ug)
        return out_lp

    prior_lp = np.zeros(len(tuples), dtype=np.float64)
    prior_summary = _summarize_posterior(prior_lp, 0)

    query_domain = val_np.copy()
    heldout = np.asarray(heldout_frame_indices,dtype=int)
    if np.any((heldout<0)|(heldout>=t_steps)):
        raise ValueError("held-out frame out of range")
    query_domain[heldout] = False
    valid_flat = np.flatnonzero(query_domain.reshape(-1))
    init_k = min(max(0, int(initial_queries)), int(valid_flat.size))
    init_chosen = rng.choice(valid_flat, size=init_k, replace=False)
    init_pairs = [(int(idx // n_pts), int(idx % n_pts)) for idx in init_chosen.tolist()]

    # Arm 1: Active disagreement query acquisition curve
    active_pairs = init_pairs.copy()
    avail_active = query_domain.copy()
    for ti, ri in init_pairs:
        avail_active[ti, ri] = False
    lp_active = _update_log_post(prior_lp, init_pairs)
    active_curve: List[Dict[str, Any]] = [prior_summary, _summarize_posterior(lp_active, len(init_pairs))]
    queried_active_count = len(init_pairs)

    for _ in range(max(0, int(active_rounds))):
        if not np.any(avail_active):
            break
        acq = select_active_disagreement_queries(
            hyp_probs, hypothesis_weights=normalize(lp_active),
            receivers_xyz=rec_np,
            valid_mask=avail_active,
            top_k=int(queries_per_round),
            spatial_suppression_radius_m=2.0,
        )
        new_pairs = [(int(t_i), int(r_i)) for t_i, r_i in acq["selected_frame_receiver_pairs"]]
        if not new_pairs:
            break
        for ti, ri in new_pairs:
            avail_active[ti, ri] = False
        lp_active = _update_log_post(lp_active, new_pairs)
        active_pairs.extend(new_pairs)
        queried_active_count += len(new_pairs)
        active_curve.append(_summarize_posterior(lp_active, queried_active_count))

    # Arm 2: Matched-budget random query baseline
    random_pairs = init_pairs.copy()
    avail_rand = query_domain.copy()
    for ti, ri in init_pairs:
        avail_rand[ti, ri] = False
    lp_rand = _update_log_post(prior_lp, init_pairs)
    random_curve: List[Dict[str, Any]] = [prior_summary, _summarize_posterior(lp_rand, len(init_pairs))]
    queried_rand_count = len(init_pairs)
    for step_i in range(2, len(active_curve)):
        need = int(active_curve[step_i]["num_queries"]) - queried_rand_count
        rem_flat = np.flatnonzero(avail_rand.reshape(-1))
        if need <= 0 or rem_flat.size == 0:
            break
        pick = rng.choice(rem_flat, size=min(need, int(rem_flat.size)), replace=False)
        r_pairs = [(int(idx // n_pts), int(idx % n_pts)) for idx in pick.tolist()]
        for ti, ri in r_pairs:
            avail_rand[ti, ri] = False
        lp_rand = _update_log_post(lp_rand, r_pairs)
        random_pairs.extend(r_pairs)
        queried_rand_count += len(r_pairs)
        random_curve.append(_summarize_posterior(lp_rand, queried_rand_count))

    final_active = active_curve[-1]
    final_random = random_curve[-1]
    prior_w = max(float(prior_summary["mean_credible_95_width_m"]), 1e-6)
    final_w = float(final_active["mean_credible_95_width_m"])

    return {
        "status": "ok" if query_domain.any() else "no_query_support",
        "sensor_eligibility": ("declared_public_elevation" if receiver_elevation_public else
                               "identified_public_surface_or_outside_unknown_building_xz_bounds"),
        "uncertified_receivers_excluded": np.flatnonzero(roof_excluded & unknown_source).tolist(),
        "query_domain_point_frames": int(query_domain.sum()),
        "anti_leakage_verified": anti_leakage_ok,
        "receiver_elevation_public": receiver_elevation_public,
        "excluded_unknown_surface_receivers": np.flatnonzero(roof_excluded).tolist(),
        "marginal_map_heights_m": final_active["marginal_map_heights_m"],
        "target_selection": "explicit_ids" if unknown_building_ids is not None else "public_component_id_order",
        "hypothesis_mode": "joint_exact" if len(tuples)==total_states else "uniform_joint_subset",
        "joint_grid_size": total_states,
        "joint_height_tuples_m": h_grid[tuples].tolist(),
        "joint_posterior": normalize(lp_active).tolist(),
        "marginal_posterior": [np.bincount(u,weights=normalize(lp_active),minlength=g_cnt).tolist() for u in tuples.T],
        "likelihood_noise":eps_lik,
        "active_observations": [[t,r,int(noisy_obs_pool_tn[t,r])] for t,r in active_pairs],
        "random_observations": [[t,r,int(noisy_obs_pool_tn[t,r])] for t,r in random_pairs],
        "heldout_frame_indices": heldout.tolist(),
        "heldout_support": int(val_np[heldout].sum()),
        "heldout_posterior_predictive_mae": (float(np.abs(np.average(hyp_probs[:,heldout],axis=0,weights=normalize(lp_active))-
            exact_sun_tn[heldout])[val_np[heldout]].mean()) if val_np[heldout].any() else None),
        "likelihood_source": "rind_checkpoint" if rind_model is not None else "differentiable_geometric_surrogate",
        "sensor_noise_flip_prob": eta,
        "neutral_reference_height_m": float(neutral_reference_height_m),
        "unknown_building_ids": unk_bids,
        "true_heights_m": [float(x) for x in true_unk_heights.tolist()],
        "posterior_mean_heights_m": final_active["posterior_mean_heights_m"],
        "posterior_map_heights_m": final_active["posterior_map_heights_m"],
        "credible_interval_95_m": final_active["credible_interval_95_m"],
        "initial_prior_mae_m": float(prior_summary["mae_m"]),
        "final_posterior_mae_m": float(final_active["mae_m"]),
        "final_random_baseline_mae_m": float(final_random["mae_m"]),
        "credible_95_coverage": float(final_active["credible_95_coverage"]),
        "height_grid_m": h_grid.tolist(),
        "prior_credible_95_width_m": float(prior_summary["mean_credible_95_width_m"]),
        "mean_credible_95_width_m": final_w,
        "uncertainty_shrinkage_ratio": float(final_w / prior_w),
        "active_query_curve": active_curve,
        "random_query_curve": random_curve,
    }


def build_layered_occlusion_benchmark_scene() -> Dict[str, Any]:
    """Construct a clean 3D multi-building urban plaza benchmark exhibiting genuine layered occlusion.

    Scene geometry (`+X` East, `+Y` Up, `+Z` North):
    - Plaza receivers `r` around `(50, 0.05, 55)`.
    - Front south-east blocker Building `A` (`x in [52, 62], z in [36, 44], y in [0, 22]`): closer to plaza.
    - Rear deeper blocker Building `B` (`x in [46, 62], z in [16, 26], y in [0, 34]`): behind `A` at `t_0`
      and extending further west so as the sun rotates from south-east (`t_0`) toward south (`t_0 + Δ`),
      `A` clears the ray while `B` governs the delayed shadow-to-sun regime transition!
    """
    from .fullday_data import _extruded_box_triangles

    ground = np.array(
        [
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [100.0, 0.0, 100.0]],
            [[0.0, 0.0, 0.0], [100.0, 0.0, 100.0], [0.0, 0.0, 100.0]],
        ],
        dtype=np.float64,
    )
    bldg_a = _extruded_box_triangles(57.0, 40.0, 5.0, 4.0, 0.0, 22.0).astype(np.float64)
    bldg_b = _extruded_box_triangles(54.0, 21.0, 8.0, 5.0, 0.0, 34.0).astype(np.float64)
    bldg_c = _extruded_box_triangles(28.0, 42.0, 4.5, 4.0, 0.0, 16.0).astype(np.float64)
    tris = np.concatenate([ground, bldg_a, bldg_b, bldg_c], axis=0)
    inst_ids = np.array(
        [-1, -1] + [0] * len(bldg_a) + [1] * len(bldg_b) + [2] * len(bldg_c),
        dtype=np.int64,
    )
    ground_mask = np.array([True, True] + [False] * (len(tris) - 2), dtype=bool)

    gx, gz = np.meshgrid(np.linspace(45.0, 55.0, 4), np.linspace(50.0, 60.0, 4))
    receivers = np.stack([gx.ravel(), np.full(16, 0.05), gz.ravel()], axis=-1).astype(np.float64)

    # Solar sweep from South-East (`az = 148 deg`) through South-West (`az = 225 deg`) at `alt = 24..32 deg`
    az_deg = np.linspace(148.0, 225.0, 16, dtype=np.float64)
    alt_deg = 24.0 + 6.0 * np.sin(np.linspace(0.0, math.pi, 16, dtype=np.float64))
    az_rad = np.deg2rad(az_deg)
    alt_rad = np.deg2rad(alt_deg)
    # Azimuth convention: 90 deg = +X (East), 180 deg = -Z (South), 270 deg = -X (West)
    suns = np.stack(
        [
            np.cos(alt_rad) * np.sin(az_rad),
            np.sin(alt_rad),
            np.cos(alt_rad) * np.cos(az_rad),
        ],
        axis=-1,
    )
    suns /= np.linalg.norm(suns, axis=-1, keepdims=True)
    return {
        "receivers_xyz": receivers,
        "sun_directions_xyz": suns,
        "triangles_xyz": tris,
        "source_instance_ids": inst_ids,
        "ground_mask": ground_mask,
    }


def _binary_f1_np(pred_bin: np.ndarray, target_bin: np.ndarray) -> Optional[float]:
    p = np.asarray(pred_bin, dtype=bool)
    y = np.asarray(target_bin, dtype=bool)
    tp = int(np.sum(p & y))
    fp = int(np.sum(p & (~y)))
    fn = int(np.sum((~p) & y))
    denom = 2 * tp + fp + fn
    return float((2.0 * tp) / denom) if denom > 0 else None


def _layered_segments(layered, eligible, adjacent):
    """First layered anchor per contiguous eligible same-day run, with exclusive end."""
    starts = eligible.copy()
    starts[1:] &= ~(eligible[:-1] & adjacent)
    ends = eligible.copy()
    ends[:-1] &= ~(eligible[1:] & adjacent)
    for first, last in zip(np.flatnonzero(starts), np.flatnonzero(ends)):
        anchors = np.flatnonzero(layered[first:last + 1])
        if len(anchors):
            yield int(first + anchors[0]), int(last + 1)


def evaluate_representation_sufficiency(
    receivers_xyz: np.ndarray,
    sun_directions_xyz: np.ndarray,
    triangles_xyz: np.ndarray,
    *,
    reference_trajectory=None,
    rind_model: Optional[torch.nn.Module] = None,
    building_ids: Optional[np.ndarray] = None,
    source_instance_ids: Optional[np.ndarray] = None,
    ground_mask: Optional[np.ndarray] = None,
    opaque_mask: Optional[np.ndarray] = None,
    source_triangle_ids: Optional[np.ndarray] = None,
    exclude_source_triangle_ids: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    same_day_adjacent_mask: Optional[np.ndarray] = None,
    edit_drop_fraction: float = 0.90,
    max_building_pairs: int = 4,
    t_max_m: float = DEFAULT_T_MAX_M,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Quantitative evaluator for Representation Sufficiency under Driver Changes & Structural Interventions.

    Implements the exact RIND G1 sufficiency analysis using double-precision `ExactSceneBVH` truth vs
    reconstructed model forward passes (ZERO fake numbers, ZERO height leakage):
      1. **Layered Occlusion Regime-Transition Gap (Driver Shift `omega_{t_0} -> omega_{t_0 + Delta}`)**:
         Identifies `(t_0, receiver, front_building A, rear_building B)` rays where BOTH `A` and `B`
         occlude at `t_0` (`y*_{t_0} = 0` leaves `B` unconstrained), then evaluates subsequent frames
         where `A` clears (`y*_{only A} = 1`), comparing Full Multi-Candidate RIND (`K=max_candidates`)
         against a Greedy First-Occluder Bottleneck (`K=1`) and Current-Output Persistence (`y_hat_{t_0}`).
      2. **Nonadditive Structural Edits & Unseen Edit Combinations (`e_12 = e_1 (+) e_2`)**:
         Applies single-building lowering `e_1` on `A`, single-building lowering `e_2` on `B`, and the
         unseen combined intervention `e_12 = e_1 (+) e_2`, computing exact physical nonadditive interaction
         residual `I*(e_1, e_2) = Delta V*(e_12) - (Delta V*(e_1) + Delta V*(e_2))` and comparing Full RIND
         `I_hat` against Naive Linear-Additive Superposition (`Delta V_hat(e_1) + Delta V_hat(e_2)`, `I_hat = 0`)
         and Greedy `K=1`.
    """
    rec_np = np.asarray(receivers_xyz, dtype=np.float64)
    sun_np = np.asarray(sun_directions_xyz, dtype=np.float64)
    tri_np = np.asarray(triangles_xyz, dtype=np.float64)
    t_steps = int(sun_np.shape[0])
    n_pts = int(rec_np.shape[0])
    m_tris = int(tri_np.shape[0])

    val_np = (
        np.asarray(valid_mask, dtype=bool)
        if valid_mask is not None
        else np.ones((t_steps, n_pts), dtype=bool)
    )
    g_mask_np = np.asarray(ground_mask, dtype=bool) if ground_mask is not None else None
    if building_ids is not None:
        b_ids_np = np.asarray(building_ids, dtype=np.int64)
    else:
        b_ids_np = assign_building_ids_from_triangles(
            tri_np, source_instance_ids=source_instance_ids, ground_mask=g_mask_np
        )
    _, heights_np, num_b = compute_building_vertical_anchors(tri_np, b_ids_np)

    if num_b < 2 or t_steps == 0 or n_pts == 0 or m_tris == 0:
        return {
            "status": "insufficient_buildings_for_layered_sufficiency",
            "num_buildings": int(num_b),
            "layered_occlusion_regime_transition": None,
            "nonadditive_unseen_edit_combinations": None,
        }

    op_full = (
        np.asarray(opaque_mask, dtype=bool)
        if opaque_mask is not None
        else np.ones((m_tris,), dtype=bool)
    )

    # Full-scene exact trajectory truth & per-building individual occlusion distances `(B, T, N)`
    full_bvh = ExactSceneBVH(tri_np, opaque_mask=op_full, source_triangle_ids=source_triangle_ids)
    y_true_base = full_bvh.query_trajectory_sunlit(
        rec_np,
        sun_np,
        t_min=DEFAULT_T_MIN_M,
        t_max=float(t_max_m),
        exclude_source_triangle_ids=exclude_source_triangle_ids,
    ).astype(np.float64)

    active_bids = [b for b in range(num_b) if heights_np[b] > 0.5 and np.any((b_ids_np == b) & op_full)]
    if len(active_bids) < 2:
        return {
            "status": "insufficient_active_buildings",
            "num_buildings": int(num_b),
            "layered_occlusion_regime_transition": None,
            "nonadditive_unseen_edit_combinations": None,
        }

    # Cap to top-16 tallest/nearest buildings on huge urban tiles so exact per-building BVH discovery stays <0.3s
    if len(active_bids) > 16:
        rec_c = rec_np[:, [0, 2]].mean(axis=0)
        scored_b = []
        for b in active_bids:
            sub_xz = tri_np[b_ids_np == b][:, :, [0, 2]]
            d_min = float(np.min(np.linalg.norm(sub_xz - rec_c[None, None, :], axis=-1)))
            scored_b.append((d_min / max(float(heights_np[b]), 1.0), b))
        scored_b.sort()
        active_bids = [b for _, b in scored_b[:16]]

    s_norm = np.linalg.norm(sun_np, axis=-1)
    above_h = (s_norm > 1e-12) & ((sun_np[:, 1] / np.maximum(s_norm, 1e-12)) > 0.0)
    o_grid = np.broadcast_to(rec_np[None, :, :], (t_steps, n_pts, 3)).reshape(t_steps * n_pts, 3)
    d_grid = np.broadcast_to(sun_np[:, None, :], (t_steps, n_pts, 3)).reshape(t_steps * n_pts, 3)
    excl_grid = (
        np.broadcast_to(np.asarray(exclude_source_triangle_ids, dtype=np.uint64)[None, :], (t_steps, n_pts)).reshape(-1)
        if exclude_source_triangle_ids is not None
        else None
    )

    b_hit_tn: Dict[int, np.ndarray] = {}
    b_dist_tn: Dict[int, np.ndarray] = {}
    for b in active_bids:
        sel = (b_ids_np == b) & op_full
        sids_b = source_triangle_ids[sel] if source_triangle_ids is not None else None
        bvh_b = ExactSceneBVH(tri_np[sel], source_triangle_ids=sids_b)
        occ_b, t_b, _, _ = bvh_b.query_first_hit(
            o_grid,
            d_grid,
            t_min=DEFAULT_T_MIN_M,
            t_max=float(t_max_m),
            exclude_source_triangle_ids=excl_grid,
        )
        occ_tn = occ_b.reshape(t_steps, n_pts) & above_h[:, None] & val_np
        b_hit_tn[b] = occ_tn
        b_dist_tn[b] = np.where(occ_tn, t_b.reshape(t_steps, n_pts), np.inf)

    # Rank ordered pairs `(A_front, B_rear)` by genuine physical layered-overlap ray count
    pair_scores: List[Tuple[int, int, int, int]] = []
    for a in active_bids:
        for b in active_bids:
            if a == b:
                continue
            overlap_tn = b_hit_tn[a] & b_hit_tn[b] & (b_dist_tn[a] < b_dist_tn[b])
            ov_cnt = int(np.sum(overlap_tn))
            if ov_cnt > 0:
                # Count subsequent frames on overlapping receivers where A clears (`~hit_A`) and B hits (`hit_B`)
                rec_with_ov = np.any(overlap_tn, axis=0)
                reveal_cnt = int(np.sum((~b_hit_tn[a][:, rec_with_ov]) & b_hit_tn[b][:, rec_with_ov] & above_h[:, None]))
                pair_scores.append((ov_cnt + 2 * reveal_cnt, ov_cnt, a, b))
    pair_scores.sort(reverse=True)

    selected_pairs = [(a, b) for _, _, a, b in pair_scores[: max(1, int(max_building_pairs))]]

    # Determine each receiver's anchored first-occluder building `A*(r)` at its initial layered anchor `t_0`
    # (retains ONLY the dominant occluder that explains current output `y_{t_0}`, leaving rear `B` unconstrained)
    dom_b_per_rec = np.full((n_pts,), -1, dtype=np.int64)
    for ni in range(n_pts):
        chosen_dom = -1
        for a, b in selected_pairs:
            ov_col = b_hit_tn[a][:, ni] & b_hit_tn[b][:, ni] & (b_dist_tn[a][:, ni] < b_dist_tn[b][:, ni])
            if np.any(ov_col):
                chosen_dom = int(a)
                break
        if chosen_dom < 0:
            # Fallback to earliest/closest occluding building across trajectory
            best_d = np.inf
            for b in active_bids:
                d_min_b = float(np.min(b_dist_tn[b][:, ni]))
                if math.isfinite(d_min_b) and d_min_b < best_d:
                    best_d = d_min_b
                    chosen_dom = int(b)
        dom_b_per_rec[ni] = chosen_dom

    # Model forward predictions on base scene: Full RIND (`K=max_candidates`) vs Greedy First-Occluder (`A*(r)` only)
    decision_threshold = float(getattr(rind_model, "calibrated_threshold", 0.5))
    probs_base_full = _reconstruct_and_predict_scene_trajectory(
        rec_np,
        sun_np,
        tri_np,
        rind_model=rind_model, reference_trajectory=reference_trajectory,
        opaque_mask=op_full,
        source_triangle_ids=source_triangle_ids,
        exclude_source_triangle_ids=exclude_source_triangle_ids,
        same_day_adjacent_mask=same_day_adjacent_mask,
        first_k_candidates_only=None,
        t_max_m=float(t_max_m),
        device=device,
    )
    def bottleneck_prediction(triangles, oracle):
        return _reconstruct_and_predict_scene_trajectory(
            rec_np, sun_np, triangles, rind_model=rind_model, reference_trajectory=reference_trajectory, opaque_mask=op_full,
            source_triangle_ids=source_triangle_ids, exclude_source_triangle_ids=exclude_source_triangle_ids,
            same_day_adjacent_mask=same_day_adjacent_mask, t_max_m=float(t_max_m), device=device,
            first_k_candidates_only=None if oracle else 1,
            building_ids_for_bottleneck=b_ids_np if oracle else None,
            dominant_building_per_receiver=dom_b_per_rec if oracle else None,
        )

    ordinary = "ranked_candidate_k1"
    privileged = "oracle_first_building"
    predictions = {"full_rind": probs_base_full, ordinary: bottleneck_prediction(tri_np, False),
                   privileged: bottleneck_prediction(tri_np, True)}

    # =========================================================================
    # Part 1: Layered Occlusion Regime-Transition Gap under Driver Shift
    # =========================================================================
    layered_t0_mask = np.zeros((t_steps, n_pts), dtype=bool)
    post_shift_mask = np.zeros((t_steps, n_pts), dtype=bool)
    persistence_pred = np.zeros((t_steps, n_pts), dtype=np.float64)
    time_errors = {name: [] for name in predictions}

    missing_events = {name: 0 for name in predictions}
    timing_support = 0
    eligible = val_np & above_h[:, None]
    adjacent = np.ones(t_steps - 1, dtype=bool) if same_day_adjacent_mask is None else np.asarray(same_day_adjacent_mask, dtype=bool).reshape(t_steps - 1)
    pair_mask = np.zeros((t_steps - 1, n_pts), dtype=bool)
    for a, b in selected_pairs:
        overlap = eligible & b_hit_tn[a] & b_hit_tn[b] & (b_dist_tn[a] < b_dist_tn[b])
        layered_t0_mask |= overlap
        for r in np.flatnonzero(overlap.any(axis=0)):
            for anchor, end in _layered_segments(overlap[:, r], eligible[:, r], adjacent):
                cleared = np.flatnonzero(~b_hit_tn[a][anchor+1:end, r]) + anchor + 1
                if not len(cleared):
                    continue
                pair_mask[anchor:end-1, r] = True
                post_shift_mask[cleared, r] = True
                persistence_pred[cleared, r] = float(probs_base_full[anchor, r] >= decision_threshold)
                truth_states = y_true_base[anchor:end, r] >= .5
                target_events = np.flatnonzero(~truth_states[:-1] & truth_states[1:])
                if not len(target_events):
                    continue
                timing_support += 1
                for name, predicted in predictions.items():
                    states = predicted[anchor:end, r] >= decision_threshold
                    events = np.flatnonzero(~states[:-1] & states[1:])
                    if len(events):
                        time_errors[name].append(abs(int(events[0]) - int(target_events[0])))
                    else:
                        missing_events[name] += 1

    true_bin_base = y_true_base >= 0.5
    layered_summary = {
        "layered_anchor_ray_count": int(layered_t0_mask.sum()),
        "post_driver_shift_ray_count": int(post_shift_mask.sum()),
        "evaluated_building_pairs": [[int(a), int(b)] for a, b in selected_pairs],
        "event_pair_count": int(pair_mask.sum()),
        "true_event_count": int(np.count_nonzero((true_bin_base[1:] != true_bin_base[:-1]) & pair_mask)),
        "transition_timing_true_events": timing_support,
        "post_driver_shift_accuracy_persistence": float(np.mean(
            (persistence_pred[post_shift_mask] >= 0.5) == true_bin_base[post_shift_mask]
        )) if post_shift_mask.any() else None,
    }
    for name, prob in predictions.items():
        pred = prob >= decision_threshold
        for label, mask in (("at_t0", layered_t0_mask), ("post_driver_shift", post_shift_mask)):
            layered_summary[f"{label}_accuracy_{name}"] = float(np.mean(pred[mask] == true_bin_base[mask])) if mask.any() else None
        layered_summary[f"predicted_event_count_{name}"] = int(np.count_nonzero((pred[1:] != pred[:-1]) & pair_mask))
        layered_summary[f"transition_timing_matched_{name}"] = len(time_errors[name])
        layered_summary[f"transition_timing_missed_{name}"] = missing_events[name]
        layered_summary[f"regime_transition_f1_{name}"] = _binary_f1_np(
            (pred[1:] != pred[:-1])[pair_mask], (true_bin_base[1:] != true_bin_base[:-1])[pair_mask]
        ) if pair_mask.any() else None
        layered_summary[f"transition_time_mae_frames_{name}"] = float(np.mean(time_errors[name])) if time_errors[name] else None
    layered_summary["sufficiency_gap_post_shift_acc_vs_ranked_candidate_k1"] = (
        layered_summary["post_driver_shift_accuracy_full_rind"] - layered_summary[f"post_driver_shift_accuracy_{ordinary}"]
        if post_shift_mask.any() else None
    )

    # =========================================================================
    # Part 2: Nonadditive Structural Edits & Unseen Edit Combinations e_12 = e_1 (+) e_2
    # =========================================================================
    if not selected_pairs:
        selected_pairs = [(active_bids[0], active_bids[1])]

    drop_frac = float(np.clip(edit_drop_fraction, 0.2, 0.98))
    pair_edit_records: List[Dict[str, Any]] = []
    all_I_true: List[np.ndarray] = []
    all_I_rind: List[np.ndarray] = []
    all_dv12_true: List[np.ndarray] = []
    all_dv12_rind: List[np.ndarray] = []
    all_dv12_add: List[np.ndarray] = []
    all_dv12_oracle: List[np.ndarray] = []
    all_dv12_k1: List[np.ndarray] = []
    ov_I_true: List[np.ndarray] = []
    ov_I_rind: List[np.ndarray] = []

    for a, b in selected_pairs:
        dh_e1 = np.zeros((num_b,), dtype=np.float64)
        dh_e1[a] = -drop_frac * float(heights_np[a])
        dh_e2 = np.zeros((num_b,), dtype=np.float64)
        dh_e2[b] = -drop_frac * float(heights_np[b])
        dh_e12 = dh_e1 + dh_e2

        tris_e1 = perturb_buildings_consistent_f64(tri_np, b_ids_np, dh_e1)
        tris_e2 = perturb_buildings_consistent_f64(tri_np, b_ids_np, dh_e2)
        tris_e12 = perturb_buildings_consistent_f64(tri_np, b_ids_np, dh_e12)

        v_star_1 = ExactSceneBVH(tris_e1, opaque_mask=op_full, source_triangle_ids=source_triangle_ids).query_trajectory_sunlit(
            rec_np, sun_np, t_min=DEFAULT_T_MIN_M, t_max=float(t_max_m), exclude_source_triangle_ids=exclude_source_triangle_ids
        ).astype(np.float64)
        v_star_2 = ExactSceneBVH(tris_e2, opaque_mask=op_full, source_triangle_ids=source_triangle_ids).query_trajectory_sunlit(
            rec_np, sun_np, t_min=DEFAULT_T_MIN_M, t_max=float(t_max_m), exclude_source_triangle_ids=exclude_source_triangle_ids
        ).astype(np.float64)
        v_star_12 = ExactSceneBVH(tris_e12, opaque_mask=op_full, source_triangle_ids=source_triangle_ids).query_trajectory_sunlit(
            rec_np, sun_np, t_min=DEFAULT_T_MIN_M, t_max=float(t_max_m), exclude_source_triangle_ids=exclude_source_triangle_ids
        ).astype(np.float64)

        p_rind_1 = _reconstruct_and_predict_scene_trajectory(
            rec_np, sun_np, tris_e1, rind_model=rind_model, reference_trajectory=reference_trajectory, opaque_mask=op_full,
            source_triangle_ids=source_triangle_ids, exclude_source_triangle_ids=exclude_source_triangle_ids,
            same_day_adjacent_mask=same_day_adjacent_mask, first_k_candidates_only=None, t_max_m=float(t_max_m), device=device
        )
        p_rind_2 = _reconstruct_and_predict_scene_trajectory(
            rec_np, sun_np, tris_e2, rind_model=rind_model, reference_trajectory=reference_trajectory, opaque_mask=op_full,
            source_triangle_ids=source_triangle_ids, exclude_source_triangle_ids=exclude_source_triangle_ids,
            same_day_adjacent_mask=same_day_adjacent_mask, first_k_candidates_only=None, t_max_m=float(t_max_m), device=device
        )
        p_rind_12 = _reconstruct_and_predict_scene_trajectory(
            rec_np, sun_np, tris_e12, rind_model=rind_model, reference_trajectory=reference_trajectory, opaque_mask=op_full,
            source_triangle_ids=source_triangle_ids, exclude_source_triangle_ids=exclude_source_triangle_ids,
            same_day_adjacent_mask=same_day_adjacent_mask, first_k_candidates_only=None, t_max_m=float(t_max_m), device=device
        )
        p_k1_12 = bottleneck_prediction(tris_e12, False)
        p_oracle_12 = bottleneck_prediction(tris_e12, True)

        dv_star_1 = v_star_1 - y_true_base
        dv_star_2 = v_star_2 - y_true_base
        dv_star_12 = v_star_12 - y_true_base
        inter_star = dv_star_12 - (dv_star_1 + dv_star_2)

        dv_rind_1 = p_rind_1 - probs_base_full
        dv_rind_2 = p_rind_2 - probs_base_full
        dv_rind_12 = p_rind_12 - probs_base_full
        dv_add_12 = dv_rind_1 + dv_rind_2
        dv_k1_12 = p_k1_12 - predictions[ordinary]
        inter_rind = dv_rind_12 - dv_add_12

        eval_mask = val_np & above_h[:, None]
        ov_ab = b_hit_tn[a] & b_hit_tn[b] & eval_mask
        all_I_true.append(inter_star[eval_mask])
        all_I_rind.append(inter_rind[eval_mask])
        all_dv12_true.append(dv_star_12[eval_mask])
        all_dv12_rind.append(dv_rind_12[eval_mask])
        all_dv12_add.append(dv_add_12[eval_mask])
        all_dv12_oracle.append((p_oracle_12 - predictions[privileged])[eval_mask])
        all_dv12_k1.append(dv_k1_12[eval_mask])
        if np.any(ov_ab):
            ov_I_true.append(inter_star[ov_ab])
            ov_I_rind.append(inter_rind[ov_ab])

        pair_edit_records.append(
            {
                "building_pair": [int(a), int(b)],
                "delta_h_e1_m": float(dh_e1[a]),
                "delta_h_e2_m": float(dh_e2[b]),
                "layered_overlap_rays": int(np.sum(ov_ab)),
                "true_mean_interaction_on_overlap": float(np.mean(inter_star[ov_ab])) if np.any(ov_ab) else 0.0,
                "rind_mean_interaction_on_overlap": float(np.mean(inter_rind[ov_ab])) if np.any(ov_ab) else 0.0,
            }
        )

    I_true_vec = np.concatenate(all_I_true, axis=0) if all_I_true else np.zeros((0,), dtype=np.float64)
    I_rind_vec = np.concatenate(all_I_rind, axis=0) if all_I_rind else np.zeros((0,), dtype=np.float64)
    dv12_true_vec = np.concatenate(all_dv12_true, axis=0) if all_dv12_true else np.zeros((0,), dtype=np.float64)
    dv12_rind_vec = np.concatenate(all_dv12_rind, axis=0) if all_dv12_rind else np.zeros((0,), dtype=np.float64)
    dv12_add_vec = np.concatenate(all_dv12_add, axis=0) if all_dv12_add else np.zeros((0,), dtype=np.float64)
    dv12_k1_vec = np.concatenate(all_dv12_k1, axis=0) if all_dv12_k1 else np.zeros((0,), dtype=np.float64)
    ov_I_true_vec = np.concatenate(ov_I_true, axis=0) if ov_I_true else np.zeros((0,), dtype=np.float64)
    ov_I_rind_vec = np.concatenate(ov_I_rind, axis=0) if ov_I_rind else np.zeros((0,), dtype=np.float64)

    true_changed = np.abs(dv12_true_vec) >= 0.5
    rind_changed = np.abs(dv12_rind_vec) >= 0.5
    add_changed = np.abs(dv12_add_vec) >= 0.5
    oracle_delta = np.concatenate(all_dv12_oracle) if all_dv12_oracle else np.zeros(0)
    k1_changed = np.abs(dv12_k1_vec) >= 0.5

    nonzero_I = np.abs(I_true_vec) >= 0.5
    sign_agree = (
        float(np.mean(np.sign(I_rind_vec[nonzero_I]) == np.sign(I_true_vec[nonzero_I])))
        if np.any(nonzero_I)
        else None
    )

    nonadditive_summary = {
        "evaluated_pair_count": len(pair_edit_records),
        "total_evaluated_point_frames": int(I_true_vec.size),
        "nonzero_nonadditive_interaction_rays": int(np.sum(nonzero_I)),
        "mean_abs_true_nonadditive_residual": float(np.mean(np.abs(I_true_vec))) if I_true_vec.size > 0 else 0.0,
        "layered_overlap_mean_true_interaction": float(np.mean(ov_I_true_vec)) if ov_I_true_vec.size > 0 else 0.0,
        "layered_overlap_mean_rind_interaction": float(np.mean(ov_I_rind_vec)) if ov_I_rind_vec.size > 0 else 0.0,
        "layered_overlap_interaction_mae_full_rind": (
            float(np.mean(np.abs(ov_I_rind_vec - ov_I_true_vec))) if ov_I_true_vec.size > 0 else None
        ),
        "layered_overlap_interaction_mae_linear_additive": (
            float(np.mean(np.abs(0.0 - ov_I_true_vec))) if ov_I_true_vec.size > 0 else None
        ),
        "interaction_residual_mae_full_rind": float(np.mean(np.abs(I_rind_vec - I_true_vec))) if I_true_vec.size > 0 else None,
        "interaction_residual_mae_linear_additive": float(np.mean(np.abs(0.0 - I_true_vec))) if I_true_vec.size > 0 else None,
        "interaction_sign_agreement_full_rind": sign_agree,
        "combined_edit_change_f1_full_rind": _binary_f1_np(rind_changed, true_changed) if dv12_true_vec.size > 0 else None,
        "combined_edit_change_f1_linear_additive": _binary_f1_np(add_changed, true_changed) if dv12_true_vec.size > 0 else None,
        "combined_edit_change_f1_oracle_first_building": _binary_f1_np(np.abs(oracle_delta) >= 0.5, true_changed) if true_changed.size else None,
        "combined_edit_mae_oracle_first_building": float(np.mean(np.abs(oracle_delta - dv12_true_vec))) if true_changed.size else None,
        "combined_edit_change_f1_ranked_candidate_k1": _binary_f1_np(k1_changed, true_changed) if dv12_true_vec.size > 0 else None,
        "combined_edit_mae_full_rind": float(np.mean(np.abs(dv12_rind_vec - dv12_true_vec))) if dv12_true_vec.size > 0 else None,
        "combined_edit_mae_linear_additive": float(np.mean(np.abs(dv12_add_vec - dv12_true_vec))) if dv12_true_vec.size > 0 else None,
        "combined_edit_mae_ranked_candidate_k1": float(np.mean(np.abs(dv12_k1_vec - dv12_true_vec))) if dv12_true_vec.size > 0 else None,
        "pair_details": pair_edit_records,
    }

    return {
        "status": "ok",
        "anti_leakage_reconstructed_inputs": True,
        "model_source": "rind_checkpoint" if rind_model is not None else "differentiable_geometric_surrogate",
        "num_receivers": n_pts,
        "num_frames": t_steps,
        "num_buildings": int(num_b),
        "baseline_schema_version": 2,
        "ranked_candidate_k1_scope": "one ranked candidate triangle per ray; neural local16 and crop use the full scene",
        "oracle_first_building_scope": "privileged whole building plus background; neural kNN receiver groups partitioned by selected building",

        "layered_occlusion_regime_transition": layered_summary,
        "nonadditive_unseen_edit_combinations": nonadditive_summary,
    }

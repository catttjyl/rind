#!/usr/bin/env python3
"""Build the shared geometry inputs H1, H2 (height-map pyramid) and P (LOD point cloud) per tile.

Inputs come only from a tile shard (no labels are read):
  `receivers_xyz`, `receiver_rc`, `static_fields_8hw` (H0 grid geometry and height datum), and the
  4 km scene mesh `scene_triangles_xyz` / `scene_opaque` / `scene_source_ids` (only `.tar.gz`
  shards carry the mesh). An optional per-triangle `scene_ground_mask` (True = ground / non-building)
  is used when present; otherwise ground is estimated from the mesh itself (see below).

Outputs `geom_{tile_id}.npz` with
  H1, H2      : (3, 512, 512) float32, channels
                0 max surface height above `ref_y` [m]
                1 max building height above estimated terrain [m]
                2 coverage fraction of the cell's horizontal area by mesh (0..1)
  P_xyz       : (M, 3) float32, world-axis offsets from the tile center (x, y - ref_y, z) [m]
  P_normal    : (M, 3) float32 unit normals (near-horizontal faces oriented upward)
  P_is_roof   : (M,) bool, normal within 30 deg of vertical
  P_opaque    : (M,) bool (always True: only opaque faces are sampled)
  P_source_ids: (M,) uint64 source triangle id of each point
  meta_json   : grid geometry, parameters, and which ground source was used

Grid geometry: H0 pixel centers are recovered from `receiver_rc -> receivers_xz` by least squares
(the shard has no world axes). H1/H2 share H0's center and row/column directions, so H0 is the
central 64 x 64 cells of H1 and H1 the central 51.2 x 51.2 cells of H2. Heights are relative to
H0's datum `ref_y` (median valid terrain), recovered as `y_receiver - 100 * field` at receivers.

Per-cell values are rasterized at `supersample x supersample` sub-cells: cell max height is the max
over sub-cell samples, triangle vertices, edge midpoints and centroids (so tall thin structures are
not averaged away); coverage is the fraction of sub-cells inside any non-vertical triangle.

Terrain (for channel 1 and for excluding ground from P): with `scene_ground_mask`, the per-cell
minimum ground height; without it, a grey-scale opening (min filter then max filter, window
`--dtm-window-m`) of the per-cell minimum surface height, the usual DSM -> DTM approach. Cells
still without terrain are filled from neighbors. Without a ground mask, ground faces for P are
near-horizontal faces whose centroid lies within `--ground-tol-m` of the estimated terrain.

Normalization (e.g. -20..200 m -> [0, 1]) is NOT applied here; all models apply the same global
constants at load time.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

KIT_ROOT = Path(__file__).resolve().parents[1]

H_SIZE = 512
H0_CELL_M = 100.0 / 512.0          # 0.1953125 m
LEVELS: Dict[str, float] = {"H1": 1.5625, "H2": 15.625}
P_S0_M = 0.5
P_DTHETA_DEG = 0.5
P_HORIZON_M = 4000.0
ROOF_COS = float(np.cos(np.radians(30.0)))
FLAT_COS = float(np.cos(np.radians(10.0)))

GEOM_KEYS = ("receivers_xyz", "receiver_rc", "static_fields_8hw",
             "scene_triangles_xyz", "scene_opaque", "scene_source_ids", "scene_ground_mask")


# --------------------------------------------------------------------------------------------
# Loading (numpy only; never reads labels)
# --------------------------------------------------------------------------------------------
def load_geometry_arrays(data_dir: Path, tile_id: str, keys: Sequence[str] = GEOM_KEYS) -> Dict[str, np.ndarray]:
    npz_path = data_dir / f"{tile_id}.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as z:
            return {k: z[k] for k in keys if k in z.files}
    for d in (data_dir / tile_id, *data_dir.glob(f"*/{tile_id}")):
        if d.is_dir():
            return {k: np.load(d / f"{k}.npy", allow_pickle=False) for k in keys if (d / f"{k}.npy").exists()}
    tar_path = data_dir / f"{tile_id}.tar.gz"
    if tar_path.exists():
        want = {f"{k}.npy": k for k in keys}
        out: Dict[str, np.ndarray] = {}
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf.getmembers():
                name = Path(m.name).name
                if name in want:
                    f = tf.extractfile(m)
                    if f is not None:
                        out[want[name]] = np.load(io.BytesIO(f.read()), allow_pickle=False)
        return out
    raise FileNotFoundError(f"Tile shard for {tile_id} not found under {data_dir}")


# --------------------------------------------------------------------------------------------
# H0 grid geometry and datum
# --------------------------------------------------------------------------------------------
def fit_h0_grid(receivers_xyz: np.ndarray, receiver_rc: np.ndarray) -> Dict[str, Any]:
    """Affine `xz = origin + row * a_row + col * a_col` from receivers at H0 pixel centers."""
    rc = np.asarray(receiver_rc, dtype=np.float64)
    xz = np.asarray(receivers_xyz, dtype=np.float64)[:, [0, 2]]
    design = np.column_stack([rc[:, 0], rc[:, 1], np.ones(len(rc))])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("receivers do not span two grid directions; cannot recover H0 geometry")
    coef, *_ = np.linalg.lstsq(design, xz, rcond=None)
    a_row, a_col, origin = coef[0], coef[1], coef[2]
    resid = float(np.abs(design @ coef - xz).max())
    s_row, s_col = float(np.linalg.norm(a_row)), float(np.linalg.norm(a_col))
    ortho = abs(float(np.dot(a_row, a_col))) / (s_row * s_col)
    if resid > 0.25 * H0_CELL_M or ortho > 1e-3 or abs(s_row - H0_CELL_M) > 1e-3 or abs(s_col - H0_CELL_M) > 1e-3:
        raise ValueError(
            f"receivers are not on a {H0_CELL_M:.7f} m orthogonal pixel-center grid "
            f"(max residual {resid:.4f} m, cell {s_row:.5f}/{s_col:.5f} m, |cos| {ortho:.2e})")
    center = origin + (H_SIZE / 2 - 0.5) * (a_row + a_col)
    return {"center_xz": center, "u_row": a_row / s_row, "u_col": a_col / s_col,
            "fit_max_residual_m": resid}


def recover_ref_y(receivers_xyz: np.ndarray, receiver_rc: np.ndarray, fields_8hw: np.ndarray) -> Dict[str, float]:
    """H0 heights are `(h - ref) / 100`; recover `ref` from receivers lying on the H0 surface."""
    r, c = np.asarray(receiver_rc, dtype=np.int64).T
    f = np.asarray(fields_8hw, dtype=np.float64)
    top_ok, ter_ok = f[4, r, c] > 0.5, f[3, r, c] > 0.5
    surf = np.where(top_ok, f[1, r, c], f[0, r, c])
    ok = top_ok | ter_ok
    if not np.any(ok):
        raise ValueError("no receiver lies on valid H0 geometry; cannot recover ref_y")
    est = np.asarray(receivers_xyz, dtype=np.float64)[ok, 1] - 100.0 * surf[ok]
    return {"ref_y": float(np.median(est)), "ref_y_spread_p5_p95_m": float(np.percentile(est, 95) - np.percentile(est, 5))}


# --------------------------------------------------------------------------------------------
# Rasterization
# --------------------------------------------------------------------------------------------
def _project(points_xz: np.ndarray, grid: Dict[str, Any], sub_cell_m: float, n_sub: int) -> np.ndarray:
    d = points_xz - grid["center_xz"]
    return np.stack([d @ grid["u_row"], d @ grid["u_col"]], axis=-1) / sub_cell_m + n_sub / 2.0


def rasterize_heights(
    tris_xyz: np.ndarray, grid: Dict[str, Any], cell_m: float, supersample: int, max_candidates: int = 4_000_000,
) -> Dict[str, np.ndarray]:
    """Sub-cell max / min surface height and hit mask of non-vertical triangles."""
    n_sub = H_SIZE * supersample
    sub_m = cell_m / supersample
    hmax = np.full((n_sub, n_sub), -np.inf, dtype=np.float64)
    hmin = np.full((n_sub, n_sub), np.inf, dtype=np.float64)
    hit = np.zeros((n_sub, n_sub), dtype=bool)
    if len(tris_xyz) == 0:
        return {"max": hmax, "min": hmin, "hit": hit}
    t = np.asarray(tris_xyz, dtype=np.float64)
    uv = _project(t[:, :, [0, 2]], grid, sub_m, n_sub)            # (M, 3, 2) continuous sub-cell coords
    y = t[:, :, 1]
    lo, hi = uv.min(axis=1), uv.max(axis=1)
    keep = (hi[:, 0] >= 0) & (lo[:, 0] < n_sub) & (hi[:, 1] >= 0) & (lo[:, 1] < n_sub)
    uv, y, lo, hi = uv[keep], y[keep], lo[keep], hi[keep]

    # 1. Point samples: vertices, edge midpoints, centroids (catch features smaller than a sub-cell).
    pts = np.concatenate([uv, 0.5 * (uv + uv[:, [1, 2, 0]]), uv.mean(axis=1, keepdims=True)], axis=1)
    py = np.concatenate([y, 0.5 * (y + y[:, [1, 2, 0]]), y.mean(axis=1, keepdims=True)], axis=1)
    pi, pj = np.floor(pts[..., 0]).astype(np.int64).ravel(), np.floor(pts[..., 1]).astype(np.int64).ravel()
    ok = (pi >= 0) & (pi < n_sub) & (pj >= 0) & (pj < n_sub)
    np.maximum.at(hmax, (pi[ok], pj[ok]), py.ravel()[ok])
    np.minimum.at(hmin, (pi[ok], pj[ok]), py.ravel()[ok])

    # 2. Sub-cell centers inside each non-vertical triangle (barycentric test).
    e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    flat = np.abs(det) > 1e-12
    i0 = np.clip(np.ceil(lo[:, 0] - 0.5), 0, n_sub - 1).astype(np.int64)
    i1 = np.clip(np.floor(hi[:, 0] - 0.5), 0, n_sub - 1).astype(np.int64)
    j0 = np.clip(np.ceil(lo[:, 1] - 0.5), 0, n_sub - 1).astype(np.int64)
    j1 = np.clip(np.floor(hi[:, 1] - 0.5), 0, n_sub - 1).astype(np.int64)
    ni, nj = i1 - i0 + 1, j1 - j0 + 1
    sel = np.flatnonzero(flat & (ni > 0) & (nj > 0) & (lo[:, 0] - 0.5 <= i1) & (lo[:, 1] - 0.5 <= j1))
    counts = (ni[sel] * nj[sel]).astype(np.int64)
    order = np.argsort(counts, kind="stable")
    sel, counts = sel[order], counts[order]
    csum = np.cumsum(counts)
    start = 0
    while start < len(sel):
        base = csum[start - 1] if start else 0
        stop = max(start + 1, int(np.searchsorted(csum, base + max_candidates, side="right")))
        idx, cnt = sel[start:stop], counts[start:stop]
        rep = np.repeat(np.arange(len(idx)), cnt)
        local = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        tri = idx[rep]
        ci = i0[tri] + local // nj[tri]
        cj = j0[tri] + local % nj[tri]
        px, pz = ci + 0.5 - uv[tri, 0, 0], cj + 0.5 - uv[tri, 0, 1]
        inv = 1.0 / det[tri]
        w1 = (px * e2[tri, 1] - pz * e2[tri, 0]) * inv
        w2 = (e1[tri, 0] * pz - e1[tri, 1] * px) * inv
        inside = (w1 >= -1e-9) & (w2 >= -1e-9) & (w1 + w2 <= 1.0 + 1e-9)
        tri, ci, cj, w1, w2 = tri[inside], ci[inside], cj[inside], w1[inside], w2[inside]
        yy = y[tri, 0] + w1 * (y[tri, 1] - y[tri, 0]) + w2 * (y[tri, 2] - y[tri, 0])
        np.maximum.at(hmax, (ci, cj), yy)
        np.minimum.at(hmin, (ci, cj), yy)
        hit[ci, cj] = True
        start = stop
    return {"max": hmax, "min": hmin, "hit": hit}


def _pool(a: np.ndarray, k: int, how: str) -> np.ndarray:
    b = a.reshape(H_SIZE, k, H_SIZE, k)
    return {"max": b.max(axis=(1, 3)), "min": b.min(axis=(1, 3)), "mean": b.mean(axis=(1, 3))}[how]


def _filter2d(a: np.ndarray, w: int, how: str) -> np.ndarray:
    """Separable square min/max filter, same size; pads with the neutral element."""
    if w <= 1:
        return a.copy()
    fill = np.inf if how == "min" else -np.inf
    red = np.min if how == "min" else np.max
    lo, hi = (w - 1) // 2, w // 2
    out = a
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (lo, hi)
        out = red(sliding_window_view(np.pad(out, pad, constant_values=fill), w, axis=axis), axis=-1)
    return out


def _fill_missing(a: np.ndarray, max_iter: int = 2048) -> np.ndarray:
    """Fill non-finite cells with the mean of finite 4-neighbors, iteratively."""
    out = a.copy()
    for _ in range(max_iter):
        bad = ~np.isfinite(out)
        if not bad.any() or bad.all():
            break
        p = np.pad(out, 1, constant_values=np.nan)
        nb = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        fin = np.isfinite(nb)
        s, n = np.where(fin, nb, 0.0).sum(axis=0), fin.sum(axis=0)
        upd = bad & (n > 0)
        out[upd] = s[upd] / n[upd]
    out[~np.isfinite(out)] = np.nan
    return out


def build_level(
    tris: np.ndarray, ground: Optional[np.ndarray], grid: Dict[str, Any], ref_y: float, cell_m: float,
    *, supersample: int, dtm_window_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the (3, 512, 512) level and its terrain (absolute y) for ground classification."""
    surf = rasterize_heights(tris, grid, cell_m, supersample)
    smax = _pool(surf["max"], supersample, "max")
    cover = _pool(surf["hit"].astype(np.float64), supersample, "mean")
    if ground is not None and ground.any():
        gr = rasterize_heights(tris[ground], grid, cell_m, supersample)
        terrain = _pool(gr["min"], supersample, "min")
        terrain[~np.isfinite(terrain)] = np.nan
    else:
        smin = _pool(surf["min"], supersample, "min")
        w = max(1, int(round(dtm_window_m / cell_m)))
        eroded = _filter2d(smin, w, "min")
        terrain = _filter2d(np.where(np.isfinite(eroded), eroded, -np.inf), w, "max")
        terrain[~np.isfinite(terrain)] = np.nan
    terrain = _fill_missing(terrain)
    terrain = np.where(np.isfinite(terrain), terrain, ref_y)  # no ground anywhere: fall back to the datum
    have = np.isfinite(smax)
    r = np.where(have, smax - ref_y, 0.0)
    g = np.where(have, np.maximum(smax - terrain, 0.0), 0.0)
    return np.stack([r, g, cover]).astype(np.float32), terrain


# --------------------------------------------------------------------------------------------
# LOD point cloud
# --------------------------------------------------------------------------------------------
def _sample_terrain(xz: np.ndarray, grid: Dict[str, Any], levels: Sequence[Tuple[float, np.ndarray]]) -> np.ndarray:
    """Nearest-cell terrain from the finest level that contains each point."""
    out = np.full(len(xz), np.nan)
    for cell_m, terr in levels:
        ij = np.floor(_project(xz, grid, cell_m, H_SIZE)).astype(np.int64)
        ok = np.isnan(out) & (ij[:, 0] >= 0) & (ij[:, 0] < H_SIZE) & (ij[:, 1] >= 0) & (ij[:, 1] < H_SIZE)
        out[ok] = terr[ij[ok, 0], ij[ok, 1]]
    return out


def sample_lod_points(
    tris: np.ndarray, opaque: np.ndarray, source_ids: np.ndarray, ground: np.ndarray,
    grid: Dict[str, Any], ref_y: float, rng: np.random.Generator,
    *, s0_m: float = P_S0_M, dtheta_deg: float = P_DTHETA_DEG, horizon_m: float = P_HORIZON_M,
) -> Dict[str, np.ndarray]:
    """Area-weighted sampling on opaque non-ground faces with spacing `max(s0, d * dtheta)`."""
    t = np.asarray(tris, dtype=np.float64)
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    cen = t.mean(axis=1)
    d = np.linalg.norm(cen[:, [0, 2]] - grid["center_xz"], axis=1)
    use = opaque & ~ground & (area > 0) & (d <= horizon_m)
    spacing = np.maximum(s0_m, d * np.radians(dtheta_deg))
    lam = np.where(use, area / spacing ** 2, 0.0)
    n = np.floor(lam).astype(np.int64) + (rng.random(len(lam)) < (lam - np.floor(lam)))
    tri = np.repeat(np.arange(len(t)), n)
    r1, r2 = np.sqrt(rng.random(len(tri))), rng.random(len(tri))
    a, b, c = t[tri, 0], t[tri, 1], t[tri, 2]
    pts = (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c
    unit = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    unit = np.where((np.abs(unit[:, 1]) >= 0.5)[:, None] & (unit[:, 1] < 0)[:, None], -unit, unit)
    offset = np.array([grid["center_xz"][0], ref_y, grid["center_xz"][1]])
    return {
        "P_xyz": (pts - offset).astype(np.float32),
        "P_normal": unit[tri].astype(np.float32),
        "P_is_roof": unit[tri, 1] > ROOF_COS,
        "P_opaque": np.ones(len(tri), dtype=bool),
        "P_source_ids": np.asarray(source_ids, dtype=np.uint64)[tri],
        "triangles_sampled": int((n > 0).sum()),
    }


# --------------------------------------------------------------------------------------------
# Per-tile driver
# --------------------------------------------------------------------------------------------
def build_tile(
    arrays: Dict[str, np.ndarray], tile_id: str, *, seed: int = 20261120, supersample: int = 4,
    dtm_window_m: float = 100.0, ground_tol_m: float = 1.0,
) -> Dict[str, Any]:
    missing = [k for k in ("receivers_xyz", "receiver_rc", "static_fields_8hw", "scene_triangles_xyz") if k not in arrays]
    if missing:
        raise KeyError(f"{tile_id}: missing {missing} (the scene mesh is only in `.tar.gz` shards)")
    tris = np.asarray(arrays["scene_triangles_xyz"], dtype=np.float64)
    m = len(tris)
    opaque = np.asarray(arrays.get("scene_opaque", np.ones(m, dtype=bool)), dtype=bool)
    sids = np.asarray(arrays.get("scene_source_ids", np.arange(m)), dtype=np.uint64)
    gmask = arrays.get("scene_ground_mask")
    gmask = None if gmask is None else np.asarray(gmask, dtype=bool)

    grid = fit_h0_grid(arrays["receivers_xyz"], arrays["receiver_rc"])
    datum = recover_ref_y(arrays["receivers_xyz"], arrays["receiver_rc"], arrays["static_fields_8hw"])
    ref_y = datum["ref_y"]

    out: Dict[str, Any] = {}
    terrains = []
    for name, cell_m in LEVELS.items():
        half = 0.5 * H_SIZE * cell_m * np.sqrt(2.0) + cell_m
        near = np.linalg.norm(tris[:, :, [0, 2]].mean(axis=1) - grid["center_xz"], axis=1) <= half + \
            np.linalg.norm(tris[:, :, [0, 2]] - tris[:, :, [0, 2]].mean(axis=1, keepdims=True), axis=2).max(axis=1)
        level, terr = build_level(tris[near], None if gmask is None else gmask[near], grid, ref_y, cell_m,
                                  supersample=supersample, dtm_window_m=dtm_window_m)
        out[name] = level
        terrains.append((cell_m, terr))

    if gmask is not None:
        ground = gmask
        ground_source = "scene_ground_mask"
    else:
        nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        ny = np.abs(nrm[:, 1]) / np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)
        cen = tris.mean(axis=1)
        terr_at = _sample_terrain(cen[:, [0, 2]], grid, terrains)
        ground = (ny > FLAT_COS) & np.isfinite(terr_at) & (cen[:, 1] - terr_at <= ground_tol_m)
        ground_source = f"heuristic_dtm_opening_{dtm_window_m:g}m_tol_{ground_tol_m:g}m"

    tile_seed = int.from_bytes(hashlib.sha256(f"{tile_id}::{seed}".encode()).digest()[:8], "little")
    pc = sample_lod_points(tris, opaque, sids, ground, grid, ref_y, np.random.default_rng(tile_seed))
    out.update({k: v for k, v in pc.items() if k.startswith("P_")})
    meta = {
        "schema": "rind.geometry_inputs.v1",
        "tile_id": tile_id,
        "center_xz_m": [float(v) for v in grid["center_xz"]],
        "u_row_xz": [float(v) for v in grid["u_row"]],
        "u_col_xz": [float(v) for v in grid["u_col"]],
        "h0_fit_max_residual_m": grid["fit_max_residual_m"],
        **datum,
        "levels": {k: {"cell_m": v, "extent_m": v * H_SIZE, "size": H_SIZE} for k, v in LEVELS.items()},
        "h_channels": ["max_surface_height_above_ref_m", "max_building_height_m", "coverage_fraction"],
        "supersample": supersample,
        "ground_source": ground_source,
        "ground_triangles": int(ground.sum()),
        "scene_triangles": int(m),
        "p_spacing": {"s0_m": P_S0_M, "dtheta_deg": P_DTHETA_DEG, "horizon_m": P_HORIZON_M},
        "p_points": int(len(pc["P_xyz"])),
        "p_triangles_sampled": pc["triangles_sampled"],
        "p_seed": seed,
        "p_xyz_frame": "world axes, origin (center_x, ref_y, center_z)",
    }
    out["meta_json"] = np.array(json.dumps(meta))
    return out


# --------------------------------------------------------------------------------------------
# Self-test on a synthetic scene
# --------------------------------------------------------------------------------------------
def _box(x0, z0, w, d, y0, y1):
    """Closed box without bottom: 2 roof + 8 wall triangles."""
    c = lambda x, y, z: [x, y, z]  # noqa: E731
    xa, xb, za, zb = x0, x0 + w, z0, z0 + d
    roof = [[c(xa, y1, za), c(xb, y1, za), c(xb, y1, zb)], [c(xa, y1, za), c(xb, y1, zb), c(xa, y1, zb)]]
    walls = []
    for (p, q) in (((xa, za), (xb, za)), ((xb, za), (xb, zb)), ((xb, zb), (xa, zb)), ((xa, zb), (xa, za))):
        walls += [[c(p[0], y0, p[1]), c(q[0], y0, q[1]), c(q[0], y1, q[1])],
                  [c(p[0], y0, p[1]), c(q[0], y1, q[1]), c(p[0], y1, p[1])]]
    return roof + walls


def _synthetic_tile() -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    center = np.array([1000.0, 2000.0])  # world (x, z)
    ref = 10.0
    rc = np.stack([rng.integers(0, 512, 256), rng.integers(0, 512, 256)], axis=1).astype(np.int32)
    # rows run along +z, columns along +x
    x = center[0] + (rc[:, 1] - 255.5) * H0_CELL_M
    z = center[1] + (rc[:, 0] - 255.5) * H0_CELL_M
    rec = np.stack([x, np.full(256, ref), z], axis=1).astype(np.float32)
    fields = np.zeros((8, 512, 512), dtype=np.float32)
    fields[3] = 1.0  # terrain valid, terrain == ref
    g = 4600.0
    ground = [[[center[0] - g, ref, center[1] - g], [center[0] + g, ref, center[1] - g], [center[0] + g, ref, center[1] + g]],
              [[center[0] - g, ref, center[1] - g], [center[0] + g, ref, center[1] + g], [center[0] - g, ref, center[1] + g]]]
    near = _box(center[0] + 200, center[1] - 20, 40, 40, ref, ref + 30)       # 30 m tall, 200 m east
    far = _box(center[0] - 20, center[1] + 3000, 60, 60, ref, ref + 210)     # 210 m tall, 3 km north
    tris = np.array(ground + near + far, dtype=np.float32)
    gm = np.zeros(len(tris), dtype=bool)
    gm[:2] = True
    return {"receivers_xyz": rec, "receiver_rc": rc, "static_fields_8hw": fields,
            "scene_triangles_xyz": tris, "scene_opaque": np.ones(len(tris), dtype=bool),
            "scene_source_ids": np.arange(len(tris), dtype=np.uint64), "_ground_truth_mask": gm}


def run_self_test() -> None:
    arr = _synthetic_tile()
    gm = arr.pop("_ground_truth_mask")
    for label, a in (("heuristic", dict(arr)), ("ground_mask", {**arr, "scene_ground_mask": gm})):
        out = build_tile(a, "SYNTH")
        meta = json.loads(str(out["meta_json"]))
        assert abs(meta["ref_y"] - 10.0) < 1e-4, meta["ref_y"]
        assert np.allclose(meta["center_xz_m"], [1000.0, 2000.0], atol=1e-3), meta["center_xz_m"]
        h1, h2 = out["H1"], out["H2"]

        def cell(level_m, dx, dz):  # world offset -> (row, col); rows along +z, cols along +x
            return int(np.floor(dz / level_m + 256)), int(np.floor(dx / level_m + 256))

        r, c = cell(1.5625, 220, 0)  # inside near building
        assert abs(h1[0, r, c] - 30) < 1e-4 and abs(h1[1, r, c] - 30) < 1e-4, (label, h1[:, r, c])
        r, c = cell(1.5625, -150, 100)  # open ground
        assert abs(h1[0, r, c]) < 1e-4 and abs(h1[1, r, c]) < 1e-4 and abs(h1[2, r, c] - 1) < 1e-9, (label, h1[:, r, c])
        r, c = cell(15.625, 10, 3030)  # inside far building
        assert abs(h2[0, r, c] - 210) < 1e-4 and abs(h2[1, r, c] - 210) < 1e-4, (label, h2[:, r, c])
        assert h1[2].min() == 1.0, "ground covers all of H1"

        # Point cloud: no ground points; counts match the area / spacing^2 expectation.
        p = out["P_xyz"]
        assert len(p) > 0 and p[:, 1].min() >= -1e-3, (label, p[:, 1].min())
        assert meta["ground_triangles"] == 2, (label, meta["ground_triangles"])  # both ground plates found
        roof_pts = out["P_is_roof"]
        assert np.all(p[roof_pts, 1] > 1.0), (label, "a roof point lies on the ground plane")
        near = (p[:, 0] > 150) & (p[:, 0] < 300) & (np.abs(p[:, 2]) < 50)
        s_near = max(P_S0_M, 220 * np.radians(P_DTHETA_DEG))
        expect = (40 * 40 + 4 * 40 * 30) / s_near ** 2
        assert abs(near.sum() / expect - 1) < 0.1, (label, near.sum(), expect)
        roof = out["P_is_roof"][near]
        assert abs(roof.mean() - 1600 / 6400) < 0.05, (label, roof.mean())
        print(f"[build_geometry_inputs --self-test] {label}: PASS "
              f"(ref_y={meta['ref_y']:.3f}, points={meta['p_points']}, ground={meta['ground_source']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build H1/H2 height-map pyramid and LOD point cloud P per tile.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data-dir", type=Path, help="Directory of tile shards (.tar.gz carry the scene mesh).")
    ap.add_argument("--out-dir", type=Path, help="Where geom_{tile_id}.npz files are written.")
    ap.add_argument("--tiles", nargs="*", default=None, help="Tile ids; default: all 93 tiles of the split.")
    ap.add_argument("--split-json", type=Path, default=KIT_ROOT / "meta" / "split.json")
    ap.add_argument("--manifest-json", type=Path, default=KIT_ROOT / "meta" / "manifest.json")
    ap.add_argument("--seed", type=int, default=20261120, help="Point-cloud sampling seed (shared by all models).")
    ap.add_argument("--supersample", type=int, default=4)
    ap.add_argument("--dtm-window-m", type=float, default=100.0)
    ap.add_argument("--ground-tol-m", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not (args.data_dir and args.out_dir):
        ap.error("--data-dir and --out-dir are required unless --self-test is set.")

    tiles = args.tiles
    if not tiles:
        sys.path.insert(0, str(KIT_ROOT / "baseline_kit"))
        from evaluate_predictions import load_split_tile_ids
        b = load_split_tile_ids(args.split_json, args.manifest_json)
        tiles = [*b["fit_67"], *b["calibration"], *b["val_spatial_clean"]]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tid in tiles:
        dst = args.out_dir / f"geom_{tid}.npz"
        if dst.exists() and not args.overwrite:
            print(f"skip {tid} (exists)")
            continue
        out = build_tile(load_geometry_arrays(args.data_dir, tid), tid, seed=args.seed,
                         supersample=args.supersample, dtm_window_m=args.dtm_window_m,
                         ground_tol_m=args.ground_tol_m)
        np.savez_compressed(dst, **out)
        meta = json.loads(str(out["meta_json"]))
        print(f"{tid}: points={meta['p_points']} ground={meta['ground_source']} "
              f"ref_spread={meta['ref_y_spread_p5_p95_m']:.3f} m fit_resid={meta['h0_fit_max_residual_m']:.4f} m")


if __name__ == "__main__":
    main()

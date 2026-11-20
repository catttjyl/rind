"""Provenance-Bound Contiguous Tensor Bank (`mmap`) & Streaming Day/Receiver Pipeline.

Implements the accelerated `MANHATTAN_4445_ANNUAL_18808_V1` data path with zero cross-split
leakage and bounded host/GPU memory (`P0-1`, `P0-2`, `P0-4`, `P1-2`, `P2-1`):

1. `ContiguousTileMmapShard`: Converts validated per-tile `DSFCH2` trajectories + full `4 km`
   azimuth-sector 3D candidate banks into contiguous `.npy` memory-mapped arrays written via
   atomic temporary directories and bound by `SHARD_PROVENANCE.json` (`benchmark_contract`,
   `feature_version`, `split_sha256`, `manifest_sha256`, `tile_id`, `canonical_split`,
   `source_core_sha256`, `mesh_provenance_sha256`, `week_schedule_sha256`, `receiver_mode`,
   `receiver_support_sha256`, `bbox_margin_m`, and mandatory per-array `shape`, `dtype`, and
   `sha256`).
2. `iter_day_batches`: Streams annual `18,808`-frame tiles by exact day-batches (`1-4` days) and
   receiver chunks (`<= 256` receivers), skipping unused geometry banks for non-candidate baselines
   and extracting exact centered `(N_chunk, 9, P, P)` spatial crops without coordinate clamping.
3. `build_canonical37_torch` & `materialize_gpu_tokens_from_azimuth_bank`: Vectorized `float32`
   token construction using canonical `float64` azimuth-bin indices (`0` sector-boundary triangle
   mismatches between CPU and GPU).
"""

from __future__ import annotations

from .label_budget import TileLabelDomain

import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import uuid

import numpy as np
import torch

from . import BENCHMARK_CONTRACT_ID, FEATURE_SCHEMA_VERSION
from .fullday_data import (
    DEFAULT_SCENE_HORIZON_M,
    FullDayTileReader,
    FullDayTileTrajectory,
    extract_exact_centered_receiver_crops,
)
from .metrics import build_receiver_sampling_disclosure, build_same_day_adjacent_mask
from .models import compute_tile_global_knn_pack_torch
from .split_contract import SplitContract, compute_file_sha256, exclusive_file_lock

_FAST_DATA_IO_STATS: Dict[str, int] = {
    "read_calls": 0,
    "bytes_read": 0,
}


def reset_fast_data_io_stats() -> None:
    _FAST_DATA_IO_STATS["read_calls"] = 0
    _FAST_DATA_IO_STATS["bytes_read"] = 0


def get_fast_data_io_stats() -> Dict[str, int]:
    return dict(_FAST_DATA_IO_STATS)


def _compute_shard_provenance_digest(prov: Mapping[str, Any]) -> str:
    """Compute canonical cryptographic SHA-256 over `SHARD_PROVENANCE.json` (protecting `optional_arrays`)."""
    body = {str(k): v for k, v in prov.items() if str(k) != "provenance_digest_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    payload = f"{BENCHMARK_CONTRACT_ID}::shard_provenance_v2::{canonical}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_week_schedule_sha256(week_indices: Sequence[int]) -> str:
    payload = ",".join(str(int(w)) for w in week_indices)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_npy_header_bytes(shape: Sequence[int], dtype: Any) -> bytes:
    dt = np.dtype(dtype)
    hdr = {
        "descr": np.lib.format.dtype_to_descr(dt),
        "fortran_order": False,
        "shape": tuple(int(d) for d in shape),
    }
    buf = io.BytesIO()
    try:
        np.lib.format.write_array_header_1_0(buf, hdr)
    except ValueError:
        buf = io.BytesIO()
        np.lib.format.write_array_header_2_0(buf, hdr)
    return buf.getvalue()


def _read_npy_header_from_stream(fh: Any, ver: Tuple[int, int]) -> Tuple[Tuple[int, ...], bool, np.dtype]:
    """Read `.npy` array header strictly via public `numpy.lib.format` APIs (NumPy 1.x & 2.x)."""
    major, minor = int(ver[0]), int(ver[1])
    if (major, minor) == (1, 0):
        shape, fortran_order, dt = np.lib.format.read_array_header_1_0(fh)
    elif (major, minor) == (2, 0):
        shape, fortran_order, dt = np.lib.format.read_array_header_2_0(fh)
    else:
        public_fn = getattr(np.lib.format, f"read_array_header_{major}_{minor}", None)
        if public_fn is not None:
            shape, fortran_order, dt = public_fn(fh)
        elif major >= 2:
            # `.npy` v3.0 shares the 4-byte uint32 header-length prefix with v2.0 (UTF-8 dict vs Latin1)
            raw_len = fh.read(4)
            if len(raw_len) != 4:
                raise ValueError("Truncated .npy v3 header length.")
            hlen = int.from_bytes(raw_len, byteorder="little", signed=False)
            import ast

            hdr_dict = ast.literal_eval(fh.read(hlen).decode("utf-8"))
            shape = hdr_dict["shape"]
            fortran_order = hdr_dict["fortran_order"]
            dt = np.lib.format.descr_to_dtype(hdr_dict["descr"])
        else:
            raise ValueError(f"Unsupported .npy format version {ver!r}")
    return tuple(int(d) for d in shape), bool(fortran_order), np.dtype(dt)


def _read_npy_header_shape_dtype(npy_path: Path) -> Tuple[List[int], str]:
    with npy_path.open("rb") as fh:
        ver = np.lib.format.read_magic(fh)
        shape, _fortran, dt = _read_npy_header_from_stream(fh, ver)
    return [int(d) for d in shape], str(dt)


def _read_npy_slice_no_mmap(npy_path: Path, sl: Any = slice(None)) -> np.ndarray:
    """Read an arbitrary `.npy` slice via direct file `pread`/`seek`+`read` (`0` `RssFile` mmap pages, `1.0x` byte read on narrow column slices)."""
    fsize = npy_path.stat().st_size
    with npy_path.open("rb") as fh:
        try:
            return _read_npy_slice_no_mmap_inner(fh, sl)
        finally:
            if fsize >= 64 * 1024 * 1024 and hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                try:
                    os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass


def _read_npy_slice_no_mmap_inner(fh: Any, sl: Any = slice(None)) -> np.ndarray:
    if True:
        ver = np.lib.format.read_magic(fh)
        shape, fortran_order, dtype = _read_npy_header_from_stream(fh, ver)
        data_offset = fh.tell()

        if len(shape) == 0 or fortran_order:
            fh.seek(0)
            arr = np.load(fh, allow_pickle=False)
            out_arr = np.array(arr[sl], copy=True)
            _FAST_DATA_IO_STATS["read_calls"] += 1
            _FAST_DATA_IO_STATS["bytes_read"] += int(arr.nbytes)
            return out_arr

        n0 = int(shape[0])
        rest_shape = tuple(int(d) for d in shape[1:])
        row_elems = math.prod(rest_shape) if rest_shape else 1
        row_bytes = row_elems * int(dtype.itemsize)

        # Case 1: full array (`sl == slice(None)`) or single axis-0 slice
        if isinstance(sl, slice):
            s0, s1, step0 = sl.indices(n0)
            if step0 == 1:
                rows = max(0, s1 - s0)
                out = np.empty((rows, *rest_shape), dtype=dtype)
                if rows == 0 or row_bytes == 0:
                    return out
                fh.seek(data_offset + s0 * row_bytes)
                blk_rows = max(1, (8 * 1024 * 1024) // max(row_bytes, 1))
                for r_off in range(0, rows, blk_rows):
                    take = min(rows - r_off, blk_rows)
                    raw = fh.read(take * row_bytes)
                    _FAST_DATA_IO_STATS["read_calls"] += 1
                    _FAST_DATA_IO_STATS["bytes_read"] += len(raw)
                    out[r_off : r_off + take] = np.frombuffer(raw, dtype=dtype).reshape(
                        take, *rest_shape
                    )
                return out

        # Case 2: 2D `(f_sl, r_sl)` slice on `(T, N, ...)` C-contiguous array
        if (
            isinstance(sl, tuple)
            and len(sl) == 2
            and isinstance(sl[0], slice)
            and isinstance(sl[1], slice)
            and len(shape) >= 2
        ):
            f0, f1, fstep = sl[0].indices(n0)
            n1 = int(shape[1])
            r0, r1, rstep = sl[1].indices(n1)
            if fstep == 1 and rstep == 1:
                f_rows = max(0, f1 - f0)
                r_cols = max(0, r1 - r0)
                tail_shape = tuple(int(d) for d in shape[2:])
                out = np.empty((f_rows, r_cols, *tail_shape), dtype=dtype)
                if f_rows == 0 or r_cols == 0 or row_bytes == 0:
                    return out

                # Eliminate O(N / r_cols) read amplification on `all_valid` shards (`r_cols << N`)!
                if r_cols * 2 <= n1:
                    col_elems = math.prod(tail_shape) if tail_shape else 1
                    col_bytes = col_elems * int(dtype.itemsize)
                    sub_bytes = r_cols * col_bytes
                    col_byte_off = r0 * col_bytes
                    fd = fh.fileno()
                    has_pread = hasattr(os, "pread")
                    for f_i in range(f_rows):
                        pos = data_offset + (f0 + f_i) * row_bytes + col_byte_off
                        if has_pread:
                            raw = os.pread(fd, sub_bytes, pos)
                        else:
                            fh.seek(pos)
                            raw = fh.read(sub_bytes)
                        _FAST_DATA_IO_STATS["read_calls"] += 1
                        _FAST_DATA_IO_STATS["bytes_read"] += len(raw)
                        out[f_i] = np.frombuffer(raw, dtype=dtype).reshape(r_cols, *tail_shape)
                    return out

                fh.seek(data_offset + f0 * row_bytes)
                blk_rows = max(1, (8 * 1024 * 1024) // max(row_bytes, 1))
                for f_off in range(0, f_rows, blk_rows):
                    take = min(f_rows - f_off, blk_rows)
                    raw = fh.read(take * row_bytes)
                    _FAST_DATA_IO_STATS["read_calls"] += 1
                    _FAST_DATA_IO_STATS["bytes_read"] += len(raw)
                    blk = np.frombuffer(raw, dtype=dtype).reshape(take, *rest_shape)
                    out[f_off : f_off + take] = blk[:, r0:r1]
                return out

        # General fallback without `mmap`
        fh.seek(0)
        arr = np.load(fh, allow_pickle=False)
        _FAST_DATA_IO_STATS["read_calls"] += 1
        _FAST_DATA_IO_STATS["bytes_read"] += int(arr.nbytes)
        return np.array(arr[sl], copy=True)


def _lexsort_triangle_vertices_torch(v: torch.Tensor) -> torch.Tensor:
    """Sort 3 vertices `(..., 3, 3)` lexicographically along `(x, y, z)` in pure `float32`."""
    out = v.clone()
    for p_i, p_j in ((0, 1), (1, 2), (0, 1)):
        vi = out[..., p_i, :]
        vj = out[..., p_j, :]
        swap = (vi[..., 0] > vj[..., 0]) | (
            (vi[..., 0] == vj[..., 0])
            & (
                (vi[..., 1] > vj[..., 1])
                | ((vi[..., 1] == vj[..., 1]) & (vi[..., 2] > vj[..., 2]))
            )
        )
        if bool(swap.any()):
            new_vi = torch.where(swap.unsqueeze(-1), vj, vi)
            new_vj = torch.where(swap.unsqueeze(-1), vi, vj)
            parts = []
            for idx in range(3):
                if idx == p_i:
                    parts.append(new_vi)
                elif idx == p_j:
                    parts.append(new_vj)
                else:
                    parts.append(out[..., idx, :])
            out = torch.stack(parts, dim=-2)
    return out


def _build_canonical37_flat_chunk_torch(
    v_flat: torch.Tensor,      # (M_flat, K, 3, 3) float32
    omat_flat: torch.Tensor,   # (M_flat, K) bool
    cmask_flat: torch.Tensor,  # (M_flat, K) bool
    s_unit_flat: torch.Tensor, # (M_flat, 3) float32
    *,
    scale_m: float = 1000.0,
) -> torch.Tensor:
    """Compute `(M_flat, K, 37)` `canonical37` tokens in pure `float32` on a bounded slice."""
    m_flat, k_cands, _, _ = v_flat.shape
    v = _lexsort_triangle_vertices_torch(v_flat)
    a = v[..., 1, :] - v[..., 0, :]
    b = v[..., 2, :] - v[..., 0, :]
    c = v[..., 2, :] - v[..., 1, :]
    normal = torch.cross(a, b, dim=-1)
    area = torch.linalg.norm(normal, dim=-1)
    edges = torch.stack(
        [
            torch.linalg.norm(a, dim=-1),
            torch.linalg.norm(b, dim=-1),
            torch.linalg.norm(c, dim=-1),
        ],
        dim=-1,
    )

    s_b = s_unit_flat[:, None, :]       # (M_flat, 1, 3)
    s_v = s_unit_flat[:, None, None, :] # (M_flat, 1, 1, 3)
    logscale = math.log1p(float(scale_m))

    along = (v * s_v).sum(dim=-1)
    radial = torch.linalg.norm(v - along.unsqueeze(-1) * s_v, dim=-1)
    lead_along = torch.sign(along) * torch.log1p(along.abs()) / logscale
    lead_radial = torch.log1p(radial) / logscale
    lead_edges = torch.log1p(edges) / logscale

    proj_area = (normal * s_b).sum(dim=-1).abs()
    incidence = proj_area / area.clamp_min(1e-20)
    tol = 1e-12 * edges.amax(dim=-1).square().clamp_min(1e-20)
    degenerate = area <= tol
    proj_degenerate = proj_area <= tol

    margin_list: List[torch.Tensor] = []
    for i0, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        edge = v[..., i1, :] - v[..., i0, :]
        edge_dot_s = (edge * s_b).sum(dim=-1, keepdim=True)
        perp = edge - edge_dot_s * s_b
        length = torch.linalg.norm(perp, dim=-1)
        orient = (torch.cross(edge, v[..., i2, :] - v[..., i0, :], dim=-1) * s_b).sum(dim=-1)
        numer = (torch.cross(edge, -v[..., i0, :], dim=-1) * s_b).sum(dim=-1) * torch.sign(orient)
        valid_m = (~proj_degenerate) & (length > torch.sqrt(tol))
        m_val = torch.where(valid_m, numer / length.clamp_min(1e-12), torch.zeros_like(length))
        margin_list.append(m_val)

    margins, _ = torch.sort(torch.stack(margin_list, dim=-1), dim=-1)
    margins = torch.sign(margins) * torch.log1p(margins.abs()) / logscale

    s_rep = s_b.expand(m_flat, k_cands, 3)
    sy_col = s_rep[..., 1:2]
    mat_col = omat_flat.to(torch.float32).unsqueeze(-1)
    log_area_col = (torch.log1p(area) / (2.0 * logscale)).unsqueeze(-1)

    out = torch.cat(
        [
            lead_along,
            lead_radial,
            lead_edges,
            incidence.unsqueeze(-1),
            mat_col,
            log_area_col,
            sy_col,
            v.reshape(m_flat, k_cands, 9) / float(scale_m),
            margins,
            s_rep,
            v.amin(dim=-2) / float(scale_m),
            v.amax(dim=-2) / float(scale_m),
            (area / (float(scale_m) ** 2)).unsqueeze(-1),
            proj_degenerate.to(torch.float32).unsqueeze(-1),
            degenerate.to(torch.float32).unsqueeze(-1),
        ],
        dim=-1,
    )
    out = torch.where(cmask_flat.unsqueeze(-1), out, torch.zeros_like(out))
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_canonical37_torch(
    rel_triangles_xyz: torch.Tensor,
    opaque_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    sun_directions_xyz: torch.Tensor,
    *,
    scale_m: float = 1000.0,
    max_flat_chunk: int = 32768,
) -> torch.Tensor:
    """Vectorized `float32` PyTorch builder for `canonical37` tokens `(B, T, N, K, 37)` or `(T, N, K, 37)`.

    Runs outside `bf16` autocast (`enabled=False`) and chunks large `(B*T*N)` inputs so
    intermediate cross-product workspaces stay `< 80 MiB` even on multi-day rollouts!
    """
    device_type = "cuda" if rel_triangles_xyz.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        suns = sun_directions_xyz.to(torch.float32)
        squeeze_batch = False
        if suns.ndim == 2:
            suns = suns.unsqueeze(0)
            squeeze_batch = True

        bsz, t_steps, _ = suns.shape
        s_norm = torch.linalg.norm(suns, dim=-1, keepdim=True).clamp_min(1e-12)
        s_unit = suns / s_norm

        v = rel_triangles_xyz.to(torch.float32)
        cmask = candidate_mask.to(torch.bool)
        omat = opaque_mask.to(torch.bool)

        if v.ndim == 5:
            v = v.unsqueeze(0)
            cmask = cmask.unsqueeze(0)
            omat = omat.unsqueeze(0)

        n_pts = v.shape[2]
        k_cands = v.shape[3]
        if n_pts == 0 or k_cands == 0 or t_steps == 0:
            empty = torch.zeros(
                (bsz, t_steps, n_pts, k_cands, 37), dtype=torch.float32, device=suns.device
            )
            return empty.squeeze(0) if squeeze_batch else empty

        total_flat = bsz * t_steps * n_pts
        v_flat = v.reshape(total_flat, k_cands, 3, 3)
        omat_flat = omat.reshape(total_flat, k_cands)
        cmask_flat = cmask.reshape(total_flat, k_cands)
        s_flat = s_unit[:, :, None, :].expand(bsz, t_steps, n_pts, 3).reshape(total_flat, 3)

        # If autograd is enabled (e.g. `rind_checkpoint_grad` inverse design) or small batch, run directly
        if v.requires_grad or total_flat <= int(max_flat_chunk):
            out_flat = _build_canonical37_flat_chunk_torch(
                v_flat, omat_flat, cmask_flat, s_flat, scale_m=scale_m
            )
        else:
            out_flat = torch.empty(
                (total_flat, k_cands, 37), dtype=torch.float32, device=suns.device
            )
            step = int(max_flat_chunk)
            for s_idx in range(0, total_flat, step):
                e_idx = min(total_flat, s_idx + step)
                out_flat[s_idx:e_idx] = _build_canonical37_flat_chunk_torch(
                    v_flat[s_idx:e_idx],
                    omat_flat[s_idx:e_idx],
                    cmask_flat[s_idx:e_idx],
                    s_flat[s_idx:e_idx],
                    scale_m=scale_m,
                )

        out = out_flat.reshape(bsz, t_steps, n_pts, k_cands, 37)
        return out.squeeze(0) if squeeze_batch else out


def compute_canonical_azimuth_bins_torch(
    sun_dirs: torch.Tensor, num_azimuth_bins: int = 24
) -> torch.Tensor:
    """Compute exact `float64` azimuth sector indices matching `sun_dirs_to_azimuth_bins` (`P1-2`)."""
    s64 = sun_dirs.to(torch.float64)
    ang = torch.remainder(torch.atan2(s64[..., 0], s64[..., 2]), 2.0 * math.pi)
    sector_w = (2.0 * math.pi) / float(num_azimuth_bins)
    return torch.remainder(torch.floor(ang / sector_w).to(torch.long), int(num_azimuth_bins))


def materialize_gpu_tokens_from_azimuth_bank(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Expand compact `az_rel_triangles` `(..., N, AzBins, K, 3, 3)` into `candidate_tokens` on device."""
    if "candidate_tokens" in batch or "az_rel_triangles" not in batch:
        return batch

    az_tris = batch["az_rel_triangles"]
    az_opaque = batch["az_opaque"]
    az_mask = batch["az_mask"]
    sun_dirs = batch["sun_dirs"]
    az_sids = batch.get("az_source_ids")

    squeeze_b = False
    if az_tris.ndim == 5:
        az_tris = az_tris.unsqueeze(0)
        az_opaque = az_opaque.unsqueeze(0)
        az_mask = az_mask.unsqueeze(0)
        if az_sids is not None and az_sids.ndim == 3:
            az_sids = az_sids.unsqueeze(0)
        squeeze_b = True
    if sun_dirs.ndim == 2:
        sun_dirs = sun_dirs.unsqueeze(0)

    bsz, n_pts, num_bins, k_cands, _, _ = az_tris.shape
    t_steps = sun_dirs.shape[1]

    if "azimuth_bin_idx" in batch and batch["azimuth_bin_idx"] is not None:
        raw_bins = batch["azimuth_bin_idx"].to(device=az_tris.device, dtype=torch.long)
        if raw_bins.ndim == 1:
            raw_bins = raw_bins.unsqueeze(0)
        bin_idx = torch.remainder(raw_bins, num_bins)
    else:
        bin_idx = compute_canonical_azimuth_bins_torch(sun_dirs, num_azimuth_bins=num_bins)

    tris_perm = az_tris.permute(0, 2, 1, 3, 4, 5)
    idx_tris = bin_idx[:, :, None, None, None, None].expand(bsz, t_steps, n_pts, k_cands, 3, 3)
    rel_t = torch.gather(tris_perm, dim=1, index=idx_tris)

    op_perm = az_opaque.permute(0, 2, 1, 3)
    mk_perm = az_mask.permute(0, 2, 1, 3)
    idx_mk = bin_idx[:, :, None, None].expand(bsz, t_steps, n_pts, k_cands)
    op_t = torch.gather(op_perm, dim=1, index=idx_mk)
    mk_t = torch.gather(mk_perm, dim=1, index=idx_mk)

    sid_t: Optional[torch.Tensor] = None
    if az_sids is not None:
        sid_perm = az_sids.to(torch.int64).permute(0, 2, 1, 3)
        sid_t = torch.gather(sid_perm, dim=1, index=idx_mk)

    tokens = build_canonical37_torch(rel_t, op_t, mk_t, sun_dirs)
    out = dict(batch)
    if squeeze_b:
        out["candidate_tokens"] = tokens.squeeze(0)
        out["candidate_mask"] = mk_t.squeeze(0)
        if sid_t is not None:
            out["candidate_source_ids"] = sid_t.squeeze(0)
    else:
        out["candidate_tokens"] = tokens
        out["candidate_mask"] = mk_t
        if sid_t is not None:
            out["candidate_source_ids"] = sid_t
    return out


RAY_ARRAYS = ("scene_triangles_xyz", "scene_opaque", "scene_source_ids", "receiver_source_triangle_ids")


def iter_objective_windows(chunks, window_frames):
    """Reassemble chronological reads into fixed loss/optimizer windows for one receiver range.

    Call with reads aligned to window boundaries (the training CLI uses their GCD).
    Static geometry is shared; only time-indexed fields are concatenated.
    """
    if window_frames <= 0:
        raise ValueError("objective window must be positive")
    temporal = ("sun_dirs", "azimuth_bin_idx", "truth", "valid_mask", "boundary_distance_m",
                "boundary_distance_valid", "response_5x5_bitpack")
    pending = []
    count = 0

    def merge():
        if len(pending) == 1:
            return pending[0]
        batch = dict(pending[0])
        for field in temporal:
            if field in batch:
                batch[field] = torch.cat([part[field] for part in pending], dim=0)
        if "ray_candidate_count" in batch:
            for field in ("az_rel_triangles", "az_opaque", "az_mask", "az_source_ids"):
                batch[field] = torch.cat([part[field] for part in pending], dim=1)
            batch["azimuth_bin_idx"] = torch.arange(count)
            batch["ray_candidate_count"] = torch.cat([part["ray_candidate_count"] for part in pending])
        for field in ("day_ids", "timestamps_unix_s"):
            batch[field] = np.concatenate([part[field] for part in pending])
        batch["solar_regimes"] = tuple(regime for part in pending for regime in part["solar_regimes"])
        batch["same_day_adjacent_mask"] = torch.from_numpy(
            build_same_day_adjacent_mask(batch["day_ids"], batch["timestamps_unix_s"]))
        batch["frame_range"] = (pending[0]["frame_range"][0], pending[-1]["frame_range"][1])
        return batch

    for part in chunks:
        if pending and not part["carry_state_from_prev_chunk"]:
            yield merge()
            pending, count = [], 0
        if pending and (part["receiver_range"] != pending[-1]["receiver_range"]
                        or part["tile_ids"] != pending[-1]["tile_ids"]
                        or part["frame_range"][0] != pending[-1]["frame_range"][1]):
            raise ValueError("non-contiguous read in objective window")
        pending.append(part)
        count += len(part["sun_dirs"])
        if count > window_frames:
            raise ValueError("physical read crosses an objective window boundary")
        if count == window_frames:
            yield merge()
            pending, count = [], 0
    if pending:
        yield merge()


class ContiguousTileMmapShard:
    """Provenance-bound `.npy` memory-mapped tile shard with full SHA/shape/dtype verification (`P0-2`)."""

    SCHEMA_VERSION = "rind.contiguous_mmap_shard.v2"

    ARRAY_NAMES: Tuple[str, ...] = (
        "day_ids",
        "timestamps_unix_s",
        "sun_dirs",
        "azimuth_bin_idx",
        "solar_altitude_deg",
        "solar_azimuth_deg",
        "receivers_xyz",
        "receiver_rc",
        "truth",
        "valid_mask",
        "boundary_distance_m",
        "boundary_distance_valid",
        "local_features",
        "static_fields_8hw",
        "geometry_known_hw",
        "az_rel_triangles",
        "az_opaque",
        "az_mask",
        "az_source_ids",
    )

    def __init__(
        self,
        shard_dir: Path | str,
        *,
        expected_split_sha256: Optional[str] = None,
        expected_manifest_sha256: Optional[str] = None,
        expected_tile_id: Optional[str] = None,
        expected_canonical_split: Optional[str] = None,
        expected_source_core_sha256: Optional[str] = None,
        expected_mesh_provenance_sha256: Optional[str] = None,
        expected_week_indices: Optional[Sequence[int]] = None,
        expected_receiver_mode: Optional[str] = None,
        expected_bbox_margin_m: Optional[float] = None,
        expected_optional_arrays: Optional[Sequence[str]] = None,
        expected_provenance_file_sha256: Optional[str] = None,
        verify_file_hashes: bool = True,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        prov_path = self.shard_dir / "SHARD_PROVENANCE.json"
        if not prov_path.exists():
            raise FileNotFoundError(f"Missing SHARD_PROVENANCE.json in {self.shard_dir}")

        prov_bytes = prov_path.read_bytes()
        if expected_provenance_file_sha256 is not None and hashlib.sha256(prov_bytes).hexdigest() != expected_provenance_file_sha256:
            raise RuntimeError("Portable provenance SHA-256 mismatch")
        self.provenance: Dict[str, Any] = json.loads(prov_bytes)
        if self.provenance.get("schema") != self.SCHEMA_VERSION:
            raise RuntimeError(
                f"Shard schema mismatch in {self.shard_dir}: "
                f"{self.provenance.get('schema')!r} != {self.SCHEMA_VERSION!r}"
            )
        if self.provenance.get("benchmark_contract") != BENCHMARK_CONTRACT_ID:
            raise RuntimeError(
                f"Shard benchmark contract mismatch: {self.provenance.get('benchmark_contract')!r}"
            )
        if self.provenance.get("feature_version") != FEATURE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Shard feature_version mismatch: {self.provenance.get('feature_version')!r} != {FEATURE_SCHEMA_VERSION!r}"
            )
        if (
            expected_split_sha256 is not None
            and self.provenance.get("split_sha256") != expected_split_sha256
        ):
            raise RuntimeError(
                f"Shard split_sha256 ({self.provenance.get('split_sha256')}) != "
                f"authoritative {expected_split_sha256}"
            )
        if (
            expected_manifest_sha256 is not None
            and self.provenance.get("manifest_sha256") != expected_manifest_sha256
        ):
            raise RuntimeError(
                f"Shard manifest_sha256 ({self.provenance.get('manifest_sha256')}) != "
                f"active {expected_manifest_sha256}"
            )
        if expected_tile_id is not None and self.provenance.get("tile_id") != expected_tile_id:
            raise RuntimeError(
                f"Shard tile_id ({self.provenance.get('tile_id')}) != expected {expected_tile_id}"
            )
        if (
            expected_canonical_split is not None
            and self.provenance.get("canonical_split") != expected_canonical_split
        ):
            raise RuntimeError(
                f"Shard canonical_split ({self.provenance.get('canonical_split')}) != "
                f"expected {expected_canonical_split}"
            )
        if (
            expected_source_core_sha256
            and self.provenance.get("source_core_sha256") != expected_source_core_sha256
        ):
            raise RuntimeError(
                f"Shard source_core_sha256 ({self.provenance.get('source_core_sha256')}) != "
                f"expected {expected_source_core_sha256}"
            )
        if (
            expected_mesh_provenance_sha256
            and self.provenance.get("mesh_provenance_sha256") != expected_mesh_provenance_sha256
        ):
            raise RuntimeError(
                f"Shard mesh_provenance_sha256 ({self.provenance.get('mesh_provenance_sha256')}) != "
                f"expected {expected_mesh_provenance_sha256}"
            )
        if expected_week_indices is not None:
            exp_w_sha = compute_week_schedule_sha256(expected_week_indices)
            if self.provenance.get("week_schedule_sha256") != exp_w_sha:
                raise RuntimeError(
                    f"Shard week schedule mismatch for {self.shard_dir}: "
                    f"{self.provenance.get('week_indices')} != {list(expected_week_indices)}"
                )
        if (
            expected_receiver_mode is not None
            and self.provenance.get("receiver_mode") != expected_receiver_mode
        ):
            raise RuntimeError(
                f"Shard receiver_mode ({self.provenance.get('receiver_mode')}) != {expected_receiver_mode}"
            )
        if expected_bbox_margin_m is not None and not math.isclose(
            float(self.provenance.get("bbox_margin_m", -1.0)),
            float(expected_bbox_margin_m),
            rel_tol=0.0,
            abs_tol=1e-3,
        ):
            raise RuntimeError(
                f"Shard bbox_margin_m ({self.provenance.get('bbox_margin_m')}) != {expected_bbox_margin_m}"
            )

        file_hashes = self.provenance.get("array_sha256")
        file_shapes = self.provenance.get("array_shapes")
        file_dtypes = self.provenance.get("array_dtypes")
        if (
            not isinstance(file_hashes, Mapping)
            or not isinstance(file_shapes, Mapping)
            or not isinstance(file_dtypes, Mapping)
        ):
            raise RuntimeError(
                f"Corrupted SHARD_PROVENANCE.json in {self.shard_dir}: missing array hash/shape/dtype maps."
            )

        if "optional_arrays" not in self.provenance or not isinstance(
            self.provenance.get("optional_arrays"), list
        ):
            raise RuntimeError(
                f"Tampered or incomplete SHARD_PROVENANCE.json in {self.shard_dir}: "
                "missing mandatory `optional_arrays` list declaration!"
            )
        recorded_prov_digest = self.provenance.get("provenance_digest_sha256")
        expected_prov_digest = _compute_shard_provenance_digest(self.provenance)
        if not recorded_prov_digest or str(recorded_prov_digest) != expected_prov_digest:
            raise RuntimeError(
                f"SHARD_PROVENANCE.json integrity digest mismatch in {self.shard_dir}: "
                f"got {recorded_prov_digest!r}, expected {expected_prov_digest!r} "
                "(tampering with `optional_arrays` or metadata is strictly blocked)."
            )

        optional_names = tuple(str(x) for x in self.provenance["optional_arrays"])
        if expected_optional_arrays is not None:
            exp_opt_set = set(str(x) for x in expected_optional_arrays)
            if set(optional_names) != exp_opt_set:
                raise RuntimeError(
                    f"Shard optional_arrays mismatch in {self.shard_dir}: "
                    f"declared={sorted(optional_names)} != expected={sorted(exp_opt_set)}"
                )
        names_to_verify = (*self.ARRAY_NAMES, *[n for n in optional_names if n not in self.ARRAY_NAMES])
        disk_npy_stems = {
            p.stem for p in self.shard_dir.glob("*.npy") if not p.name.startswith(".")
        }
        undeclared_npy = disk_npy_stems - set(names_to_verify)
        if undeclared_npy:
            raise RuntimeError(
                f"Undeclared `.npy` files present in shard {self.shard_dir} "
                f"(`optional_arrays` deletion/tampering detected): {sorted(undeclared_npy)}"
            )

        self.array_paths: Dict[str, Path] = {}
        for name in names_to_verify:
            if name not in file_hashes or not file_hashes[name]:
                raise RuntimeError(
                    f"SHARD_PROVENANCE.json omitted required SHA-256 for `{name}.npy` (`P0-2`)!"
                )
            if name not in file_shapes or name not in file_dtypes:
                raise RuntimeError(
                    f"SHARD_PROVENANCE.json omitted required shape/dtype for `{name}.npy` (`P0-2`)!"
                )
            npy_path = self.shard_dir / f"{name}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"Missing mmap shard array {npy_path}")
            if verify_file_hashes:
                actual = compute_file_sha256(npy_path)
                if actual != file_hashes[name]:
                    raise RuntimeError(
                        f"Mmap shard checksum mismatch for {npy_path.name}: {actual} != {file_hashes[name]}"
                    )
            # Validate .npy header shape & dtype without mapping multi-GB files into process RSS
            actual_shape, actual_dtype = _read_npy_header_shape_dtype(npy_path)
            if actual_shape != [int(d) for d in file_shapes[name]]:
                raise RuntimeError(
                    f"Shard array shape mismatch on {npy_path.name}: {actual_shape} != {file_shapes[name]}"
                )
            if actual_dtype != str(file_dtypes[name]):
                raise RuntimeError(
                    f"Shard array dtype mismatch on {npy_path.name}: {actual_dtype} != {file_dtypes[name]}"
                )
            self.array_paths[name] = npy_path

        self.tile_id = str(self.provenance["tile_id"])
        self.canonical_split = str(self.provenance["canonical_split"])
        self.spatial_group = str(self.provenance["spatial_group"])
        self.source_core_sha256 = str(self.provenance.get("source_core_sha256", ""))
        self.mesh_provenance_sha256 = str(self.provenance.get("mesh_provenance_sha256", ""))
        self.has_response_5x5_bitpack = "response_5x5_bitpack" in self.array_paths
        self.receiver_mode = str(self.provenance["receiver_mode"])
        self.total_eligible_receivers = int(self.provenance["total_eligible_receivers"])
        self.meters_per_pixel = float(self.provenance["meters_per_pixel"])
        self.bbox_margin_m = float(self.provenance["bbox_margin_m"])
        self.solar_regimes: Tuple[str, ...] = tuple(self.provenance["solar_regimes"])
        self.num_frames = int(self.provenance["num_frames"])
        self.num_receivers = int(self.provenance["num_receivers"])
        self._day_ranges_cache: Optional[List[Tuple[int, int, int]]] = None
        self._global_knn_cache: Optional[Dict[str, torch.Tensor]] = None

    def _get_global_knn_pack(self) -> Dict[str, torch.Tensor]:
        """Compute & cache tile-global valid `k`-NN static geometry descriptors (`< 55 MiB` peak RAM)."""
        if self._global_knn_cache is not None:
            return self._global_knn_cache
        all_rc = self._mmap_slice_copy("receiver_rc")
        all_xyz = self._mmap_slice_copy("receivers_xyz")
        known_hw = self._mmap_slice_copy("geometry_known_hw")
        all_lf = self._mmap_slice_copy("local_features")
        obs_all = np.ascontiguousarray(known_hw[all_rc[:, 0], all_rc[:, 1]].astype(bool))
        knn_dev = (
            torch.device("cuda")
            if (all_xyz.shape[0] >= 4096 and torch.cuda.is_available())
            else torch.device("cpu")
        )
        pack = compute_tile_global_knn_pack_torch(
            torch.from_numpy(all_xyz).to(knn_dev),
            torch.from_numpy(obs_all).to(knn_dev),
            torch.from_numpy(all_lf).to(knn_dev),
            k_neighbors=8,
            query_chunk=2048 if knn_dev.type == "cuda" else 512,
        )
        self._global_knn_cache = {k: v.cpu() for k, v in pack.items()}
        return self._global_knn_cache

    def _mmap_slice_copy(self, name: str, sl: Any = slice(None)) -> np.ndarray:
        """Read a contiguous slice via direct `.npy` offset `seek`+`read` (`0` `RssFile` mmap bloat)."""
        return _read_npy_slice_no_mmap(self.array_paths[name], sl)

    def close(self) -> None:
        return None

    def receiver_sampling_disclosure(self) -> Dict[str, Any]:
        rc = self._mmap_slice_copy("receiver_rc")
        return build_receiver_sampling_disclosure(
            receiver_mode=self.receiver_mode,
            eligible_receiver_count=self.total_eligible_receivers,
            evaluated_receiver_count=self.num_receivers,
            receiver_rc=rc,
        )

    def get_contiguous_day_ranges(self) -> List[Tuple[int, int, int]]:
        """Return `(day_id, start_frame, stop_frame)` for every day in the shard."""
        if self._day_ranges_cache is not None:
            return list(self._day_ranges_cache)
        d_ids = self._mmap_slice_copy("day_ids")
        if d_ids.size == 0:
            self._day_ranges_cache = []
            return []
        changes = np.flatnonzero(d_ids[1:] != d_ids[:-1]) + 1
        starts = [0, *changes.tolist()]
        stops = [*changes.tolist(), int(d_ids.size)]
        ranges = [(int(d_ids[s]), int(s), int(e)) for s, e in zip(starts, stops)]
        self._day_ranges_cache = ranges
        return list(ranges)

    @classmethod
    def write_from_trajectory(
        cls,
        shard_dir: Path | str,
        traj: FullDayTileTrajectory,
        *,
        contract: SplitContract,
    ) -> "ContiguousTileMmapShard":
        """Write shard atomically via `.tmp_shard_*` + `os.replace` (`P0-2`)."""
        out_dir = Path(shard_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)

        rec = contract.tiles[traj.tile_id]
        if rec.canonical_split != traj.canonical_split:
            raise RuntimeError(
                f"Role mismatch while writing shard for {traj.tile_id}: "
                f"{rec.canonical_split} vs {traj.canonical_split}"
            )
        if rec.source_core_sha256 != traj.source_core_sha256:
            raise RuntimeError(
                f"source_core_sha256 mismatch while writing shard for {traj.tile_id}"
            )

        tmp_dir = (
            out_dir.parent
            / f".tmp_shard_{out_dir.name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        )
        tmp_dir.mkdir(parents=True, exist_ok=False)

        try:
            data_map: Dict[str, np.ndarray] = {
                "day_ids": np.ascontiguousarray(traj.day_ids, dtype=np.int32),
                "timestamps_unix_s": np.ascontiguousarray(traj.timestamps_unix_s, dtype=np.int64),
                "sun_dirs": np.ascontiguousarray(traj.sun_dirs, dtype=np.float32),
                "azimuth_bin_idx": np.ascontiguousarray(traj.azimuth_bin_idx, dtype=np.int16),
                "solar_altitude_deg": np.ascontiguousarray(
                    traj.solar_altitude_deg, dtype=np.float32
                ),
                "solar_azimuth_deg": np.ascontiguousarray(
                    traj.solar_azimuth_deg, dtype=np.float32
                ),
                "receivers_xyz": np.ascontiguousarray(traj.receivers_xyz, dtype=np.float32),
                "receiver_rc": np.ascontiguousarray(traj.receiver_rc, dtype=np.int64),
                "truth": np.ascontiguousarray(traj.truth, dtype=np.int8),
                "valid_mask": np.ascontiguousarray(traj.valid_mask, dtype=bool),
                "boundary_distance_m": np.ascontiguousarray(
                    traj.boundary_distance_m, dtype=np.float32
                ),
                "boundary_distance_valid": np.ascontiguousarray(
                    traj.boundary_distance_valid, dtype=bool
                ),
                "local_features": np.ascontiguousarray(traj.local_features, dtype=np.float32),
                "static_fields_8hw": np.ascontiguousarray(
                    traj.static_fields_8hw, dtype=np.float32
                ),
                "geometry_known_hw": np.ascontiguousarray(traj.geometry_known_hw, dtype=bool),
                "az_rel_triangles": np.ascontiguousarray(
                    traj.az_rel_triangles, dtype=np.float32
                ),
                "az_opaque": np.ascontiguousarray(traj.az_opaque, dtype=bool),
                "az_mask": np.ascontiguousarray(traj.az_mask, dtype=bool),
                "az_source_ids": np.ascontiguousarray(
                    traj.az_source_ids.view(np.int64), dtype=np.int64
                ),
            }

            optional_arrays: List[str] = []
            if traj.ray_candidates:
                for name in RAY_ARRAYS:
                    data_map[name] = np.ascontiguousarray(getattr(traj,name))
                optional_arrays.extend(RAY_ARRAYS)
            if traj.response_5x5_bitpack is not None:
                data_map["response_5x5_bitpack"] = np.ascontiguousarray(
                    traj.response_5x5_bitpack, dtype=np.int64
                )
                optional_arrays.append("response_5x5_bitpack")

            hashes: Dict[str, str] = {}
            shapes: Dict[str, List[int]] = {}
            dtypes: Dict[str, str] = {}
            total_bytes = 0
            for name in (*cls.ARRAY_NAMES, *optional_arrays):
                arr = data_map[name]
                npy_path = tmp_dir / f"{name}.npy"
                mm = np.lib.format.open_memmap(
                    npy_path, mode="w+", dtype=arr.dtype, shape=arr.shape
                )
                try:
                    mm[...] = arr
                    mm.flush()
                finally:
                    raw_mm = getattr(mm, "_mmap", None)
                    if raw_mm is not None:
                        raw_mm.close()
                    del mm
                hashes[name] = compute_file_sha256(npy_path)
                shapes[name] = [int(d) for d in arr.shape]
                dtypes[name] = str(arr.dtype)
                total_bytes += int(npy_path.stat().st_size)

            rc_disclosure = traj.receiver_sampling_disclosure()
            provenance = {
                "schema": cls.SCHEMA_VERSION,
                "benchmark_contract": BENCHMARK_CONTRACT_ID,
                "feature_version": FEATURE_SCHEMA_VERSION,
                "split_sha256": contract.split_sha256,
                "manifest_sha256": contract.manifest_sha256,
                "tile_id": traj.tile_id,
                "canonical_split": traj.canonical_split,
                "spatial_group": traj.spatial_group,
                "source_core_sha256": traj.source_core_sha256,
                "mesh_provenance_sha256": traj.mesh_provenance_sha256,
                "week_indices": list(traj.week_indices),
                "week_schedule_sha256": compute_week_schedule_sha256(traj.week_indices),
                "window_ids": list(traj.window_ids),
                "receiver_mode": traj.receiver_mode,
                "total_eligible_receivers": int(traj.total_eligible_receivers),
                "receiver_support_sha256": rc_disclosure["receiver_support_sha256"],
                "max_candidates": int(traj.az_rel_triangles.shape[2]),
                "num_azimuth_bins": int(traj.az_rel_triangles.shape[1]),
                "bbox_margin_m": float(traj.bbox_margin_m),
                "num_frames": int(traj.sun_dirs.shape[0]),
                "num_receivers": int(traj.receivers_xyz.shape[0]),
                "point_frames": int(traj.point_frames),
                "meters_per_pixel": float(traj.meters_per_pixel),
                "solar_regimes": list(traj.solar_regimes),
                "optional_arrays": optional_arrays,
                "total_shard_bytes": total_bytes,
                "array_shapes": shapes,
                "array_dtypes": dtypes,
                "array_sha256": hashes,
            }
            provenance["provenance_digest_sha256"] = _compute_shard_provenance_digest(provenance)
            prov_path = tmp_dir / "SHARD_PROVENANCE.json"
            prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

            if out_dir.exists():
                backup_dir = (
                    out_dir.parent
                    / f".old_shard_{out_dir.name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
                )
                os.replace(out_dir, backup_dir)
                shutil.rmtree(backup_dir, ignore_errors=True)
            os.replace(tmp_dir, out_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        return cls(
            out_dir,
            expected_split_sha256=contract.split_sha256,
            expected_manifest_sha256=contract.manifest_sha256,
            expected_tile_id=traj.tile_id,
            expected_canonical_split=traj.canonical_split,
            expected_source_core_sha256=traj.source_core_sha256,
            expected_mesh_provenance_sha256=traj.mesh_provenance_sha256,
            expected_week_indices=traj.week_indices,
            expected_receiver_mode=traj.receiver_mode,
            expected_bbox_margin_m=traj.bbox_margin_m,
            verify_file_hashes=True,
        )

    @classmethod
    def write_streaming_from_reader(
        cls,
        shard_dir: Path | str,
        reader: FullDayTileReader,
        tile_id: str,
        *,
        week_indices: Sequence[int],
        receiver_mode: str = "sampled",
        max_receivers: int = 256,
        receiver_seed: int = 20261111,
        max_candidates: int = 32,
        num_azimuth_bins: int = 0,
        bbox_margin_m: float = DEFAULT_SCENE_HORIZON_M,
    ) -> "ContiguousTileMmapShard":
        """Stream week-by-week directly into `.npy` memmaps (`< 450 MiB` host RAM on `all_valid`)."""
        out_dir = Path(shard_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        contract = reader.contract

        static_ctx, iter_weeks = reader.prepare_tile_streaming_context(
            tile_id,
            week_indices=week_indices,
            receiver_mode=receiver_mode,
            max_receivers=max_receivers,
            max_candidates=max_candidates,
            num_azimuth_bins=num_azimuth_bins,
            bbox_margin_m=bbox_margin_m,
            receiver_seed=receiver_seed,
        )

        rec = contract.tiles[tile_id]
        if rec.canonical_split != static_ctx["canonical_split"]:
            raise RuntimeError(f"Role mismatch while streaming shard for {tile_id}")
        if rec.source_core_sha256 != static_ctx["source_core_sha256"]:
            raise RuntimeError(f"source_core_sha256 mismatch while streaming shard for {tile_id}")

        tmp_dir = (
            out_dir.parent
            / f".tmp_shard_{out_dir.name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        )
        tmp_dir.mkdir(parents=True, exist_ok=False)

        try:
            static_arrays: Dict[str, np.ndarray] = {
                "day_ids": np.ascontiguousarray(static_ctx["day_ids"], dtype=np.int32),
                "timestamps_unix_s": np.ascontiguousarray(
                    static_ctx["timestamps_unix_s"], dtype=np.int64
                ),
                "sun_dirs": np.ascontiguousarray(static_ctx["sun_dirs"], dtype=np.float32),
                "azimuth_bin_idx": np.ascontiguousarray(
                    static_ctx["azimuth_bin_idx"], dtype=np.int16
                ),
                "solar_altitude_deg": np.ascontiguousarray(
                    static_ctx["solar_altitude_deg"], dtype=np.float32
                ),
                "solar_azimuth_deg": np.ascontiguousarray(
                    static_ctx["solar_azimuth_deg"], dtype=np.float32
                ),
                "receivers_xyz": np.ascontiguousarray(
                    static_ctx["receivers_xyz"], dtype=np.float32
                ),
                "receiver_rc": np.ascontiguousarray(static_ctx["receiver_rc"], dtype=np.int64),
                "local_features": np.ascontiguousarray(
                    static_ctx["local_features"], dtype=np.float32
                ),
                "static_fields_8hw": np.ascontiguousarray(
                    static_ctx["static_fields_8hw"], dtype=np.float32
                ),
                "geometry_known_hw": np.ascontiguousarray(
                    static_ctx["geometry_known_hw"], dtype=bool
                ),
                "az_rel_triangles": np.ascontiguousarray(
                    static_ctx["az_rel_triangles"], dtype=np.float32
                ),
                "az_opaque": np.ascontiguousarray(static_ctx["az_opaque"], dtype=bool),
                "az_mask": np.ascontiguousarray(static_ctx["az_mask"], dtype=bool),
                "az_source_ids": np.ascontiguousarray(
                    static_ctx["az_source_ids"].view(np.int64), dtype=np.int64
                ),
            }

            hashes: Dict[str, str] = {}
            shapes: Dict[str, List[int]] = {}
            dtypes: Dict[str, str] = {}
            total_bytes = 0

            if not num_azimuth_bins:
                static_arrays.update({name:static_ctx.pop(name) for name in RAY_ARRAYS})
            for name, arr in static_arrays.items():
                npy_path = tmp_dir / f"{name}.npy"
                c_arr = np.ascontiguousarray(arr)
                hdr_b = _write_npy_header_bytes(c_arr.shape, c_arr.dtype)
                sha_s = hashlib.sha256()
                sha_s.update(hdr_b)
                with npy_path.open("wb") as fh_s:
                    fh_s.write(hdr_b)
                    flat_u8 = c_arr.view(np.uint8).reshape(-1)
                    step_b = 16 * 1024 * 1024
                    for off_b in range(0, int(flat_u8.size), step_b):
                        chunk_b = flat_u8[off_b : off_b + step_b].tobytes()
                        fh_s.write(chunk_b)
                        sha_s.update(chunk_b)
                hashes[name] = sha_s.hexdigest()
                shapes[name] = [int(d) for d in c_arr.shape]
                dtypes[name] = str(c_arr.dtype)
                total_bytes += int(npy_path.stat().st_size)

            # Release large candidate banks from RAM before streaming 52 weeks of labels
            max_cands_actual = int(static_ctx["az_rel_triangles"].shape[2])
            num_az_actual = int(static_ctx["az_rel_triangles"].shape[1])
            static_arrays.clear()
            static_ctx.pop("az_rel_triangles", None)
            static_ctx.pop("az_opaque", None)
            static_ctx.pop("az_mask", None)
            static_ctx.pop("az_source_ids", None)

            t_frames = int(static_ctx["num_frames"])
            n_recs = int(static_ctx["num_receivers"])
            has_bpack = bool(static_ctx["has_response_5x5_bitpack"])
            optional_arrays: List[str] = (["response_5x5_bitpack"] if has_bpack else []) + ([] if num_azimuth_bins else list(RAY_ARRAYS))

            time_specs: List[Tuple[str, Any]] = [
                ("truth", np.dtype(np.int8)),
                ("valid_mask", np.dtype(bool)),
                ("boundary_distance_m", np.dtype(np.float32)),
                ("boundary_distance_valid", np.dtype(bool)),
            ]
            if has_bpack:
                time_specs.append(("response_5x5_bitpack", np.dtype(np.int64)))

            open_fhs: Dict[str, Any] = {}
            open_shas: Dict[str, Any] = {}
            expected_cursor = 0
            try:
                for t_name, t_dt in time_specs:
                    hdr_b = _write_npy_header_bytes((t_frames, n_recs), t_dt)
                    fh_t = (tmp_dir / f"{t_name}.npy").open("wb")
                    fh_t.write(hdr_b)
                    sha_t = hashlib.sha256()
                    sha_t.update(hdr_b)
                    open_fhs[t_name] = fh_t
                    open_shas[t_name] = sha_t

                for f0, f1, y_fn, v_fn, b_dist_fn, b_val_fn, bpack_fn in iter_weeks():
                    if int(f0) != expected_cursor:
                        raise RuntimeError(
                            f"Non-contiguous week cursor in {tile_id}: f0={f0} != {expected_cursor}"
                        )
                    expected_cursor = int(f1)
                    week_items: List[Tuple[str, Any, np.ndarray]] = [
                        ("truth", np.int8, y_fn),
                        ("valid_mask", bool, v_fn),
                        ("boundary_distance_m", np.float32, b_dist_fn),
                        ("boundary_distance_valid", bool, b_val_fn),
                    ]
                    if has_bpack and bpack_fn is not None:
                        week_items.append(("response_5x5_bitpack", np.int64, bpack_fn))
                    for w_name, w_dt, w_arr in week_items:
                        raw_w = np.ascontiguousarray(w_arr, dtype=w_dt).tobytes()
                        open_fhs[w_name].write(raw_w)
                        open_shas[w_name].update(raw_w)
                if expected_cursor != t_frames:
                    raise RuntimeError(
                        f"Incomplete week stream in {tile_id}: wrote {expected_cursor} != {t_frames} frames"
                    )
                for fh_t in open_fhs.values():
                    fh_t.flush()
            finally:
                for fh_t in list(open_fhs.values()):
                    fh_t.close()
                open_fhs.clear()

            for t_name, t_dt in time_specs:
                npy_path = tmp_dir / f"{t_name}.npy"
                hashes[t_name] = open_shas[t_name].hexdigest()
                shapes[t_name] = [t_frames, n_recs]
                dtypes[t_name] = str(t_dt)
                total_bytes += int(npy_path.stat().st_size)

            rc_disclosure = build_receiver_sampling_disclosure(
                receiver_mode=str(static_ctx["receiver_mode"]),
                eligible_receiver_count=int(static_ctx["total_eligible_receivers"]),
                evaluated_receiver_count=n_recs,
                receiver_rc=static_ctx["receiver_rc"],
            )
            provenance = {
                "schema": cls.SCHEMA_VERSION,
                "benchmark_contract": BENCHMARK_CONTRACT_ID,
                "feature_version": FEATURE_SCHEMA_VERSION,
                "split_sha256": contract.split_sha256,
                "manifest_sha256": contract.manifest_sha256,
                "tile_id": tile_id,
                "canonical_split": static_ctx["canonical_split"],
                "spatial_group": static_ctx["spatial_group"],
                "source_core_sha256": static_ctx["source_core_sha256"],
                "mesh_provenance_sha256": static_ctx["mesh_provenance_sha256"],
                "week_indices": list(static_ctx["week_indices"]),
                "week_schedule_sha256": compute_week_schedule_sha256(static_ctx["week_indices"]),
                "window_ids": list(static_ctx["window_ids"]),
                "receiver_mode": static_ctx["receiver_mode"],
                "total_eligible_receivers": int(static_ctx["total_eligible_receivers"]),
                "receiver_support_sha256": rc_disclosure["receiver_support_sha256"],
                "max_candidates": max_cands_actual,
                "num_azimuth_bins": num_az_actual,
                "bbox_margin_m": float(static_ctx["bbox_margin_m"]),
                "num_frames": t_frames,
                "num_receivers": n_recs,
                "point_frames": int(t_frames * n_recs),
                "meters_per_pixel": float(static_ctx["meters_per_pixel"]),
                "solar_regimes": list(static_ctx["solar_regimes"]),
                "optional_arrays": optional_arrays,
                "total_shard_bytes": total_bytes,
                "array_shapes": shapes,
                "array_dtypes": dtypes,
                "array_sha256": hashes,
            }
            provenance["provenance_digest_sha256"] = _compute_shard_provenance_digest(provenance)
            prov_path = tmp_dir / "SHARD_PROVENANCE.json"
            prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

            if out_dir.exists():
                backup_dir = (
                    out_dir.parent
                    / f".old_shard_{out_dir.name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
                )
                os.replace(out_dir, backup_dir)
                shutil.rmtree(backup_dir, ignore_errors=True)
            os.replace(tmp_dir, out_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        return cls(
            out_dir,
            expected_split_sha256=contract.split_sha256,
            expected_manifest_sha256=contract.manifest_sha256,
            expected_tile_id=tile_id,
            expected_canonical_split=static_ctx["canonical_split"],
            expected_source_core_sha256=static_ctx["source_core_sha256"],
            expected_mesh_provenance_sha256=static_ctx["mesh_provenance_sha256"],
            expected_week_indices=static_ctx["week_indices"],
            expected_receiver_mode=static_ctx["receiver_mode"],
            expected_bbox_margin_m=float(static_ctx["bbox_margin_m"]),
            verify_file_hashes=True,
        )

    def _set_ray_bank(self, batch, receivers, *, candidate_stride=1,
                      candidate_provider=None, anchor_mask=None, next_phase=None,
                      candidate_mode="causal_rollout"):
        """Retrieve only planned anchor Suns; materialize no non-anchor BVH work."""
        if "scene_triangles_xyz" not in self.array_paths and candidate_provider is None:
            return
        from .ray_candidates import load_shared_scene, plan_anchor_frames
        provider = candidate_provider
        if provider is None:
            provider = load_shared_scene(
                tuple(self.array_paths[key] for key in RAY_ARRAYS[:3]),
                tuple(self.provenance["array_sha256"][key] for key in RAY_ARRAYS[:3]))
        total = len(batch["sun_dirs"])
        if anchor_mask is None:
            anchor_mask, next_phase = plan_anchor_frames(
                batch["day_ids"], batch["timestamps_unix_s"], candidate_stride,
                initial_phase=int(batch.get("intra_day_step_offset", 0)),
                carry_from_previous=bool(batch.get("carry_state_from_prev_chunk", False)),
                prev_frame_same_day=bool(batch.get("prev_frame_same_day", True)))
        anchor_mask = np.asarray(anchor_mask, dtype=bool)
        if anchor_mask.shape != (total,):
            raise ValueError("Candidate anchor plan must match the full chunk length")
        indices = np.flatnonzero(anchor_mask)
        own = self._mmap_slice_copy("receiver_source_triangle_ids", receivers)
        capacity = self.provenance["max_candidates"]
        if len(indices):
            values = provider.query(batch["receivers_xyz"].numpy(),
                                    batch["sun_dirs"].numpy()[indices],
                                    self.bbox_margin_m, capacity, own)
        else:
            # A single padding bin keeps compact-bank gather well-defined. No
            # provider call occurs, and rollout uses its previous anchor state.
            shape = (len(batch["receivers_xyz"]), 1, capacity)
            values = (np.zeros((*shape, 3, 3), np.float32),
                      np.zeros(shape, bool), np.zeros(shape, bool),
                      np.full(shape, -1, np.int64), np.zeros((shape[0], 0), np.int64))
        for key, value in zip(("az_rel_triangles", "az_opaque", "az_mask", "az_source_ids"), values[:4]):
            batch[key] = torch.from_numpy(value)
        # Non-anchor entries reference a past anchor only; they are never
        # independently encoded by the strided model.
        bins = np.maximum(np.cumsum(anchor_mask, dtype=np.int64) - 1, 0)
        batch["azimuth_bin_idx"] = torch.from_numpy(bins)
        batch["ray_candidate_count"] = torch.from_numpy(values[4].T.copy())
        batch["candidate_anchor_mask"] = torch.from_numpy(anchor_mask.copy())
        batch["candidate_anchor_next_phase"] = int(next_phase or 0)
        batch["candidate_anchor_stride"] = max(1, int(candidate_stride))
        batch["candidate_anchor_mode"] = str(candidate_mode)
        batch["candidate_query_frames"] = int(len(indices))
        batch["candidate_total_frames"] = int(total)
        batch["candidate_bvh_savings"] = 1.0 - len(indices) / float(max(total, 1))
        batch["candidate_geometry_identity"] = getattr(provider, "geometry_identity", None)

    @staticmethod
    def _candidate_plan_for_frames(day_ids, timestamps, frame_indices, stride, mode, schedule=None):
        from .ray_candidates import plan_anchor_schedule
        ids = np.asarray(frame_indices, dtype=np.int64)
        if schedule is None:
            schedule = plan_anchor_schedule(day_ids, timestamps, stride,
                                             offline_interp=mode == "offline_interp")
        full_mask, phases = schedule
        return full_mask[ids], int(phases[ids[-1]]) if len(ids) else 0

    def load_compact_batch(
        self,
        *,
        frame_slice: Optional[slice] = None,
        receiver_slice: Optional[slice] = None,
        include_candidate_bank: bool = True,
        candidate_stride: int = 1,
        candidate_provider: Any = None,
        candidate_mode: str = "causal_rollout",
        include_receiver_crops: bool = True,
        include_labels: bool = True,
        crop_size: int = 64,
        convgru_execution_mode: str = "full_tile_512_halo",
    ) -> Dict[str, Any]:
        """Slice contiguous `mmap` views into compact CPU tensors for a bounded `(frame, receiver)` block."""
        f_sl = slice(None) if frame_slice is None else frame_slice
        r_sl = slice(None) if receiver_slice is None else receiver_slice

        day_ids = self._mmap_slice_copy("day_ids", f_sl)
        ts = self._mmap_slice_copy("timestamps_unix_s", f_sl)
        same_day_adj = build_same_day_adjacent_mask(day_ids, ts)
        full_rc = self._mmap_slice_copy("receiver_rc")
        domain = TileLabelDomain(self.tile_id, full_rc) if include_labels else None
        rc_sub = full_rc[r_sl]
        known_hw = self._mmap_slice_copy("geometry_known_hw")
        obs_r = known_hw[rc_sub[:, 0], rc_sub[:, 1]].astype(bool, copy=False)
        gknn = self._get_global_knn_pack()

        out: Dict[str, Any] = {
            "tile_ids": [self.tile_id],
            "sun_dirs": torch.from_numpy(self._mmap_slice_copy("sun_dirs", f_sl)),
            "azimuth_bin_idx": torch.from_numpy(
                self._mmap_slice_copy("azimuth_bin_idx", f_sl).astype(np.int64)
            ),
            "same_day_adjacent_mask": torch.from_numpy(same_day_adj),
            "local_features": torch.from_numpy(self._mmap_slice_copy("local_features", r_sl)),
            "receivers_xyz": torch.from_numpy(self._mmap_slice_copy("receivers_xyz", r_sl)),
            "tile_label_domain": domain,
            "receiver_rc": torch.from_numpy(rc_sub),
            "meters_per_pixel": self.meters_per_pixel,
            "receiver_obs_mask": torch.from_numpy(np.ascontiguousarray(obs_r)),
            "global_knn_local_features": gknn["knn_local_features"][r_sl],
            "global_knn_rel_xyz": gknn["knn_rel_xyz"][r_sl],
            "global_knn_dist": gknn["knn_dist"][r_sl],
            "global_knn_obs_valid": gknn["knn_obs_valid"][r_sl],
            "solar_regimes": self.solar_regimes[f_sl],
            "day_ids": day_ids,
            "timestamps_unix_s": ts,
        }
        if include_labels:
            out["truth"] = torch.from_numpy(self._mmap_slice_copy("truth", (f_sl, r_sl)))
            out["valid_mask"] = torch.from_numpy(self._mmap_slice_copy("valid_mask", (f_sl, r_sl)))
            out["boundary_distance_m"] = torch.from_numpy(
                self._mmap_slice_copy("boundary_distance_m", (f_sl, r_sl))
            )
            out["boundary_distance_valid"] = torch.from_numpy(
                self._mmap_slice_copy("boundary_distance_valid", (f_sl, r_sl))
            )
            if self.has_response_5x5_bitpack:
                out["response_5x5_bitpack"] = torch.from_numpy(
                    self._mmap_slice_copy("response_5x5_bitpack", (f_sl, r_sl))
                )

        if include_candidate_bank:
            out["az_rel_triangles"] = torch.from_numpy(
                self._mmap_slice_copy("az_rel_triangles", r_sl)
            )
            out["az_opaque"] = torch.from_numpy(self._mmap_slice_copy("az_opaque", r_sl))
            out["az_mask"] = torch.from_numpy(self._mmap_slice_copy("az_mask", r_sl))
            out["az_source_ids"] = torch.from_numpy(
                self._mmap_slice_copy("az_source_ids", r_sl)
            )

        if include_candidate_bank:
            all_days = self._mmap_slice_copy("day_ids")
            all_times = self._mmap_slice_copy("timestamps_unix_s")
            plan, phase = self._candidate_plan_for_frames(
                all_days, all_times, np.arange(len(all_days))[f_sl], candidate_stride, candidate_mode)
            self._set_ray_bank(out, r_sl, candidate_stride=candidate_stride,
                               candidate_provider=candidate_provider, anchor_mask=plan, next_phase=phase, candidate_mode=candidate_mode)
        if include_receiver_crops:
            fields_8hw = self._mmap_slice_copy("static_fields_8hw")
            out["static_fields_8hw"] = torch.from_numpy(fields_8hw)
            out["geometry_known_hw"] = torch.from_numpy(known_hw)
            if convgru_execution_mode == "per_receiver_crop":
                crops = extract_exact_centered_receiver_crops(
                    fields_8hw, known_hw, rc_sub, crop_size=int(crop_size)
                )
                out["receiver_crops"] = torch.from_numpy(crops)

        return out

    def iter_day_batches(
        self,
        *,
        days_per_batch: int = 2,
        receiver_chunk_size: int = 256,
        physical_receiver_read: int = 0,
        model_type: str = "rind_full_geom",
        crop_size: int = 64,
        shuffle_days: bool = False,
        rng: Optional[np.random.Generator] = None,
        max_intra_day_frames: int = 0,
        convgru_execution_mode: str = "full_tile_512_halo",
        force_subchunk_full_tile_convgru: bool = False,
        include_labels: bool = True,
        candidate_stride: int = 1,
        candidate_provider: Any = None,
        candidate_mode: str = "causal_rollout",
    ) -> Iterator[Dict[str, Any]]:
        """Stream annual tile by bounded `(day_group, receiver_chunk)` windows (`P0-1`).

        - Skips reading `az_rel_triangles` for `pointwise_mlp` and `local_convgru_grid`!
        - For `local_convgru_grid`: defaults to `"full_tile_512_halo"` (`8 MiB` static grid, `<1s/step`
          across the full tile in ONE single `O(1)` grid pass per day across all `N` receivers), or
          extracts centered `receiver_crops` when `"per_receiver_crop"`.
        - When `max_intra_day_frames == 0` (default), groups whole days (`1..days_per_batch` days per
          batch), where recurrence resets cleanly across distinct sampling dates (`0` state truncation).
        - When `max_intra_day_frames > 0`, splits individual days into chronological sub-windows and
          marks `carry_state_from_prev_chunk=True` + `intra_day_step_offset` + `prev_frame_truth` /
          `prev_frame_valid_mask` so callers thread `(z_state, prev_prob)` across intra-day boundaries seamlessly!
        """
        need_cands = model_type.startswith(("rind_","canonical37_"))
        need_crops = model_type in ("local_convgru_grid", "full_tile_convgru")
        eff_convgru_mode = (
            "full_tile_512_halo"
            if model_type == "full_tile_convgru"
            or convgru_execution_mode in ("full_tile_512_halo", "full_tile_convgru")
            else convgru_execution_mode
        )

        day_ranges = self.get_contiguous_day_ranges()
        if not day_ranges:
            return

        if (
            need_crops
            and eff_convgru_mode == "full_tile_512_halo"
            and (not force_subchunk_full_tile_convgru)
        ):
            # Single O(1) full-grid pass gathers all receivers simultaneously (`0` repeated grid work!)
            r_step = max(1, self.num_receivers)
        else:
            r_step = (
                max(1, self.num_receivers)
                if int(receiver_chunk_size) <= 0
                else max(1, int(receiver_chunk_size))
            )
        d_step = max(1, int(days_per_batch))

        if shuffle_days and rng is not None and max_intra_day_frames <= 0:
            order = rng.permutation(len(day_ranges)).tolist()
            ordered_days = [day_ranges[i] for i in order]
        else:
            ordered_days = day_ranges

        # Cache lightweight 1D temporal schedules ONCE per shard iteration (`< 500 KB` for 18,808 frames)
        all_day_ids = self._mmap_slice_copy("day_ids")
        all_ts = self._mmap_slice_copy("timestamps_unix_s")
        all_suns = self._mmap_slice_copy("sun_dirs")
        all_azb = self._mmap_slice_copy("azimuth_bin_idx").astype(np.int64)
        from .ray_candidates import plan_anchor_schedule
        candidate_schedule = plan_anchor_schedule(all_day_ids, all_ts, candidate_stride,
            offline_interp=candidate_mode == "offline_interp") if need_cands else None

        cached_fields = self._mmap_slice_copy("static_fields_8hw") if need_crops else None
        cached_known = self._mmap_slice_copy("geometry_known_hw")
        cached_fields_t = torch.from_numpy(cached_fields) if cached_fields is not None else None
        cached_known_t = torch.from_numpy(cached_known) if need_crops else None
        all_rc = self._mmap_slice_copy("receiver_rc")
        domain = TileLabelDomain(self.tile_id, all_rc) if include_labels else None
        all_xyz = self._mmap_slice_copy("receivers_xyz")
        gknn = self._get_global_knn_pack() if need_cands else None

        def read_receivers(name, selection):
            slices = selection if isinstance(selection,tuple) else (selection,)
            start, stop = slices[-1].start, slices[-1].stop
            width = physical_receiver_read
            if width <= 0 or stop-start <= width:
                return self._mmap_slice_copy(name,selection)
            pieces = [self._mmap_slice_copy(name,slices[:-1]+(slice(r,min(r+width,stop)),))
                      for r in range(start,stop,width)]
            return np.concatenate(pieces,axis=len(slices)-1)

        for r0 in range(0, self.num_receivers, r_step):
            r1 = min(self.num_receivers, r0 + r_step)
            r_sl = slice(r0, r1)
            rc_sub = all_rc[r_sl]
            xyz_sub = torch.from_numpy(np.ascontiguousarray(all_xyz[r_sl]))
            obs_sub = torch.from_numpy(
                np.ascontiguousarray(cached_known[rc_sub[:, 0], rc_sub[:, 1]].astype(bool))
            )
            gknn_lf_sub = gknn["knn_local_features"][r_sl] if gknn is not None else None
            gknn_rel_sub = gknn["knn_rel_xyz"][r_sl] if gknn is not None else None
            gknn_dist_sub = gknn["knn_dist"][r_sl] if gknn is not None else None
            gknn_obs_sub = gknn["knn_obs_valid"][r_sl] if gknn is not None else None

            az_rel = (
                torch.from_numpy(read_receivers("az_rel_triangles", r_sl))
                if need_cands
                else None
            )
            az_op = (
                torch.from_numpy(read_receivers("az_opaque", r_sl))
                if need_cands
                else None
            )
            az_mk = (
                torch.from_numpy(read_receivers("az_mask", r_sl))
                if need_cands
                else None
            )
            az_sid = (
                torch.from_numpy(read_receivers("az_source_ids", r_sl))
                if need_cands
                else None
            )
            lf_sub = torch.from_numpy(read_receivers("local_features", r_sl))
            crops_sub = (
                torch.from_numpy(
                    extract_exact_centered_receiver_crops(
                        cached_fields, cached_known, rc_sub, crop_size=int(crop_size)
                    )
                )
                if (
                    need_crops
                    and cached_fields is not None
                    and eff_convgru_mode == "per_receiver_crop"
                )
                else None
            )

            if max_intra_day_frames > 0:
                sub_len = int(max_intra_day_frames)
                for _day_id, ds, de in ordered_days:
                    for fs in range(ds, de, sub_len):
                        fe = min(de, fs + sub_len)
                        f_sl = slice(fs, fe)
                        day_sub = np.array(all_day_ids[f_sl], copy=True)
                        ts_sub = np.array(all_ts[f_sl], copy=True)
                        same_day_adj = build_same_day_adjacent_mask(day_sub, ts_sub)
                        b_dict: Dict[str, Any] = {
                            "tile_ids": [self.tile_id],
                            "receiver_range": (r0, r1),
                            "frame_range": (fs, fe),
                            "carry_state_from_prev_chunk": bool(fs > ds),
                            "intra_day_step_offset": int(fs - ds),
                            "sun_dirs": torch.from_numpy(np.array(all_suns[f_sl], copy=True)),
                            "azimuth_bin_idx": torch.from_numpy(np.array(all_azb[f_sl], copy=True)),
                            "same_day_adjacent_mask": torch.from_numpy(same_day_adj),
                            "local_features": lf_sub,
                            "receivers_xyz": xyz_sub,
                            "tile_label_domain": domain,
                            "receiver_rc": torch.from_numpy(rc_sub),
                            "meters_per_pixel": self.meters_per_pixel,
                            "receiver_obs_mask": obs_sub,
                            "solar_regimes": self.solar_regimes[f_sl],
                            "day_ids": day_sub,
                            "timestamps_unix_s": ts_sub,
                        }
                        if include_labels:
                            b_dict["truth"] = torch.from_numpy(
                                read_receivers("truth", (f_sl, r_sl))
                            )
                            b_dict["valid_mask"] = torch.from_numpy(
                                read_receivers("valid_mask", (f_sl, r_sl))
                            )
                            b_dict["boundary_distance_m"] = torch.from_numpy(
                                read_receivers("boundary_distance_m", (f_sl, r_sl))
                            )
                            b_dict["boundary_distance_valid"] = torch.from_numpy(
                                read_receivers("boundary_distance_valid", (f_sl, r_sl))
                            )
                            if self.has_response_5x5_bitpack:
                                b_dict["response_5x5_bitpack"] = torch.from_numpy(
                                    read_receivers("response_5x5_bitpack", (f_sl, r_sl))
                                )
                        if fs > ds:
                            b_dict["prev_chunk_obs_mask"] = obs_sub
                            prev_sl = slice(fs - 1, fs)
                            b_dict["prev_frame_sun_dir"] = torch.from_numpy(
                                np.array(all_suns[fs - 1], copy=True)
                            )
                            b_dict["prev_frame_same_day"] = bool(
                                int(all_day_ids[fs]) == int(all_day_ids[fs - 1])
                                and int(all_ts[fs] - all_ts[fs - 1]) == 120
                            )
                            if include_labels:
                                b_dict["prev_frame_truth"] = torch.from_numpy(
                                    read_receivers("truth", (prev_sl, r_sl))[0]
                                )
                                b_dict["prev_frame_valid_mask"] = torch.from_numpy(
                                    read_receivers("valid_mask", (prev_sl, r_sl))[0]
                                )
                                b_dict["prev_frame_boundary_distance_m"] = torch.from_numpy(
                                    read_receivers("boundary_distance_m", (prev_sl, r_sl))[0]
                                )
                                b_dict["prev_frame_boundary_distance_valid"] = torch.from_numpy(
                                    read_receivers("boundary_distance_valid", (prev_sl, r_sl))[0]
                                )
                                if self.has_response_5x5_bitpack:
                                    b_dict["prev_frame_response_5x5_bitpack"] = torch.from_numpy(
                                        read_receivers("response_5x5_bitpack", (prev_sl, r_sl))[0]
                                    )
                        if need_cands:
                            b_dict["az_rel_triangles"] = az_rel
                            b_dict["az_opaque"] = az_op
                            b_dict["az_mask"] = az_mk
                            b_dict["az_source_ids"] = az_sid
                            if gknn_lf_sub is not None:
                                b_dict["global_knn_local_features"] = gknn_lf_sub
                                b_dict["global_knn_rel_xyz"] = gknn_rel_sub
                                b_dict["global_knn_dist"] = gknn_dist_sub
                                b_dict["global_knn_obs_valid"] = gknn_obs_sub
                        if need_crops and cached_fields_t is not None and cached_known_t is not None:
                            b_dict["static_fields_8hw"] = cached_fields_t
                            b_dict["geometry_known_hw"] = cached_known_t
                        if crops_sub is not None:
                            b_dict["receiver_crops"] = crops_sub
                        if need_cands:
                            plan, phase = self._candidate_plan_for_frames(
                                all_day_ids, all_ts, np.arange(fs, fe), candidate_stride, candidate_mode, candidate_schedule)
                            self._set_ray_bank(b_dict, r_sl, candidate_stride=candidate_stride,
                                               candidate_provider=candidate_provider,
                                               anchor_mask=plan, next_phase=phase, candidate_mode=candidate_mode)
                        yield b_dict
                continue

            # Whole-day batching (`d_step` days per batch)
            for d_idx in range(0, len(ordered_days), d_step):
                grp = ordered_days[d_idx : d_idx + d_step]
                is_contiguous = all(
                    grp[k + 1][1] == grp[k][2] for k in range(len(grp) - 1)
                )
                if is_contiguous:
                    fs, fe = grp[0][1], grp[-1][2]
                    f_sl = slice(fs, fe)
                    day_sub = np.array(all_day_ids[f_sl], copy=True)
                    ts_sub = np.array(all_ts[f_sl], copy=True)
                    suns_sub = np.array(all_suns[f_sl], copy=True)
                    azb_sub = np.array(all_azb[f_sl], copy=True)
                    if include_labels:
                        truth_sub = read_receivers("truth", (f_sl, r_sl))
                        val_sub = read_receivers("valid_mask", (f_sl, r_sl))
                        bd_sub = read_receivers("boundary_distance_m", (f_sl, r_sl))
                        bv_sub = read_receivers("boundary_distance_valid", (f_sl, r_sl))
                        bpack_sub = (
                            read_receivers("response_5x5_bitpack", (f_sl, r_sl))
                            if self.has_response_5x5_bitpack
                            else None
                        )
                    else:
                        truth_sub = val_sub = bd_sub = bv_sub = bpack_sub = None
                    regs_sub = self.solar_regimes[f_sl]
                else:
                    day_sub = np.concatenate([all_day_ids[s:e] for _, s, e in grp], axis=0)
                    ts_sub = np.concatenate([all_ts[s:e] for _, s, e in grp], axis=0)
                    suns_sub = np.concatenate([all_suns[s:e] for _, s, e in grp], axis=0)
                    azb_sub = np.concatenate([all_azb[s:e] for _, s, e in grp], axis=0)
                    if include_labels:
                        truth_sub = np.concatenate(
                            [read_receivers("truth", (slice(s, e), r_sl)) for _, s, e in grp],
                            axis=0,
                        )
                        val_sub = np.concatenate(
                            [
                                read_receivers("valid_mask", (slice(s, e), r_sl))
                                for _, s, e in grp
                            ],
                            axis=0,
                        )
                        bd_sub = np.concatenate(
                            [
                                read_receivers("boundary_distance_m", (slice(s, e), r_sl))
                                for _, s, e in grp
                            ],
                            axis=0,
                        )
                        bv_sub = np.concatenate(
                            [
                                read_receivers("boundary_distance_valid", (slice(s, e), r_sl))
                                for _, s, e in grp
                            ],
                            axis=0,
                        )
                        bpack_sub = (
                            np.concatenate(
                                [
                                    read_receivers("response_5x5_bitpack", (slice(s, e), r_sl))
                                    for _, s, e in grp
                                ],
                                axis=0,
                            )
                            if self.has_response_5x5_bitpack
                            else None
                        )
                    else:
                        truth_sub = val_sub = bd_sub = bv_sub = bpack_sub = None
                    regs_list: List[str] = []
                    for _, s, e in grp:
                        regs_list.extend(self.solar_regimes[s:e])
                    regs_sub = tuple(regs_list)

                same_day_adj = build_same_day_adjacent_mask(day_sub, ts_sub)
                b_dict = {
                    "tile_ids": [self.tile_id],
                    "receiver_range": (r0, r1),
                    "carry_state_from_prev_chunk": False,
                    "intra_day_step_offset": 0,
                    "sun_dirs": torch.from_numpy(suns_sub),
                    "azimuth_bin_idx": torch.from_numpy(azb_sub),
                    "same_day_adjacent_mask": torch.from_numpy(same_day_adj),
                    "local_features": lf_sub,
                    "receivers_xyz": xyz_sub,
                    "tile_label_domain": domain,
                    "receiver_rc": torch.from_numpy(rc_sub),
                    "meters_per_pixel": self.meters_per_pixel,
                    "receiver_obs_mask": obs_sub,
                    "solar_regimes": regs_sub,
                    "day_ids": day_sub,
                    "timestamps_unix_s": ts_sub,
                }
                if is_contiguous:
                    b_dict["frame_range"] = (fs, fe)
                if include_labels and truth_sub is not None:
                    b_dict["truth"] = torch.from_numpy(truth_sub)
                    b_dict["valid_mask"] = torch.from_numpy(val_sub)
                    b_dict["boundary_distance_m"] = torch.from_numpy(bd_sub)
                    b_dict["boundary_distance_valid"] = torch.from_numpy(bv_sub)
                    if bpack_sub is not None:
                        b_dict["response_5x5_bitpack"] = torch.from_numpy(bpack_sub)
                if need_cands:
                    b_dict["az_rel_triangles"] = az_rel
                    b_dict["az_opaque"] = az_op
                    b_dict["az_mask"] = az_mk
                    b_dict["az_source_ids"] = az_sid
                    if gknn_lf_sub is not None:
                        b_dict["global_knn_local_features"] = gknn_lf_sub
                        b_dict["global_knn_rel_xyz"] = gknn_rel_sub
                        b_dict["global_knn_dist"] = gknn_dist_sub
                        b_dict["global_knn_obs_valid"] = gknn_obs_sub
                if need_crops and cached_fields_t is not None and cached_known_t is not None:
                    b_dict["static_fields_8hw"] = cached_fields_t
                    b_dict["geometry_known_hw"] = cached_known_t
                if crops_sub is not None:
                    b_dict["receiver_crops"] = crops_sub
                if need_cands:
                    selected_frames = np.concatenate([np.arange(s, e) for _, s, e in grp])
                    plan, phase = self._candidate_plan_for_frames(
                        all_day_ids, all_ts, selected_frames, candidate_stride, candidate_mode, candidate_schedule)
                    self._set_ray_bank(b_dict, r_sl, candidate_stride=candidate_stride,
                                       candidate_provider=candidate_provider,
                                       anchor_mask=plan, next_phase=phase, candidate_mode=candidate_mode)
                yield b_dict


def _build_shard_identity_tag(
    *,
    contract: SplitContract,
    week_indices: Sequence[int],
    receiver_mode: str,
    max_receivers: int,
    receiver_seed: int,
    max_candidates: int,
    num_azimuth_bins: int,
    bbox_margin_m: float,
    mesh_provenance_sha256: str = "",
) -> str:
    w_sha = compute_week_schedule_sha256(week_indices)[:10]
    ident_str = (
        f"candidate_source_v5|{FEATURE_SCHEMA_VERSION}|{contract.split_sha256}|{contract.manifest_sha256}|"
        f"{mesh_provenance_sha256}|{w_sha}|{receiver_mode}|{int(max_receivers)}|{int(receiver_seed)}|"
        f"{int(max_candidates)}|{int(num_azimuth_bins)}|{float(bbox_margin_m):.1f}"
    )
    digest12 = hashlib.sha256(ident_str.encode("utf-8")).hexdigest()[:12]
    r_lbl = "all" if receiver_mode == "all_valid" else f"r{int(max_receivers)}"
    return f"candidate_source_v5_w{len(week_indices)}_{r_lbl}_k{int(max_candidates)}_az{int(num_azimuth_bins)}_m{int(round(bbox_margin_m))}_{digest12}"


def build_or_open_single_tile_shard(
    cache_root: Path | str,
    reader: FullDayTileReader,
    tile_id: str,
    *,
    week_indices: Sequence[int],
    receiver_mode: str = "sampled",
    max_receivers: int = 256,
    receiver_seed: int = 20261111,
    max_candidates: int = 32,
    num_azimuth_bins: int = 0,
    bbox_margin_m: float = DEFAULT_SCENE_HORIZON_M,
) -> ContiguousTileMmapShard:
    """Build or open a single verified `ContiguousTileMmapShard` under a per-shard inter-process lock (`P2-2`)."""
    root = Path(cache_root)
    rec = reader.contract.assert_tile_readable(tile_id)
    portable = getattr(reader, "portable_index", None)
    if portable is not None:
        # The index SHA must be pinned out-of-band; self-hashes alone cannot
        # authenticate a rewritten provenance document or establish origin.
        entry = portable.get("tiles", {}).get(tile_id)
        if not isinstance(entry, dict):
            raise RuntimeError(f"Portable index has no tile {tile_id}")
        rel = Path(entry.get("path", ""))
        if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 2 or rel.name != tile_id:
            raise RuntimeError("Unsafe portable shard path")
        root_resolved = root.resolve(strict=True)
        s_dir = root_resolved / rel
        if any(part.is_symlink() for part in (s_dir.parent, s_dir)) or not s_dir.resolve(strict=True).is_relative_to(root_resolved):
            raise RuntimeError("Portable shard must remain inside cache root without symlinks")
        prov_path = s_dir / "SHARD_PROVENANCE.json"
        if prov_path.is_symlink() or compute_file_sha256(prov_path) != entry.get("provenance_sha256"):
            raise RuntimeError("Portable provenance SHA-256 mismatch")
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        mesh_sha = str(prov.get("mesh_provenance_sha256", ""))
        if len(mesh_sha) != 64 or any(c not in "0123456789abcdef" for c in mesh_sha):
            raise RuntimeError("Portable mesh provenance SHA-256 missing or malformed")
        tag = _build_shard_identity_tag(contract=reader.contract, week_indices=week_indices,
            receiver_mode=receiver_mode, max_receivers=max_receivers, receiver_seed=receiver_seed,
            max_candidates=max_candidates, num_azimuth_bins=num_azimuth_bins,
            bbox_margin_m=bbox_margin_m, mesh_provenance_sha256=mesh_sha)
        if rel.parts[0] != tag:
            raise RuntimeError("Portable shard identity does not match requested configuration")
        if any(p.is_symlink() for p in s_dir.iterdir()):
            raise RuntimeError("Portable shard contains a symlink")
        eff_mode = "all_valid" if receiver_mode == "all_valid" or max_receivers <= 0 else "sampled"
        shard = ContiguousTileMmapShard(s_dir,
            expected_split_sha256=reader.contract.split_sha256,
            expected_manifest_sha256=reader.contract.manifest_sha256,
            expected_tile_id=tile_id, expected_canonical_split=rec.canonical_split,
            expected_source_core_sha256=rec.source_core_sha256,
            expected_mesh_provenance_sha256=mesh_sha, expected_week_indices=week_indices,
            expected_receiver_mode=eff_mode, expected_bbox_margin_m=bbox_margin_m,
            expected_optional_arrays=("response_5x5_bitpack",) + (() if num_azimuth_bins else RAY_ARRAYS),
            expected_provenance_file_sha256=entry["provenance_sha256"], verify_file_hashes=True)
        if shard.spatial_group != rec.spatial_group or shard.provenance["max_candidates"] != max_candidates or shard.provenance["num_azimuth_bins"] != max(1, num_azimuth_bins):
            raise RuntimeError("Portable shard geometry configuration mismatch")
        return shard
    root.mkdir(parents=True, exist_ok=True)
    mesh_sha = reader.get_mesh_provenance_sha256_for_tile(tile_id)
    tag = _build_shard_identity_tag(
        contract=reader.contract,
        week_indices=week_indices,
        receiver_mode=receiver_mode,
        max_receivers=max_receivers,
        receiver_seed=receiver_seed,
        max_candidates=max_candidates,
        num_azimuth_bins=num_azimuth_bins,
        bbox_margin_m=bbox_margin_m,
        mesh_provenance_sha256=mesh_sha,
    )
    s_dir = root / tag / tile_id
    prov_file = s_dir / "SHARD_PROVENANCE.json"
    eff_mode = (
        "all_valid" if (receiver_mode == "all_valid" or max_receivers <= 0) else "sampled"
    )

    def _try_open_existing() -> Optional[ContiguousTileMmapShard]:
        if not prov_file.exists():
            return None
        shard = ContiguousTileMmapShard(
            s_dir,
            expected_split_sha256=reader.contract.split_sha256,
            expected_manifest_sha256=reader.contract.manifest_sha256,
            expected_tile_id=tile_id,
            expected_canonical_split=rec.canonical_split,
            expected_source_core_sha256=rec.source_core_sha256,
            expected_mesh_provenance_sha256=mesh_sha,
            expected_week_indices=week_indices,
            expected_bbox_margin_m=bbox_margin_m,
            expected_optional_arrays=("response_5x5_bitpack",) + (() if num_azimuth_bins else RAY_ARRAYS),
            verify_file_hashes=True,
        )
        if eff_mode == "all_valid" and shard.receiver_mode != "all_valid":
            return None
        return shard

    # Fast path when shard is already published; otherwise serialize build under per-shard lock!
    lock_path = s_dir.parent / f".{tile_id}.shard.lock"
    with exclusive_file_lock(lock_path):
        existing = _try_open_existing()
        if existing is not None:
            return existing
        return ContiguousTileMmapShard.write_streaming_from_reader(
            s_dir,
            reader,
            tile_id,
            week_indices=week_indices,
            receiver_mode=receiver_mode,
            max_receivers=max_receivers,
            receiver_seed=receiver_seed,
            max_candidates=max_candidates,
            num_azimuth_bins=num_azimuth_bins,
            bbox_margin_m=bbox_margin_m,
        )


def build_or_open_mmap_cache(
    cache_root: Path | str,
    reader: FullDayTileReader,
    tile_ids: Sequence[str],
    *,
    week_indices: Sequence[int],
    receiver_mode: str = "sampled",
    max_receivers: int = 256,
    receiver_seed: int = 20261111,
    max_candidates: int = 32,
    num_azimuth_bins: int = 0,
    bbox_margin_m: float = DEFAULT_SCENE_HORIZON_M,
) -> Dict[str, ContiguousTileMmapShard]:
    """Build or open verified `ContiguousTileMmapShard` handles (each handle holds only path + metadata)."""
    shards: Dict[str, ContiguousTileMmapShard] = {}
    for tid in tile_ids:
        reader.cache.check_ram_guard(f"shard_prep:{tid}")
        shards[tid] = build_or_open_single_tile_shard(
            cache_root,
            reader,
            tid,
            week_indices=week_indices,
            receiver_mode=receiver_mode,
            max_receivers=max_receivers,
            receiver_seed=receiver_seed,
            max_candidates=max_candidates,
            num_azimuth_bins=num_azimuth_bins,
            bbox_margin_m=bbox_margin_m,
        )
    return shards

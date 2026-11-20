"""True Sunrise-to-Sunset Full-Day Streaming Reader & Full-Scene 4km Geometry Builder.

Implements the canonical `MANHATTAN_4445_ANNUAL_18808_V1` data pipeline:
- Reads full 52-week sunrise-to-sunset trajectories (`18,808` frames/tile, `272-448` frames/day
  at exact `120s` intra-day cadence) across all storage layouts (`shadow-jepa.stream.v1`
  `MANIFEST.json` + `DSFCH2` Deflate/LZ4 hybrid containers, `npy`, `npz`).
- Enforces payload provenance (`tile_id`, `source_core_sha256`, `STATIC_CASES.npz` SHA-256,
  mandatory `MODEL_GEOMETRY.npz` SHA-256, `condition` SHA-256, contiguous `chunk` SHA-256,
  `native_origin_policy` coordinate contract, unit upward `sun_dirs`, and raw `states in {1,2,3}`).
- Loads full `4 km` (`bbox_margin_m = 4000.0`) `DS3MESH` scene triangles with chunk SHA-256
  verification, `opaque` semantics, and deterministic `uint64` `source_triangle_ids`.
- Computes exact centered per-receiver spatial patches `(N, 9, P, P)` without coordinate clamping
  and exact planar shadow-boundary distances (`boundary_distance_m`, `boundary_distance_valid`).
- Enforces `<= 4.0 GiB` LRU host cache + `HostResourceGuard` (`< 95%` real host physical RAM).
"""

from __future__ import annotations

from .label_budget import TileLabelDomain

from collections import OrderedDict
from datetime import datetime, timezone
import dataclasses
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
import zlib

import numpy as np

from . import BENCHMARK_CONTRACT_ID, FEATURE_SCHEMA_VERSION
from .metrics import (
    SOLAR_REGIME_ORDER,
    build_receiver_sampling_disclosure,
    build_same_day_adjacent_mask,
    classify_solar_regime,
    compute_planar_shadow_boundary_distance_grid,
)
from .split_contract import (
    ADDITIONAL_FIT_HOLDOUT,
    EXPECTED_ANNUAL_FRAMES_PER_TILE,
    EXPECTED_ANNUAL_WEEKS_PER_TILE,
    SplitContract,
    TileSplitRecord,
    compute_file_sha256,
)
from .system_guard import HostResourceGuard, sample_host_resources

MAX_DEFAULT_HOST_CACHE_BYTES = int(3.0 * (1024**3))  # Strictly <= 4.0 GiB
DEFAULT_SCENE_HORIZON_M: float = 4000.0
CANONICAL_4SEASON_WEEK_INDICES: Tuple[int, ...] = (0, 13, 26, 39)
STRATIFIED_13WEEK_INDICES: Tuple[int, ...] = tuple(range(0, 52, 4))
ANNUAL_52WEEK_INDICES: Tuple[int, ...] = tuple(range(52))


def physical_oracle_sunlit(*args, **kwargs):
    """Load exact ray/triangle physics only when an explicit Oracle is requested."""
    from .inverse_design import exact_moller_trumbore_trajectory_sunlit_f64
    return exact_moller_trumbore_trajectory_sunlit_f64(*args, **kwargs)


def get_system_ram_usage_percent() -> float:
    """Return real host physical RAM usage percentage via `system_guard` (fail-closed)."""
    snap = sample_host_resources(stage="ram_probe")
    return float(snap.physical_used_percent)


def translate_storage_path(raw_path: str | Path) -> Path:
    """Translate Windows `G:\\...` <-> WSL `/mnt/g/...` paths seamlessly based on current OS."""
    s = str(raw_path).strip()
    if not s:
        return Path(".")
    if os.name == "nt":
        if s.startswith("/mnt/") and len(s) > 6 and s[6] == "/":
            drive = s[5].upper()
            rest = s[7:].replace("/", "\\")
            return Path(f"{drive}:\\{rest}")
        return Path(s)
    if len(s) >= 3 and s[1] == ":" and s[2] in ("\\", "/"):
        drive = s[0].lower()
        rest = s[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(s.replace("\\", "/"))


class BoundedHostByteCache:
    """Strict LRU host RAM cache (`<= 4.0 GiB`) wired to `HostResourceGuard` (`< 95%` physical RAM)."""

    def __init__(
        self,
        max_bytes: int = MAX_DEFAULT_HOST_CACHE_BYTES,
        max_ram_percent: float = 92.0,
        resource_guard: Optional[HostResourceGuard] = None,
    ) -> None:
        if max_bytes <= 0 or max_bytes > 4 * (1024**3):
            raise ValueError(f"max_bytes must be in (0, 4 GiB], got {max_bytes}")
        self.max_bytes = int(max_bytes)
        self.max_ram_percent = float(max_ram_percent)
        self.current_bytes = 0
        self.peak_bytes = 0
        self._items: OrderedDict[Any, Tuple[Any, int]] = OrderedDict()
        self.guard = resource_guard or HostResourceGuard(max_physical_ram_percent=max_ram_percent)

    @staticmethod
    def _estimate_nbytes(obj: Any) -> int:
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
        if isinstance(obj, Mapping):
            return sum(BoundedHostByteCache._estimate_nbytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(BoundedHostByteCache._estimate_nbytes(v) for v in obj)
        return 64

    def clear(self) -> None:
        self._items.clear()
        self.current_bytes = 0
        gc.collect()

    def check_ram_guard(self, stage: str = "cache_check") -> None:
        self.guard.check(stage, cleanup_callbacks=[self.clear])

    def get_or_load(self, key: Any, loader_fn: Any) -> Any:
        self.check_ram_guard("cache_get_or_load")
        if key in self._items:
            val, sz = self._items.pop(key)
            self._items[key] = (val, sz)
            return val
        val = loader_fn()
        sz = max(self._estimate_nbytes(val), 1)
        if sz <= self.max_bytes:
            while self._items and (self.current_bytes + sz > self.max_bytes):
                _, (_, evicted_sz) = self._items.popitem(last=False)
                self.current_bytes -= evicted_sz
            self._items[key] = (val, sz)
            self.current_bytes += sz
            self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        return val


def decode_dsfch2_container_cpu(
    chunk_path: Path,
    *,
    expected_frames: int,
    grid_shape: Tuple[int, int] = (512, 512),
    verify_compressed_sha256: bool = True,
    decode_reasons: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode `DSFCH2` (`Deflate` or `LZ4`, 64KiB zero/literal bitmap + `xor_prev`) on CPU.

    Returns:
        states: `(F, H, W)` uint8 strictly validated in `{1=SHADOW, 2=SUN, 3=UNKNOWN}`
        reasons: `(F, H, W)` uint16
    """
    raw = chunk_path.read_bytes()
    if len(raw) < 16:
        raise ValueError(f"Truncated DSFCH2 chunk: {chunk_path}")
    magic, header_len = struct.unpack("<8sQ", raw[:16])
    if magic != b"DSFCH2\x00\x00":
        raise ValueError(f"Invalid DSFCH2 magic {magic!r} in {chunk_path}")
    header_end = 16 + int(header_len)
    header = json.loads(raw[16:header_end].decode("utf-8"))
    compressed = memoryview(raw)[header_end:]

    if "compressed_bytes" in header and len(compressed) != int(header["compressed_bytes"]):
        raise ValueError(
            f"Compressed byte length mismatch in {chunk_path}: {len(compressed)} != {header['compressed_bytes']}"
        )
    if verify_compressed_sha256 and "compressed_sha256" in header:
        actual_sha = hashlib.sha256(compressed).hexdigest()
        if actual_sha != header["compressed_sha256"]:
            raise ValueError(f"DSFCH2 compressed SHA-256 mismatch in {chunk_path}")

    algo = header.get("codec", {}).get("algorithm", "Deflate")
    if algo == "Deflate":
        payload_bytes = zlib.decompress(compressed, -zlib.MAX_WBITS)
    elif algo == "LZ4":
        import lz4.block  # type: ignore

        if "segments" in header:
            parts: List[bytes] = []
            offset = 0
            for seg in header["segments"]:
                c_len = int(seg["compressed_bytes"])
                u_len = int(seg["uncompressed_bytes"])
                parts.append(
                    lz4.block.decompress(
                        bytes(compressed[offset : offset + c_len]), uncompressed_size=u_len
                    )
                )
                offset += c_len
            payload_bytes = b"".join(parts)
        else:
            payload_bytes = lz4.block.decompress(
                bytes(compressed), uncompressed_size=int(header["payload_bytes"])
            )
    else:
        raise ValueError(f"Unsupported DSFCH2 compression algorithm {algo!r} in {chunk_path}")

    if "payload_bytes" in header and len(payload_bytes) != int(header["payload_bytes"]):
        raise ValueError(
            f"DSFCH2 uncompressed byte count mismatch: {len(payload_bytes)} != {header['payload_bytes']}"
        )

    h, w = int(grid_shape[0]), int(grid_shape[1])
    states_out = np.empty((expected_frames, h, w), dtype=np.uint8)
    reasons_out = np.zeros((expected_frames, h, w), dtype=np.uint16)

    reconstructed_raw: Dict[str, np.ndarray] = {}
    cursor = 0
    block_bytes = 65536
    rows = header["fields"]
    row_by_name = {str(r["name"]): r for r in rows}
    needed_names: Set[str] = set()
    for r in rows:
        nm = str(r["name"])
        if nm.startswith("states_") or (decode_reasons and nm.startswith("reasons_")):
            cur_nm: Optional[str] = nm
            while cur_nm is not None and cur_nm not in needed_names:
                needed_names.add(cur_nm)
                prev_nm = row_by_name[cur_nm].get("previous")
                cur_nm = str(prev_nm) if prev_nm is not None else None

    seen_states = 0
    for row in rows:
        name = str(row["name"])
        bc = int(row["block_count"])
        lit_blocks = int(row["literal_blocks"])
        n_bytes = int(row["bytes"])
        bm_len = (bc + 7) // 8
        lit_total = lit_blocks * block_bytes

        if name not in needed_names:
            cursor += bm_len + lit_total
            continue

        bm_bits = np.unpackbits(
            np.frombuffer(payload_bytes, dtype=np.uint8, count=bm_len, offset=cursor),
            bitorder="big",
        )[:bc].astype(bool)
        cursor += bm_len

        lit_buf = np.frombuffer(payload_bytes, dtype=np.uint8, count=lit_total, offset=cursor)
        cursor += lit_total

        padded = np.zeros((bc, block_bytes), dtype=np.uint8)
        if lit_blocks > 0:
            padded[bm_bits] = lit_buf.reshape(lit_blocks, block_bytes)
        transformed = padded.reshape(-1)[:n_bytes]

        enc = row.get("encoding", "base")
        if enc == "xor_prev":
            prev_bytes = reconstructed_raw[str(row["previous"])]
            restored = np.bitwise_xor(transformed, prev_bytes)
        elif enc == "base":
            restored = transformed.copy()
        else:
            raise ValueError(f"Unknown field encoding {enc!r} in {chunk_path}")

        reconstructed_raw[name] = restored
        if name.startswith("states_"):
            f_local = int(name.split("_")[1])
            if 0 <= f_local < expected_frames:
                states_out[f_local] = restored.view(np.dtype(row["dtype"])).reshape(h, w)
                seen_states += 1
        elif decode_reasons and name.startswith("reasons_"):
            f_local = int(name.split("_")[1])
            if 0 <= f_local < expected_frames:
                reasons_out[f_local] = restored.view(np.dtype(row["dtype"])).reshape(h, w)

    if seen_states != expected_frames:
        raise ValueError(
            f"DSFCH2 decoded {seen_states} state frames != expected {expected_frames} in {chunk_path}"
        )

    # Validate raw state domain BEFORE any downstream receiver eligibility masking (`P1-8`)
    if not bool(np.all((states_out >= 1) & (states_out <= 3))):
        bad_vals = sorted(int(v) for v in np.unique(states_out) if v not in (1, 2, 3))
        raise ValueError(
            f"Corrupted raw DSFCH2 state codes {bad_vals} outside {{1,2,3}} in {chunk_path}"
        )
    if decode_reasons and bool(np.any((states_out != 3) & (reasons_out != 0))):
        raise ValueError(
            f"State/reason contract violation in {chunk_path}: non-UNKNOWN state has nonzero reason."
        )

    return states_out, reasons_out


def _validate_and_cast_raw_state_array(raw_states: Any, source_label: str) -> np.ndarray:
    """Validate raw state codes on their ORIGINAL dtype BEFORE casting to `np.uint8` (`P0-3`/`P1`).

    Rejects `bool`, `NaN`/`Inf`, non-integer floats (e.g. `1.9`), and out-of-range ints (e.g. `257`, `-255`, `0`, `4`)
    that would otherwise wrap or truncate silently into `{1, 2, 3}` under `astype(np.uint8)`.
    """
    arr = np.asarray(raw_states)
    if arr.dtype.kind == "b":
        raise ValueError(
            f"Boolean array is invalid for raw state codes in {source_label}; expected integer codes in {{1,2,3}}."
        )
    if arr.dtype.kind in ("f", "c"):
        if arr.dtype.kind == "c" or (not bool(np.all(np.isfinite(arr)))):
            raise ValueError(f"Non-finite or complex raw state values found in {source_label}.")
        if not bool(np.all(arr == np.floor(arr))):
            raise ValueError(f"Non-integer floating raw state values found in {source_label}.")
    elif arr.dtype.kind not in ("i", "u"):
        raise ValueError(f"Unsupported raw state dtype {arr.dtype} in {source_label}.")

    in_domain = (arr >= 1) & (arr <= 3)
    if not bool(np.all(in_domain)):
        bad_sample = arr[~in_domain].reshape(-1)[:8].tolist()
        raise ValueError(
            f"Invalid raw state codes {bad_sample} outside {{1,2,3}} (dtype={arr.dtype}) in {source_label}"
        )
    return np.ascontiguousarray(arr, dtype=np.uint8)


def encode_dsfch2_container_bytes(states: np.ndarray, reasons: Optional[np.ndarray] = None) -> bytes:
    """Encode `(F, H, W)` uint8 states (`1=SHADOW, 2=SUN, 3=UNKNOWN`) into exact `DSFCH2` Deflate container."""
    st = _validate_and_cast_raw_state_array(states, "encode_dsfch2_container_bytes")
    if st.ndim != 3:
        raise ValueError(f"states must be (F, H, W), got {st.shape}")
    f_cnt, h, w = st.shape
    rs = (
        np.zeros((f_cnt, h, w), dtype=np.uint16)
        if reasons is None
        else np.asarray(reasons, dtype=np.uint16)
    )

    block_bytes = 65536
    fields_meta: List[Dict[str, Any]] = []
    payload_parts: List[bytes] = []

    def _pack_series(prefix: str, series: np.ndarray, dtype_str: str) -> None:
        prev_raw: Optional[np.ndarray] = None
        prev_name: Optional[str] = None
        for idx in range(f_cnt):
            name = f"{prefix}_{idx:03d}"
            cur_raw = np.ascontiguousarray(series[idx]).view(np.uint8).reshape(-1)
            n_b = int(cur_raw.size)
            if prev_raw is None:
                diff = cur_raw
                enc = "base"
            else:
                diff = np.bitwise_xor(cur_raw, prev_raw)
                enc = "xor_prev"

            bc = (n_b + block_bytes - 1) // block_bytes
            padded = np.zeros((bc * block_bytes,), dtype=np.uint8)
            padded[:n_b] = diff
            blocks = padded.reshape(bc, block_bytes)
            lit_mask = np.any(blocks != 0, axis=1)
            lit_blocks = int(np.sum(lit_mask))

            bm_packed = np.packbits(lit_mask.astype(np.uint8), bitorder="big")
            payload_parts.append(bm_packed.tobytes())
            if lit_blocks > 0:
                payload_parts.append(blocks[lit_mask].tobytes())

            entry: Dict[str, Any] = {
                "name": name,
                "dtype": dtype_str,
                "shape": [h * w],
                "bytes": n_b,
                "block_count": bc,
                "literal_blocks": lit_blocks,
                "encoding": enc,
            }
            if prev_name is not None:
                entry["previous"] = prev_name
            fields_meta.append(entry)
            prev_raw = cur_raw
            prev_name = name

    _pack_series("states", st, "uint8")
    _pack_series("reasons", rs, "uint16")

    payload = b"".join(payload_parts)
    comp_obj = zlib.compressobj(level=6, wbits=-zlib.MAX_WBITS)
    compressed = comp_obj.compress(payload) + comp_obj.flush()
    header = {
        "schema": "dsfch2.v2",
        "codec": {"algorithm": "Deflate", "container_compression": "single_raw_deflate_stream"},
        "payload_bytes": len(payload),
        "compressed_bytes": len(compressed),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "fields": fields_meta,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<8sQ", b"DSFCH2\x00\x00", len(header_bytes)) + header_bytes + compressed


def build_canonical37_numpy(
    rel_triangles_xyz: np.ndarray,
    opaque_mask: np.ndarray,
    candidate_mask: np.ndarray,
    sun_directions_xyz: np.ndarray,
    *,
    scale_m: float = 1000.0,
) -> np.ndarray:
    """Vectorized NumPy computation of exact `canonical37` tokens `(T, N, K, 37)`."""
    suns = np.asarray(sun_directions_xyz, dtype=np.float64)
    if suns.ndim != 2 or suns.shape[1] != 3:
        raise ValueError(f"sun_directions_xyz must be (T, 3), got {suns.shape}")
    t_steps = suns.shape[0]
    s_norm = np.linalg.norm(suns, axis=-1, keepdims=True)
    s_unit = np.where(s_norm > 1e-12, suns / np.maximum(s_norm, 1e-12), 0.0)

    t_rel = np.asarray(rel_triangles_xyz, dtype=np.float64)
    if t_rel.ndim == 4:
        t_rel = t_rel[None, ...]
    n_pts = t_rel.shape[1]
    k_cands = t_rel.shape[2]
    if t_rel.shape[0] == 1 and t_steps > 1:
        t_rel = np.broadcast_to(t_rel, (t_steps, n_pts, k_cands, 3, 3))

    cmask = np.asarray(candidate_mask, dtype=bool)
    if cmask.ndim == 2:
        cmask = cmask[None, ...]
    if cmask.shape[0] == 1 and t_steps > 1:
        cmask = np.broadcast_to(cmask, (t_steps, n_pts, k_cands))

    omat = np.asarray(opaque_mask, dtype=bool)
    if omat.ndim == 2:
        omat = omat[None, ...]
    if omat.shape[0] == 1 and t_steps > 1:
        omat = np.broadcast_to(omat, (t_steps, n_pts, k_cands))

    if n_pts == 0 or k_cands == 0:
        return np.zeros((t_steps, n_pts, k_cands, 37), dtype=np.float32)

    v = t_rel.copy()
    for p_i, p_j in ((0, 1), (1, 2), (0, 1)):
        vi = v[..., p_i, :]
        vj = v[..., p_j, :]
        swap = (vi[..., 0] > vj[..., 0]) | (
            (vi[..., 0] == vj[..., 0])
            & ((vi[..., 1] > vj[..., 1]) | ((vi[..., 1] == vj[..., 1]) & (vi[..., 2] > vj[..., 2])))
        )
        if np.any(swap):
            tmp = vi[swap].copy()
            v[..., p_i, :][swap] = vj[swap]
            v[..., p_j, :][swap] = tmp

    a = v[..., 1, :] - v[..., 0, :]
    b = v[..., 2, :] - v[..., 0, :]
    c = v[..., 2, :] - v[..., 1, :]
    normal = np.cross(a, b)
    area = np.linalg.norm(normal, axis=-1)
    edges = np.stack(
        [np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1), np.linalg.norm(c, axis=-1)],
        axis=-1,
    )

    s_b = s_unit[:, None, None, :]
    s_v = s_unit[:, None, None, None, :]
    logscale = math.log1p(float(scale_m))

    along = np.sum(v * s_v, axis=-1)
    radial = np.linalg.norm(v - along[..., None] * s_v, axis=-1)
    lead_along = np.sign(along) * np.log1p(np.abs(along)) / logscale
    lead_radial = np.log1p(radial) / logscale
    lead_edges = np.log1p(edges) / logscale

    proj_area = np.abs(np.sum(normal * s_b, axis=-1))
    incidence = proj_area / np.maximum(area, 1e-20)
    tol = 1e-12 * np.maximum(edges.max(axis=-1) ** 2, 1e-20)
    degenerate = area <= tol
    proj_degenerate = proj_area <= tol

    margin_list = []
    for i0, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        edge = v[..., i1, :] - v[..., i0, :]
        edge_dot_s = np.sum(edge * s_b, axis=-1, keepdims=True)
        perp = edge - edge_dot_s * s_b
        length = np.linalg.norm(perp, axis=-1)
        orient = np.sum(np.cross(edge, v[..., i2, :] - v[..., i0, :]) * s_b, axis=-1)
        numer = np.sum(np.cross(edge, -v[..., i0, :]) * s_b, axis=-1) * np.sign(orient)
        valid_m = (~proj_degenerate) & (length > np.sqrt(tol))
        m_val = np.zeros_like(length)
        np.divide(numer, length, out=m_val, where=valid_m)
        margin_list.append(m_val)

    margins = np.sort(np.stack(margin_list, axis=-1), axis=-1)
    margins = np.sign(margins) * np.log1p(np.abs(margins)) / logscale

    s_rep = np.broadcast_to(s_b, (t_steps, n_pts, k_cands, 3))
    sy_col = s_rep[..., 1:2]
    mat_col = omat.astype(np.float64)[..., None]
    log_area_col = (np.log1p(area) / (2.0 * logscale))[..., None]

    out = np.concatenate(
        [
            lead_along,                                       # 0..2   (3)
            lead_radial,                                      # 3..5   (3)
            lead_edges,                                       # 6..8   (3)
            incidence[..., None],                             # 9      (1)
            mat_col,                                          # 10     (1)
            log_area_col,                                     # 11     (1)
            sy_col,                                           # 12     (1)
            v.reshape(t_steps, n_pts, k_cands, 9) / float(scale_m),  # 13..21 (9)
            margins,                                          # 22..24 (3)
            s_rep,                                            # 25..27 (3)
            v.min(axis=-2) / float(scale_m),                  # 28..30 (3)
            v.max(axis=-2) / float(scale_m),                  # 31..33 (3)
            (area / (float(scale_m) ** 2))[..., None],        # 34     (1)
            proj_degenerate.astype(np.float64)[..., None],    # 35     (1)
            degenerate.astype(np.float64)[..., None],         # 36     (1)
        ],
        axis=-1,
    )
    out = np.where(cmask[..., None], out, 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def select_azimuth_sector_candidates(
    receivers_xyz: np.ndarray,
    scene_triangles_xyz: np.ndarray,
    *,
    scene_opaque: Optional[np.ndarray] = None,
    scene_source_triangle_ids: Optional[np.ndarray] = None,
    exclude_source_triangle_ids: Optional[np.ndarray] = None,
    num_azimuth_bins: int = 24,
    max_candidates: int = 8,
    horizon_m: float = DEFAULT_SCENE_HORIZON_M,
    return_source_ids: bool = False,
    tri_chunk_size: int = 32768,
    use_spatial_bnb: bool = True,
) -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray]
    | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Build deterministic static azimuth-sector top-`K` 3D candidate triangle bank per receiver.

    Angular eligibility is conservative for the full projected triangle. Ranking uses
    geometry only. It has no access to visibility labels, sun frames or first-hit IDs.
    """
    rec = np.asarray(receivers_xyz, dtype=np.float64)
    tris = np.asarray(scene_triangles_xyz, dtype=np.float64)
    n_pts = rec.shape[0]
    m_tris = tris.shape[0]
    az_bins = int(num_azimuth_bins)
    k_max = int(max_candidates)
    t_chunk = max(1024, int(tri_chunk_size))

    az_tris = np.zeros((n_pts, az_bins, k_max, 3, 3), dtype=np.float32)
    az_opaque = np.zeros((n_pts, az_bins, k_max), dtype=bool)
    az_mask = np.zeros((n_pts, az_bins, k_max), dtype=bool)
    az_sids = np.zeros((n_pts, az_bins, k_max), dtype=np.uint64)
    if n_pts == 0 or m_tris == 0 or k_max <= 0:
        return (az_tris, az_opaque, az_mask, az_sids) if return_source_ids else (az_tris, az_opaque, az_mask)

    op_arr = (
        np.asarray(scene_opaque, dtype=bool)
        if scene_opaque is not None
        else np.ones((m_tris,), dtype=bool)
    )
    sids_arr = (
        np.asarray(scene_source_triangle_ids, dtype=np.uint64)
        if scene_source_triangle_ids is not None
        else np.arange(m_tris, dtype=np.uint64)
    )

    excluded = None
    if exclude_source_triangle_ids is not None:
        excluded = np.asarray(exclude_source_triangle_ids, dtype=np.uint64)
        if excluded.shape != (n_pts,):
            raise ValueError("one excluded source triangle id is required per receiver")

    opaque_idx = np.flatnonzero(op_arr)
    if opaque_idx.size == 0:
        return (az_tris, az_opaque, az_mask, az_sids) if return_source_ids else (az_tris, az_opaque, az_mask)

    sector_width = (2.0 * math.pi) / float(az_bins)
    two_pi = 2.0 * math.pi

    tri_cent = tris.mean(axis=1)
    tri_ymax = tris[:, :, 1].max(axis=1)
    tri_r_xz = np.linalg.norm(tris[:, :, [0, 2]] - tri_cent[:, None, [0, 2]], axis=-1).max(axis=1)

    normals = np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0])
    normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
    range_edges = np.array([100.,400.,800.,1600.])
    n_ranges = len(range_edges)+1
    reserve = k_max//2

    if use_spatial_bnb:
        centers = tri_cent[opaque_idx][:, [0, 2]]
        bounds = rec[:, [0, 2]]
        exterior = np.maximum(np.maximum(bounds.min(axis=0)-centers, centers-bounds.max(axis=0)), 0.)
        keep = (np.linalg.norm(exterior, axis=1) <= horizon_m + tri_r_xz[opaque_idx])
        opaque_idx = opaque_idx[keep & (tri_ymax[opaque_idx] > rec[:,1].min() + 1e-3)]

    groups = None
    if use_spatial_bnb and opaque_idx.size >= 1024:
        # Share 64 m cells across receiver queries. Each bounding circle encloses
        # the centroid disks used by the exact ranking, including wide faces.
        cells = np.floor(tri_cent[opaque_idx][:,[0,2]] / 64.).astype(np.int64)
        order = np.lexsort((cells[:,1],cells[:,0]))
        sorted_cells = cells[order]
        starts = np.r_[0,np.flatnonzero(np.any(np.diff(sorted_cells,axis=0),axis=1))+1]
        faces = opaque_idx[order]
        centers = (sorted_cells[starts]+.5)*64.
        disk_radius = np.linalg.norm((sorted_cells+.5)*64.-tri_cent[faces][:,[0,2]],axis=1) + tri_r_xz[faces]
        radii = np.maximum.reduceat(disk_radius,starts) + 1e-8
        top_y = np.maximum.reduceat(tri_ymax[faces],starts)
        groups = np.split(faces,starts[1:])

    def candidate_blocks(q, current_scores, range_scores):
        if groups is None:
            for start in range(0,len(opaque_idx),t_chunk):
                yield opaque_idx[start:start+t_chunk]
            return
        displacement = centers-q[[0,2]]
        distance = np.linalg.norm(displacement,axis=1)
        lower = distance-radii
        upper = (top_y-q[1])/np.maximum(lower,.25)
        possible = np.flatnonzero((lower <= horizon_m) & (top_y > q[1]+1e-3))
        ranked = possible[np.argsort(-upper[possible],kind="stable")]
        angles = np.arctan2(displacement[:,0],displacement[:,1])
        half = np.arcsin(np.minimum(1.,radii/np.maximum(distance,1e-20)))
        half[distance <= radii] = math.pi
        mids = (np.arange(az_bins)+.5)*sector_width
        for off in range(0,len(ranked),64):
            ids = ranked[off:off+64]
            cuts = np.array([sc[-1] if len(sc)==k_max else -np.inf for sc in current_scores])
            angular_distance = np.abs((angles[ids,None]-mids+math.pi)%two_pi-math.pi)
            can_win = (angular_distance <= half[ids,None]+sector_width/2+1e-10) & (upper[ids,None]+1e-10 >= cuts)
            if reserve:
                # The same score upper bound covers the foreshortened reserve score.
                # A cell can feed more than one distance tier.
                for band in range(n_ranges):
                    d_lo = range_edges[band-1] if band else 0.
                    d_hi = range_edges[band] if band<len(range_edges) else np.inf
                    in_band = (lower[ids] <= d_hi) & (distance[ids]+radii[ids] >= d_lo)
                    if in_band.any():
                        cuts = np.array([sc[-1] if len(sc)==reserve else -np.inf for sc in range_scores[band]])
                        can_win |= in_band[:,None] & (angular_distance <= half[ids,None]+sector_width/2+1e-10) & (upper[ids,None]+1e-10 >= cuts)
            survivors = ids[can_win.any(axis=1)]
            if survivors.size:
                yield np.concatenate([groups[g] for g in survivors])

    for i in range(n_pts):
        q = rec[i]
        qx = float(q[0])
        qy = float(q[1])
        qz = float(q[2])

        # Maintain bounded top-K per sector across triangle chunks (`<= 2*K` merged at each chunk)
        bin_best_g: List[np.ndarray] = [np.empty((0,), dtype=np.int64) for _ in range(az_bins)]
        bin_best_sc: List[np.ndarray] = [np.empty((0,), dtype=np.float64) for _ in range(az_bins)]

        range_g = [[np.empty(0,dtype=np.int64) for _ in range(az_bins)] for _ in range(n_ranges)]
        range_scores = [[np.empty(0) for _ in range(az_bins)] for _ in range(n_ranges)]
        for cand_slice in candidate_blocks(q, bin_best_sc, range_scores):
            above_m = tri_ymax[cand_slice] > (qy + 1e-3)
            if excluded is not None:
                above_m &= sids_arr[cand_slice] != excluded[i]
            if not np.any(above_m):
                continue
            sub_idx = cand_slice[above_m]
            dx = tri_cent[sub_idx, 0] - qx
            dz = tri_cent[sub_idx, 2] - qz
            dist = np.hypot(dx, dz)
            r_sub = tri_r_xz[sub_idx]
            in_horizon = (dist - r_sub) <= float(horizon_m)
            if not np.any(in_horizon):
                continue
            sub_idx = sub_idx[in_horizon]
            dx = dx[in_horizon]
            dz = dz[in_horizon]
            r_sub = r_sub[in_horizon]
            eff_dist = np.maximum(dist[in_horizon] - r_sub, 0.25)
            elev_score = (tri_ymax[sub_idx] - qy) / eff_dist

            band_idx = np.searchsorted(range_edges,dist[in_horizon],side="right")
            radial = tri_cent[sub_idx]-q
            facing = np.abs(np.sum(normals[sub_idx]*radial,axis=1)) / np.maximum(np.linalg.norm(radial,axis=1),1e-20)
            reserve_score = elev_score * facing
            # A convex triangle outside the receiver spans the complement of the
            # largest circular gap between vertex bearings. If it contains or
            # touches the origin, all bearings are conservatively eligible.
            planar = tris[sub_idx][:, :, [0, 2]] - q[[0, 2]]
            bearings = np.sort(np.mod(np.arctan2(planar[..., 0], planar[..., 1]), two_pi), axis=1)
            gaps = np.diff(np.column_stack((bearings, bearings[:, 0] + two_pi)), axis=1)
            largest = np.argmax(gaps, axis=1)
            rows = np.arange(len(sub_idx))
            arc_start = bearings[rows, (largest + 1) % 3]
            arc_width = two_pi - gaps[rows, largest]
            surrounds = (arc_width >= math.pi - 1e-12) | (np.linalg.norm(planar, axis=-1).min(axis=1) < 1e-9)
            for b_id in range(az_bins):
                midpoint = (b_id + 0.5) * sector_width
                delta = np.abs((arc_start + arc_width/2 - midpoint + math.pi) % two_pi - math.pi)
                uniq_local = np.flatnonzero(surrounds | (delta <= (arc_width + sector_width)/2 + 1e-12))
                if not uniq_local.size:
                    continue
                new_g = sub_idx[uniq_local]
                new_sc = elev_score[uniq_local]
                if reserve:
                    for band in np.unique(band_idx[uniq_local]):
                        subset = uniq_local[band_idx[uniq_local] == band]
                        g = np.r_[range_g[band][b_id],sub_idx[subset]]
                        sc = np.r_[range_scores[band][b_id],reserve_score[subset]]
                        sel = np.lexsort((sids_arr[g],-sc))[:reserve]
                        range_g[band][b_id],range_scores[band][b_id] = g[sel],sc[sel]
                if bin_best_g[b_id].size > 0:
                    merged_g = np.concatenate([bin_best_g[b_id], new_g], axis=0)
                    merged_sc = np.concatenate([bin_best_sc[b_id], new_sc], axis=0)
                else:
                    merged_g = new_g
                    merged_sc = new_sc
                merged_sids = sids_arr[merged_g]
                order = np.lexsort((merged_sids, -merged_sc))[:k_max]
                bin_best_g[b_id] = merged_g[order]
                bin_best_sc[b_id] = merged_sc[order]

        for b_id in range(az_bins):
            primary = bin_best_g[b_id].tolist()
            chosen = primary[:k_max-reserve]
            # Round-robin range tiers with the farthest represented first.
            for rank in range(reserve):
                for band in reversed(range(n_ranges)):
                    pool = range_g[band][b_id]
                    if rank < len(pool) and len(chosen)<k_max and int(pool[rank]) not in chosen:
                        chosen.append(int(pool[rank]))
            chosen_g = np.array((chosen+[g for g in primary if g not in chosen])[:k_max],dtype=np.int64)
            take = int(chosen_g.size)
            if take == 0:
                continue
            az_tris[i, b_id, :take] = (tris[chosen_g] - q[None, None, :]).astype(np.float32)
            az_opaque[i, b_id, :take] = op_arr[chosen_g]
            az_mask[i, b_id, :take] = True
            az_sids[i, b_id, :take] = sids_arr[chosen_g]

    if return_source_ids:
        return az_tris, az_opaque, az_mask, az_sids
    return az_tris, az_opaque, az_mask


def sun_dirs_to_azimuth_bins(sun_directions_xyz: np.ndarray, num_azimuth_bins: int = 24) -> np.ndarray:
    """Single canonical `float64` azimuth sector mapping shared across CPU and GPU pipelines (`P1-2`)."""
    s = np.asarray(sun_directions_xyz, dtype=np.float64)
    ang = np.mod(np.arctan2(s[:, 0], s[:, 2]), 2.0 * math.pi)
    sector_width = (2.0 * math.pi) / float(num_azimuth_bins)
    return (np.floor(ang / sector_width).astype(np.int64)) % int(num_azimuth_bins)


def extract_exact_centered_receiver_crops(
    static_fields_8hw: np.ndarray,
    geometry_known_hw: Optional[np.ndarray],
    receiver_rc_n2: np.ndarray,
    *,
    crop_size: int = 64,
) -> np.ndarray:
    """Extract exact per-receiver centered `(N, 9, P, P)` spatial patches (`P0-4`).

    - Channels `0..7`: `static_fields_8hw` centered on `(r_i, c_i)` with zero-padding outside `[0, H)x[0, W)`.
    - Channel `8`: binary `in_bounds & geometry_known` mask (`1.0` valid grid cell, `0.0` padded halo).
    - Guarantees `crops[i, :8, P // 2, P // 2] == static_fields_8hw[:, r_i, c_i]` for ALL `(r_i, c_i)`
      across the entire `512 x 512` tile with ZERO coordinate clamping!
    """
    fields = np.asarray(static_fields_8hw, dtype=np.float32)
    if fields.ndim != 3 or fields.shape[0] != 8:
        raise ValueError(f"static_fields_8hw must have shape (8, H, W), got {fields.shape}")
    _, h_grid, w_grid = fields.shape
    known = (
        np.asarray(geometry_known_hw, dtype=np.float32)
        if geometry_known_hw is not None
        else np.ones((h_grid, w_grid), dtype=np.float32)
    )
    if known.shape != (h_grid, w_grid):
        raise ValueError(f"geometry_known_hw shape {known.shape} != ({h_grid}, {w_grid})")

    rc = np.asarray(receiver_rc_n2, dtype=np.int64)
    if rc.ndim != 2 or rc.shape[1] != 2:
        raise ValueError(f"receiver_rc_n2 must have shape (N, 2), got {rc.shape}")
    n_pts = rc.shape[0]
    side = int(crop_size)
    if side < 3:
        raise ValueError(f"crop_size must be >= 3, got {side}")
    if n_pts == 0:
        return np.zeros((0, 9, side, side), dtype=np.float32)

    if np.any((rc[:, 0] < 0) | (rc[:, 0] >= h_grid) | (rc[:, 1] < 0) | (rc[:, 1] >= w_grid)):
        raise ValueError(f"receiver_rc out of grid bounds (0..{h_grid - 1}, 0..{w_grid - 1}).")

    pad_before = side // 2
    pad_after = side - 1 - pad_before
    stacked_9hw = np.concatenate([fields, known[None, :, :]], axis=0)  # (9, H, W)
    padded = np.pad(
        stacked_9hw,
        ((0, 0), (pad_before, pad_after), (pad_before, pad_after)),
        mode="constant",
        constant_values=0.0,
    )
    # Center of receiver (r, c) in padded array is (r + pad_before, c + pad_before),
    # so slice [r : r + side, c : c + side] places (r, c) at exact index (pad_before, pad_before) == (side//2, side//2)!
    out = np.empty((n_pts, 9, side, side), dtype=np.float32)
    for idx in range(n_pts):
        r0 = int(rc[idx, 0])
        c0 = int(rc[idx, 1])
        out[idx] = padded[:, r0 : r0 + side, c0 : c0 + side]
    return out


def extract_receiver_local_features_and_crops(
    fields_8hw: np.ndarray,
    receiver_xyz_hw3: np.ndarray,
    receiver_rc_n2: np.ndarray,
    world_x_m: np.ndarray,
    world_z_m: np.ndarray,
    *,
    geometry_known_hw: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Extract 16-dim multi-scale local features + exact grid metadata (zero receiver clamping)."""
    fields = np.asarray(fields_8hw, dtype=np.float32)
    _, h_grid, w_grid = fields.shape
    rc = np.asarray(receiver_rc_n2, dtype=np.int64)

    wx = np.asarray(world_x_m, dtype=np.float64).reshape(-1)
    wz = np.asarray(world_z_m, dtype=np.float64).reshape(-1)
    dx = float(np.median(np.abs(np.diff(wx)))) if wx.size >= 2 else (100.0 / 512.0)
    dz = float(np.median(np.abs(np.diff(wz)))) if wz.size >= 2 else (100.0 / 512.0)
    meters_per_pixel = 0.5 * (dx + dz)

    if rc.shape[0] > 0 and np.any(
        (rc[:, 0] < 0) | (rc[:, 0] >= h_grid) | (rc[:, 1] < 0) | (rc[:, 1] >= w_grid)
    ):
        raise ValueError(f"receiver_rc_n2 out of bounds for grid ({h_grid}, {w_grid})")
    rows = rc[:, 0]
    cols = rc[:, 1]
    exact_8 = fields[:, rows, cols].T if rc.shape[0] > 0 else np.zeros((0, 8), dtype=np.float32)

    pad_r = 4
    padded = np.pad(fields, ((0, 0), (pad_r, pad_r), (pad_r, pad_r)), mode="edge")
    csum = np.pad(padded.astype(np.float64), ((0, 0), (1, 0), (1, 0))).cumsum(axis=1).cumsum(axis=2)
    win = 2 * pad_r + 1
    r0 = rows
    c0 = cols
    r1 = rows + win
    c1 = cols + win
    box_sum = csum[:, r1, c1] - csum[:, r0, c1] - csum[:, r1, c0] + csum[:, r0, c0]
    mean_8 = (box_sum / float(win * win)).T.astype(np.float32) if rc.shape[0] > 0 else np.zeros((0, 8), dtype=np.float32)
    local_features = np.concatenate([exact_8, mean_8], axis=-1)

    known_hw = (
        np.asarray(geometry_known_hw, dtype=bool)
        if geometry_known_hw is not None
        else np.ones((h_grid, w_grid), dtype=bool)
    )
    return {
        "local_features": local_features,
        "static_fields_8hw": fields.astype(np.float32),
        "geometry_known_hw": known_hw,
        "meters_per_pixel": float(meters_per_pixel),
        "receptive_field_m_21": 21.0 * float(meters_per_pixel),
        "receptive_field_m_64": 64.0 * float(meters_per_pixel),
    }


def _compute_receiver_boundary_distances_from_states(
    states_fhw: np.ndarray,
    sel_rc: np.ndarray,
    *,
    grid_spacing_m: float = 0.1953125,
    max_search_radius_m: float = 2.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact `(F, N)` planar shadow boundary distances & `distance_valid` around `sel_rc`."""
    f_cnt, h_grid, w_grid = states_fhw.shape
    rc = np.asarray(sel_rc, dtype=np.int64)
    n_pts = rc.shape[0]
    if f_cnt == 0 or n_pts == 0:
        return (
            np.full((f_cnt, n_pts), np.nan, dtype=np.float32),
            np.zeros((f_cnt, n_pts), dtype=bool),
        )

    # If grid is small (or dense receiver set covers >25% of grid), run grid pass directly
    if h_grid * w_grid <= 8192 or n_pts * 4 >= h_grid * w_grid:
        v_fhw = states_fhw != 3
        gt_fhw = np.where(v_fhw, (states_fhw == 2).astype(np.int8), np.int8(-1))
        return compute_planar_shadow_boundary_distance_grid(
            gt_fhw,
            v_fhw,
            grid_spacing_m=grid_spacing_m,
            max_search_radius_m=max_search_radius_m,
            receiver_rc=rc,
        )

    # Sparse receiver extraction (`N=256` on `512x512`): evaluate exact `(2R+1)x(2R+1)` local neighborhoods!
    rad_px = max(1, int(math.ceil(max_search_radius_m / max(float(grid_spacing_m), 1e-6))))
    offsets: List[Tuple[int, int, float]] = []
    for dr in range(-rad_px, rad_px + 1):
        for dc in range(-rad_px, rad_px + 1):
            if dr == 0 and dc == 0:
                continue
            d_m = math.hypot(dr * grid_spacing_m, dc * grid_spacing_m)
            if d_m <= max_search_radius_m + 1e-9:
                offsets.append((dr, dc, d_m))

    rows = rc[:, 0]
    cols = rc[:, 1]
    c_states = states_fhw[:, rows, cols]  # (F, N)
    c_valid = c_states != 3

    inf32 = np.float32(np.inf)
    min_opp = np.full((f_cnt, n_pts), inf32, dtype=np.float32)
    border_d = np.minimum(
        np.minimum(rows + 1, h_grid - rows),
        np.minimum(cols + 1, w_grid - cols),
    ).astype(np.float32) * np.float32(grid_spacing_m)
    min_unk = np.broadcast_to(border_d[None, :], (f_cnt, n_pts)).copy()

    for dr, dc, dist_m in offsets:
        d32 = np.float32(dist_m)
        nr = rows + dr
        nc = cols + dc
        in_b = (nr >= 0) & (nr < h_grid) & (nc >= 0) & (nc < w_grid)
        if not np.any(in_b):
            continue
        ib_idx = np.flatnonzero(in_b)
        nb_st = states_fhw[:, nr[ib_idx], nc[ib_idx]]
        nb_val = nb_st != 3
        src_val = c_valid[:, ib_idx]

        unk_hit = src_val & (~nb_val) & (d32 < min_unk[:, ib_idx])
        if np.any(unk_hit):
            sub_u = min_unk[:, ib_idx]
            sub_u[unk_hit] = d32
            min_unk[:, ib_idx] = sub_u

        opp_hit = src_val & nb_val & (nb_st != c_states[:, ib_idx]) & (d32 < min_opp[:, ib_idx])
        if np.any(opp_hit):
            sub_o = min_opp[:, ib_idx]
            sub_o[opp_hit] = d32
            min_opp[:, ib_idx] = sub_o

    cert = np.minimum(min_opp, np.float32(2.0001))
    d_valid = c_valid & (min_unk > cert)
    d_out = np.where(np.isfinite(min_opp), min_opp, np.float32(max_search_radius_m))
    d_out = np.where(d_valid, d_out, np.float32(np.nan))
    return d_out, d_valid


def pack_receiver_5x5_response_bitpack(
    states_fhw: np.ndarray,
    sel_rc: np.ndarray,
    *,
    receiver_chunk: int = 4096,
) -> np.ndarray:
    """Losslessly pack exact `5x5` grid `(sunlit, valid)` neighborhood around `sel_rc` into `int64`.

    Bits `0..24` store `sunlit` (`states == 2`) for `(dr, dc) in {-2..+2} x {-2..+2}` row-major;
    bits `25..49` store `valid` (`states != 3` and inside grid bounds).
    Chunked across `receiver_chunk` so even `all_valid` (`N ~ 27,000`) uses `< 16 MiB` transient RAM.
    Used strictly by training-only `ResponseTargetTeacher` under `FitFirewallToken` (`0` inference reads).
    """
    f_cnt, h_grid, w_grid = states_fhw.shape
    rc = np.asarray(sel_rc, dtype=np.int64)
    n_pts = rc.shape[0]
    packed = np.zeros((f_cnt, n_pts), dtype=np.int64)
    if f_cnt == 0 or n_pts == 0:
        return packed

    step = max(int(receiver_chunk), 256)
    for c0 in range(0, n_pts, step):
        c1 = min(n_pts, c0 + step)
        rows = rc[c0:c1, 0]
        cols = rc[c0:c1, 1]
        sub_p = packed[:, c0:c1]
        bit_idx = 0
        for dr in range(-2, 3):
            nr = rows + dr
            r_ok = (nr >= 0) & (nr < h_grid)
            nr_safe = np.clip(nr, 0, h_grid - 1)
            for dc in range(-2, 3):
                nc = cols + dc
                in_b = r_ok & (nc >= 0) & (nc < w_grid)
                if np.any(in_b):
                    nc_safe = np.clip(nc, 0, w_grid - 1)
                    nb_st = states_fhw[:, nr_safe, nc_safe]
                    v_mask = (nb_st != 3) & in_b[None, :]
                    s_mask = (nb_st == 2) & v_mask
                    sub_p |= s_mask.astype(np.int64) << np.int64(bit_idx)
                    sub_p |= v_mask.astype(np.int64) << np.int64(25 + bit_idx)
                bit_idx += 1
    return packed


@dataclasses.dataclass
class FullDayTileTrajectory:
    tile_id: str
    canonical_split: str
    spatial_group: str
    source_core_sha256: str
    mesh_provenance_sha256: str
    receiver_mode: str
    total_eligible_receivers: int
    week_indices: Tuple[int, ...]
    window_ids: Tuple[str, ...]
    day_ids: np.ndarray                 # (T,) int32
    timestamps_unix_s: np.ndarray       # (T,) int64
    same_day_adjacent_mask: np.ndarray  # (T - 1,) bool
    sun_dirs: np.ndarray                # (T, 3) float32
    azimuth_bin_idx: np.ndarray         # (T,) int16 (canonical float64 sector assignment)
    solar_altitude_deg: np.ndarray      # (T,) float32
    solar_azimuth_deg: np.ndarray       # (T,) float32
    solar_regimes: Tuple[str, ...]      # len T
    receivers_xyz: np.ndarray           # (N, 3) float32
    receiver_rc: np.ndarray             # (N, 2) int64
    truth: np.ndarray                   # (T, N) int8 (-1=UNKNOWN, 0=SHADOW, 1=SUN)
    valid_mask: np.ndarray              # (T, N) bool
    boundary_distance_m: np.ndarray     # (T, N) float32
    boundary_distance_valid: np.ndarray # (T, N) bool
    local_features: np.ndarray          # (N, 16) float32
    static_fields_8hw: np.ndarray       # (8, H, W) float32
    geometry_known_hw: np.ndarray       # (H, W) bool
    meters_per_pixel: float
    bbox_margin_m: float
    az_rel_triangles: np.ndarray        # (N, AzBins, K, 3, 3) float32
    az_opaque: np.ndarray               # (N, AzBins, K) bool
    az_mask: np.ndarray                 # (N, AzBins, K) bool
    az_source_ids: np.ndarray           # (N, AzBins, K) uint64
    scene_triangles_xyz: np.ndarray     # (M_scene, 3, 3) float32
    scene_opaque: np.ndarray            # (M_scene,) bool
    scene_source_ids: np.ndarray        # (M_scene,) uint64
    reasons: Optional[np.ndarray] = None  # optional (T, N) uint16
    response_5x5_bitpack: Optional[np.ndarray] = None  # optional (T, N) int64
    world_x_m: Optional[np.ndarray] = None  # optional (W,) float64 pixel x-centers
    world_z_m: Optional[np.ndarray] = None  # optional (H,) float64 pixel z-centers
    receiver_source_triangle_ids: Optional[np.ndarray] = None  # optional (N,) uint64 surface face IDs
    ray_candidates: bool = False
    scene_ground_mask: Optional[np.ndarray] = None  # optional (M_scene,) bool true for ground/non-building
    scene_source_instance_ids: Optional[np.ndarray] = None  # optional (M_scene,) int64 building instance IDs

    @property
    def point_frames(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def receiver_obs_mask(self) -> np.ndarray:
        """Label-free static geometry observability mask `(N,)` bool (`0` reads of `truth`/`valid_mask`)."""
        return np.ascontiguousarray(
            self.geometry_known_hw[self.receiver_rc[:, 0], self.receiver_rc[:, 1]].astype(bool)
        )

    def receiver_sampling_disclosure(self) -> Dict[str, Any]:
        return build_receiver_sampling_disclosure(
            receiver_mode=self.receiver_mode,
            eligible_receiver_count=self.total_eligible_receivers,
            evaluated_receiver_count=int(self.receiver_rc.shape[0]),
            receiver_rc=self.receiver_rc,
        )

    def extract_centered_crops(
        self, crop_size: int = 64, receiver_slice: slice = slice(None)
    ) -> np.ndarray:
        return extract_exact_centered_receiver_crops(
            self.static_fields_8hw,
            self.geometry_known_hw,
            self.receiver_rc[receiver_slice],
            crop_size=crop_size,
        )

    def materialize_canonical37_batch(
        self,
        *,
        frame_slice: slice = slice(None),
        receiver_slice: slice = slice(None),
        crop_size: int = 64,
        candidate_stride: int = 1,
        candidate_provider: Any = None,
        candidate_mode: str = "causal_rollout",
    ) -> Dict[str, Any]:
        """Materialize `(T_sub, N_sub, K, 37)` CPU batch dictionary with exact candidate IDs & centered crops."""
        import torch
        from .models import compute_tile_global_knn_pack_torch

        bin_idx = self.azimuth_bin_idx[frame_slice].astype(np.int64)
        az_rel_sub = self.az_rel_triangles[receiver_slice]
        az_op_sub = self.az_opaque[receiver_slice]
        az_mk_sub = self.az_mask[receiver_slice]
        az_sid_sub = self.az_source_ids[receiver_slice]

        rel_t = np.transpose(az_rel_sub, (1, 0, 2, 3, 4))[bin_idx]
        op_t = np.transpose(az_op_sub, (1, 0, 2))[bin_idx]
        mk_t = np.transpose(az_mk_sub, (1, 0, 2))[bin_idx]
        sid_t = np.transpose(az_sid_sub, (1, 0, 2))[bin_idx]

        suns_sub = self.sun_dirs[frame_slice]
        counts = None
        anchor_plan = None
        next_phase = 0
        if self.ray_candidates:
            from .ray_candidates import BVHCandidateProvider, plan_anchor_frames
            full_plan, _ = plan_anchor_frames(self.day_ids, self.timestamps_unix_s,
                                               candidate_stride,
                                               offline_interp=candidate_mode == "offline_interp")
            selected_frames = np.arange(len(self.day_ids))[frame_slice]
            anchor_plan = full_plan[selected_frames]
            if len(selected_frames):
                _, next_phase = plan_anchor_frames(self.day_ids[:int(selected_frames[-1]) + 1],
                    self.timestamps_unix_s[:int(selected_frames[-1]) + 1], candidate_stride)
            provider = candidate_provider
            if provider is None:
                if not hasattr(self, "_ray_scene"):
                    self._ray_scene = BVHCandidateProvider(self.scene_triangles_xyz, self.scene_opaque, self.scene_source_ids)
                else:
                    self._ray_scene.update_geometry(self.scene_triangles_xyz, self.scene_opaque, self.scene_source_ids)
                provider = self._ray_scene
            anchor_indices = np.flatnonzero(anchor_plan)
            shape = (len(suns_sub), len(self.receivers_xyz[receiver_slice]), self.az_mask.shape[-1])
            rel_t = np.zeros((*shape, 3, 3), np.float32)
            op_t, mk_t = np.zeros(shape, bool), np.zeros(shape, bool)
            sid_t = np.full(shape, -1, np.int64)
            counts = np.zeros((shape[1], 0), np.int64)
            if len(anchor_indices):
                relative, opaque, mask, refs, counts = provider.query(
                    self.receivers_xyz[receiver_slice], suns_sub[anchor_indices], self.bbox_margin_m,
                    self.az_mask.shape[-1], None if self.receiver_source_triangle_ids is None
                    else self.receiver_source_triangle_ids[receiver_slice])
                for target, value in zip((rel_t, op_t, mk_t, sid_t), (relative, opaque, mask, refs)):
                    target[anchor_indices] = value.swapaxes(0, 1)

        tokens = build_canonical37_numpy(rel_t, op_t, mk_t, suns_sub)

        day_sub = self.day_ids[frame_slice]
        ts_sub = self.timestamps_unix_s[frame_slice]
        same_day_adj = build_same_day_adjacent_mask(day_sub, ts_sub)
        centered_crops = self.extract_centered_crops(
            crop_size=crop_size, receiver_slice=receiver_slice
        )
        rc_sub = self.receiver_rc[receiver_slice]
        all_obs_t = torch.from_numpy(
            np.ascontiguousarray(
                self.geometry_known_hw[self.receiver_rc[:, 0], self.receiver_rc[:, 1]].astype(bool)
            )
        )
        global_knn = compute_tile_global_knn_pack_torch(
            torch.from_numpy(self.receivers_xyz),
            all_obs_t,
            torch.from_numpy(self.local_features),
            k_neighbors=8,
        )

        out: Dict[str, Any] = {
            "tile_ids": [self.tile_id],
            "meters_per_pixel": float(self.meters_per_pixel),
            "candidate_tokens": torch.from_numpy(tokens),
            "candidate_mask": torch.from_numpy(mk_t),
            "candidate_source_ids": torch.from_numpy(sid_t.astype(np.int64)),
            "azimuth_bin_idx": torch.from_numpy(bin_idx.astype(np.int64)),
            "local_features": torch.from_numpy(self.local_features[receiver_slice]),
            "tile_label_domain": TileLabelDomain(self.tile_id, self.receiver_rc),
            "receiver_rc": torch.from_numpy(rc_sub),
            "receiver_obs_mask": torch.from_numpy(
                np.ascontiguousarray(self.geometry_known_hw[rc_sub[:, 0], rc_sub[:, 1]].astype(bool))
            ),
            "receivers_xyz": torch.from_numpy(self.receivers_xyz[receiver_slice]),
            "global_knn_local_features": global_knn["knn_local_features"][receiver_slice],
            "global_knn_rel_xyz": global_knn["knn_rel_xyz"][receiver_slice],
            "global_knn_dist": global_knn["knn_dist"][receiver_slice],
            "global_knn_obs_valid": global_knn["knn_obs_valid"][receiver_slice],
            "static_fields_8hw": torch.from_numpy(self.static_fields_8hw),
            "geometry_known_hw": torch.from_numpy(self.geometry_known_hw),
            "receiver_crops": torch.from_numpy(centered_crops),
            "sun_dirs": torch.from_numpy(suns_sub),
            "same_day_adjacent_mask": torch.from_numpy(same_day_adj),
            "truth": torch.from_numpy(self.truth[frame_slice, receiver_slice]),
            "valid_mask": torch.from_numpy(self.valid_mask[frame_slice, receiver_slice]),
            "boundary_distance_m": torch.from_numpy(
                self.boundary_distance_m[frame_slice, receiver_slice]
            ),
            "boundary_distance_valid": torch.from_numpy(
                self.boundary_distance_valid[frame_slice, receiver_slice]
            ),
            "day_ids": day_sub,
            "timestamps_unix_s": ts_sub,
            "solar_regimes": self.solar_regimes[frame_slice],
        }
        if self.response_5x5_bitpack is not None:
            out["response_5x5_bitpack"] = torch.from_numpy(
                self.response_5x5_bitpack[frame_slice, receiver_slice]
            )
        if counts is not None:
            out["ray_candidate_count"] = torch.from_numpy(counts.T.copy())
            out["candidate_anchor_mask"] = torch.from_numpy(anchor_plan.copy())
            out["candidate_anchor_next_phase"] = int(next_phase)
            out["candidate_anchor_stride"] = max(1, int(candidate_stride))
            out["candidate_anchor_mode"] = str(candidate_mode)
            out["candidate_query_frames"] = int(anchor_plan.sum())
            out["candidate_total_frames"] = len(anchor_plan)
            out["candidate_bvh_savings"] = 1.0 - int(anchor_plan.sum()) / float(max(len(anchor_plan), 1))
            out["candidate_geometry_identity"] = getattr(provider, "geometry_identity", None)
        return out


def resolve_week_indices(temporal_protocol: str) -> Tuple[int, ...]:
    if temporal_protocol == "annual_52week":
        return ANNUAL_52WEEK_INDICES
    if temporal_protocol == "canonical_4season":
        return CANONICAL_4SEASON_WEEK_INDICES
    if temporal_protocol == "stratified_13week":
        return STRATIFIED_13WEEK_INDICES
    raise ValueError(
        f"Unknown temporal_protocol {temporal_protocol!r}. "
        f"Valid: ('annual_52week', 'canonical_4season', 'stratified_13week')"
    )


class FullDayTileReader:
    """Memory-bounded (`<= 4.0 GiB`) full sunrise-to-sunset reader with strict payload SHA validation."""

    def __init__(
        self,
        contract: SplitContract,
        *,
        max_host_ram_bytes: int = MAX_DEFAULT_HOST_CACHE_BYTES,
        global_mesh_shared_dir: Optional[Path | str] = None,
        verify_payload_sha256: bool = True,
        resource_guard: Optional[HostResourceGuard] = None,
    ) -> None:
        self.contract = contract
        self.guard = resource_guard or HostResourceGuard()
        self.cache = BoundedHostByteCache(
            max_bytes=max_host_ram_bytes, resource_guard=self.guard
        )
        self.verify_payload_sha256 = bool(verify_payload_sha256)
        self._explicit_mesh_shared_dir = (
            Path(global_mesh_shared_dir) if global_mesh_shared_dir is not None else None
        )
        self._spatial_index_cache: Dict[
            str, Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], Path, str]
        ] = {}
        self._chunk_ground_mask_cache: Dict[str, np.ndarray] = {}
        self._verified_file_shas: Dict[str, str] = {}

    def _assert_file_sha256(self, file_path: Path, expected_sha: Optional[str], label: str) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required {label} file: {file_path}")
        key = str(file_path.resolve())
        actual = self._verified_file_shas.get(key)
        if actual is None:
            actual = compute_file_sha256(file_path)
            self._verified_file_shas[key] = actual
        if self.verify_payload_sha256 and expected_sha:
            if actual != str(expected_sha).strip().lower():
                raise ValueError(
                    f"Payload SHA-256 mismatch for {label} ({file_path}): "
                    f"actual={actual} != expected={expected_sha}"
                )
        return actual

    def _resolve_tile_manifest_and_roots(
        self, rec: TileSplitRecord
    ) -> Tuple[Path, Dict[str, Any], Dict[str, Path], Optional[Path]]:
        win_or_posix_data = translate_storage_path(
            rec.win_data_dir if os.name == "nt" else (rec.wsl_data_dir or rec.win_data_dir)
        )
        win_or_posix_entry = translate_storage_path(
            rec.win_reader_entry
            if os.name == "nt"
            else (rec.wsl_reader_entry or rec.win_reader_entry)
        )

        if not win_or_posix_entry.exists() and Path(rec.win_reader_entry).exists():
            win_or_posix_entry = Path(rec.win_reader_entry)
        if not win_or_posix_data.exists() and Path(rec.win_data_dir).exists():
            win_or_posix_data = Path(rec.win_data_dir)

        bundle_root: Optional[Path] = None
        if win_or_posix_entry.name == "MANIFEST.json" and win_or_posix_entry.exists():
            manifest_path = win_or_posix_entry
        elif (win_or_posix_data / "MANIFEST.json").exists():
            manifest_path = win_or_posix_data / "MANIFEST.json"
            if win_or_posix_entry.suffix == ".py":
                bundle_root = win_or_posix_entry.parent
        elif (win_or_posix_data / "reader-annual-r01" / "MANIFEST.json").exists():
            manifest_path = win_or_posix_data / "reader-annual-r01" / "MANIFEST.json"
        else:
            raise FileNotFoundError(
                f"Could not locate tile MANIFEST.json for {rec.tile_id} "
                f"(checked {win_or_posix_entry} and {win_or_posix_data})"
            )

        manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8-sig"))

        # Verify top-level tile identity & source_core_sha256 if present (`P0-2`)
        for top_key in ("tile_id", "family_id"):
            doc_tid = manifest_doc.get(top_key)
            if doc_tid is not None and self._normalize_tile_id_alias(str(doc_tid)) != rec.tile_id:
                raise ValueError(
                    f"Tile payload mismatch in {manifest_path}: MANIFEST {top_key}={doc_tid!r} != "
                    f"requested {rec.tile_id!r}"
                )
        doc_core_sha = manifest_doc.get("source_core_sha256")
        if (
            rec.source_core_sha256
            and doc_core_sha
            and str(doc_core_sha).strip().lower() != rec.source_core_sha256.lower()
        ):
            raise ValueError(
                f"Tile source_core_sha256 mismatch in {manifest_path}: {doc_core_sha} != {rec.source_core_sha256}"
            )

        roots: Dict[str, Path] = {"dataset": manifest_path.parent}
        for k, raw_v in manifest_doc.get("object_roots", {}).items():
            cand = translate_storage_path(str(raw_v))
            if cand.is_absolute() and cand.exists():
                roots[k] = cand
            elif bundle_root is not None and (bundle_root / str(raw_v).replace("\\", "/")).exists():
                roots[k] = (bundle_root / str(raw_v).replace("\\", "/")).resolve()
            else:
                rel_cand = manifest_path.parent / str(raw_v).replace("\\", "/")
                roots[k] = rel_cand.resolve() if rel_cand.exists() else cand

        raw_records = manifest_doc.get("records")
        if not isinstance(raw_records, list) or len(raw_records) == 0:
            raise ValueError(f"Tile MANIFEST {manifest_path} contains no `records`!")
        for r_item in raw_records:
            if not isinstance(r_item, Mapping):
                raise ValueError(f"Malformed record entry in {manifest_path}: expected mapping.")
            self._validate_record_identity(r_item, rec)

        return manifest_path, manifest_doc, roots, bundle_root

    @staticmethod
    def _normalize_tile_id_alias(raw_id: str) -> str:
        val = str(raw_id).strip()
        prefix = "manhattan/full-v3/"
        if val.startswith(prefix) and val[len(prefix) :].startswith("fi-"):
            return "core_" + val[len(prefix) :]
        return val

    @classmethod
    def _validate_record_identity(cls, record: Mapping[str, Any], rec: TileSplitRecord) -> None:
        saw_tile_id = False
        for id_key in ("tile_id", "family_id"):
            raw_val = record.get(id_key)
            if raw_val is not None and str(raw_val).strip() != "":
                saw_tile_id = True
                norm_id = cls._normalize_tile_id_alias(str(raw_val))
                if norm_id != rec.tile_id:
                    raise ValueError(
                        f"Record {id_key}={raw_val!r} (normalized={norm_id!r}) in tile manifest "
                        f"does not match requested tile_id={rec.tile_id!r} (mispointed tile directory!)."
                    )
        if not saw_tile_id:
            raise ValueError(
                f"Record in tile {rec.tile_id} is missing mandatory `tile_id` / `family_id` provenance!"
            )

        rec_core = record.get("source_core_sha256")
        if not rec_core and isinstance(record.get("source_core"), Mapping):
            rec_core = record["source_core"].get("sha256")
        if rec.source_core_sha256:
            if not rec_core or str(rec_core).strip() == "":
                raise ValueError(
                    f"Record in tile {rec.tile_id} is missing mandatory `source_core_sha256`!"
                )
            if str(rec_core).strip().lower() != rec.source_core_sha256.lower():
                raise ValueError(
                    f"Record source_core_sha256={rec_core!r} != SplitContract {rec.source_core_sha256!r} "
                    f"on tile {rec.tile_id}."
                )

    def get_mesh_provenance_sha256_for_tile(self, tile_id: str) -> str:
        """Compute deterministic `mesh_provenance_sha256` for `tile_id` without decoding 52 weeks."""
        rec = self.contract.assert_tile_readable(tile_id)
        _, m_doc, roots, bundle_root = self._resolve_tile_manifest_and_roots(rec)
        is_synth = bool(m_doc.get("synthetic") is True)
        records = sorted(m_doc["records"], key=lambda r: str(r["window_id"]))
        r0 = records[0]
        static_path = roots[r0["static"].get("root", "dataset")] / r0["static"]["file"]
        local_scene_npz = static_path.parent / "SCENE_TRIANGLES.npz"
        if is_synth and local_scene_npz.exists():
            return self._assert_file_sha256(local_scene_npz, None, "SCENE_TRIANGLES.npz")

        shared_dir = self._discover_mesh_shared_dir(bundle_root)
        if shared_dir is None:
            raise FileNotFoundError(
                "Formal 4km shared DS3MESH root (`mesh_spatial_index.npz` + `mesh_source_index.json`) "
                "was not found!"
            )
        cache_key = str(shared_dir.resolve())
        if cache_key in self._spatial_index_cache:
            return self._spatial_index_cache[cache_key][3]
        src_idx_path = shared_dir / "mesh_source_index.json"
        sp_idx_path = shared_dir / "mesh_spatial_index.npz"
        if not src_idx_path.exists() or not sp_idx_path.exists():
            raise FileNotFoundError(
                f"Incomplete shared mesh root {shared_dir}: missing source or spatial index."
            )
        bundle_manifest = shared_dir / "mesh" / "mesh-bundle-manifest.json"
        prov_parts = [
            compute_file_sha256(src_idx_path),
            compute_file_sha256(sp_idx_path),
        ]
        if bundle_manifest.exists():
            prov_parts.append(compute_file_sha256(bundle_manifest))
        return hashlib.sha256("|".join(prov_parts).encode("utf-8")).hexdigest()

    def _discover_mesh_shared_dir(self, bundle_root: Optional[Path]) -> Optional[Path]:
        if self._explicit_mesh_shared_dir is not None:
            if not self._explicit_mesh_shared_dir.exists():
                raise FileNotFoundError(
                    f"Explicit global_mesh_shared_dir does not exist: {self._explicit_mesh_shared_dir}"
                )
            return self._explicit_mesh_shared_dir
        if bundle_root is not None:
            cand = bundle_root / "shared" / "static" / "dataset" / "shared"
            if (cand / "mesh_spatial_index.npz").exists():
                return cand
        for raw_g in (
            r"G:\Deepsunlight\generated\static-geometry-v2-20261009-r01\dataset\shared",
            "/mnt/g/Deepsunlight/generated/static-geometry-v2-20261009-r01/dataset/shared",
        ):
            p = Path(raw_g)
            if (p / "mesh_spatial_index.npz").exists():
                return p
        return None

    def _get_chunk_ground_mask(self, mesh_path: Path, nt: int) -> np.ndarray:
        """Resolve per-triangle ground/non-building mask from `triangle-lineage.ndjson.gz` (cached)."""
        key = f"{mesh_path.resolve()}::{int(nt)}"
        cached = self._chunk_ground_mask_cache.get(key)
        if cached is not None:
            return cached
        lineage_gz = mesh_path.parent / "triangle-lineage.ndjson.gz"
        if not lineage_gz.exists():
            out = np.zeros((int(nt),), dtype=bool)
            self._chunk_ground_mask_cache[key] = out
            return out
        is_bldg = np.zeros((int(nt),), dtype=bool)
        saw_chunk = False
        target_name = mesh_path.name
        with gzip.open(lineage_gz, "rt", encoding="utf-8") as fh_lin:
            for raw_line in fh_lin:
                line = raw_line.strip()
                if not line:
                    continue
                rec_lin = json.loads(line)
                if str(rec_lin.get("chunk_path", "")) != target_name:
                    continue
                saw_chunk = True
                t_start = int(rec_lin.get("triangle_start", 0))
                t_cnt = int(rec_lin.get("triangle_count", 0))
                layer_name = str(rec_lin.get("source_layer", ""))
                if layer_name.startswith("Buildings::") and t_cnt > 0:
                    is_bldg[t_start : min(int(nt), t_start + t_cnt)] = True
        out = (~is_bldg) if saw_chunk else np.zeros((int(nt),), dtype=bool)
        self._chunk_ground_mask_cache[key] = out
        return out

    def _load_scene_triangles_for_tile(
        self,
        receivers_xyz: np.ndarray,
        static_dir: Path,
        bundle_root: Optional[Path],
        *,
        is_synthetic_manifest: bool = False,
        bbox_margin_m: float = DEFAULT_SCENE_HORIZON_M,
        return_lineage: bool = False,
    ) -> (
        Tuple[np.ndarray, np.ndarray, np.ndarray, str]
        | Tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray, Optional[np.ndarray]]
    ):
        """Load full `4 km` (`bbox_margin_m = 4000.0`) scene mesh with chunk SHA-256 verification (`P0-5`)."""
        local_scene_npz = static_dir / "SCENE_TRIANGLES.npz"
        if is_synthetic_manifest and local_scene_npz.exists():
            scene_sha = self._assert_file_sha256(local_scene_npz, None, "SCENE_TRIANGLES.npz")
            with np.load(local_scene_npz, allow_pickle=False) as d:
                tris = np.asarray(d["triangles"], dtype=np.float32)
                op = (
                    np.asarray(d["opaque"], dtype=bool)
                    if "opaque" in d.files
                    else np.ones((tris.shape[0],), dtype=bool)
                )
                sids = (
                    np.asarray(d["source_triangle_ids"], dtype=np.uint64)
                    if "source_triangle_ids" in d.files
                    else np.arange(tris.shape[0], dtype=np.uint64)
                )
                gmask = (
                    np.asarray(d["ground_mask"], dtype=bool)
                    if "ground_mask" in d.files
                    else np.zeros((tris.shape[0],), dtype=bool)
                )
                inst_ids = (
                    np.asarray(d["source_instance_ids"], dtype=np.int64)
                    if "source_instance_ids" in d.files
                    else None
                )
            if return_lineage:
                return tris, op, sids, scene_sha, gmask, inst_ids
            return tris, op, sids, scene_sha

        shared_dir = self._discover_mesh_shared_dir(bundle_root)
        if shared_dir is None:
            raise FileNotFoundError(
                "Formal 4km shared DS3MESH root (`mesh_spatial_index.npz` + `mesh_source_index.json`) "
                "was not found! Silent fallback to empty scene triangles is strictly prohibited (`P0-5`)."
            )
        if receivers_xyz.shape[0] == 0:
            raise ValueError("Cannot load 4km scene mesh for 0 receivers.")

        cache_key = str(shared_dir.resolve())
        if cache_key not in self._spatial_index_cache:
            src_idx_path = shared_dir / "mesh_source_index.json"
            sp_idx_path = shared_dir / "mesh_spatial_index.npz"
            if not src_idx_path.exists() or not sp_idx_path.exists():
                raise FileNotFoundError(
                    f"Incomplete shared mesh root {shared_dir}: missing source or spatial index."
                )
            bundle_manifest = shared_dir / "mesh" / "mesh-bundle-manifest.json"
            prov_parts = [
                compute_file_sha256(src_idx_path),
                compute_file_sha256(sp_idx_path),
            ]
            if bundle_manifest.exists():
                prov_parts.append(compute_file_sha256(bundle_manifest))
            mesh_prov_sha = hashlib.sha256("|".join(prov_parts).encode("utf-8")).hexdigest()

            sources = json.loads(src_idx_path.read_text(encoding="utf-8"))["chunks"]
            with np.load(sp_idx_path, allow_pickle=False) as sp:
                sp_dict = {k: sp[k] for k in sp.files}
            self._spatial_index_cache[cache_key] = (sources, sp_dict, shared_dir, mesh_prov_sha)

        sources, idx, sh_root, mesh_prov_sha = self._spatial_index_cache[cache_key]
        xmin = float(receivers_xyz[:, 0].min() - float(bbox_margin_m))
        xmax = float(receivers_xyz[:, 0].max() + float(bbox_margin_m))
        zmin = float(receivers_xyz[:, 2].min() - float(bbox_margin_m))
        zmax = float(receivers_xyz[:, 2].max() + float(bbox_margin_m))

        edge = np.array([-48.741, -21.863], dtype=np.float64)
        low = np.floor((np.array([xmin, zmin]) - edge) / 200.0).astype(np.int64)
        high = np.floor((np.array([xmax, zmax]) - edge) / 200.0).astype(np.int64)
        tiles = idx["tile_indices"]
        sel = np.flatnonzero(
            (tiles[:, 0] >= low[0])
            & (tiles[:, 0] <= high[0])
            & (tiles[:, 1] >= low[1])
            & (tiles[:, 1] <= high[1])
        )
        if sel.size == 0:
            empty_t = np.zeros((0, 3, 3), dtype=np.float32)
            empty_b = np.zeros((0,), dtype=bool)
            empty_u = np.zeros((0,), dtype=np.uint64)
            if return_lineage:
                return empty_t, empty_b, empty_u, mesh_prov_sha, empty_b.copy(), None
            return empty_t, empty_b, empty_u, mesh_prov_sha

        refs = np.unique(
            np.concatenate(
                [idx["source_triangle_refs"][idx["offsets"][q] : idx["offsets"][q + 1]] for q in sel]
            )
        )
        tris_parts: List[np.ndarray] = []
        op_parts: List[np.ndarray] = []
        sid_parts: List[np.ndarray] = []
        gmask_parts: List[np.ndarray] = []

        for chunk_id, src_info in enumerate(sources):
            mask_c = (refs >> np.uint64(32)) == np.uint64(chunk_id)
            if not np.any(mask_c):
                continue
            face_ids = (refs[mask_c] & np.uint64(0xFFFFFFFF)).astype(np.int64)
            stable = str(src_info["stable_id"])
            sub_rel = (
                stable.split("deepsunlight://nyc-dcp-v3/")[1]
                if "nyc-dcp-v3/" in stable
                else stable
            )
            mesh_path = sh_root / "mesh" / sub_rel
            # NEVER skip a missing or corrupted referenced DS3MESH chunk (`P0-5`)!
            self._assert_file_sha256(mesh_path, src_info.get("sha256"), f"DS3MESH[{chunk_id}]")

            with mesh_path.open("rb") as f:
                hdr = f.read(24)
            if len(hdr) < 24:
                raise ValueError(f"Truncated DS3MESH header in {mesh_path}")
            magic, version, nv, nt, _ = struct.unpack("<8sIIII", hdr)
            if magic != b"DS3MESH\x00" or version != 1:
                raise ValueError(f"Invalid DS3MESH header in {mesh_path}")
            if "vertex_count" in src_info and int(src_info["vertex_count"]) != int(nv):
                raise ValueError(
                    f"DS3MESH vertex_count mismatch in {mesh_path}: {nv} != {src_info['vertex_count']}"
                )
            if "triangle_count" in src_info and int(src_info["triangle_count"]) != int(nt):
                raise ValueError(
                    f"DS3MESH triangle_count mismatch in {mesh_path}: {nt} != {src_info['triangle_count']}"
                )

            verts = np.memmap(mesh_path, dtype="<f4", mode="r", offset=24, shape=(nv, 3))
            faces = np.memmap(
                mesh_path, dtype="<u4", mode="r", offset=24 + 12 * nv, shape=(nt, 3)
            )
            try:
                tri = np.array(verts[faces[face_ids]], dtype=np.float32)
            finally:
                raw_vm = getattr(verts, "_mmap", None)
                if raw_vm is not None:
                    raw_vm.close()
                raw_fm = getattr(faces, "_mmap", None)
                if raw_fm is not None:
                    raw_fm.close()
                del verts, faces

            xz = tri[:, :, [0, 2]]
            use = (
                (xz[:, :, 0].max(axis=1) >= xmin)
                & (xz[:, :, 0].min(axis=1) <= xmax)
                & (xz[:, :, 1].max(axis=1) >= zmin)
                & (xz[:, :, 1].min(axis=1) <= zmax)
            )
            if np.any(use):
                used_tris = tri[use]
                used_faces_i64 = face_ids[use]
                used_faces = used_faces_i64.astype(np.uint64)
                chunk_op = bool(src_info.get("opaque", True))
                tris_parts.append(used_tris)
                op_parts.append(np.full((used_tris.shape[0],), chunk_op, dtype=bool))
                sid_parts.append((np.uint64(chunk_id) << np.uint64(32)) | used_faces)
                if return_lineage:
                    chunk_gmask = self._get_chunk_ground_mask(mesh_path, int(nt))
                    gmask_parts.append(chunk_gmask[used_faces_i64])

        if not tris_parts:
            empty_t = np.zeros((0, 3, 3), dtype=np.float32)
            empty_b = np.zeros((0,), dtype=bool)
            empty_u = np.zeros((0,), dtype=np.uint64)
            if return_lineage:
                return empty_t, empty_b, empty_u, mesh_prov_sha, empty_b.copy(), None
            return empty_t, empty_b, empty_u, mesh_prov_sha
        cat_tris = np.concatenate(tris_parts, axis=0)
        cat_op = np.concatenate(op_parts, axis=0)
        cat_sids = np.concatenate(sid_parts, axis=0)
        if return_lineage:
            cat_gmask = (
                np.concatenate(gmask_parts, axis=0)
                if gmask_parts
                else np.zeros((cat_tris.shape[0],), dtype=bool)
            )
            return cat_tris, cat_op, cat_sids, mesh_prov_sha, cat_gmask, None
        return cat_tris, cat_op, cat_sids, mesh_prov_sha

    def _decode_record_states(
        self,
        record: Mapping[str, Any],
        roots: Mapping[str, Path],
        eligible_hw: np.ndarray,
        *,
        decode_reasons: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode all frames of a single full-day record into `(F, H, W)` states & reasons with SHA & contiguity checks."""
        h, w = int(record["shape"][0]), int(record["shape"][1])
        total_f = int(record["frames"])
        states = np.full((total_f, h, w), 3, dtype=np.uint8)
        reasons = np.zeros((total_f, h, w), dtype=np.uint16)

        chunks_sorted = sorted(record["chunks"], key=lambda c: int(c["start"]))
        expected_cursor = 0
        for chunk in chunks_sorted:
            start = int(chunk["start"])
            stop = int(chunk["stop"])
            if start != expected_cursor or stop <= start or stop > total_f:
                raise ValueError(
                    f"Non-contiguous or invalid chunk range [{start}, {stop}) for total_f={total_f} "
                    f"(expected start={expected_cursor}) in record {record.get('record_id')}"
                )
            expected_cursor = stop
            f_cnt = stop - start
            fmt = str(chunk.get("format", "dsfch2")).lower()
            root_dir = roots[chunk.get("root", "dataset")]

            if fmt in ("dsfch2", "hybrid_v2", "gpu_final_hybrid_v2"):
                c_path = root_dir / chunk["file"]
                self._assert_file_sha256(c_path, chunk.get("sha256"), f"chunk[{start}:{stop}]")
                st_part, rs_part = decode_dsfch2_container_cpu(
                    c_path,
                    expected_frames=f_cnt,
                    grid_shape=(h, w),
                    decode_reasons=decode_reasons,
                )
            elif fmt == "npy":
                s_ref = chunk["state"]
                s_path = roots[s_ref.get("root", "dataset")] / s_ref["file"]
                self._assert_file_sha256(s_path, s_ref.get("sha256"), f"state_npy[{start}:{stop}]")
                raw_st = np.load(s_path, allow_pickle=False, mmap_mode="r")
                st_part = _validate_and_cast_raw_state_array(
                    raw_st, f"state_npy[{start}:{stop}] ({record.get('record_id')})"
                )
                rs_part = np.zeros((f_cnt, h, w), dtype=np.uint16)
                if decode_reasons and "reason" in chunk:
                    r_ref = chunk["reason"]
                    r_path = roots[r_ref.get("root", "dataset")] / r_ref["file"]
                    self._assert_file_sha256(r_path, r_ref.get("sha256"), f"reason_npy[{start}:{stop}]")
                    rs_part = np.array(np.load(r_path, allow_pickle=False, mmap_mode="r"), dtype=np.uint16)
            elif fmt == "npz":
                c_path = root_dir / chunk["file"]
                self._assert_file_sha256(c_path, chunk.get("sha256"), f"chunk_npz[{start}:{stop}]")
                with np.load(c_path, allow_pickle=False) as d:
                    st_part = _validate_and_cast_raw_state_array(
                        d["ray_status_code"], f"chunk_npz[{start}:{stop}] ({record.get('record_id')})"
                    )
                    rs_part = (
                        np.asarray(d["unknown_reason_code"], dtype=np.uint16)
                        if (decode_reasons and "unknown_reason_code" in d.files)
                        else np.zeros((f_cnt, h, w), dtype=np.uint16)
                    )
            else:
                c_path = root_dir / chunk["file"]
                self._assert_file_sha256(c_path, chunk.get("sha256"), f"chunk[{start}:{stop}]")
                with c_path.open("rb") as f:
                    head8 = f.read(8)
                if head8 == b"DSFCH2\x00\x00":
                    st_part, rs_part = decode_dsfch2_container_cpu(
                        c_path,
                        expected_frames=f_cnt,
                        grid_shape=(h, w),
                        decode_reasons=decode_reasons,
                    )
                else:
                    raise ValueError(f"Unsupported chunk format {fmt!r} ({head8!r}) in {c_path}")

            st_reshaped = st_part.reshape(f_cnt, h, w)
            if not bool(np.all((st_reshaped >= 1) & (st_reshaped <= 3))):
                raise ValueError(f"Invalid raw state codes outside {{1,2,3}} in {record.get('record_id')}")
            # Verify no non-eligible pixel claims SUN/SHADOW in raw stream (`baseline_streaming` contract)
            if bool(np.any(st_reshaped[:, ~eligible_hw] != 3)):
                raise ValueError(
                    f"Raw stream contains non-UNKNOWN state on ineligible static receiver in {record.get('record_id')}"
                )
            states[start:stop] = st_reshaped
            if decode_reasons:
                reasons[start:stop] = rs_part.reshape(f_cnt, h, w)

        if expected_cursor != total_f:
            raise ValueError(
                f"Incomplete chunk coverage {expected_cursor} != {total_f} in {record.get('record_id')}"
            )
        return states, reasons

    @staticmethod
    def _validate_and_load_static_geometry(
        static_path: Path,
        mg_path: Path,
        *,
        return_surface_ids: bool = False,
    ) -> (
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        """Load and validate `STATIC_CASES.npz` & mandatory `MODEL_GEOMETRY.npz` (`P0-2`, `P1-8`)."""
        if not mg_path.exists() or mg_path == static_path:
            raise FileNotFoundError(
                f"Mandatory `MODEL_GEOMETRY.npz` is missing ({mg_path}); silent ground-only "
                f"fallback is strictly prohibited (`P0-2` / `P1-8`)."
            )
        with np.load(static_path, allow_pickle=False) as sc:
            eligible_hw = np.asarray(sc["engineering_receiver_valid"], dtype=bool)
            xyz_hw3 = np.asarray(sc["receiver_world_xyz_m"], dtype=np.float32)
            world_x_m = np.asarray(sc["world_x_m"], dtype=np.float64)
            world_z_m = np.asarray(sc["world_z_m"], dtype=np.float64)
            surface_sid_hw = np.full(eligible_hw.shape, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
            if (
                "first_surface_chunk_index" in sc.files
                and "first_surface_triangle_index" in sc.files
            ):
                f_chunk = np.asarray(sc["first_surface_chunk_index"], dtype=np.int64)
                f_tri = np.asarray(sc["first_surface_triangle_index"], dtype=np.int64)
                valid_surf = (f_chunk >= 0) & (f_tri >= 0)
                if np.any(valid_surf):
                    surface_sid_hw[valid_surf] = (
                        np.uint64(f_chunk[valid_surf]) << np.uint64(32)
                    ) | np.uint64(f_tri[valid_surf])
            elif "receiver_source_triangle_id" in sc.files:
                surface_sid_hw = np.asarray(sc["receiver_source_triangle_id"], dtype=np.uint64)
            if "native_receiver_world_xyz_m" in sc.files:
                native_xyz = np.asarray(sc["native_receiver_world_xyz_m"], dtype=np.float32)
                if not np.array_equal(xyz_hw3[eligible_hw], native_xyz[eligible_hw]):
                    raise ValueError(
                        f"Coordinate contract violation in {static_path}: "
                        f"`receiver_world_xyz_m` != `native_receiver_world_xyz_m` on eligible receivers."
                    )
                if "source_dem_surface_y_m" in sc.files:
                    surf_y = np.asarray(sc["source_dem_surface_y_m"], dtype=np.float32)
                    expected_y = (surf_y + np.float32(0.05))[eligible_hw]
                    if not np.allclose(native_xyz[..., 1][eligible_hw], expected_y, atol=1e-4):
                        raise ValueError(
                            f"Elevation contract violation in {static_path}: "
                            f"`native_receiver_world_xyz_m[..., 1]` != `source_dem_surface_y_m + 0.05`."
                        )

        with np.load(mg_path, allow_pickle=False) as mg:
            fields_8hw = np.asarray(mg["fields"], dtype=np.float32)
            geometry_known_hw = (
                np.asarray(mg["geometry_known"], dtype=bool)
                if "geometry_known" in mg.files
                else np.ones(eligible_hw.shape, dtype=bool)
            )
        if fields_8hw.shape != (8, eligible_hw.shape[0], eligible_hw.shape[1]):
            raise ValueError(
                f"MODEL_GEOMETRY fields shape {fields_8hw.shape} != (8, {eligible_hw.shape[0]}, {eligible_hw.shape[1]})"
            )
        if not np.all(np.isfinite(fields_8hw)):
            raise ValueError(f"Non-finite values found in {mg_path} `fields`!")

        if return_surface_ids:
            return (
                eligible_hw,
                xyz_hw3,
                world_x_m,
                world_z_m,
                fields_8hw,
                geometry_known_hw,
                surface_sid_hw,
            )
        return eligible_hw, xyz_hw3, world_x_m, world_z_m, fields_8hw, geometry_known_hw

    @staticmethod
    def _parse_and_validate_condition_frames(
        c_path: Path,
        expected_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Parse condition JSON and enforce `120s` cadence + unit upward sun directions (`P1-8`)."""
        c_frames = json.loads(c_path.read_text(encoding="utf-8-sig"))["frames"]
        f_cnt = len(c_frames)
        if f_cnt != int(expected_frames):
            raise ValueError(
                f"Condition frame count {f_cnt} != record frames {expected_frames} in {c_path}"
            )
        if f_cnt < 2:
            raise ValueError(f"Condition window {c_path} has < 2 frames ({f_cnt}).")

        tss = np.asarray([int(f["timestamp_unix_seconds"]) for f in c_frames], dtype=np.int64)
        diffs = np.diff(tss)
        if not bool(np.all(diffs == 120)):
            bad_dt = sorted(set(int(x) for x in diffs if x != 120))[:5]
            raise ValueError(
                f"Intra-day timestamp cadence violation in {c_path}: expected exact 120s, saw {bad_dt}"
            )

        raw_scene_suns = np.asarray(
            [f["native_direction"] if "native_direction" in f else f["direction"] for f in c_frames],
            dtype=np.float64,
        )
        s_norms = np.linalg.norm(raw_scene_suns, axis=-1, keepdims=True)
        if not bool(np.allclose(s_norms[:, 0], 1.0, atol=2e-3)):
            raise ValueError(f"Non-unit sun_dirs norm in {c_path}")
        if not bool(np.all(raw_scene_suns[:, 1] > 0.0)):
            raise ValueError(f"Non-upward (below-horizon) sun vector `sy <= 0` in {c_path}")
        scene_suns = (raw_scene_suns / s_norms).astype(np.float32)

        eun_suns = np.asarray(
            [
                f["direction"]
                if "direction" in f
                else [-f["native_direction"][0], f["native_direction"][1], -f["native_direction"][2]]
                for f in c_frames
            ],
            dtype=np.float64,
        )
        enorms = np.linalg.norm(eun_suns, axis=-1, keepdims=True)
        if not bool(np.allclose(enorms[:, 0], 1.0, atol=2e-3)) or not bool(np.all(eun_suns[:, 1] > 0.0)):
            raise ValueError(f"Invalid geographic EUN sun direction in {c_path}")
        eun_unit = eun_suns / enorms

        alts = np.asarray(
            [
                float(f["solar_altitude_deg"])
                if "solar_altitude_deg" in f
                else math.degrees(math.asin(float(np.clip(eun_unit[i, 1], -1.0, 1.0))))
                for i, f in enumerate(c_frames)
            ],
            dtype=np.float32,
        )
        azs = ((np.degrees(np.arctan2(eun_unit[:, 0], eun_unit[:, 2])) + 360.0) % 360.0).astype(
            np.float32
        )
        regs = [classify_solar_regime(float(a), float(z)) for a, z in zip(alts, azs)]
        return scene_suns, alts, azs, regs, tss

    def inspect_tile_annual_summary(
        self, tile_id: str, *, decode_week_sample: Optional[Sequence[int]] = None
    ) -> Dict[str, Any]:
        """Verify all 52 weeks (`18,808` frames, `52` unique dates, `18,756` same-day edges) on CPU."""
        if tile_id not in self.contract.tiles:
            raise KeyError(f"Tile {tile_id!r} not registered in SplitContract")
        rec = self.contract.tiles[tile_id]
        manifest_path, m_doc, roots, _ = self._resolve_tile_manifest_and_roots(rec)
        records = sorted(m_doc["records"], key=lambda r: str(r["window_id"]))
        if len(records) != EXPECTED_ANNUAL_WEEKS_PER_TILE:
            raise RuntimeError(f"Tile {tile_id} has {len(records)} weeks != 52")

        window_ids_seen: Set[str] = set()
        utc_dates_seen: Set[str] = set()
        win_ts_pairs: Set[Tuple[str, int]] = set()
        all_day_ids: List[np.ndarray] = []
        all_tss: List[np.ndarray] = []

        total_frames = 0
        regime_frame_counts: Dict[str, int] = {r: 0 for r in SOLAR_REGIME_ORDER}
        min_day_frames = 10**9
        max_day_frames = 0

        r0 = records[0]
        self._validate_record_identity(r0, rec)
        st_path = roots[r0["static"].get("root", "dataset")] / r0["static"]["file"]
        self._assert_file_sha256(st_path, r0["static"].get("sha256"), "STATIC_CASES.npz")
        if "model_geometry" not in r0:
            raise FileNotFoundError(f"Tile {tile_id} record is missing mandatory `model_geometry` ref.")
        mg_path = roots[r0["model_geometry"].get("root", "dataset")] / r0["model_geometry"]["file"]
        self._assert_file_sha256(mg_path, r0["model_geometry"].get("sha256"), "MODEL_GEOMETRY.npz")
        elig, _, _, _, _, _ = self._validate_and_load_static_geometry(st_path, mg_path)

        for w_idx, r in enumerate(records):
            self._validate_record_identity(r, rec)
            wid = str(r["window_id"])
            if wid in window_ids_seen:
                raise ValueError(f"Duplicate window_id {wid!r} in {tile_id}")
            window_ids_seen.add(wid)

            c_ref = r["condition"]
            c_path = roots[c_ref.get("root", "dataset")] / c_ref["file"]
            self._assert_file_sha256(c_path, c_ref.get("sha256"), f"condition[{wid}]")
            _, _, _, regs, tss = self._parse_and_validate_condition_frames(c_path, int(r["frames"]))

            f_len = int(tss.size)
            total_frames += f_len
            min_day_frames = min(min_day_frames, f_len)
            max_day_frames = max(max_day_frames, f_len)

            # Mid-window timestamp determines unique local/UTC sampling day
            mid_dt = datetime.fromtimestamp(int(tss[f_len // 2]), tz=timezone.utc).strftime("%Y-%m-%d")
            utc_dates_seen.add(mid_dt)
            for ts_v in tss.tolist():
                pair = (wid, int(ts_v))
                if pair in win_ts_pairs:
                    raise ValueError(f"Duplicate (window_id, timestamp) {pair} in {tile_id}")
                win_ts_pairs.add(pair)

            all_day_ids.append(np.full((f_len,), w_idx, dtype=np.int32))
            all_tss.append(tss)
            for reg_name in regs:
                regime_frame_counts[reg_name] += 1

        if len(utc_dates_seen) != EXPECTED_ANNUAL_WEEKS_PER_TILE:
            raise ValueError(
                f"Tile {tile_id} has {len(utc_dates_seen)} unique sampling dates != 52!"
            )
        if total_frames != EXPECTED_ANNUAL_FRAMES_PER_TILE:
            raise ValueError(
                f"Tile {tile_id} total frames {total_frames} != {EXPECTED_ANNUAL_FRAMES_PER_TILE}"
            )
        same_day_adj = build_same_day_adjacent_mask(
            np.concatenate(all_day_ids), np.concatenate(all_tss)
        )
        same_day_edges = int(same_day_adj.sum())
        if same_day_edges != (EXPECTED_ANNUAL_FRAMES_PER_TILE - EXPECTED_ANNUAL_WEEKS_PER_TILE):
            raise ValueError(
                f"Expected {EXPECTED_ANNUAL_FRAMES_PER_TILE - EXPECTED_ANNUAL_WEEKS_PER_TILE} "
                f"same-day adjacent edges, got {same_day_edges}"
            )

        decoded_check: Dict[str, Any] = {}
        if decode_week_sample:
            sampled_pf = 0
            sampled_unk = 0
            for w_i in decode_week_sample:
                st, _ = self._decode_record_states(
                    records[int(w_i)], roots, elig, decode_reasons=True
                )
                sampled_pf += int(np.sum(st != 3))
                sampled_unk += int(np.sum(st == 3))
            decoded_check = {
                "decoded_weeks": [int(x) for x in decode_week_sample],
                "valid_point_frames": sampled_pf,
                "unknown_masked_slots": sampled_unk,
            }

        return {
            "tile_id": tile_id,
            "canonical_split": rec.canonical_split,
            "spatial_group": rec.spatial_group,
            "source_batch": rec.source_batch,
            "source_core_sha256": rec.source_core_sha256,
            "manifest_path": str(manifest_path),
            "annual_weeks": len(records),
            "unique_sampling_dates": len(utc_dates_seen),
            "annual_frames": total_frames,
            "unique_window_timestamp_frames": len(win_ts_pairs),
            "same_day_adjacent_edges": same_day_edges,
            "eligible_receivers": int(elig.sum()),
            "min_frames_per_day": min_day_frames,
            "max_frames_per_day": max_day_frames,
            "regime_frame_counts": regime_frame_counts,
            "all_5_solar_regimes_present": all(v > 0 for v in regime_frame_counts.values()),
            "decoded_sample_verification": decoded_check,
        }

    def load_tile_fullday(
        self,
        tile_id: str,
        *,
        week_indices: Sequence[int] = ANNUAL_52WEEK_INDICES,
        receiver_mode: str = "sampled",
        max_receivers: int = 256,
        max_candidates: int = 32,
        num_azimuth_bins: int = 0,
        bbox_margin_m: float = DEFAULT_SCENE_HORIZON_M,
        receiver_seed: int = 20261111,
        include_reasons: bool = False,
        include_labels: bool = True,
    ) -> FullDayTileTrajectory:
        """Load complete sunrise-to-sunset daily trajectories + full `4 km` geometry for `tile_id`."""
        self.cache.check_ram_guard("load_tile_fullday:start")
        rec = self.contract.assert_tile_readable(tile_id)
        if receiver_mode not in ("sampled", "all_valid"):
            raise ValueError(f"Unknown receiver_mode {receiver_mode!r}; use 'sampled' or 'all_valid'.")

        _, m_doc, roots, bundle_root = self._resolve_tile_manifest_and_roots(rec)
        is_synth = bool(m_doc.get("synthetic") is True)
        records = sorted(m_doc["records"], key=lambda r: str(r["window_id"]))

        r0 = records[0]
        self._validate_record_identity(r0, rec)
        static_path = roots[r0["static"].get("root", "dataset")] / r0["static"]["file"]
        self._assert_file_sha256(static_path, r0["static"].get("sha256"), "STATIC_CASES.npz")
        if "model_geometry" not in r0:
            raise FileNotFoundError(f"Tile {tile_id} missing mandatory `model_geometry` entry.")
        mg_ref = r0["model_geometry"]
        mg_path = roots[mg_ref.get("root", "dataset")] / mg_ref["file"]
        self._assert_file_sha256(mg_path, mg_ref.get("sha256"), "MODEL_GEOMETRY.npz")

        (
            eligible_hw,
            xyz_hw3,
            world_x_m,
            world_z_m,
            fields_8hw,
            geometry_known_hw,
            surface_sid_hw,
        ) = self._validate_and_load_static_geometry(static_path, mg_path, return_surface_ids=True)

        elig_rc = np.argwhere(eligible_hw)
        total_elig = int(elig_rc.shape[0])
        if total_elig == 0:
            raise RuntimeError(f"Tile {tile_id} has 0 eligible static receivers!")

        if receiver_mode == "all_valid" or max_receivers <= 0 or total_elig <= int(max_receivers):
            sel_rc = elig_rc
            eff_mode = "all_valid" if total_elig == elig_rc.shape[0] and (receiver_mode == "all_valid" or max_receivers <= 0) else receiver_mode
        else:
            eff_mode = "sampled"
            tile_hash = int.from_bytes(
                hashlib.sha256(f"{tile_id}::{receiver_seed}".encode("utf-8")).digest()[:4], "little"
            )
            stride = total_elig / float(max_receivers)
            offset = (tile_hash % 997) / 997.0
            pick_idx = np.floor((np.arange(int(max_receivers)) + offset) * stride).astype(np.int64)
            pick_idx = np.clip(pick_idx, 0, total_elig - 1)
            # Guarantee strictly unique receiver coordinates
            pick_idx = np.unique(pick_idx)
            if pick_idx.size < int(max_receivers):
                rem = np.setdiff1d(np.arange(total_elig, dtype=np.int64), pick_idx, assume_unique=True)
                need = int(max_receivers) - int(pick_idx.size)
                pick_idx = np.sort(np.concatenate([pick_idx, rem[:need]]))
            sel_rc = elig_rc[pick_idx]

        receivers_xyz = xyz_hw3[sel_rc[:, 0], sel_rc[:, 1]].astype(np.float32)
        local_pack = extract_receiver_local_features_and_crops(
            fields_8hw,
            xyz_hw3,
            sel_rc,
            world_x_m,
            world_z_m,
            geometry_known_hw=geometry_known_hw,
        )

        rc_sha = hashlib.sha256(np.ascontiguousarray(sel_rc, dtype=np.int32).tobytes()).hexdigest()
        geom_key = (
            "tile_static_geom_v3",
            tile_id,
            rc_sha,
            int(max_candidates),
            int(num_azimuth_bins),
            float(bbox_margin_m),
        )

        def _load_geom() -> Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            str,
            np.ndarray,
            Optional[np.ndarray],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]:
            tris, s_op, s_ids, m_sha, s_gmask, s_inst = self._load_scene_triangles_for_tile(
                receivers_xyz,
                static_path.parent,
                bundle_root,
                is_synthetic_manifest=is_synth,
                bbox_margin_m=float(bbox_margin_m),
                return_lineage=True,
            )
            az_t, az_o, az_m, az_sids = select_azimuth_sector_candidates(
                receivers_xyz,
                tris if num_azimuth_bins else tris[:0],
                scene_opaque=s_op,
                scene_source_triangle_ids=s_ids,
                num_azimuth_bins=max(1,num_azimuth_bins),
                max_candidates=max_candidates,
                horizon_m=float(bbox_margin_m),
                return_source_ids=True,
            )
            return tris, s_op, s_ids, m_sha, s_gmask, s_inst, az_t, az_o, az_m, az_sids

        (
            scene_tris,
            scene_opaque,
            scene_sids,
            mesh_prov_sha,
            scene_gmask,
            scene_inst_ids,
            az_tris,
            az_opaque,
            az_mask,
            az_sids,
        ) = self.cache.get_or_load(geom_key, _load_geom)

        sun_list: List[np.ndarray] = []
        alt_list: List[np.ndarray] = []
        az_list: List[np.ndarray] = []
        reg_list: List[str] = []
        ts_list: List[np.ndarray] = []
        day_id_list: List[np.ndarray] = []
        truth_list: List[np.ndarray] = []
        valid_list: List[np.ndarray] = []
        bdist_list: List[np.ndarray] = []
        bvalid_list: List[np.ndarray] = []
        bpack_list: List[np.ndarray] = []
        reasons_list: List[np.ndarray] = []
        win_ids: List[str] = []
        seen_wids: Set[str] = set()
        seen_dates: Set[str] = set()
        pack_5x5 = bool(include_labels)

        for w_idx in week_indices:
            rec_w = records[int(w_idx)]
            self._validate_record_identity(rec_w, rec)
            wid = str(rec_w["window_id"])
            if wid in seen_wids:
                raise ValueError(f"Duplicate week window {wid!r} requested in {tile_id}")
            seen_wids.add(wid)
            win_ids.append(wid)

            c_ref = rec_w["condition"]
            c_path = roots[c_ref.get("root", "dataset")] / c_ref["file"]
            self._assert_file_sha256(c_path, c_ref.get("sha256"), f"condition[{wid}]")
            scene_suns, alts, azs, regs, tss = self._parse_and_validate_condition_frames(
                c_path, int(rec_w["frames"])
            )
            f_cnt = int(tss.size)
            mid_dt = datetime.fromtimestamp(int(tss[f_cnt // 2]), tz=timezone.utc).strftime("%Y-%m-%d")
            seen_dates.add(mid_dt)

            if include_labels:
                states_fhw, reasons_fhw = self._decode_record_states(
                    rec_w, roots, eligible_hw, decode_reasons=include_reasons
                )
                b_dist_fn, b_val_fn = _compute_receiver_boundary_distances_from_states(
                    states_fhw,
                    sel_rc,
                    grid_spacing_m=float(local_pack["meters_per_pixel"]),
                )
                states_fn = states_fhw[:, sel_rc[:, 0], sel_rc[:, 1]]
                v_fn = states_fn != 3
                y_fn = np.where(v_fn, (states_fn == 2).astype(np.int8), np.int8(-1))
                if pack_5x5:
                    bpack_list.append(pack_receiver_5x5_response_bitpack(states_fhw, sel_rc))
            else:
                n_sel = int(sel_rc.shape[0])
                y_fn = np.full((f_cnt, n_sel), -1, dtype=np.int8)
                v_fn = np.ones((f_cnt, n_sel), dtype=bool)
                b_dist_fn = np.full((f_cnt, n_sel), np.nan, dtype=np.float32)
                b_val_fn = np.zeros((f_cnt, n_sel), dtype=bool)
                reasons_fhw = None

            sun_list.append(scene_suns)
            alt_list.append(alts)
            az_list.append(azs)
            reg_list.extend(regs)
            ts_list.append(tss)
            day_id_list.append(np.full((f_cnt,), int(w_idx), dtype=np.int32))
            truth_list.append(y_fn)
            valid_list.append(v_fn)
            bdist_list.append(b_dist_fn)
            bvalid_list.append(b_val_fn)
            if include_reasons and reasons_fhw is not None:
                reasons_list.append(reasons_fhw[:, sel_rc[:, 0], sel_rc[:, 1]])

        if len(week_indices) == EXPECTED_ANNUAL_WEEKS_PER_TILE:
            if len(seen_dates) != EXPECTED_ANNUAL_WEEKS_PER_TILE:
                raise ValueError(
                    f"Annual 52-week load on {tile_id} saw {len(seen_dates)} unique dates != 52"
                )

        day_ids_all = np.concatenate(day_id_list, axis=0)
        ts_all = np.concatenate(ts_list, axis=0)
        if len(week_indices) == EXPECTED_ANNUAL_WEEKS_PER_TILE and ts_all.size != EXPECTED_ANNUAL_FRAMES_PER_TILE:
            raise ValueError(
                f"Annual 52-week load on {tile_id} produced {ts_all.size} frames != {EXPECTED_ANNUAL_FRAMES_PER_TILE}"
            )
        same_day_adj = build_same_day_adjacent_mask(day_ids_all, ts_all)
        all_suns = np.concatenate(sun_list, axis=0)
        az_bin_idx = sun_dirs_to_azimuth_bins(all_suns, num_azimuth_bins=max(1,num_azimuth_bins)).astype(
            np.int16
        )

        self.cache.check_ram_guard("load_tile_fullday:done")
        return FullDayTileTrajectory(
            tile_id=tile_id,
            canonical_split=rec.canonical_split,
            spatial_group=rec.spatial_group,
            source_core_sha256=rec.source_core_sha256,
            mesh_provenance_sha256=mesh_prov_sha,
            receiver_mode=eff_mode,
            total_eligible_receivers=total_elig,
            week_indices=tuple(int(w) for w in week_indices),
            window_ids=tuple(win_ids),
            day_ids=day_ids_all,
            timestamps_unix_s=ts_all,
            same_day_adjacent_mask=same_day_adj,
            sun_dirs=all_suns,
            azimuth_bin_idx=az_bin_idx,
            solar_altitude_deg=np.concatenate(alt_list, axis=0),
            solar_azimuth_deg=np.concatenate(az_list, axis=0),
            solar_regimes=tuple(reg_list),
            receivers_xyz=receivers_xyz,
            receiver_rc=sel_rc.astype(np.int64),
            truth=np.concatenate(truth_list, axis=0),
            valid_mask=np.concatenate(valid_list, axis=0),
            boundary_distance_m=np.concatenate(bdist_list, axis=0),
            boundary_distance_valid=np.concatenate(bvalid_list, axis=0),
            local_features=local_pack["local_features"],
            static_fields_8hw=local_pack["static_fields_8hw"],
            geometry_known_hw=local_pack["geometry_known_hw"],
            meters_per_pixel=local_pack["meters_per_pixel"],
            bbox_margin_m=float(bbox_margin_m),
            az_rel_triangles=az_tris,
            az_opaque=az_opaque,
            az_mask=az_mask,
            az_source_ids=az_sids,
            ray_candidates=num_azimuth_bins == 0,
            scene_triangles_xyz=scene_tris,
            scene_opaque=scene_opaque,
            scene_source_ids=scene_sids,
            reasons=np.concatenate(reasons_list, axis=0) if include_reasons else None,
            response_5x5_bitpack=np.concatenate(bpack_list, axis=0) if bpack_list else None,
            world_x_m=np.asarray(world_x_m, dtype=np.float64),
            world_z_m=np.asarray(world_z_m, dtype=np.float64),
            receiver_source_triangle_ids=np.ascontiguousarray(
                surface_sid_hw[sel_rc[:, 0], sel_rc[:, 1]], dtype=np.uint64
            ),
            scene_ground_mask=np.ascontiguousarray(scene_gmask, dtype=bool),
            scene_source_instance_ids=(
                np.ascontiguousarray(scene_inst_ids, dtype=np.int64)
                if scene_inst_ids is not None
                else None
            ),
        )

    def prepare_tile_streaming_context(
        self,
        tile_id: str,
        *,
        week_indices: Optional[Sequence[int]] = None,
        receiver_mode: str = "sampled",
        max_receivers: int = 256,
        max_candidates: int = 32,
        num_azimuth_bins: int = 0,
        bbox_margin_m: float = DEFAULT_SCENE_HORIZON_M,
        receiver_seed: int = 20261111,
    ) -> Tuple[Dict[str, Any], Any]:
        """Prepare static arrays + solar trajectory metadata and return a week-by-week generator.

        Eliminates holding 52 weeks of `(T=18808, N=27000)` `truth`/`boundary_distance_m` arrays
        simultaneously in host RAM (`< 450 MiB` peak RSS even in `all_valid` mode).
        """
        rec = self.contract.assert_tile_readable(tile_id)
        if float(bbox_margin_m) <= 0.0:
            raise ValueError(f"bbox_margin_m must be > 0, got {bbox_margin_m}")

        self.cache.check_ram_guard(f"stream_tile:{tile_id}:start")
        _, m_doc, roots, bundle_root = self._resolve_tile_manifest_and_roots(rec)
        is_synth = bool(m_doc.get("synthetic") is True)
        records = sorted(m_doc["records"], key=lambda r: str(r["window_id"]))
        if week_indices is None:
            week_indices = list(range(len(records)))
        if not week_indices:
            raise ValueError(f"Empty week_indices requested for {tile_id}")

        rec0 = records[int(week_indices[0])]
        self._validate_record_identity(rec0, rec)
        static_ref = rec0["static"]
        static_path = roots[static_ref.get("root", "dataset")] / static_ref["file"]
        self._assert_file_sha256(static_path, static_ref.get("sha256"), f"static[{tile_id}]")
        if "model_geometry" not in rec0:
            raise FileNotFoundError(f"Tile {tile_id} missing mandatory `model_geometry` entry.")
        mg_ref = rec0["model_geometry"]
        mg_path = roots[mg_ref.get("root", "dataset")] / mg_ref["file"]
        self._assert_file_sha256(mg_path, mg_ref.get("sha256"), f"model_geometry[{tile_id}]")

        (
            eligible_hw,
            xyz_hw3,
            world_x_m,
            world_z_m,
            fields_8hw,
            geometry_known_hw,
            surface_sid_hw,
        ) = self._validate_and_load_static_geometry(static_path, mg_path, return_surface_ids=True)
        elig_rc = np.argwhere(eligible_hw)
        total_elig = int(elig_rc.shape[0])
        if total_elig == 0:
            raise RuntimeError(f"Tile {tile_id} has 0 eligible static receivers!")

        if receiver_mode == "all_valid" or max_receivers <= 0 or total_elig <= int(max_receivers):
            sel_rc = elig_rc
            eff_mode = (
                "all_valid"
                if total_elig == elig_rc.shape[0] and (receiver_mode == "all_valid" or max_receivers <= 0)
                else receiver_mode
            )
        else:
            eff_mode = "sampled"
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
            sel_rc = elig_rc[pick_idx]

        receivers_xyz = xyz_hw3[sel_rc[:, 0], sel_rc[:, 1]].astype(np.float32)
        local_pack = extract_receiver_local_features_and_crops(
            fields_8hw,
            xyz_hw3,
            sel_rc,
            world_x_m,
            world_z_m,
            geometry_known_hw=geometry_known_hw,
        )

        def _compute_geom() -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            tris, s_op, s_ids, m_sha = self._load_scene_triangles_for_tile(
                receivers_xyz,
                static_path.parent,
                bundle_root,
                is_synthetic_manifest=is_synth,
                bbox_margin_m=float(bbox_margin_m),
            )
            az_t, az_o, az_m, az_sids = select_azimuth_sector_candidates(
                receivers_xyz,
                tris if num_azimuth_bins else tris[:0],
                scene_opaque=s_op,
                scene_source_triangle_ids=s_ids,
                num_azimuth_bins=max(1,num_azimuth_bins),
                max_candidates=max_candidates,
                horizon_m=float(bbox_margin_m),
                return_source_ids=True,
            )
            return m_sha, az_t, az_o, az_m, az_sids, (tris, s_op, s_ids) if not num_azimuth_bins else None

        # Avoid retaining large `all_valid` candidate banks inside the in-memory LRU cache
        if int(sel_rc.shape[0]) <= 2048:
            rc_sha = hashlib.sha256(np.ascontiguousarray(sel_rc, dtype=np.int32).tobytes()).hexdigest()
            geom_key = (
                "tile_stream_geom_v2",
                tile_id,
                rc_sha,
                int(max_candidates),
                int(num_azimuth_bins),
                float(bbox_margin_m),
            )
            mesh_prov_sha, az_tris, az_opaque, az_mask, az_sids, ray_source = self.cache.get_or_load(
                geom_key, _compute_geom
            )
        else:
            mesh_prov_sha, az_tris, az_opaque, az_mask, az_sids, ray_source = _compute_geom()

        # Pass 1: validate tiny solar condition files across weeks (< 1 MiB total) to know exact num_frames
        sun_list: List[np.ndarray] = []
        alt_list: List[np.ndarray] = []
        az_list: List[np.ndarray] = []
        reg_list: List[str] = []
        ts_list: List[np.ndarray] = []
        day_id_list: List[np.ndarray] = []
        win_ids: List[str] = []
        week_frame_spans: List[Tuple[int, int, int]] = []
        seen_wids: Set[str] = set()
        seen_dates: Set[str] = set()
        cursor_f = 0

        for w_idx in week_indices:
            rec_w = records[int(w_idx)]
            self._validate_record_identity(rec_w, rec)
            wid = str(rec_w["window_id"])
            if wid in seen_wids:
                raise ValueError(f"Duplicate week window {wid!r} requested in {tile_id}")
            seen_wids.add(wid)
            win_ids.append(wid)

            c_ref = rec_w["condition"]
            c_path = roots[c_ref.get("root", "dataset")] / c_ref["file"]
            self._assert_file_sha256(c_path, c_ref.get("sha256"), f"condition[{wid}]")
            scene_suns, alts, azs, regs, tss = self._parse_and_validate_condition_frames(
                c_path, int(rec_w["frames"])
            )
            f_cnt = int(tss.size)
            mid_dt = datetime.fromtimestamp(int(tss[f_cnt // 2]), tz=timezone.utc).strftime("%Y-%m-%d")
            seen_dates.add(mid_dt)

            sun_list.append(scene_suns)
            alt_list.append(alts)
            az_list.append(azs)
            reg_list.extend(regs)
            ts_list.append(tss)
            day_id_list.append(np.full((f_cnt,), int(w_idx), dtype=np.int32))
            week_frame_spans.append((int(w_idx), cursor_f, cursor_f + f_cnt))
            cursor_f += f_cnt

        if len(week_indices) == EXPECTED_ANNUAL_WEEKS_PER_TILE:
            if len(seen_dates) != EXPECTED_ANNUAL_WEEKS_PER_TILE:
                raise ValueError(
                    f"Annual 52-week load on {tile_id} saw {len(seen_dates)} unique dates != 52"
                )

        day_ids_all = np.concatenate(day_id_list, axis=0)
        ts_all = np.concatenate(ts_list, axis=0)
        if len(week_indices) == EXPECTED_ANNUAL_WEEKS_PER_TILE and ts_all.size != EXPECTED_ANNUAL_FRAMES_PER_TILE:
            raise ValueError(
                f"Annual 52-week load on {tile_id} produced {ts_all.size} frames != {EXPECTED_ANNUAL_FRAMES_PER_TILE}"
            )
        all_suns = np.concatenate(sun_list, axis=0)
        az_bin_idx = sun_dirs_to_azimuth_bins(all_suns, num_azimuth_bins=max(1,num_azimuth_bins)).astype(
            np.int16
        )
        # Streaming week-by-week writes directly to disk memmap, so 5x5 response bitpacks are
        # always available for `all_valid` fit shards (`N ~ 27,000`) with `< 16 MiB` transient RAM!
        pack_5x5 = True
        mpp = float(local_pack["meters_per_pixel"])

        static_ctx: Dict[str, Any] = {
            "tile_id": tile_id,
            "canonical_split": rec.canonical_split,
            "spatial_group": rec.spatial_group,
            "source_core_sha256": rec.source_core_sha256,
            "mesh_provenance_sha256": mesh_prov_sha,
            "receiver_mode": eff_mode,
            "total_eligible_receivers": total_elig,
            "week_indices": tuple(int(w) for w in week_indices),
            "window_ids": tuple(win_ids),
            "day_ids": day_ids_all,
            "timestamps_unix_s": ts_all,
            "sun_dirs": all_suns,
            "azimuth_bin_idx": az_bin_idx,
            "solar_altitude_deg": np.concatenate(alt_list, axis=0),
            "solar_azimuth_deg": np.concatenate(az_list, axis=0),
            "solar_regimes": tuple(reg_list),
            "receivers_xyz": receivers_xyz,
            "receiver_rc": sel_rc.astype(np.int64),
            "local_features": local_pack["local_features"],
            "static_fields_8hw": local_pack["static_fields_8hw"],
            "geometry_known_hw": local_pack["geometry_known_hw"],
            "meters_per_pixel": mpp,
            "bbox_margin_m": float(bbox_margin_m),
            "az_rel_triangles": az_tris,
            "az_opaque": az_opaque,
            "az_mask": az_mask,
            "az_source_ids": az_sids,
            "has_response_5x5_bitpack": pack_5x5,
            "num_frames": int(ts_all.size),
            "num_receivers": int(sel_rc.shape[0]),
        }

        if ray_source is not None:
            static_ctx.update(zip(("scene_triangles_xyz","scene_opaque","scene_source_ids"),ray_source))
            static_ctx["receiver_source_triangle_ids"] = surface_sid_hw[sel_rc[:, 0], sel_rc[:, 1]]
        def _iter_weeks() -> Iterator[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
            for w_idx, f0, f1 in week_frame_spans:
                self.cache.check_ram_guard(f"stream_tile:{tile_id}:w{w_idx}")
                rec_w = records[int(w_idx)]
                states_fhw, _ = self._decode_record_states(
                    rec_w, roots, eligible_hw, decode_reasons=False
                )
                if states_fhw.shape[0] != (f1 - f0):
                    raise ValueError(
                        f"Frame count mismatch in week {w_idx} of {tile_id}: {states_fhw.shape[0]} != {f1 - f0}"
                    )
                b_dist_fn, b_val_fn = _compute_receiver_boundary_distances_from_states(
                    states_fhw,
                    sel_rc,
                    grid_spacing_m=mpp,
                )
                states_fn = states_fhw[:, sel_rc[:, 0], sel_rc[:, 1]]
                v_fn = states_fn != 3
                y_fn = np.where(v_fn, (states_fn == 2).astype(np.int8), np.int8(-1))
                bpack_fn = (
                    pack_receiver_5x5_response_bitpack(states_fhw, sel_rc)
                    if pack_5x5
                    else None
                )
                del states_fhw, states_fn
                yield f0, f1, y_fn, v_fn, b_dist_fn, b_val_fn, bpack_fn

        return static_ctx, _iter_weeks


def _extruded_box_triangles(
    cx: float, cz: float, half_w: float, half_d: float, y0: float, y1: float
) -> np.ndarray:
    """Create 10 watertight triangles (4 walls x 2 + 1 flat roof x 2) for a 3D building."""
    x0, x1 = cx - half_w, cx + half_w
    z0, z1 = cz - half_d, cz + half_d
    b00 = [x0, y0, z0]
    b10 = [x1, y0, z0]
    b11 = [x1, y0, z1]
    b01 = [x0, y0, z1]
    t00 = [x0, y1, z0]
    t10 = [x1, y1, z0]
    t11 = [x1, y1, z1]
    t01 = [x0, y1, z1]
    tris = [
        [t00, t10, t11],
        [t00, t11, t01],
        [b00, b10, t10],
        [b00, t10, t00],
        [b10, b11, t11],
        [b10, t11, t10],
        [b11, b01, t01],
        [b11, t01, t11],
        [b01, b00, t00],
        [b01, t00, t01],
    ]
    return np.asarray(tris, dtype=np.float32)


def create_synthetic_fullday_fixture(
    root_dir: Path | str,
    *,
    grid_side: int = 24,
    frames_per_day: int = 30,
    num_fit_tiles: int = 8,
    num_val_tiles: int = 2,
    num_cal_tiles: int = 2,
) -> Tuple[Path, Path]:
    """Build a self-contained `MANHATTAN_4445_ANNUAL_18808_V1`-compatible synthetic smoke fixture (`P2-3`).

    Explicitly labeled `"fixture_kind": "synthetic_smoke"` and `"synthetic": True`, with explicit
    tri-state `"historical_exposure_status"` (`"exposed"` / `"unexposed"`), `source_core_sha256`,
    mandatory `MODEL_GEOMETRY.npz`, 52 unique weekly dates summing to `18,808` frames, all 5 solar
    regimes, and exact `float64` Möller–Trumbore shadows + `UNKNOWN` (`states == 3`) cells.
    """
    base = Path(root_dir)
    base.mkdir(parents=True, exist_ok=True)

    c_weeks = set(CANONICAL_4SEASON_WEEK_INDICES)
    rem_total = EXPECTED_ANNUAL_FRAMES_PER_TILE - 4 * int(frames_per_day)
    q_rem, r_rem = divmod(rem_total, 48)
    week_lengths = []
    rem_seen = 0
    for w in range(52):
        if w in c_weeks:
            week_lengths.append(int(frames_per_day))
        else:
            week_lengths.append(q_rem + (1 if rem_seen < r_rem else 0))
            rem_seen += 1
    assert sum(week_lengths) == EXPECTED_ANNUAL_FRAMES_PER_TILE

    specs: List[Tuple[str, str, str, str, str]] = []
    for i in range(int(num_fit_tiles)):
        specs.append(
            (f"core_fi-01{i:02d}-fj-0100", "fit", f"blk_r1{i}_c10", "unexposed", "fresh_unexposed")
        )
    specs.append(("core_fi-0029-fj-0043", "fit", "blk_r29_c43", "exposed", "legacy_archive"))
    for i in range(int(num_cal_tiles)):
        specs.append(
            (f"core_fi-02{i+1:02d}-fj-0200", "calibration", f"blk_r2{i}_c20", "unexposed", "fresh_unexposed")
        )
    for i in range(int(num_val_tiles)):
        specs.append(
            (f"core_fi-03{i+1:02d}-fj-0300", "val", f"blk_r3{i}_c30", "unexposed", "r05-3")
        )
    specs.extend(
        [
            ("core_fi-0401-fj-0400", "test", "blk_r40_c40", "unexposed", "expansion"),
            ("core_fi-0501-fj-0500", "test", "blk_r50_c50", "exposed", "r05aa"),
        ]
    )

    split_tiles: Dict[str, Any] = {}
    manifest_tiles: Dict[str, Any] = {}

    shared_cond_dir = base / "shared_conditions"
    shared_cond_dir.mkdir(parents=True, exist_ok=True)
    cond_meta_by_week: List[Tuple[str, str, np.ndarray]] = []
    base_unix = 1704283200
    for w_idx, f_len in enumerate(week_lengths):
        u_grid = np.linspace(0.0, 1.0, f_len)
        az_deg = 95.0 + 170.0 * u_grid
        alt_deg = 4.0 + 51.0 * np.sin(math.pi * u_grid)
        alt_rad = np.radians(alt_deg)
        az_rad = np.radians(az_deg)
        sx_e = np.cos(alt_rad) * np.sin(az_rad)
        sy_u = np.sin(alt_rad)
        sz_n = np.cos(alt_rad) * np.cos(az_rad)
        eun = np.stack([sx_e, sy_u, sz_n], axis=-1)
        eun /= np.linalg.norm(eun, axis=-1, keepdims=True)
        frames_json = []
        w_ts = base_unix + w_idx * 7 * 86400
        for fi in range(f_len):
            vec = [float(eun[fi, 0]), float(eun[fi, 1]), float(eun[fi, 2])]
            frames_json.append(
                {
                    "frame_index": fi,
                    "timestamp_unix_seconds": int(w_ts + fi * 120),
                    "direction": vec,
                    "native_direction": vec,
                    "solar_altitude_deg": float(alt_deg[fi]),
                }
            )
        fname = f"weekly_{w_idx:02d}.json"
        fpath = shared_cond_dir / fname
        fpath.write_text(json.dumps({"frames": frames_json}), encoding="utf-8")
        cond_meta_by_week.append((fname, compute_file_sha256(fpath), eun.astype(np.float64)))

    cell_m = 100.0 / float(grid_side)
    wx = (np.arange(grid_side, dtype=np.float64) + 0.5) * cell_m
    wz = (np.arange(grid_side, dtype=np.float64) + 0.5) * cell_m
    gx, gz = np.meshgrid(wx, wz)

    for idx_t, (tid, auth_role, sgroup, exp_status, sbatch) in enumerate(specs):
        t_dir = base / "tiles" / tid
        st_dir = t_dir / "static"
        ch_dir = t_dir / "chunks"
        st_dir.mkdir(parents=True, exist_ok=True)
        ch_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(1000 + idx_t)
        b1 = _extruded_box_triangles(
            32.0 + rng.uniform(-3, 3), 45.0, 7.0, 7.0, 0.0, 24.0 + idx_t % 5
        )
        b2 = _extruded_box_triangles(
            68.0 + rng.uniform(-3, 3), 45.0, 6.5, 8.0, 0.0, 20.0 + (idx_t * 2) % 7
        )
        b3 = _extruded_box_triangles(50.0, 30.0 + rng.uniform(-2, 2), 8.0, 6.0, 0.0, 28.0)
        scene_tris = np.concatenate([b1, b2, b3], axis=0)
        sids = np.arange(scene_tris.shape[0], dtype=np.uint64) + np.uint64((idx_t + 1) << 16)
        inst_ids = np.concatenate(
            [
                np.zeros((b1.shape[0],), dtype=np.int64),
                np.ones((b2.shape[0],), dtype=np.int64),
                np.full((b3.shape[0],), 2, dtype=np.int64),
            ],
            axis=0,
        )
        np.savez_compressed(
            st_dir / "SCENE_TRIANGLES.npz",
            triangles=scene_tris,
            opaque=np.ones((scene_tris.shape[0],), dtype=bool),
            source_triangle_ids=sids,
            ground_mask=np.zeros((scene_tris.shape[0],), dtype=bool),
            source_instance_ids=inst_ids,
        )

        terrain_y = np.zeros((grid_side, grid_side), dtype=np.float32)
        xyz_hw3 = np.stack([gx, terrain_y + np.float32(0.05), gz], axis=-1).astype(np.float32)
        elig_hw = np.ones((grid_side, grid_side), dtype=bool)
        elig_hw[:2, :2] = False

        b_height = np.zeros((grid_side, grid_side), dtype=np.float32)
        for tri in scene_tris:
            cx_t, cz_t = float(tri[:, 0].mean()), float(tri[:, 2].mean())
            ci = int(np.clip(cx_t / cell_m, 0, grid_side - 1))
            ri = int(np.clip(cz_t / cell_m, 0, grid_side - 1))
            b_height[ri, ci] = max(b_height[ri, ci], float(tri[:, 1].max()))

        sc_path = st_dir / "STATIC_CASES.npz"
        np.savez_compressed(
            sc_path,
            terrain_world_y_m=terrain_y,
            terrain_valid=np.ones_like(elig_hw),
            building_top_world_y_m=b_height,
            building_top_valid=(b_height > 0),
            building_height_m=b_height,
            building_height_valid=np.ones_like(elig_hw),
            building_mask=(b_height > 0),
            unknown_geometry_mask=np.zeros_like(elig_hw),
            geometry_coverage_mask=np.ones_like(elig_hw),
            engineering_receiver_valid=elig_hw,
            receiver_valid=elig_hw,
            receiver_world_xyz_m=xyz_hw3,
            native_receiver_world_xyz_m=xyz_hw3,
            source_dem_surface_y_m=terrain_y,
            world_x_m=wx,
            world_z_m=wz,
        )
        fields_8 = np.zeros((8, grid_side, grid_side), dtype=np.float32)
        # Put distinct coordinate ramps in channels 0 and 4 so centered crop exactness is verifiable!
        fields_8[0] = (gx / 100.0).astype(np.float32)
        fields_8[1] = b_height / 100.0
        fields_8[2] = b_height / 100.0
        fields_8[3] = 1.0
        fields_8[4] = (gz / 100.0).astype(np.float32)
        fields_8[5] = 1.0
        fields_8[7] = 1.0
        mg_path = st_dir / "MODEL_GEOMETRY.npz"
        np.savez_compressed(
            mg_path,
            fields=fields_8,
            roof=b_height,
            geometry_known=np.ones_like(elig_hw),
            context_boxes_m=np.zeros((512, 6), dtype=np.float32),
            context_box_valid=np.zeros((512,), dtype=bool),
            query_source_indices=np.zeros((grid_side, grid_side, 2), dtype=np.int32),
            source_world_x_m=wx,
            source_world_z_m=wz,
        )

        sc_sha = compute_file_sha256(sc_path)
        mg_sha = compute_file_sha256(mg_path)
        core_sha = hashlib.sha256(f"synthetic_core::{tid}::{sc_sha}::{mg_sha}".encode("utf-8")).hexdigest()

        flat_pts = xyz_hw3.reshape(-1, 3).astype(np.float64)
        records_json = []
        placeholder_cache: Dict[int, Tuple[Path, str]] = {}
        for w_idx, (c_fname, c_sha, eun_w) in enumerate(cond_meta_by_week):
            f_len = week_lengths[w_idx]
            if w_idx in c_weeks:
                sunlit = physical_oracle_sunlit(flat_pts, eun_w, scene_tris)
                st_u8 = np.where(sunlit.reshape(f_len, grid_side, grid_side) == 1, 2, 1).astype(
                    np.uint8
                )
                rs_u16 = np.zeros((f_len, grid_side, grid_side), dtype=np.uint16)
                st_u8[:, ~elig_hw] = 3
                rs_u16[:, ~elig_hw] = 1
                if grid_side > 2:
                    st_u8[:, 2, 2] = 3
                    rs_u16[:, 2, 2] = 2
                c_bytes = encode_dsfch2_container_bytes(st_u8, rs_u16)
                ch_file = f"week_{w_idx:02d}.dsfch2"
                ch_full = ch_dir / ch_file
                ch_full.write_bytes(c_bytes)
                ch_sha = hashlib.sha256(c_bytes).hexdigest()
                ch_root = "chunks"
            else:
                if f_len not in placeholder_cache:
                    st_u8 = np.ones((f_len, grid_side, grid_side), dtype=np.uint8)
                    rs_u16 = np.zeros((f_len, grid_side, grid_side), dtype=np.uint16)
                    st_u8[:, ~elig_hw] = 3
                    rs_u16[:, ~elig_hw] = 1
                    c_bytes = encode_dsfch2_container_bytes(st_u8, rs_u16)
                    shared_ch = st_dir / f"const_{f_len}.dsfch2"
                    shared_ch.write_bytes(c_bytes)
                    placeholder_cache[f_len] = (shared_ch, hashlib.sha256(c_bytes).hexdigest())
                shared_ch, ch_sha = placeholder_cache[f_len]
                ch_file = shared_ch.name
                ch_root = "static"

            records_json.append(
                {
                    "record_id": f"{tid}/weekly_{w_idx:02d}",
                    "family_id": tid,
                    "tile_id": tid,
                    "source_core_sha256": core_sha,
                    "window_id": f"weekly_{w_idx:02d}",
                    "role": "train",
                    "shape": [grid_side, grid_side],
                    "frames": f_len,
                    "condition": {
                        "root": "condition",
                        "file": c_fname,
                        "format": "json",
                        "sha256": c_sha,
                    },
                    "static": {"root": "static", "file": "STATIC_CASES.npz", "sha256": sc_sha},
                    "model_geometry": {
                        "root": "static",
                        "file": "MODEL_GEOMETRY.npz",
                        "sha256": mg_sha,
                    },
                    "chunks": [
                        {
                            "start": 0,
                            "stop": f_len,
                            "format": "dsfch2",
                            "root": ch_root,
                            "file": ch_file,
                            "sha256": ch_sha,
                        }
                    ],
                }
            )

        tile_manifest_path = t_dir / "MANIFEST.json"
        tile_manifest_path.write_text(
            json.dumps(
                {
                    "schema": "shadow-jepa.stream.v1",
                    "units": "metres_seconds",
                    "synthetic": True,
                    "fixture_kind": "synthetic_smoke",
                    "tile_id": tid,
                    "source_core_sha256": core_sha,
                    "object_roots": {
                        "condition": str(shared_cond_dir.resolve()),
                        "static": str(st_dir.resolve()),
                        "chunks": str(ch_dir.resolve()),
                    },
                    "records": records_json,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if tid in ADDITIONAL_FIT_HOLDOUT:
            c_role = "student_holdout"
        elif auth_role == "val":
            c_role = "val_spatial_clean"
        elif auth_role == "test":
            c_role = (
                "test_spatial_unexposed"
                if exp_status == "unexposed"
                else "test_spatial_legacy_reassigned"
            )
        else:
            c_role = auth_role

        split_tiles[tid] = {
            "tile_id": tid,
            "dataset_role": "train" if auth_role in ("fit", "calibration") else auth_role,
            "training_partition": auth_role if auth_role in ("fit", "calibration") else None,
            "role": auth_role,
            "spatial_group": sgroup,
            "tile_index": [idx_t, 0],
        }
        manifest_tiles[tid] = {
            "tile_id": tid,
            "authoritative_dataset_role": "train" if auth_role in ("fit", "calibration") else auth_role,
            "authoritative_training_partition": auth_role if auth_role in ("fit", "calibration") else None,
            "canonical_role": c_role,
            "historical_exposure_status": exp_status,
            "has_historical_exposure": exp_status == "exposed",
            "spatial_group": sgroup,
            "tile_index": [idx_t, 0],
            "source_batch": sbatch,
            "source_core_sha256": core_sha,
            "annual_weeks": 52,
            "annual_frames": EXPECTED_ANNUAL_FRAMES_PER_TILE,
            "win_data_dir": str(t_dir.resolve()),
            "wsl_data_dir": str(t_dir.resolve()),
            "win_reader_entry": str(tile_manifest_path.resolve()),
            "wsl_reader_entry": str(tile_manifest_path.resolve()),
        }

    split_json_path = base / "SYNTHETIC_SPLIT_V1.json"
    split_json_path.write_text(
        json.dumps(
            {
                "benchmark_contract": BENCHMARK_CONTRACT_ID,
                "fixture_kind": "synthetic_smoke",
                "synthetic": True,
                "tiles": list(split_tiles.values()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    split_sha = compute_file_sha256(split_json_path)

    manifest_json_path = base / "SYNTHETIC_ANNUAL_18808_MANIFEST_V1.json"
    manifest_json_path.write_text(
        json.dumps(
            {
                "benchmark_contract": BENCHMARK_CONTRACT_ID,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "fixture_kind": "synthetic_smoke",
                "synthetic": True,
                "authoritative_split_contract": {"sha256": split_sha},
                "tiles": list(manifest_tiles.values()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return split_json_path, manifest_json_path

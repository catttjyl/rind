"""Direction-conditioned broad phase over shared v08 source geometry.

No ray/triangle test, labels, or predicted visibility participate in retrieval.
"""
from contextvars import ContextVar
from functools import lru_cache
from typing import Protocol, runtime_checkable
import hashlib

import numpy as np

from .v08_geometry_index import ConservativeRayIndex


require_complete = ContextVar("require_complete_ray_candidates", default=False)


@lru_cache(maxsize=1)
def load_shared_scene(paths, hashes):
    """Reuse one verified public scene across batches, releasing previous tiles."""
    return BVHCandidateProvider(*(np.load(path, mmap_mode="r") for path in paths))


@runtime_checkable
class CandidateProvider(Protocol):
    """Source-only broad phase. Results are (N,T,K) with exact pre-cap counts.

    Implementations must preserve source IDs, entry-distance/source-ID ordering,
    self-face exclusion and the requested physical horizon; no visibility Oracle
    or labels are part of this interface.
    """

    def query(self, receivers, suns, horizon, capacity, receiver_sources=None): ...


def plan_anchor_schedule(day_ids, timestamps_unix_s, stride, *, initial_phase=0,
                         carry_from_previous=False, prev_frame_same_day=True,
                         offline_interp=False):
    """Return the anchor mask and phase after every frame, in one O(T) pass."""
    days = np.asarray(day_ids).reshape(-1)
    times = np.asarray(timestamps_unix_s).reshape(-1)
    if days.shape != times.shape:
        raise ValueError("day_ids and timestamps_unix_s must have equal lengths")
    stride = max(1, int(stride))
    phase = int(initial_phase) % stride if carry_from_previous and prev_frame_same_day else 0
    mask = np.zeros(len(days), dtype=bool)
    phases = np.zeros(len(days), dtype=np.int64)
    for t in range(len(days)):
        if t and (days[t] != days[t - 1] or times[t] - times[t - 1] != 120):
            if offline_interp:
                mask[t - 1] = True
            phase = 0
        mask[t] = phase == 0
        phase = (phase + 1) % stride
        phases[t] = phase
    if offline_interp and len(mask):
        mask[-1] = True
    return mask, phases


def plan_anchor_frames(day_ids, timestamps_unix_s, stride, *, initial_phase=0,
                       carry_from_previous=False, prev_frame_same_day=True,
                       offline_interp=False):
    """Plan segment-relative absolute day phases, independently of chunk length.

    Supply the previous plan's ``next_phase`` only for a contiguous continuation.
    Each changed day or non-120-second gap starts a fresh segment and anchor.
    A continuation may contain no anchor at all.
    """
    mask, phases = plan_anchor_schedule(day_ids, timestamps_unix_s, stride,
        initial_phase=initial_phase, carry_from_previous=carry_from_previous,
        prev_frame_same_day=prev_frame_same_day, offline_interp=offline_interp)
    return mask, int(phases[-1]) if len(phases) else (
        int(initial_phase) % max(1, int(stride)) if carry_from_previous and prev_frame_same_day else 0)


def _geometry_digest(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(value)
        h.update(arr.dtype.str.encode())
        h.update(str(arr.shape).encode())
        if arr.nbytes:
            h.update(memoryview(arr).cast("B"))
    return h.hexdigest()


class BVHCandidateProvider:
    """Default conservative BVH backend with explicit geometry cache identity.

    ``update_geometry`` is required after an edit. Its content identity catches
    in-place edits as well as replacement arrays, rebuilding only when changed.
    Query results themselves are never cached across receiver/Sun changes.
    """

    def __init__(self, triangles, opaque, source_ids):
        self.geometry_identity = None
        self.scene = None
        self.update_geometry(triangles, opaque, source_ids)

    def update_geometry(self, triangles, opaque, source_ids):
        identity = _geometry_digest(triangles, opaque, source_ids)
        if identity != self.geometry_identity:
            self.scene = RayCandidateScene(triangles, opaque, source_ids)
            self.geometry_identity = identity
        return self

    @property
    def index(self):
        return self.scene.index

    def query(self, receivers, suns, horizon, capacity, receiver_sources=None):
        return self.scene.query(receivers, suns, horizon, capacity, receiver_sources)


def record_budget(summary, counts, capacity):
    summary["queries"] = summary.get("queries", 0) + counts.numel()
    summary["truncated"] = summary.get("truncated", 0) + int((counts > capacity).sum())
    summary["empty"] = summary.get("empty", 0) + int((counts == 0).sum())
    summary["max_before_cap"] = max(summary.get("max_before_cap", 0), int(counts.max()) if counts.numel() else 0)
    summary["capacity"] = capacity


def _slab(origins, directions, low, high, horizon):
    """Pairwise v08 slab bounds, retaining entry distance for budget ordering."""
    enter = np.zeros(len(origins))
    leave = np.full(len(origins), float(horizon))
    for axis in range(3):
        ray = directions[:, axis]
        q = origins[:, axis]
        parallel = ray == 0
        divisor = np.where(parallel, 1., ray)
        a = (low[:, axis] - q) / divisor
        b = (high[:, axis] - q) / divisor
        enter = np.maximum(enter, np.where(parallel, 0., np.minimum(a, b)))
        outside = (q < low[:, axis]) | (q > high[:, axis])
        leave = np.minimum(leave, np.where(parallel, np.where(outside, -1., horizon), np.maximum(a, b)))
    return enter <= leave, enter


class RayCandidateScene:
    def __init__(self, triangles, opaque, source_ids):
        self.index = ConservativeRayIndex(triangles, opaque, np.asarray(source_ids, dtype=np.uint64))
        if self.index.global_unresolved:
            raise ValueError("Unresolved nonfinite source geometry")

    def query(self, receivers, suns, horizon, capacity, receiver_sources=None):
        """Return (receiver,time,K) public geometry and exact pre-budget counts."""
        receivers = np.asarray(receivers, np.float64)
        suns = np.asarray(suns, np.float64)
        norms = np.linalg.norm(suns, axis=1)
        if (not np.isfinite(receivers).all() or not np.isfinite(suns).all()
                or (abs(norms - 1.) > 1e-3).any() or not np.isfinite(horizon)
                or horizon <= 0 or capacity < 1):
            raise ValueError("Finite origins, unit directions and positive budget/horizon required")
        suns = suns / norms[:, None]
        index = self.index
        shape = (len(receivers), len(suns), capacity)
        ids = np.zeros(shape, np.int64)
        counts = np.zeros(shape[:2], np.int64)
        if receiver_sources is not None:
            receiver_sources = np.asarray(receiver_sources, dtype=np.uint64)
        # Keep traversal arrays bounded independently of the training window.
        for start in range(0, counts.size, 128):
            rays = np.arange(start, min(start + 128, counts.size))
            nodes = np.full(len(rays), index.root)
            hits = []
            while len(nodes) and index.root >= 0:
                rx, tm = np.divmod(rays, len(suns))
                ok, _ = _slab(receivers[rx], suns[tm], index._low[nodes], index._high[nodes], horizon)
                rays, nodes = rays[ok], nodes[ok]
                left, right, lo, hi = index._links[nodes].T
                leaf = left < 0
                if leaf.any():
                    offsets = np.arange(32)
                    slots = lo[leaf, None] + offsets
                    row, col = np.nonzero(slots < hi[leaf, None])
                    face = index._order[slots[row, col]]
                    rr = rays[leaf][row]
                    rx, tm = np.divmod(rr, len(suns))
                    ok, entry = _slab(receivers[rx], suns[tm], index._face_low[face], index._face_high[face], horizon)
                    if (index._reasons[face[ok]] == 2).any():
                        raise ValueError("Unresolved near-degenerate source face on ray")
                    if receiver_sources is not None:
                        ok &= index._refs[face] != receiver_sources[rx]
                    if ok.any():
                        hits.append((rr[ok], entry[ok], face[ok]))
                rays = np.tile(rays[~leaf], 2)
                nodes = np.concatenate((left[~leaf], right[~leaf]))
            if hits:
                rr, entry, faces = (np.concatenate(items) for items in zip(*hits))
                order = np.lexsort((index._refs[faces], entry, rr))
                rr, faces = rr[order], faces[order]
                unique, first, size = np.unique(rr, return_index=True, return_counts=True)
                counts.flat[unique] = size
                slot = np.arange(len(rr)) - np.repeat(first, size)
                keep = slot < capacity
                ids.reshape(-1, capacity)[rr[keep], slot[keep]] = faces[keep]
        if require_complete.get() and np.any(counts > capacity):
            raise ValueError(f"requires_more_candidates: capacity={capacity}, required={counts.max()}")
        mask = np.arange(capacity) < counts[..., None]
        relative = np.zeros((*shape, 3, 3), np.float32)
        opaque = np.zeros(shape, bool)
        refs = np.full(shape, -1, np.int64)
        rx = np.nonzero(mask)[0]
        selected = ids[mask]
        relative[mask] = index._triangles[selected] - receivers[rx, None, :]
        opaque[mask] = index._opaque[selected]
        refs[mask] = index._refs[selected].view(np.int64)
        return relative, opaque, mask, refs, counts

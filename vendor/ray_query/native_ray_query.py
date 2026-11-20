"""Use a compiled traversal of the existing immutable c47 scene index."""
import ctypes
from pathlib import Path

import numpy as np

from rind import ray_candidates


def install(library=None):
    path = library or Path(__file__).with_name("ray_query.so")
    kernel = ctypes.CDLL(str(path)).query_rays
    ptr = ctypes.c_void_p
    kernel.argtypes = [ctypes.c_int64] * 4 + [ctypes.c_double] + [ptr] * 14
    kernel.restype = ctypes.c_int

    def query(self, receivers, suns, horizon, capacity, receiver_sources=None):
        xyz = np.ascontiguousarray(receivers, np.float64)
        sun = np.ascontiguousarray(suns, np.float64)
        if xyz.ndim != 2 or sun.ndim != 2 or xyz.shape[1] != 3 or sun.shape[1] != 3:
            raise ValueError("3D receiver coordinates and sun directions required")
        norms = np.linalg.norm(sun, axis=1)
        if (not np.isfinite(xyz).all() or not np.isfinite(sun).all()
                or (abs(norms - 1.) > 1e-3).any() or not np.isfinite(horizon)
                or horizon <= 0 or capacity < 1):
            raise ValueError("Finite origins, unit directions and positive budget/horizon required")
        sun = np.ascontiguousarray(sun / norms[:, None])
        own = None
        if receiver_sources is not None:
            own = np.ascontiguousarray(receiver_sources, np.uint64)
            if own.shape != (len(xyz),):
                raise ValueError("One original source ID per receiver required")
        index = self.index
        shape = (len(xyz), len(sun), capacity)
        ids = np.zeros(shape, np.int64)
        counts = np.zeros(shape[:2], np.int64)
        scratch = np.empty(capacity, np.float64)
        arrays = (index._low, index._high, index._links, index._order,
                  index._face_low, index._face_high, index._reasons, index._refs)
        expected = (np.float64, np.float64, np.int64, np.int64,
                    np.float64, np.float64, np.uint8, np.uint64)
        if any(not a.flags.c_contiguous or a.dtype != dt for a, dt in zip(arrays, expected)):
            raise ValueError("Unexpected c47 scene index layout")
        status = kernel(len(xyz), len(sun), capacity, index.root, horizon,
                        xyz.ctypes.data, sun.ctypes.data, own.ctypes.data if own is not None else None,
                        *(a.ctypes.data for a in arrays), ids.ctypes.data,
                        counts.ctypes.data, scratch.ctypes.data)
        if status == 1:
            raise ValueError("Unresolved near-degenerate source face on ray")
        if status:
            raise ValueError("Scene index traversal stack exhausted")
        if ray_candidates.require_complete.get() and np.any(counts > capacity):
            raise ValueError(f"requires_more_candidates: capacity={capacity}, required={counts.max()}")
        mask = np.arange(capacity) < counts[..., None]
        relative = np.zeros((*shape, 3, 3), np.float32)
        opaque = np.zeros(shape, bool)
        refs = np.full(shape, -1, np.int64)
        rx = np.nonzero(mask)[0]
        selected = ids[mask]
        relative[mask] = index._triangles[selected] - xyz[rx, None, :]
        opaque[mask] = index._opaque[selected]
        refs[mask] = index._refs[selected].view(np.int64)
        return relative, opaque, mask, refs, counts

    ray_candidates.RayCandidateScene.query = query

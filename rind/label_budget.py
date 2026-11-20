"""Receiver identity and seed-specific ranks on a fixed tile support."""
import hashlib

import numpy as np


def coordinate_keys(rc):
    rc = np.asarray(rc, dtype=np.int64)
    if rc.ndim != 2 or rc.shape[1] != 2 or np.any(rc < 0) or np.any(rc > 0x7fffffff):
        raise ValueError("expected nonnegative int32 receiver coordinates [N, 2]")
    return (rc[:, 0] << 32) | rc[:, 1]


def coordinate_scores(keys, tile_id, seed):
    salt = int.from_bytes(hashlib.sha256(f"{int(seed)}::{tile_id}".encode()).digest()[:8], "little")
    row = keys >> 32
    col = keys & 0xffffffff
    scores = (row * 6364136223846793005 + col * 1442695040888963407 + (salt & 0x7fffffff)) & 0x7fffffff
    return ((scores ^ (scores >> 15)) * 2246822519) & 0x7fffffff


class TileLabelDomain:
    """Rank once per seed on the full receiver list, then look up any subset."""

    def __init__(self, tile_id, receiver_rc):
        self.tile_id = str(tile_id)
        self.keys = np.sort(coordinate_keys(receiver_rc))
        if np.any(self.keys[1:] == self.keys[:-1]):
            raise ValueError("duplicate coordinates in tile label support")
        self.ranks = {}

    def select(self, keys, tile_id, seed, fraction):
        if str(tile_id) != self.tile_id:
            raise ValueError("tile label support belongs to another tile")
        idx = np.searchsorted(self.keys, keys)
        if np.any(idx >= len(self.keys)) or np.any(self.keys[idx] != keys):
            raise ValueError("receiver outside declared tile label support")
        if seed not in self.ranks:
            order = np.argsort(coordinate_scores(self.keys, tile_id, seed), kind="stable")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(order))
            self.ranks[seed] = ranks
        return self.ranks[seed][idx] < max(1, round(fraction * len(self.keys)))

"""Retrieval contracts: true anchor-only work, immutable geometry identities, lazy Oracle."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rind.fast_data import ContiguousTileMmapShard, build_or_open_single_tile_shard
from rind.fullday_data import FullDayTileReader, create_synthetic_fullday_fixture
from rind.ray_candidates import (BVHCandidateProvider, CandidateProvider, RayCandidateScene,
                                 plan_anchor_frames)
from rind.split_contract import SplitContract


class SpyCandidateProvider:
    def __init__(self, provider):
        self.provider = provider
        self.suns = []
        self.calls = 0

    def query(self, receivers, suns, horizon, capacity, receiver_sources=None):
        self.calls += 1
        self.suns.extend(np.asarray(suns).copy())
        return self.provider.query(receivers, suns, horizon, capacity, receiver_sources)


class BVHProviderContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        split, manifest = create_synthetic_fullday_fixture(root / "fixture", grid_side=10, frames_per_day=11)
        contract = SplitContract(split, manifest, enforce_pinned_split_sha256=False)
        cls.shard = build_or_open_single_tile_shard(root / "cache", FullDayTileReader(contract),
            contract.get_fit_tiles("fit_8")[0], week_indices=[0], max_receivers=8, max_candidates=32)
        arrays = [cls.shard._mmap_slice_copy(key) for key in
                  ("scene_triangles_xyz", "scene_opaque", "scene_source_ids")]
        cls.provider = BVHCandidateProvider(*arrays)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_stride_one_exact_legacy_bvh_parity(self):
        self.assertIsInstance(self.provider, CandidateProvider)
        shard = self.shard
        batch = shard.load_compact_batch(include_labels=False, include_receiver_crops=False)
        expected = RayCandidateScene(*[shard._mmap_slice_copy(key) for key in
            ("scene_triangles_xyz", "scene_opaque", "scene_source_ids")]).query(
                batch["receivers_xyz"].numpy(), batch["sun_dirs"].numpy(), 4000, 32,
                shard._mmap_slice_copy("receiver_source_triangle_ids"))
        for key, result in zip(("az_rel_triangles", "az_opaque", "az_mask", "az_source_ids"), expected[:4]):
            np.testing.assert_array_equal(batch[key], result)
        np.testing.assert_array_equal(batch["ray_candidate_count"], expected[4].T)

    def test_actual_queries_are_exact_global_anchors_across_chunks(self):
        shard = self.shard
        days = shard._mmap_slice_copy("day_ids")
        times = shard._mmap_slice_copy("timestamps_unix_s")
        suns = shard._mmap_slice_copy("sun_dirs")
        for stride in (2, 4):
            expected, _ = plan_anchor_frames(days, times, stride)
            for width in (1, 3, 7):
                with self.subTest(stride=stride, width=width):
                    spy = SpyCandidateProvider(self.provider)
                    batches = list(shard.iter_day_batches(days_per_batch=1,
                        receiver_chunk_size=8, max_intra_day_frames=width,
                        model_type="rind_full_geom", include_labels=False,
                        candidate_stride=stride, candidate_provider=spy))
                    queried = np.asarray(spy.suns)
                    np.testing.assert_array_equal(queried, suns[expected])
                    np.testing.assert_array_equal(torch.cat([b["candidate_anchor_mask"] for b in batches]), expected)
                    self.assertEqual(sum(b["candidate_query_frames"] for b in batches), int(expected.sum()))
                    self.assertEqual(spy.calls, sum(bool(b["candidate_query_frames"]) for b in batches))
                    if width == 1:
                        self.assertTrue(any(b["candidate_query_frames"] == 0 for b in batches))
                    for batch in batches:
                        self.assertEqual(batch["candidate_bvh_savings"],
                            1 - batch["candidate_query_frames"] / len(batch["sun_dirs"]))

    def test_real_iterated_gap_at_chunk_boundary_and_inside_chunk(self):
        shard = self.shard
        days = shard._mmap_slice_copy("day_ids")
        times = shard._mmap_slice_copy("timestamps_unix_s")
        times[3:] += 120  # frame 3 starts a 240-second gap within the same day
        suns = shard._mmap_slice_copy("sun_dirs")
        original_read = shard._mmap_slice_copy

        def read_with_gap(name, sl=slice(None)):
            if name == "timestamps_unix_s":
                return times[sl].copy()
            return original_read(name, sl)

        for stride in (2, 4):
            expected, _ = plan_anchor_frames(days, times, stride)
            self.assertTrue(expected[3])
            for width in (3, 4):  # gap exactly at boundary, then inside a chunk
                with self.subTest(stride=stride, width=width), \
                     patch.object(shard, "_mmap_slice_copy", side_effect=read_with_gap):
                    spy = SpyCandidateProvider(self.provider)
                    batches = list(shard.iter_day_batches(days_per_batch=1,
                        receiver_chunk_size=8, max_intra_day_frames=width,
                        model_type="rind_full_geom", include_labels=False,
                        candidate_stride=stride, candidate_provider=spy))
                np.testing.assert_array_equal(np.asarray(spy.suns), suns[expected])
                np.testing.assert_array_equal(torch.cat([b["candidate_anchor_mask"] for b in batches]), expected)
                self.assertEqual(sum(b["candidate_query_frames"] for b in batches), expected.sum())
                for start, _stop in [(s, e) for _d, s, e in shard.get_contiguous_day_ranges()]:
                    self.assertTrue(expected[start])  # each new day restarts at phase zero
                if width == 3:
                    after_gap = next(b for b in batches if b["frame_range"][0] == 3)
                    self.assertFalse(after_gap["prev_frame_same_day"])
                    self.assertTrue(after_gap["candidate_anchor_mask"][0])

    def test_gap_day_resets_and_empty_anchor_continuation(self):
        days = np.array([0] * 5 + [1] * 4)
        times = np.array([0, 120, 240, 600, 720, 1000, 1120, 1240, 1360])
        full, _ = plan_anchor_frames(days, times, 4)
        np.testing.assert_array_equal(np.flatnonzero(full), [0, 3, 5])
        first, phase = plan_anchor_frames(days[:1], times[:1], 4)
        empty, phase = plan_anchor_frames(days[1:3], times[1:3], 4,
            initial_phase=phase, carry_from_previous=True)
        self.assertTrue(first[0])
        self.assertFalse(empty.any())
        self.assertEqual(phase, 3)
        reset, _ = plan_anchor_frames(days[3:5], times[3:5], 4,
            initial_phase=phase, carry_from_previous=True, prev_frame_same_day=False)
        self.assertTrue(reset[0])

    def test_geometry_edit_invalidates_provider_snapshot(self):
        tri = np.array([[[-1., 1., 2.], [1., 1., 2.], [0., 3., 2.]]], np.float32)
        op = np.ones(1, bool)
        refs = np.array([17], np.uint64)
        provider = BVHCandidateProvider(tri, op, refs)
        scene = provider.scene
        old_identity = provider.geometry_identity
        receiver = np.zeros((1, 3))
        sun = np.array([[0., 1., 1.]]) / np.sqrt(2)
        self.assertEqual(provider.query(receiver, sun, 4000, 32)[4][0, 0], 1)
        provider.update_geometry(tri.copy(), op.copy(), refs.copy())
        self.assertIs(provider.scene, scene)
        tri[:, :, 0] += 100
        provider.update_geometry(tri, op, refs)
        self.assertNotEqual(provider.geometry_identity, old_identity)
        self.assertIsNot(provider.scene, scene)
        self.assertEqual(provider.query(receiver, sun, 4000, 32)[4][0, 0], 0)
        refs[0] = 99
        provider.update_geometry(tri, op, refs)
        self.assertEqual(provider.index._refs[0], 99)
        provider.update_geometry(np.empty((0, 3, 3), np.float32),
                                 np.empty(0, bool), np.empty(0, np.uint64))
        self.assertEqual(provider.query(receiver, sun, 4000, 32)[4][0, 0], 0)

    def test_4km_self_face_order_and_precap_count(self):
        x = np.arange(35, dtype=float) * 10 + 10
        tri = np.stack([np.array([[-1., a - 1, a], [1., a - 1, a], [0., a + 1, a]]) for a in x])
        # A farther source lies beyond the physical 4km ray horizon.
        tri = np.concatenate([tri, tri[:1] * 10000])
        refs = np.arange(len(tri), dtype=np.uint64)[::-1].copy()
        provider = BVHCandidateProvider(tri, np.ones(len(tri), bool), refs)
        own = refs[:1]
        _, _, mask, chosen, counts = provider.query(np.zeros((1, 3)),
            np.array([[0., 1., 1.]]) / np.sqrt(2), 4000, 32, own)
        self.assertEqual(counts[0, 0], 34)
        self.assertEqual(mask.sum(), 32)
        np.testing.assert_array_equal(chosen[0, 0], refs[1:33].astype(np.int64))

    def test_training_import_does_not_load_physical_oracle(self):
        code = "import sys, rind.fast_train, rind.cli; print('rind.inverse_design' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()

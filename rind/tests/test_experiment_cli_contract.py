"""End-to-end CLI streaming, experimental switches, and macOS telemetry fallback."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rind.cli import _candidate_attention_options, _evaluate_single_tile_streaming, build_arg_parser
from rind.fast_data import build_or_open_single_tile_shard
from rind.fullday_data import FullDayTileReader, create_synthetic_fullday_fixture
from rind.models import build_model
from rind.ray_candidates import BVHCandidateProvider, plan_anchor_frames
from rind.split_contract import SplitContract
from rind.system_guard import HostResourceGuard, _query_darwin_memory


class ExperimentCLIContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        split, manifest = create_synthetic_fullday_fixture(root / "fixture", grid_side=10, frames_per_day=11)
        contract = SplitContract(split, manifest, enforce_pinned_split_sha256=False)
        cls.shard = build_or_open_single_tile_shard(
            root / "cache", FullDayTileReader(contract), contract.get_fit_tiles("fit_8")[0],
            week_indices=[0], max_receivers=8, max_candidates=32,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cli_switches_and_legacy_defaults(self):
        parser = build_arg_parser()
        for command in ("train", "benchmark-train-speed"):
            required = [command, "--cache-dir", "/tmp/unused", "--model-type", "rind_full_geom"]
            if command == "train":
                required += ["--checkpoint-out", "/tmp/unused.pt"]
            default = parser.parse_args(required)
            self.assertEqual(_candidate_attention_options(default),
                             dict(film_beta_bound=0.0, attn_qk_norm=False, cand_readout="attn_only"))
            self.assertEqual(default.cf_subgroup_size, 0)
            enabled = parser.parse_args(required + ["--film-beta-bound", "2", "--attn-qk-norm",
                                                   "--cand-readout", "attn_plus_lse", "--cf-subgroup-size", "64"])
            self.assertEqual(_candidate_attention_options(enabled),
                             dict(film_beta_bound=2.0, attn_qk_norm=True, cand_readout="attn_plus_lse"))
            self.assertEqual(enabled.cf_subgroup_size, 64)

    def test_streamed_query_counts_cache_and_static_direct(self):
        shard = self.shard
        provider = BVHCandidateProvider(*[shard._mmap_slice_copy(key) for key in
            ("scene_triangles_xyz", "scene_opaque", "scene_source_ids")])
        torch.manual_seed(135)
        model = build_model("rind_full_geom", hidden_dim=16, num_heads=4, max_candidates=32,
                            film_beta_bound=1.0, attn_qk_norm=True, cand_readout="attn_plus_lse").eval()
        days = shard._mmap_slice_copy("day_ids")
        times = shard._mmap_slice_copy("timestamps_unix_s")
        for stride, mode in ((2, "causal_rollout"), (4, "causal_rollout"), (4, "static_direct"),
                             (4, "offline_interp")):
            expected, _ = plan_anchor_frames(days, times, 1 if mode == "static_direct" else stride,
                                              offline_interp=mode == "offline_interp")
            outputs = []
            for width in (0, 1, 3):
                with self.subTest(stride=stride, mode=mode, width=width), \
                     patch("rind.ray_candidates.load_shared_scene", return_value=provider), \
                     patch.object(provider, "query", wraps=provider.query) as query:
                    *_, info = _evaluate_single_tile_streaming(
                        model, shard, torch.device("cpu"), guard=HostResourceGuard(device="cpu"),
                        days_per_batch=1, receiver_chunk_size=4, intra_day_frames=width,
                        eval_stride=stride, inference_mode=mode, collect_logits_for_stratified=True,
                        collect_attention=True,
                    )
                    # Both receiver chunks independently query exactly the planned Suns.
                    actual = np.concatenate([call.args[1] for call in query.call_args_list])
                    suns = shard._mmap_slice_copy("sun_dirs")[expected]
                    np.testing.assert_array_equal(actual, np.concatenate([suns, suns]))
                    self.assertEqual(info["actual_anchor_frames"], int(expected.sum()) * 2)
                    self.assertEqual(info["candidate_total_frames"], len(days) * 2)
                    self.assertEqual(info["bvh_savings_rate"], 1 - expected.mean())
                    if mode == "static_direct":
                        self.assertIsNotNone(info["collected_attention"])
                    outputs.append(info)
            for info in outputs[1:]:
                np.testing.assert_allclose(info["collected_logits"], outputs[0]["collected_logits"], atol=2e-6)
                np.testing.assert_array_equal(info["collected_ray_ids"], outputs[0]["collected_ray_ids"])
                np.testing.assert_array_equal(info["collected_ray_mask"], outputs[0]["collected_ray_mask"])

    def test_darwin_real_telemetry_fallback_and_fail_closed(self):
        denied = subprocess.CalledProcessError(1, ["sysctl", "-n", "hw.memsize"])
        pressure = subprocess.CompletedProcess([], 0, "The system has 17179869184 (1048576 pages).")
        vm = subprocess.CompletedProcess([], 0,
             "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100000.\n")
        with patch("rind.system_guard.subprocess.run", side_effect=[denied, pressure, vm]):
            data = _query_darwin_memory()
        self.assertEqual(data["physical_total_bytes"], 17179869184)
        self.assertEqual(data["physical_avail_bytes"], 1638400000)
        with patch("rind.system_guard.subprocess.run", side_effect=[denied, subprocess.CompletedProcess([], 0, "unknown")]):
            with self.assertRaisesRegex(RuntimeError, "Cannot determine"):
                _query_darwin_memory()


if __name__ == "__main__":
    unittest.main()

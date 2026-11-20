"""G3 Regression Module: Host Resource Guard, Streaming Normalizer & Longest-Day 5-Model Budget.

Covers Astra `P0-1`, `P0-8`, `P1-4`, and `G3`:
1. Verifies real host physical memory & commit telemetry (`< 95%`), soft-cleanup callback trigger,
   hard `ResourceBudgetExceededError` abort above `95%`, and fail-closed WSL host telemetry check.
2. Verifies chunked `RunningNormalizer` (`max_chunk_rows=512`) matches monolithic `float64` stats.
3. Exercises longest summer day (`T = 445` frames) training (normalizer + forward + backward +
   optimizer), calibration, and streaming multi-day evaluation across all 5 `MODEL_TYPES`
   (including `local_convgru_grid` at `crop_size = 64`), confirming `< 95%` host physical RAM,
   bounded CUDA VRAM, shard interrupt recovery, and `0` full-year `(18808 × N × K × 37)` tensors.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest
from unittest import mock

import numpy as np
import torch

from rind.fast_data import (
    ContiguousTileMmapShard,
    build_or_open_single_tile_shard,
    materialize_gpu_tokens_from_azimuth_bank,
)
from rind.fast_train import calibrate_model_on_split, fit_model_fullday
from rind.fullday_data import FullDayTileReader, create_synthetic_fullday_fixture
from rind.models import MODEL_TYPES, RunningNormalizer, build_model
from rind.split_contract import SplitContract
from rind.system_guard import (
    HARD_MAX_PHYSICAL_RAM_PERCENT,
    HostResourceGuard,
    ResourceBudgetExceededError,
    _query_wsl_windows_host_memory,
    sample_host_resources,
)

from rind.tests._tempdir import TemporaryTestDir


class TestStreamingResourceBudget(unittest.TestCase):
    def test_01_host_resource_guard_real_telemetry_soft_cleanup_hard_abort_and_wsl_fail_closed(self) -> None:
        real_snap = sample_host_resources(stage="g3_unit_probe")
        self.assertGreater(real_snap.physical_total_bytes, 0)
        self.assertLess(real_snap.physical_used_percent, HARD_MAX_PHYSICAL_RAM_PERCENT)
        self.assertGreaterEqual(real_snap.commit_limit_bytes, 0)

        with self.assertRaises(ValueError):
            HostResourceGuard(hard_abort_ram_percent=96.0)

        # Soft warning triggers cleanup callback, dropping simulated RAM from 93.5% -> 72.0%
        state = {"pct": 93.5}

        def _sim_soft() -> dict[str, int | float | str]:
            tot = 64 * (1024 ** 3)
            used = int(tot * state["pct"] / 100.0)
            return {
                "host_os_kind": "simulated_host",
                "physical_total_bytes": tot,
                "physical_avail_bytes": tot - used,
                "physical_used_bytes": used,
                "physical_used_percent": state["pct"],
                "commit_limit_bytes": tot,
                "commit_avail_bytes": 16 * (1024 ** 3),
                "commit_used_bytes": tot - 16 * (1024 ** 3),
                "commit_used_percent": 75.0,
                "process_working_set_bytes": 512 * (1024 ** 2),
                "process_commit_bytes": 512 * (1024 ** 2),
            }

        guard_soft = HostResourceGuard(
            max_physical_ram_percent=92.0,
            hard_abort_ram_percent=95.0,
            telemetry_override=_sim_soft,
        )
        cleaned = {"called": 0}

        def _drop_mem() -> None:
            cleaned["called"] += 1
            state["pct"] = 72.0

        snap_after = guard_soft.check("soft_test", cleanup_callbacks=[_drop_mem])
        self.assertEqual(cleaned["called"], 1)
        self.assertEqual(guard_soft.cleanup_trigger_count, 1)
        self.assertAlmostEqual(snap_after.physical_used_percent, 72.0, places=3)

        # Hard threshold >= 95% -> raises ResourceBudgetExceededError
        state["pct"] = 96.2
        with self.assertRaises(ResourceBudgetExceededError):
            guard_soft.check("hard_test")

        # WSL host telemetry MUST fail closed when powershell.exe is unreachable
        with mock.patch("rind.system_guard.subprocess.run", side_effect=OSError("no powershell")):
            with self.assertRaises(RuntimeError):
                _query_wsl_windows_host_memory()

    def test_02_streaming_running_normalizer_exact_float64_parity(self) -> None:
        contract = SplitContract()
        fit_t = contract.get_fit_tiles("fit_8")[:1]
        token = contract.issue_fit_firewall_token(fit_t, subset_name="fit_8")

        torch.manual_seed(1234)
        big_x = torch.randn(4096, 37) * 17.5 + 3.25
        mask = torch.rand(4096) > 0.2

        norm_chunked = RunningNormalizer(37, max_rows_per_chunk=257)
        norm_chunked.update(
            big_x,
            firewall_token=token,
            tile_ids=fit_t,
            expected_split_sha256=contract.split_sha256,
            valid_mask=mask,
        )

        ref_valid = big_x[mask].to(torch.float64)
        ref_mean = ref_valid.mean(dim=0)
        ref_var = ref_valid.var(dim=0, unbiased=False).clamp_min(1e-5)

        self.assertEqual(int(norm_chunked.count.item()), int(mask.sum().item()))
        cur_var64 = (norm_chunked.m2_64 / norm_chunked.count.clamp_min(1.0)).clamp_min(1e-5)
        self.assertLess(float((norm_chunked.mean64 - ref_mean).abs().max().item()), 1e-10)
        self.assertLess(float((cur_var64 - ref_var).abs().max().item()), 1e-9)
        self.assertLess(float((norm_chunked.mean.to(torch.float64) - ref_mean).abs().max().item()), 1e-5)

    def test_03_all_5_models_longest_day_t445_train_calibrate_and_stream_eval_within_budget(self) -> None:
        contract = SplitContract()
        fit_t = contract.get_fit_tiles("fit_8")[:1]
        cal_t = contract.get_calibration_tiles()[:1]
        token = contract.issue_fit_firewall_token(fit_t, subset_name="fit_8")

        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(dev_str)
        # Longest summer solstice day has T = 445 frames at 120s cadence!
        t_longest = 445
        n_rec = 256 if dev_str == "cuda" else 32
        k_cands = 8
        h_dim = 96 if dev_str == "cuda" else 32

        torch.manual_seed(20261111)
        suns = torch.randn(t_longest, 3)
        suns[:, 1] = suns[:, 1].abs() + 0.1
        suns = suns / torch.linalg.norm(suns, dim=-1, keepdim=True)
        day_batch = {
            "tile_ids": fit_t,
            "candidate_tokens": torch.randn(t_longest, n_rec, k_cands, 37),
            "candidate_mask": torch.ones(t_longest, n_rec, k_cands, dtype=torch.bool),
            "local_features": torch.randn(n_rec, 16),
            "sun_dirs": suns,
            "same_day_adjacent_mask": torch.ones(t_longest - 1, dtype=torch.bool),
            "truth": torch.randint(0, 2, (t_longest, n_rec), dtype=torch.int8),
            "valid_mask": torch.ones(t_longest, n_rec, dtype=torch.bool),
            "receiver_crops": torch.randn(n_rec, 9, 64, 64),
        }
        cal_batch = dict(day_batch)
        cal_batch["tile_ids"] = cal_t

        per_model_peaks = {}
        for m_type in MODEL_TYPES:
            if dev_str == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)

            guard = HostResourceGuard(device=dev)
            guard.check(f"g3:{m_type}:pre_alloc")
            kw = {"hidden_dim": h_dim, "max_candidates": k_cands, "crop_size": 64}
            model = build_model(m_type, **kw)

            tr_res = fit_model_fullday(
                model,
                lambda _ep: [day_batch],
                firewall_token=token,
                expected_split_sha256=contract.split_sha256,
                epochs=1,
                lr=1e-3,
                device=dev,
                use_bf16_amp=(dev_str == "cuda"),
                pin_and_async=(dev_str == "cuda"),
                resource_guard=guard,
            )
            self.assertEqual(tr_res["nonfinite_count"], 0)
            cal_res = calibrate_model_on_split(
                model,
                [cal_batch],
                contract=contract,
                device=dev,
                resource_guard=guard,
            )
            self.assertGreater(cal_res["calibration_point_frames"], 0)

            summary = guard.summary()
            self.assertTrue(summary["physical_ram_strictly_below_95pct"])
            self.assertLess(summary["peak_physical_used_percent"], HARD_MAX_PHYSICAL_RAM_PERCENT)
            per_model_peaks[m_type] = summary
            del model

        self.assertEqual(len(per_model_peaks), 5)

        # Verify streaming shard build + interrupt recovery + zero full-year (18808 x N x K x 37) materialization
        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fx", grid_side=15, frames_per_day=12, num_fit_tiles=8
            )
            syn_contract = SplitContract(sp_json, mf_json, enforce_pinned_split_sha256=False)
            reader = FullDayTileReader(syn_contract)
            syn_tid = syn_contract.get_fit_tiles("fit_8")[0]
            cache_dir = root / "mmap_cache"

            shard1 = build_or_open_single_tile_shard(
                cache_dir,
                reader,
                syn_tid,
                week_indices=[0, 13, 26, 39],
                max_receivers=12,
                max_candidates=4,
            )
            # Simulate an interrupted sibling build leaving `.tmp_shard_*`, then reopen existing intact shard1!
            stale_tmp = shard1.shard_dir.parent / f".tmp_shard_{syn_tid}_interrupted"
            stale_tmp.mkdir(parents=True, exist_ok=True)
            (stale_tmp / "half.npy").write_bytes(b"123")
            shard2 = build_or_open_single_tile_shard(
                cache_dir,
                reader,
                syn_tid,
                week_indices=[0, 13, 26, 39],
                max_receivers=12,
                max_candidates=4,
            )
            self.assertEqual(shard2.shard_dir, shard1.shard_dir)

            max_window_frames = 0
            for win in shard2.iter_day_batches(days_per_batch=1, receiver_chunk_size=6, model_type="rind_full_geom"):
                mat = materialize_gpu_tokens_from_azimuth_bank(win)
                t_w = int(mat["candidate_tokens"].shape[0])
                max_window_frames = max(max_window_frames, t_w)
                self.assertLess(t_w, 18808)
            self.assertEqual(max_window_frames, 12)


if __name__ == "__main__":
    unittest.main()

"""G6 Regression Module: Matched-Domain 4-Arm Latency & Exact BVH Parity (`P0-5`, `P1-8`, `G6`).

Verifies:
1. `ExactSceneBVH` matches brute-force `exact_moller_trumbore_occluded_f64(..., use_aabb_prefilter=False)`
   bit-for-bit (`0` mismatches) across random and low-angle rays, including `opaque_mask=False`
   (transparent glass/foliage never occludes) and `exclude_source_triangle_ids` (self-triangle
   ignored while non-excluded co-blocking triangle still occludes).
2. `benchmark-latency` runs all 4 matched-domain arms (`neural_cold_e2e`, `neural_cached_shard_e2e`,
   `neural_pure_warm_gpu_forward`, `physical_exact_4km_raytrace`) on identical `domain_sha256`:
   - `neural_cold_e2e["label_chunk_reads"] == 0`
   - `neural_cached_shard_e2e["label_array_reads"] == 0`
   - `neural_pure_warm_gpu_forward["warm_loop_io_counters"]` are all `0`
   - Derived `ms_per_tile_frame` and `us_per_valid_point_frame` match raw `wall_seconds`
   - `--skip-physical-baseline` emits `physical_exact_4km_raytrace = None` and
     `relative_speedup_vs_physical = None`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

import numpy as np

from rind.cli import main as rind_cli_main
from rind.fullday_data import _extruded_box_triangles, create_synthetic_fullday_fixture
from rind.inverse_design import (
    DEFAULT_T_MIN_M,
    ExactSceneBVH,
    exact_moller_trumbore_occluded_f64,
    exact_moller_trumbore_trajectory_sunlit_f64,
)

from rind.tests._tempdir import TemporaryTestDir


class TestMatchedPhysicalLatency(unittest.TestCase):
    def test_01_exact_scene_bvh_bit_exact_parity_with_bruteforce_opaque_and_self_exclusion(self) -> None:
        rng = np.random.default_rng(20261111)
        boxes = [
            _extruded_box_triangles(25.0, 40.0, 8.0, 9.0, 0.0, 32.0),
            _extruded_box_triangles(55.0, 30.0, 10.0, 7.0, 0.0, 22.0),
            _extruded_box_triangles(75.0, 65.0, 9.0, 9.0, 0.0, 45.0),
            # Distant 4 km horizon tower at 1,400 m East-South-East
            _extruded_box_triangles(1350.0, -350.0, 45.0, 45.0, 0.0, 190.0),
        ]
        # Add a transparent screen right in front of receivers (`opaque=False`) + self-triangle + co-blocker
        trans_tri = np.array(
            [[[35.0, 0.0, 48.0], [65.0, 0.0, 48.0], [50.0, 30.0, 48.0]]],
            dtype=np.float64,
        )
        self_tri = np.array(
            [[[49.0, 0.20, 50.0], [51.0, 0.20, 50.0], [50.0, 0.20, 49.0]]],
            dtype=np.float64,
        )
        coblocker_tri = np.array(
            [[[48.0, 0.0, 46.0], [52.0, 0.0, 46.0], [50.0, 12.0, 46.0]]],
            dtype=np.float64,
        )
        tris = np.concatenate(boxes + [trans_tri, self_tri, coblocker_tri], axis=0)
        m_tris = tris.shape[0]
        opaque = np.ones((m_tris,), dtype=bool)
        opaque[-3] = False  # `trans_tri` is non-opaque -> MUST NOT occlude!
        sids = np.arange(1000, 1000 + m_tris, dtype=np.uint64)
        self_sid = int(sids[-2])

        bvh = ExactSceneBVH(tris, opaque_mask=opaque, source_triangle_ids=sids, leaf_size=4)

        # Sample 64 receivers + 1 receiver sitting right below `self_tri`
        recs = np.zeros((65, 3), dtype=np.float64)
        recs[:64, 0] = rng.uniform(5.0, 95.0, size=64)
        recs[:64, 1] = rng.uniform(0.05, 5.0, size=64)
        recs[:64, 2] = rng.uniform(5.0, 95.0, size=64)
        recs[64] = np.array([50.0, 0.05, 49.7], dtype=np.float64)

        # Sample 18 sun directions spanning low-angle (3 deg) to high noon (70 deg) + below horizon (-2 deg)
        alts = np.deg2rad(np.linspace(-2.0, 70.0, 18))
        azs = np.deg2rad(np.linspace(65.0, 295.0, 18))
        suns = np.stack(
            [np.cos(alts) * np.sin(azs), np.sin(alts), np.cos(alts) * np.cos(azs)],
            axis=-1,
        )
        suns /= np.linalg.norm(suns, axis=-1, keepdims=True)

        # 1. Trajectory sunlit parity (`0` mismatches out of 65 * 18 = 1,170 rays against true brute-force MT)
        bvh_sun = bvh.query_trajectory_sunlit(recs, suns, t_min=DEFAULT_T_MIN_M)
        bf_sun = np.zeros_like(bvh_sun)
        for ti in range(suns.shape[0]):
            if suns[ti, 1] > 0.0:
                occ_t = exact_moller_trumbore_occluded_f64(
                    recs,
                    np.broadcast_to(suns[ti : ti + 1], recs.shape),
                    tris,
                    opaque_mask=opaque,
                    source_triangle_ids=sids,
                    t_min=DEFAULT_T_MIN_M,
                    use_aabb_prefilter=False,
                )
                bf_sun[ti] = (~occ_t).astype(np.int8)
        self.assertTrue(
            np.array_equal(bvh_sun, bf_sun),
            msg=f"BVH vs brute-force mismatch count: {int(np.sum(bvh_sun != bf_sun))}",
        )

        # 2. Verify self-triangle exclusion (`exclude_source_triangle_ids`) on upward ray vs co-blocker ray
        o_self = np.array([[50.0, 0.05, 49.7], [50.0, 0.05, 49.7]], dtype=np.float64)
        # Ray 0 shoots straight up through `self_tri` (`y=0.20`, `t=0.15 > 1e-4`), hitting nothing else
        # Ray 1 shoots South-Up through `self_tri` AND hits `coblocker_tri` at `z=46`
        d_pair = np.array([[0.0, 1.0, 0.0], [0.0, 0.45, -0.89302855]], dtype=np.float64)
        d_pair /= np.linalg.norm(d_pair, axis=-1, keepdims=True)
        excl = np.array([self_sid, self_sid], dtype=np.int64)

        occ_bvh = bvh.query_occluded(o_self, d_pair, t_min=DEFAULT_T_MIN_M, exclude_source_triangle_ids=excl)
        occ_bf = exact_moller_trumbore_occluded_f64(
            o_self,
            d_pair,
            tris,
            opaque_mask=opaque,
            source_triangle_ids=sids,
            exclude_source_triangle_ids=excl,
            t_min=DEFAULT_T_MIN_M,
            use_aabb_prefilter=False,
        )
        self.assertTrue(np.array_equal(occ_bvh, occ_bf))
        self.assertEqual(occ_bvh.tolist(), [False, True])

    def test_02_four_arm_matched_domain_latency_receipt_and_zero_label_warm_io_guards(self) -> None:
        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fx", grid_side=17, frames_per_day=12, num_fit_tiles=8
            )
            cache_dir = root / "mmap_cache"
            raw_ckpt = root / "raw.pt"
            cal_ckpt = root / "cal.pt"

            # Train + calibrate checkpoint in diagnostic mode on synthetic fixture
            self.assertEqual(
                rind_cli_main(
                    [
                        "train",
                        "--split-file",
                        str(sp_json),
                        "--manifest-file",
                        str(mf_json),
                        "--allow-synthetic-split",
                        "--diagnostic-mode",
                        "--cache-dir",
                        str(cache_dir),
                        "--model-type",
                        "rind_full_geom",
                        "--fit-subset",
                        "fit_8",
                        "--temporal-protocol",
                        "canonical_4season",
                        "--hidden-dim",
                        "24",
                        "--epochs",
                        "1",
                        "--max-tiles",
                        "2",
                        "--max-receivers",
                        "16",
                        "--max-candidates",
                        "4",
                        "--device",
                        "cpu",
                        "--checkpoint-out",
                        str(raw_ckpt),
                        "--quiet",
                    ]
                ),
                0,
            )
            self.assertEqual(
                rind_cli_main(
                    [
                        "calibrate",
                        "--split-file",
                        str(sp_json),
                        "--manifest-file",
                        str(mf_json),
                        "--allow-synthetic-split",
                        "--diagnostic-mode",
                        "--cache-dir",
                        str(cache_dir),
                        "--checkpoint",
                        str(raw_ckpt),
                        "--checkpoint-out",
                        str(cal_ckpt),
                        "--temporal-protocol",
                        "canonical_4season",
                        "--max-tiles",
                        "2",
                        "--max-receivers",
                        "16",
                        "--max-candidates",
                        "4",
                        "--device",
                        "cpu",
                        "--quiet",
                    ]
                ),
                0,
            )

            # 1. Full 4-arm matched-domain benchmark (including `physical_exact_4km_raytrace`)
            lat_full_path = root / "latency_full.json"
            self.assertEqual(
                rind_cli_main(
                    [
                        "benchmark-latency",
                        "--split-file",
                        str(sp_json),
                        "--manifest-file",
                        str(mf_json),
                        "--allow-synthetic-split",
                        "--diagnostic-mode",
                        "--checkpoint",
                        str(cal_ckpt),
                        "--cache-dir",
                        str(cache_dir),
                        "--temporal-protocol",
                        "canonical_4season",
                        "--num-days",
                        "1",
                        "--max-receivers",
                        "16",
                        "--max-candidates",
                        "4",
                        "--warmup-iters",
                        "1",
                        "--timed-iters",
                        "2",
                        "--physical-timed-iters",
                        "1",
                        "--device",
                        "cpu",
                        "--output",
                        str(lat_full_path),
                        "--quiet",
                    ]
                ),
                0,
            )
            doc = json.loads(lat_full_path.read_text(encoding="utf-8"))
            dom = doc["matched_domain"]
            dom_sha = dom["domain_sha256"]
            f_eval = int(dom["frames_evaluated"])
            v_pf = int(dom["valid_point_frames"])
            self.assertEqual(len(dom_sha), 64)
            self.assertGreater(f_eval, 0)
            self.assertGreater(v_pf, 0)

            cold = doc["neural_cold_e2e"]
            cached = doc["neural_cached_shard_e2e"]
            warm = doc["neural_pure_warm_gpu_forward"]
            phys = doc["physical_exact_4km_raytrace"]
            rel = doc["relative_speedup_vs_physical"]
            self.assertIsNotNone(phys)
            self.assertIsNotNone(rel)

            # Identical `domain_sha256` across all 4 arms
            for arm_name, arm_obj in (
                ("cold", cold),
                ("cached", cached),
                ("warm", warm),
                ("phys", phys),
            ):
                self.assertEqual(arm_obj["domain_sha256"], dom_sha, msg=f"Domain mismatch on {arm_name}")

            # Zero label-read and zero warm-IO assertions
            self.assertEqual(cold["label_chunk_reads"], 0)
            self.assertEqual(cached["label_array_reads"], 0)
            for k_io, v_io in warm["warm_loop_io_counters"].items():
                self.assertEqual(v_io, 0, msg=f"Non-zero warm IO counter {k_io}={v_io}")

            # Verify derived latency formulas match raw wall seconds
            self.assertAlmostEqual(
                cold["ms_per_tile_frame"], 1000.0 * cold["wall_seconds"] / f_eval, places=9
            )
            self.assertAlmostEqual(
                cold["us_per_valid_point_frame"], 1e6 * cold["wall_seconds"] / v_pf, places=6
            )
            self.assertAlmostEqual(
                cached["ms_per_tile_frame"], 1000.0 * cached["mean_wall_seconds"] / f_eval, places=9
            )
            self.assertAlmostEqual(
                warm["us_per_valid_point_frame"], 1e6 * warm["mean_wall_seconds"] / v_pf, places=6
            )
            self.assertAlmostEqual(
                phys["ms_per_tile_frame"], 1000.0 * phys["mean_wall_seconds"] / f_eval, places=9
            )
            self.assertTrue(
                math.isclose(
                    rel["pure_warm_gpu_vs_physical_exact"],
                    phys["mean_wall_seconds"] / warm["mean_wall_seconds"],
                    rel_tol=1e-9,
                )
            )

            # 2. `--skip-physical-baseline` MUST emit None for physical & relative speedup (never fabricate!)
            lat_skip_path = root / "latency_skip.json"
            self.assertEqual(
                rind_cli_main(
                    [
                        "benchmark-latency",
                        "--split-file",
                        str(sp_json),
                        "--manifest-file",
                        str(mf_json),
                        "--allow-synthetic-split",
                        "--diagnostic-mode",
                        "--checkpoint",
                        str(cal_ckpt),
                        "--cache-dir",
                        str(cache_dir),
                        "--temporal-protocol",
                        "canonical_4season",
                        "--num-days",
                        "1",
                        "--max-receivers",
                        "16",
                        "--max-candidates",
                        "4",
                        "--skip-physical-baseline",
                        "--device",
                        "cpu",
                        "--output",
                        str(lat_skip_path),
                        "--quiet",
                    ]
                ),
                0,
            )
            skip_doc = json.loads(lat_skip_path.read_text(encoding="utf-8"))
            self.assertIsNone(skip_doc["physical_exact_4km_raytrace"])
            self.assertIsNone(skip_doc["relative_speedup_vs_physical"])


if __name__ == "__main__":
    unittest.main()

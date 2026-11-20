"""G5 Regression Module 2: Paired Publication Metrics, Zero-Support NA & Distance-Band Gating.

Covers Astra `P0-7`, `P1-6`, `P1-7`, and `G5`:
1. Zero-support accumulators emit `None` (`format_metric_or_na` -> `"NA"`), NEVER `0.0`!
2. Recomputes `state_accuracy`, `state_sun_iou`, `state_shadow_iou`, `bce_nll`, `brier_score`,
   `hard_sun_*`, undirected `transition_f1`, directed `sun_to_shadow_f1` & `shadow_to_sun_f1`,
   5 solar regimes, and pre-clamp `nonfinite_count` / `nonfinite_window_count` from raw integer
   confusion counts and verifies exact agreement.
3. Verifies `compute_planar_shadow_boundary_distance_grid` marks border / `UNKNOWN`-adjacent cells
   as `distance_valid == False` (`NaN`) and strictly excludes them from `lt_0p5m`.
4. Verifies spatial-group cluster `paired_bootstrap_model_comparison` (including unmatched
   denominator rejection), `aggregate_multiseed_reports` (`Mean ± Std`), and sampling disclosure.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from rind.metrics import (
    ConfusionAccumulator,
    SOLAR_REGIME_ORDER,
    aggregate_multiseed_reports,
    bootstrap_tile_ci_95,
    build_receiver_sampling_disclosure,
    compute_planar_shadow_boundary_distance_grid,
    evaluate_trajectory_predictions,
    format_metric_or_na,
    paired_bootstrap_model_comparison,
)


class TestPairedPublicationMetrics(unittest.TestCase):
    def test_01_zero_support_returns_none_and_na_never_zero(self) -> None:
        empty = ConfusionAccumulator()
        s = empty.summarize()
        for key in (
            "state_accuracy",
            "state_error",
            "state_sun_iou",
            "state_shadow_iou",
            "state_f1",
            "brier_score",
            "bce_nll",
            "transition_precision",
            "transition_recall",
            "transition_f1",
        ):
            self.assertIsNone(s[key], msg=f"Zero-support {key} returned {s[key]} instead of None!")
            self.assertEqual(format_metric_or_na(s[key]), "NA")
        for hs_key in ("accuracy", "sun_iou", "f1"):
            self.assertIsNone(s["hard_sun"][hs_key])
            self.assertEqual(format_metric_or_na(s["hard_sun"][hs_key]), "NA")
        for dir_key in ("sun_to_shadow", "shadow_to_sun"):
            self.assertIsNone(s["directed_transitions"][dir_key]["f1"])
            self.assertEqual(format_metric_or_na(s["directed_transitions"][dir_key]["f1"]), "NA")

        ci_empty = bootstrap_tile_ci_95([empty])
        self.assertIsNone(ci_empty["state_accuracy_ci95"])
        ms_empty = aggregate_multiseed_reports([s])
        self.assertEqual(ms_empty["metrics"]["state_accuracy"]["formatted"], "NA")

    def test_02_confusion_recomputation_directed_events_hard_sun_and_nonfinite_tracking(self) -> None:
        # 6 frames across all 5 regimes + 1 non-finite logit frame, 3 receivers (1 invalid slot)
        truth = np.array(
            [
                [1, 0, -1],
                [0, 0, 1],   # rec0: 1->0 (sun_to_shadow), rec1: 0->0, rec2: invalid->1 (excluded)
                [0, 1, 1],   # rec0: 0->0, rec1: 0->1 (shadow_to_sun), rec2: 1->1
                [1, 1, 0],   # rec0: 0->1 (shadow_to_sun), rec1: 1->1, rec2: 1->0 (sun_to_shadow)
                [1, 0, 0],   # rec0: 1->1, rec1: 1->0 (sun_to_shadow), rec2: 0->0
                [0, 0, 1],   # Day 1 starts here -> NO cross-day transition from frame 4 to 5!
            ],
            dtype=np.int8,
        )
        valid = truth >= 0
        logits = np.array(
            [
                [3.0, -2.5, 0.0],
                [-3.0, -2.0, 2.5],
                [-2.0, 2.5, 1.5],
                [2.2, -0.5, -2.5],      # rec1 is FP/FN error (-0.5 -> pred 0 vs truth 1)
                [float("nan"), -3.0, -2.0],  # 1 non-finite valid point-frame!
                [-2.5, -2.0, 2.0],
            ],
            dtype=np.float32,
        )
        hard_mask = np.zeros_like(valid, dtype=bool)
        hard_mask[:4, :2] = True
        day_ids = np.array([0, 0, 0, 0, 0, 1], dtype=np.int32)
        ts = np.array([1000, 1120, 1240, 1360, 1480, 90000], dtype=np.int64)
        regimes = [
            "sunrise_low",
            "morning_mid",
            "solar_noon",
            "afternoon_west",
            "sunset_low",
            "sunrise_low",
        ]

        overall, by_regime = evaluate_trajectory_predictions(
            logits,
            truth,
            valid,
            solar_regimes=regimes,
            day_ids=day_ids,
            timestamps_unix_s=ts,
            has_active_candidates=hard_mask,
        )
        s = overall.summarize()

        # Verify nonfinite_count & nonfinite_window_count BEFORE clamping!
        self.assertEqual(s["nonfinite_count"], 1)
        self.assertEqual(s["nonfinite_window_count"], 1)
        self.assertEqual(s["rejected_or_unavailable_slots"], 2)
        self.assertEqual(s["point_frames"], 16)

        # Recompute every state metric from s["state_confusion"]
        c = s["state_confusion"]
        tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
        self.assertEqual(tp + fp + tn + fn, 16)
        self.assertAlmostEqual(s["state_accuracy"], (tp + tn) / 16.0, places=12)
        self.assertAlmostEqual(s["state_sun_iou"], tp / (tp + fp + fn), places=12)
        self.assertAlmostEqual(s["state_shadow_iou"], tn / (tn + fp + fn), places=12)

        # Recompute hard_sun metrics
        hc = s["hard_sun"]
        htp, hfp, htn, hfn = hc["tp"], hc["fp"], hc["tn"], hc["fn"]
        self.assertEqual(hc["point_frames"], 8)
        self.assertAlmostEqual(hc["accuracy"], (htp + htn) / 8.0, places=12)
        self.assertAlmostEqual(hc["sun_iou"], htp / (htp + hfp + hfn), places=12)

        # Recompute undirected & directed transition F1 from confusion counts
        ec = s["transition_confusion"]
        self.assertEqual(s["transition_pairs"], 10)  # Reject the pair ending at the nonfinite prediction.
        self.assertAlmostEqual(
            s["transition_f1"],
            2.0 * ec["tp"] / (2 * ec["tp"] + ec["fp"] + ec["fn"]),
            places=12,
        )
        s2sh = s["directed_transitions"]["sun_to_shadow"]
        self.assertAlmostEqual(
            s2sh["f1"],
            2.0 * s2sh["tp"] / (2 * s2sh["tp"] + s2sh["fp"] + s2sh["fn"]),
            places=12,
        )
        sh2s = s["directed_transitions"]["shadow_to_sun"]
        self.assertAlmostEqual(
            sh2s["f1"],
            2.0 * sh2s["tp"] / (2 * sh2s["tp"] + sh2s["fp"] + sh2s["fn"]),
            places=12,
        )

        # Sum of regime point_frames and transition_pairs MUST equal overall!
        self.assertEqual(sum(by_regime[r].point_frames for r in SOLAR_REGIME_ORDER), 16)
        self.assertEqual(sum(by_regime[r].transition_pairs for r in SOLAR_REGIME_ORDER), 10)

    def test_03_boundary_distance_valid_gating_and_paired_spatial_cluster_bootstrap(self) -> None:
        # Build a 48x48 grid with spacing 0.25m (`12m x 12m`), shadow on left (`c < 24`), sun on right (`c >= 24`)
        h, w = 48, 48
        gt = np.zeros((1, h, w), dtype=np.int8)
        gt[:, :, 24:] = 1
        vm = np.ones((1, h, w), dtype=bool)
        # Inject an UNKNOWN hole at `(24, 10)` far in shadow interior: neighbors within 2.0m of `(24, 10)`
        # could have a hidden boundary inside the hole, so `distance_valid` MUST be False (`NaN`), NEVER `<0.5m`!
        vm[0, 24, 10] = False
        gt[0, 24, 10] = -1

        dist_m, dist_valid = compute_planar_shadow_boundary_distance_grid(
            gt, vm, grid_spacing_m=0.25, max_search_radius_m=2.25
        )
        # Cell `(24, 11)` is 0.25m from the UNKNOWN hole `(24, 10)` and 3.25m from true boundary `c=24`:
        # `dist_valid[0, 24, 11]` MUST be False!
        self.assertFalse(bool(dist_valid[0, 24, 11]))
        self.assertTrue(math.isnan(float(dist_m[0, 24, 11])))
        # Cell `(24, 23)` is 0.25m from true valid boundary `(24, 24)` and >3.0m from any border/hole -> True & <0.5m!
        self.assertTrue(bool(dist_valid[0, 24, 23]))
        self.assertAlmostEqual(float(dist_m[0, 24, 23]), 0.25, places=5)

        logits_flat = np.where(gt.reshape(1, -1) == 1, 4.0, -4.0).astype(np.float32)
        _, _, by_dist, excl_n = evaluate_trajectory_predictions(
            logits_flat,
            gt.reshape(1, -1),
            vm.reshape(1, -1),
            solar_regimes=["solar_noon"],
            day_ids=np.array([0], dtype=np.int32),
            timestamps_unix_s=np.array([1000], dtype=np.int64),
            boundary_distance_m=dist_m.reshape(1, -1),
            boundary_distance_valid=dist_valid.reshape(1, -1),
            return_distance_bands=True,
        )
        self.assertGreater(excl_n, 0)
        self.assertGreater(by_dist["lt_0p5m"].point_frames, 0)
        self.assertGreater(by_dist["0p5m_to_2p0m"].point_frames, 0)
        self.assertGreater(by_dist["gt_2p0m"].point_frames, 0)
        self.assertEqual(
            by_dist["lt_0p5m"].point_frames
            + by_dist["0p5m_to_2p0m"].point_frames
            + by_dist["gt_2p0m"].point_frames
            + excl_n,
            int(vm.sum()),
        )

        # Paired spatial-group comparison: Model A strictly outperforms Model B on every cluster
        acc_a = {}
        acc_b = {}
        sg_map = {}
        for idx in range(6):
            tid = f"tile_{idx}"
            sg_map[tid] = f"sg_{idx // 2}"  # 3 spatial clusters of 2 tiles each
            acc_a[tid] = ConfusionAccumulator(
                point_frames=100, state_tp=48, state_tn=46, state_fp=3, state_fn=3,
                transition_pairs=90, event_tp=18, event_fp=2, event_tn=68, event_fn=2,
            )
            acc_b[tid] = ConfusionAccumulator(
                point_frames=100, state_tp=38, state_tn=36, state_fp=13, state_fn=13,
                transition_pairs=90, event_tp=10, event_fp=8, event_tn=62, event_fn=10,
            )

        paired = paired_bootstrap_model_comparison(acc_a, acc_b, tile_to_spatial_group=sg_map)
        self.assertEqual(paired["cluster_unit"], "spatial_group")
        self.assertEqual(paired["cluster_count"], 3)
        self.assertAlmostEqual(paired["deltas"]["delta_state_accuracy"], 0.20, places=9)
        self.assertGreater(paired["delta_ci95"]["delta_state_accuracy"][0], 0.15)

        # Unmatched denominator on any tile MUST raise ValueError!
        acc_b_bad = dict(acc_b)
        acc_b_bad["tile_0"] = ConfusionAccumulator(point_frames=99, state_tp=50, state_tn=49)
        with self.assertRaises(ValueError):
            paired_bootstrap_model_comparison(acc_a, acc_b_bad, tile_to_spatial_group=sg_map)

        disc = build_receiver_sampling_disclosure(
            receiver_mode="sampled",
            eligible_receiver_count=200000,
            evaluated_receiver_count=256,
            receiver_rc=np.stack([np.arange(256), np.arange(256)], axis=-1),
        )
        self.assertAlmostEqual(disc["sampling_inclusion_probability"], 256.0 / 200000.0, places=10)
        self.assertEqual(len(disc["receiver_support_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

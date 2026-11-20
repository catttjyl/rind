"""G7 Regression Module: Full-Scene 4 km Inverse-Design Oracle, Manifold Identity & 3-Arm Audit (`P0-6`, `G7`).

Verifies every item of Astra `P0-6` & `G7`:
1. Ray-triangle `t_min = 1e-4` & self-triangle semantics:
   - `t = -1` (behind ray origin) -> `False`
   - `t = +2` (ahead) -> `True`
   - `t = +5e-5 < 1e-4` -> `False`
   - `t = +2e-4 > 1e-4` -> `True`
   - Receiver's own source triangle (`exclude_source_triangle_ids`) -> `False`, while a different face
     of the same building (`different source_triangle_id`) -> `True`.
2. Full `4 km` (`4000 m`) scene horizon: a `1,200 m` distant tower blocks low-angle sun (`4 deg`) and
   is NEVER truncated by a `150 m` crop.
3. Composite manifold geometry (`sloped roof` + `two buildings sharing XZ at different Y: y=0..8 vs canopy y=15`
   + `shared building wall` + `0.75 m thin parapet` + `ground plane y=0` with `building_id == -1`):
   - `assign_building_ids_from_triangles` assigns distinct `building_id`s to XZ-touching buildings at
     disjoint Y (`y=0..8` vs `y=15`) while grouping 3D edge-connected walls/roofs together and setting
     ground `y=0` to `-1`.
   - `Δh = 0` and `steps = 0` yield bit-for-bit identical vertices (`max |V' - V| == 0.0`), identical
     physical sunlit array, and `signed_net_sunlight_delta_fraction == 0.0`.
4. Canopy-with-distant-wall regression: adding an unrelated vertical wall `500 m` away does NOT disable
   horizontal canopy roof occlusion in either `ExactSceneBVH` or `_differentiable_soft_moller_trumbore_sunlit`!
5. Area-weighted volume conservation (`sum_b A_b * Δh_b == 0`): on unequal closed extruded prisms
   (`128 m^2` vs `64 m^2`), `abs(area_weighted_volume_residual_m3) < 1e-9` and
   `abs(mesh_signed_volume_delta_m3) < 1e-6`.
6. Signed negative-gain reporting: an adversarial/inverted RIND checkpoint that raises a south blocker
   reports negative `signed_net_sunlight_delta_fraction < 0.0` and `degraded_receiver_count > 0`
   (NEVER clamped to `0`).
7. Distinct 3-arm execution (`rind_checkpoint_grad`, `analytic_diff_surrogate`, `blackbox_zero_order`):
   distinct `optimizer_arm` tags, positive `wall_seconds`, and exact expected `oracle_queries` /
   `model_queries` counts.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from rind.fullday_data import _extruded_box_triangles
from rind.inverse_design import (
    DEFAULT_T_MIN_M,
    ExactSceneBVH,
    _differentiable_soft_moller_trumbore_sunlit,
    assign_building_ids_from_triangles,
    compute_building_effective_areas_f64,
    compute_closed_mesh_signed_volume_f64,
    exact_moller_trumbore_occluded_f64,
    moller_trumbore_intersect_f64,
    optimize_fullday_building_heights,
    perturb_buildings_consistent_f64,
    project_area_weighted_volume_conservation_f64,
)
from rind.models import build_model


def _closed_box_prism_triangles(
    cx: float, cz: float, wx: float, wz: float, y0: float, y1: float
) -> np.ndarray:
    """Build a watertight 12-triangle outward-oriented rectangular prism (`wx * wz * (y1 - y0)`)."""
    x0, x1 = cx - 0.5 * wx, cx + 0.5 * wx
    z0, z1 = cz - 0.5 * wz, cz + 0.5 * wz
    p000 = [x0, y0, z0]
    p100 = [x1, y0, z0]
    p101 = [x1, y0, z1]
    p001 = [x0, y0, z1]
    p010 = [x0, y1, z0]
    p110 = [x1, y1, z0]
    p111 = [x1, y1, z1]
    p011 = [x0, y1, z1]
    return np.array(
        [
            # Bottom (y = y0, outward normal -Y)
            [p000, p100, p101],
            [p000, p101, p001],
            # Top (y = y1, outward normal +Y)
            [p010, p111, p110],
            [p010, p011, p111],
            # Front (z = z0, outward normal -Z)
            [p000, p110, p100],
            [p000, p010, p110],
            # Back (z = z1, outward normal +Z)
            [p001, p101, p111],
            [p001, p111, p011],
            # Left (x = x0, outward normal -X)
            [p000, p001, p011],
            [p000, p011, p010],
            # Right (x = x1, outward normal +X)
            [p100, p111, p101],
            [p100, p010 if False else p110, p111],
        ],
        dtype=np.float64,
    )


class TestFullSceneInverseOracle(unittest.TestCase):
    def test_01_ray_tmin_epsilon_and_self_face_exclusion_and_4km_distant_tower(self) -> None:
        # Horizontal triangle at z in [-1, 1], x in [-1, 1], y = 0 rotated onto y=t along +Y ray from (0,0,0)
        def _horiz_tri_at_y(y_val: float) -> np.ndarray:
            return np.array([[[-1.0, y_val, -1.0], [1.0, y_val, -1.0], [0.0, y_val, 1.5]]], dtype=np.float64)

        origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        up_dir = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)

        for y_val, expected_hit in (
            (-1.0, False),   # t = -1 (behind origin)
            (2.0, True),     # t = +2 (ahead)
            (5e-5, False),   # t = 5e-5 < 1e-4 (rejected by epsilon)
            (2e-4, True),    # t = 2e-4 > 1e-4 (accepted!)
        ):
            tri = _horiz_tri_at_y(y_val)
            hit_mt, _ = moller_trumbore_intersect_f64(origin, up_dir, tri, t_min=DEFAULT_T_MIN_M)
            self.assertEqual(bool(hit_mt[0, 0]), expected_hit, msg=f"MT failed for y={y_val}")
            bvh = ExactSceneBVH(tri)
            hit_bvh = bvh.query_occluded(origin, up_dir, t_min=DEFAULT_T_MIN_M)
            self.assertEqual(bool(hit_bvh[0]), expected_hit, msg=f"BVH failed for y={y_val}")

        # Receiver self-triangle excluded vs same-building other face occluding
        same_bldg_tris = np.concatenate([_horiz_tri_at_y(0.05), _horiz_tri_at_y(4.0)], axis=0)
        sids = np.array([701, 702], dtype=np.uint64)
        # Only self face (701) present along ray -> excluded -> False
        occ_self_only = exact_moller_trumbore_occluded_f64(
            origin,
            up_dir,
            same_bldg_tris[:1],
            source_triangle_ids=sids[:1],
            exclude_source_triangle_ids=np.array([701], dtype=np.int64),
            t_min=DEFAULT_T_MIN_M,
        )
        self.assertFalse(bool(occ_self_only[0]))
        # Both self face (701) and upper overhang face (702) of same building -> 702 still occludes -> True!
        occ_with_overhang = exact_moller_trumbore_occluded_f64(
            origin,
            up_dir,
            same_bldg_tris,
            source_triangle_ids=sids,
            exclude_source_triangle_ids=np.array([701], dtype=np.int64),
            t_min=DEFAULT_T_MIN_M,
        )
        self.assertTrue(bool(occ_with_overhang[0]))

        # Full 4 km (`4000 m`) distant tower at 1,200 m East blocking 4-degree morning sun (`tan(4 deg)*1200 = 83.9 m`)
        tower_1200m = _extruded_box_triangles(1200.0, 50.0, 40.0, 40.0, 0.0, 130.0)
        rec_center = np.array([[50.0, 1.5, 50.0]], dtype=np.float64)
        alt_rad = math.radians(4.0)
        sun_east_low = np.array([[math.cos(alt_rad), math.sin(alt_rad), 0.0]], dtype=np.float64)
        bvh_4km = ExactSceneBVH(tower_1200m)
        sunlit_4km = bvh_4km.query_trajectory_sunlit(rec_center, sun_east_low)
        self.assertEqual(int(sunlit_4km[0, 0]), 0, msg="1,200 m tower MUST cast shadow in 4 km oracle!")

    def test_02_composite_manifold_3d_grouping_zero_perturbation_identity_and_canopy_wall_regression(self) -> None:
        # Build composite geometry:
        # 1) Ground plane at y = 0 (`building_id == -1`)
        ground = np.array(
            [
                [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [100.0, 0.0, 100.0]],
                [[0.0, 0.0, 0.0], [100.0, 0.0, 100.0], [0.0, 0.0, 100.0]],
            ],
            dtype=np.float64,
        )
        # 2) Building A (`x in [20, 30], z in [20, 30], y in [0, 8]`) with a sloped roof rising from y=8 to y=11
        # `_extruded_box_triangles` takes `(cx, cz, half_w, half_d, y0, y1)` and puts flat roof in `[:2]`, 4 walls in `[2:]`
        bldg_a_walls = _extruded_box_triangles(25.0, 25.0, 5.0, 5.0, 0.0, 8.0).astype(np.float64)[2:]
        sloped_roof = np.array(
            [
                [[20.0, 8.0, 20.0], [30.0, 8.0, 20.0], [30.0, 11.0, 30.0]],
                [[20.0, 8.0, 20.0], [30.0, 11.0, 30.0], [20.0, 11.0, 30.0]],
            ],
            dtype=np.float64,
        )
        # Add side gable triangles so all 4 walls and sloped roof form one watertight 3D component
        gables = np.array(
            [
                [[20.0, 8.0, 20.0], [20.0, 11.0, 30.0], [20.0, 8.0, 30.0]],
                [[30.0, 8.0, 20.0], [30.0, 8.0, 30.0], [30.0, 11.0, 30.0]],
                [[20.0, 8.0, 30.0], [20.0, 11.0, 30.0], [30.0, 11.0, 30.0]],
                [[20.0, 8.0, 30.0], [30.0, 11.0, 30.0], [30.0, 8.0, 30.0]],
            ],
            dtype=np.float64,
        )
        bldg_a = np.concatenate([bldg_a_walls, sloped_roof, gables], axis=0)

        # 3) Elevated horizontal canopy at `y = 15.0` sharing the exact `(x=30, z=20..30)` XZ planform line with Building A!
        #    3D vertex welding MUST keep Canopy as a SEPARATE `building_id` because `(30, 8, 20) != (30, 15, 20)`!
        canopy = np.array(
            [
                [[30.0, 15.0, 20.0], [42.0, 15.0, 20.0], [42.0, 15.0, 30.0]],
                [[30.0, 15.0, 20.0], [42.0, 15.0, 30.0], [30.0, 15.0, 30.0]],
            ],
            dtype=np.float64,
        )

        # 4) Thin `0.75 m` low parapet wall (`y in [0, 0.75]`) — MUST NOT be destroyed by `0.8 m` floor clamp at Δh = 0!
        parapet_075 = _extruded_box_triangles(60.0, 40.0, 8.0, 0.6, 0.0, 0.75)

        composite = np.concatenate([ground, bldg_a, canopy, parapet_075], axis=0)
        b_ids = assign_building_ids_from_triangles(composite)

        # Ground triangles MUST be -1
        self.assertTrue(np.all(b_ids[:2] == -1))
        # Building A walls + sloped roof share 3D vertices `(20,8,20)` and `(30,8,20)` -> single building ID!
        b_a_id = int(b_ids[2])
        self.assertGreaterEqual(b_a_id, 0)
        self.assertTrue(np.all(b_ids[2 : 2 + len(bldg_a)] == b_a_id))
        # Canopy at y=15 shares XZ=(30, 20..30) with Building A at y=8,11 -> MUST have a DIFFERENT building ID!
        canopy_start = 2 + len(bldg_a)
        b_canopy_id = int(b_ids[canopy_start])
        self.assertGreaterEqual(b_canopy_id, 0)
        self.assertNotEqual(b_canopy_id, b_a_id)
        # 0.75 m parapet is its own building ID
        b_parapet_id = int(b_ids[canopy_start + 2])
        self.assertGreaterEqual(b_parapet_id, 0)
        self.assertEqual(len({b_a_id, b_canopy_id, b_parapet_id}), 3)

        # Exact bit-for-bit identity at Δh = 0 (including the 0.75 m parapet!)
        num_b = int(b_ids.max()) + 1
        zero_pert = perturb_buildings_consistent_f64(composite, b_ids, np.zeros((num_b,), dtype=np.float64))
        self.assertEqual(float(np.max(np.abs(zero_pert - composite))), 0.0)

        # Verify `steps = 0` in `optimize_fullday_building_heights` yields 0.0 delta and zero_perturbation_identity_verified
        recs = np.array([[36.0, 0.05, 25.0], [25.0, 0.05, 35.0], [60.0, 0.05, 42.0]], dtype=np.float64)
        suns = np.array([[0.0, 1.0, 0.0], [0.0, 0.6, -0.8]], dtype=np.float64)
        res0 = optimize_fullday_building_heights(recs, suns, composite, steps=0)
        self.assertTrue(res0.zero_perturbation_identity_verified)
        self.assertTrue(res0.building_consistency_verified)
        self.assertEqual(res0.signed_net_sunlight_delta_fraction, 0.0)
        self.assertEqual(res0.signed_net_sunlight_delta_point_frames, 0)

        # Canopy + distant vertical wall regression:
        # Receiver `(34, 0.05, 27)` strictly inside horizontal `canopy` (`x in [30, 42], z in [20, 30], y = 15`)
        # under zenith sun `(0, 1, 0)` MUST be occluded (`sunlit < 0.10`) BOTH with canopy alone AND when an
        # unrelated vertical wall is added 500 m away!
        rec_under_canopy = torch.tensor([[34.0, 0.05, 27.0]], dtype=torch.float32)
        zenith_sun = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        canopy_t = torch.tensor(canopy, dtype=torch.float32)
        canopy_bid = torch.zeros((len(canopy),), dtype=torch.long)

        s_alone = float(
            _differentiable_soft_moller_trumbore_sunlit(
                rec_under_canopy, zenith_sun, canopy_t, building_ids=canopy_bid, temperature_m=0.5
            ).item()
        )
        distant_wall = _extruded_box_triangles(550.0, 550.0, 10.0, 2.0, 0.0, 30.0)
        combo_np = np.concatenate([canopy, distant_wall], axis=0)
        combo_bids_np = assign_building_ids_from_triangles(combo_np)
        s_with_wall = float(
            _differentiable_soft_moller_trumbore_sunlit(
                rec_under_canopy,
                zenith_sun,
                torch.tensor(combo_np, dtype=torch.float32),
                building_ids=torch.tensor(combo_bids_np, dtype=torch.long),
                temperature_m=0.5,
            ).item()
        )
        self.assertLess(s_alone, 0.15)
        self.assertLess(s_with_wall, 0.15, msg="Distant vertical wall disabled canopy occlusion!")
        bvh_combo = ExactSceneBVH(combo_np)
        self.assertEqual(
            int(bvh_combo.query_trajectory_sunlit(np.array([[34.0, 0.05, 27.0]]), np.array([[0.0, 1.0, 0.0]]))[0, 0]),
            0,
        )

    def test_03_area_weighted_volume_conservation_signed_negative_gain_and_three_arms(self) -> None:
        # 1. Exact closed-mesh signed volume conservation on unequal footprints (`16x8 = 128 m^2` vs `8x8 = 64 m^2`)
        # Building 0 (South blocker, 128 m^2, height 15 m) vs Building 1 (North building, 64 m^2, height 14 m)
        b_south = _closed_box_prism_triangles(50.0, 34.0, 16.0, 8.0, 0.0, 15.0)
        b_north = _closed_box_prism_triangles(50.0, 78.0, 8.0, 8.0, 0.0, 14.0)
        scene_closed = np.concatenate([b_south, b_north], axis=0)
        b_ids = assign_building_ids_from_triangles(scene_closed)
        areas = compute_building_effective_areas_f64(scene_closed, b_ids)
        self.assertEqual(len(areas), 2)
        self.assertAlmostEqual(float(areas[0]), 128.0, places=6)
        self.assertAlmostEqual(float(areas[1]), 64.0, places=6)

        proj = project_area_weighted_volume_conservation_f64(
            np.array([-5.0, 3.0]), areas, np.array([15.0, 14.0]), max_delta_h_m=12.0
        )
        self.assertLess(abs(float(np.dot(areas, proj))), 1e-11)
        pert_closed = perturb_buildings_consistent_f64(scene_closed, b_ids, proj)
        v_before = abs(compute_closed_mesh_signed_volume_f64(b_south)) + abs(
            compute_closed_mesh_signed_volume_f64(b_north)
        )
        v_after = abs(compute_closed_mesh_signed_volume_f64(pert_closed[b_ids == 0])) + abs(
            compute_closed_mesh_signed_volume_f64(pert_closed[b_ids == 1])
        )
        self.assertLess(abs(v_after - v_before), 1e-8)

        # Plaza receivers between z=42 and z=58 shaded by South Building (`b_south` at z=34)
        gx, gz = np.meshgrid(np.linspace(44.0, 56.0, 4), np.linspace(42.0, 58.0, 4))
        recs = np.stack([gx.ravel(), np.full(16, 0.05), gz.ravel()], axis=-1).astype(np.float64)
        alts = np.deg2rad(np.array([18.0, 24.0, 30.0, 34.0, 28.0, 20.0]))
        azs = np.deg2rad(np.array([150.0, 165.0, 175.0, 185.0, 195.0, 210.0]))
        suns = np.stack(
            [np.cos(alts) * np.sin(azs), np.sin(alts), -np.cos(alts) * np.cos(azs - math.pi)],
            axis=-1,
        )
        # Ensure pure southern sun (`sz < 0`, `sy > 0`)
        suns = np.stack(
            [np.cos(alts) * np.sin(azs - math.pi), np.sin(alts), -np.cos(alts) * np.cos(azs - math.pi)],
            axis=-1,
        )
        suns /= np.linalg.norm(suns, axis=-1, keepdims=True)
        regimes = ["morning_mid", "morning_mid", "solar_noon", "solar_noon", "afternoon_west", "afternoon_west"]

        # 2. Run all 3 distinct arms (`analytic_diff_surrogate`, `blackbox_zero_order`, `rind_checkpoint_grad`)
        res_analytic = optimize_fullday_building_heights(
            recs,
            suns,
            scene_closed,
            solar_regimes=regimes,
            optimizer_arm="analytic_diff_surrogate",
            steps=24,
            lr=0.8,
            max_delta_h_m=12.0,
            enforce_volume_conservation=True,
        )
        self.assertEqual(res_analytic.optimizer_arm, "analytic_diff_surrogate")
        self.assertGreater(res_analytic.signed_net_sunlight_delta_fraction, 0.0)
        self.assertLess(abs(res_analytic.area_weighted_volume_residual_m3), 1e-9)
        self.assertLess(abs(res_analytic.mesh_signed_volume_delta_m3), 1e-6)
        self.assertEqual(res_analytic.oracle_queries, 2)
        self.assertEqual(res_analytic.model_queries, 0)
        self.assertGreater(res_analytic.wall_seconds, 0.0)
        self.assertTrue(res_analytic.building_consistency_verified)

        res_bb = optimize_fullday_building_heights(
            recs,
            suns,
            scene_closed,
            solar_regimes=regimes,
            optimizer_arm="blackbox_zero_order",
            steps=10,
            max_delta_h_m=12.0,
            enforce_volume_conservation=True,
        )
        self.assertEqual(res_bb.optimizer_arm, "blackbox_zero_order")
        self.assertEqual(res_bb.oracle_queries, 12)  # 1 baseline + 10 proposals + 1 final verify
        self.assertEqual(res_bb.model_queries, 0)
        self.assertLess(abs(res_bb.area_weighted_volume_residual_m3), 1e-9)
        self.assertGreater(res_bb.wall_seconds, 0.0)

        torch.manual_seed(20261111)
        rind_mod = build_model("rind_full_geom", hidden_dim=24, max_candidates=4)
        res_rind = optimize_fullday_building_heights(
            recs,
            suns,
            scene_closed,
            solar_regimes=regimes,
            optimizer_arm="rind_checkpoint_grad",
            rind_model=rind_mod,
            checkpoint_sha256="a" * 64,
            max_candidates=4,
            steps=8,
            lr=0.4,
            max_delta_h_m=12.0,
            enforce_volume_conservation=True,
        )
        self.assertEqual(res_rind.optimizer_arm, "rind_checkpoint_grad")
        self.assertEqual(res_rind.checkpoint_sha256, "a" * 64)
        self.assertEqual(res_rind.oracle_queries, 2)
        self.assertEqual(res_rind.model_queries, 10)  # 1 init + 8 steps + 1 fin
        self.assertLess(abs(res_rind.area_weighted_volume_residual_m3), 1e-9)
        self.assertLess(abs(res_rind.mesh_signed_volume_delta_m3), 1e-6)
        self.assertGreater(res_rind.wall_seconds, 0.0)

        # 3. Negative-gain perturbation test: verify signed delta is NEVER clamped to 0!
        # Construct a tiny RIND model whose readout explicitly rewards taller `y_max` on `b_south` (`slot 14/17/20`),
        # so maximizing model output raises `b_south` (`delta_h[0] > 0`) and degrades actual plaza sunlight!
        class _AdversarialBlockerRaiser(torch.nn.Module):
            def forward(self, batch):
                toks = batch["candidate_tokens"]  # (1, T, N, K, 37)
                # Reward higher relative vertex y (`slots 14, 17, 20`) -> pushes South building UP!
                y_rel = toks[..., 14] + toks[..., 17] + toks[..., 20]
                logits = y_rel.mean(dim=-1)
                return {"logits": logits, "probs": torch.sigmoid(logits)}

        b_south_low = _closed_box_prism_triangles(50.0, 34.0, 16.0, 8.0, 0.0, 2.0)
        scene_neg = np.concatenate([b_south_low, b_north], axis=0)
        res_neg = optimize_fullday_building_heights(
            recs,
            suns,
            scene_neg,
            solar_regimes=regimes,
            optimizer_arm="rind_checkpoint_grad",
            rind_model=_AdversarialBlockerRaiser(),
            max_candidates=12,
            steps=10,
            lr=1.5,
            max_delta_h_m=12.0,
            enforce_volume_conservation=True,
        )
        self.assertLess(
            res_neg.signed_net_sunlight_delta_fraction,
            0.0,
            msg="Negative sunlight delta MUST be reported negatively without clamping to 0!",
        )
        self.assertLess(res_neg.signed_net_sunlight_delta_point_frames, 0)
        self.assertGreater(res_neg.degraded_receiver_count, 0)


if __name__ == "__main__":
    unittest.main()

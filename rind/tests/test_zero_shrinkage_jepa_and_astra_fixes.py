"""Comprehensive Regression Suite for Zero-Shrinkage ST-JEPA Re-Implantation & Astra Audit Fixes.

Verifies:
1. Smooth C^inf hemisphere tangent basis (zero zenith branch switch / zero horizon kink),
   strict post-sanitizer isolation (Tier-G1/G0/CoarseAABB zero NaN/Inf leakage), StaticGeometryMemory
   with Solar-FiLM, FourierSphereTangentEncoder102, RayAnisotropicAttentionBias, and valid-only
   LocalValidNeighborhoodReadout.
2. Full ST-JEPA stack (`ResponseTargetTeacher` with FitFirewallToken guard, label-free
   `EMAGeometricTarget`, 5-mask `MultiHorizonLatentPredictor` with Context Sensitivity D_X,
   event-velocity matching, Dual-Path consistency), `VICReg` + official Epps-Pulley `SIGReg`,
   5 latent collapse diagnostics, `<0.5m` `MetricBoundaryLoss`, hard-sunlight directional
   contrastive loss, detached 7-cue focal hard-negative mining, and counterfactual height
   monotonicity loss.
3. Nonlocal 4km counterexamples, `reanchor_every`, `evaluate_strided_rollout`, and label-free
   `receiver_obs_mask` per-receiver state reset (`0` label reads).
4. Closed-box horizontal roof occlusion fix (`(0, 1, 0)` upward ray inside closed box -> sunlit ~ 0),
   unified `DEFAULT_T_MAX_M = 4000.0` cutoff, exact `< 1e-14` O(B log B) breakpoint KKT volume
   conservation + `DifferentiableAreaWeightedVolumeKKT` autograd check, calibrated RIND inverse
   design with `AmortizedHeightSolver` & `abs(mesh_signed_volume_delta_m3) <= 1e-5`, and
   `select_active_disagreement_queries`.
5. Streaming shard `mesh_provenance_sha256` & `response_5x5_bitpack` verification, stripped-record
   rejection, chunked `_evaluate_single_tile_streaming` exact parity (`chunk=5` vs `chunk=256`),
   `LocalConvGRUGridModel` `full_tile_512_halo` execution, and formal CLI `--receiver-mode sampled`
   rejection without `--diagnostic-mode`.
"""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np
import torch

from rind.cli import _evaluate_single_tile_streaming, main as rind_cli_main
from rind.fast_data import (
    ContiguousTileMmapShard,
    _read_npy_slice_no_mmap,
    build_canonical37_torch,
    build_or_open_single_tile_shard,
    materialize_gpu_tokens_from_azimuth_bank,
)
from rind.fast_train import (
    LabelReadForbiddenBatchView,
    calibrate_model_on_split,
    fit_model_fullday,
    move_batch_to_device,
)
from rind.fullday_data import (
    FullDayTileReader,
    build_canonical37_numpy,
    create_synthetic_fullday_fixture,
    extract_receiver_local_features_and_crops,
    pack_receiver_5x5_response_bitpack,
    select_azimuth_sector_candidates,
)
from rind.inverse_design import (
    DEFAULT_T_MAX_M,
    AmortizedHeightSolver,
    DifferentiableAreaWeightedVolumeKKT,
    ExactSceneBVH,
    _build_local_feature_height_couplings,
    _differentiable_soft_moller_trumbore_sunlit,
    _differentiably_shift_local_features,
    assign_building_ids_from_triangles,
    optimize_fullday_building_heights,
    project_area_weighted_volume_conservation_f64,
    select_active_disagreement_queries,
)
from rind.models import (
    ALL_MODEL_TYPES,
    MODEL_TYPES,
    FocalEventTransitionGate,
    FourierSphereTangentEncoder102,
    build_model,
    compute_smooth_hemisphere_tangent_basis,
    compute_tile_global_knn_pack_torch,
)
from rind.objectives_jepa import (
    RindJepaAuxiliaryModule,
    build_height_raised_counterfactual_tokens_torch,
    compute_counterfactual_height_monotonicity_loss,
    compute_focal_event_hard_negative_loss,
    compute_hard_sunlight_and_contrastive_loss,
    compute_latent_health_diagnostics,
    compute_metric_boundary_loss,
    sketched_isotropic_gaussian_reg,
    vicreg_variance_covariance_loss,
)
from rind.split_contract import BENCHMARK_CONTRACT_ID, SplitContract
from rind.system_guard import HostResourceGuard
from rind.tests._tempdir import TemporaryTestDir


def _make_closed_axis_aligned_box(
    xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float
) -> np.ndarray:
    """Return `(12, 3, 3)` watertight outward-oriented closed box triangles (walls + floor + roof)."""
    p000 = [xmin, ymin, zmin]
    p100 = [xmax, ymin, zmin]
    p101 = [xmax, ymin, zmax]
    p001 = [xmin, ymin, zmax]
    p010 = [xmin, ymax, zmin]
    p110 = [xmax, ymax, zmin]
    p111 = [xmax, ymax, zmax]
    p011 = [xmin, ymax, zmax]
    tris = [
        # Roof (y = ymax)
        [p010, p111, p110],
        [p010, p011, p111],
        # Floor (y = ymin)
        [p000, p100, p101],
        [p000, p101, p001],
        # 4 vertical walls
        [p000, p010, p110],
        [p000, p110, p100],
        [p100, p110, p111],
        [p100, p111, p101],
        [p101, p111, p011],
        [p101, p011, p001],
        [p001, p011, p010],
        [p001, p010, p000],
    ]
    return np.asarray(tris, dtype=np.float64)


class TestZeroShrinkageJepaAndAstraFixes(unittest.TestCase):
    def test_01_smooth_hemisphere_basis_and_strict_sanitizer_isolation(self) -> None:
        self.assertEqual(len(MODEL_TYPES), 5)
        self.assertIn("rind_coarse_aabb", ALL_MODEL_TYPES)

        # Check orthonormality & smoothness across exact zenith (0, 1, 0) and horizon s_y = 0
        eps = 1e-4
        s_dirs = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [-eps, math.sqrt(1.0 - eps * eps), 0.0],
                [eps, math.sqrt(1.0 - eps * eps), 0.0],
                [1.0, 0.0, 0.0],
                [0.6, 1e-5, -0.8],
                [0.6, 0.0, -0.8],
            ],
            dtype=torch.float64,
        )
        s_u, e1, e2 = compute_smooth_hemisphere_tangent_basis(s_dirs)
        self.assertTrue(torch.allclose(torch.linalg.norm(e1, dim=-1), torch.ones(6, dtype=torch.float64), atol=1e-12))
        self.assertTrue(torch.allclose(torch.linalg.norm(e2, dim=-1), torch.ones(6, dtype=torch.float64), atol=1e-12))
        self.assertLess(float((e1 * s_u).sum(dim=-1).abs().max().item()), 1e-12)
        self.assertLess(float((e2 * s_u).sum(dim=-1).abs().max().item()), 1e-12)
        self.assertLess(float((e1 * e2).sum(dim=-1).abs().max().item()), 1e-12)
        # Right-handed orientation e1 x e2 == s_u
        cross_12 = torch.linalg.cross(e1, e2, dim=-1)
        self.assertTrue(torch.allclose(cross_12, s_u, atol=1e-11))
        # Smoothness across zenith: jump across 2*eps is O(eps), never O(1) branch flip!
        self.assertLess(float(torch.linalg.norm(e1[2] - e1[1]).item()), 5e-4)
        self.assertLess(float(torch.linalg.norm(e1[5] - e1[4]).item()), 5e-4)

        # Verify Tier-G1 (slots 22..24) and Tier-G0 / CoarseAABB have ZERO NaN/Inf leakage
        torch.manual_seed(7)
        tok = torch.randn(1, 3, 4, 4, 37)
        cmask = torch.ones(1, 3, 4, 4, dtype=torch.bool)
        lf = torch.randn(1, 4, 16)
        r_xyz = torch.randn(1, 4, 3) * 10.0
        suns = torch.tensor([[[0.2, 0.8, 0.5], [0.3, 0.7, 0.4], [0.4, 0.6, 0.3]]], dtype=torch.float32)
        adj = torch.ones(1, 2, dtype=torch.bool)

        m_g1 = build_model("rind_no_edge_margin", hidden_dim=32, max_candidates=4)
        m_g1.eval()
        base_out = m_g1(
            {
                "candidate_tokens": tok,
                "candidate_mask": cmask,
                "local_features": lf,
                "receivers_xyz": r_xyz,
                "sun_dirs": suns,
                "same_day_adjacent_mask": adj,
            }
        )["logits"]
        tok_poison = tok.clone()
        tok_poison[..., 22:25] = float("nan")
        poison_out = m_g1(
            {
                "candidate_tokens": tok_poison,
                "candidate_mask": cmask,
                "local_features": lf,
                "receivers_xyz": r_xyz,
                "sun_dirs": suns,
                "same_day_adjacent_mask": adj,
            }
        )["logits"]
        self.assertEqual(float((base_out - poison_out).abs().max().item()), 0.0)

        m_g0 = build_model("rind_zero_margin", hidden_dim=32, max_candidates=4)
        m_g0.eval()
        tok_g0_poison = tok.clone()
        for sl in (slice(0, 10), slice(12, 13), slice(22, 25), slice(28, 37)):
            tok_g0_poison[..., sl] = float("inf")
        out_g0_a = m_g0(
            {
                "candidate_tokens": tok,
                "candidate_mask": cmask,
                "local_features": torch.full_like(lf, float("nan")),
                "receivers_xyz": r_xyz,
                "sun_dirs": suns,
                "same_day_adjacent_mask": adj,
            }
        )["logits"]
        out_g0_b = m_g0(
            {
                "candidate_tokens": tok_g0_poison,
                "candidate_mask": cmask,
                "local_features": lf,
                "receivers_xyz": r_xyz,
                "sun_dirs": suns,
                "same_day_adjacent_mask": adj,
            }
        )["logits"]
        self.assertTrue(torch.isfinite(out_g0_a).all().item())
        self.assertEqual(float((out_g0_a - out_g0_b).abs().max().item()), 0.0)

        m_aabb = build_model("rind_coarse_aabb", hidden_dim=32, max_candidates=4)
        out_aabb = m_aabb(
            {
                "candidate_tokens": tok_poison,
                "candidate_mask": cmask,
                "local_features": lf,
                "receivers_xyz": r_xyz,
                "sun_dirs": suns,
                "same_day_adjacent_mask": adj,
            }
        )["logits"]
        self.assertTrue(torch.isfinite(out_aabb).all().item())

    def test_02_st_jepa_teachers_epps_pulley_sigreg_and_physical_losses(self) -> None:
        torch.manual_seed(42)
        # 1. Official Epps-Pulley SIGReg: isotropic Gaussian has ~0 loss; collapsed rank-1 has high loss
        z_iso = torch.randn(512, 32, requires_grad=True)
        loss_iso = sketched_isotropic_gaussian_reg(z_iso, num_slices=64)
        loss_iso.backward()
        self.assertIsNotNone(z_iso.grad)
        self.assertGreater(float(z_iso.grad.norm().item()), 0.0)

        base_col = torch.randn(512, 1)
        z_collapsed = base_col.repeat(1, 32).detach().requires_grad_(True)
        loss_col = sketched_isotropic_gaussian_reg(z_collapsed, num_slices=64)
        self.assertGreater(float(loss_col.item()), float(loss_iso.item()) * 5.0)

        vc_loss = vicreg_variance_covariance_loss(z_collapsed)
        self.assertGreater(float(vc_loss.item()), 0.1)

        health = compute_latent_health_diagnostics(z_iso.detach())
        for key in (
            "effective_rank",
            "participation_ratio",
            "min_std",
            "offdiag_cov_rms",
            "cosine_diversity",
        ):
            self.assertIn(key, health)
        self.assertGreater(health["effective_rank"], 15.0)
        health_col = compute_latent_health_diagnostics(z_collapsed.detach())
        self.assertLess(health_col["effective_rank"], 2.0)

        # 2. Bitpack packing & ResponseTargetTeacher FitFirewallToken guard + MultiHorizon predictor
        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fx", grid_side=12, frames_per_day=8, num_fit_tiles=8
            )
            contract = SplitContract(
                split_path=sp_json, manifest_path=mf_json, enforce_pinned_split_sha256=False
            )
            fit_tiles = contract.get_fit_tiles("fit_8")
            val_tile = contract.get_eval_tiles("val_spatial_clean")[0]
            fw_tok = contract.issue_fit_firewall_token(fit_tiles, subset_name="fit_8")

            states = np.ones((8, 12, 12), dtype=np.uint8)
            states[2:6, 3:9, 3:9] = 2
            rc = np.array([[4, 4], [5, 5], [6, 6], [7, 7]], dtype=np.int64)
            bitpack = pack_receiver_5x5_response_bitpack(states, rc)
            self.assertEqual(bitpack.shape, (8, 4))
            self.assertEqual(bitpack.dtype, np.int64)

            aux = RindJepaAuxiliaryModule(
                latent_dim=32,
                solar_dim=28,
                jepa_mode="response",
                reg_mode="vicreg_sigreg",
            )
            z_seq = torch.randn(1, 8, 4, 32, requires_grad=True)
            z_dual = torch.randn(1, 8, 4, 32, requires_grad=True)
            s_emb = torch.randn(1, 8, 28)
            g_probs = torch.sigmoid(torch.randn(1, 7, 4))
            truth_t = torch.randint(0, 2, (1, 8, 4), dtype=torch.float32)
            valid_t = torch.ones(1, 8, 4, dtype=torch.bool)
            bdist_t = torch.rand(1, 8, 4) * 1.5
            bval_t = torch.ones(1, 8, 4, dtype=torch.bool)
            bpack_t = torch.from_numpy(bitpack).unsqueeze(0)
            adj_t = torch.ones(1, 7, dtype=torch.bool)
            r_xyz_t = torch.randn(1, 4, 3)

            # Non-fit tile MUST be rejected by ResponseTargetTeacher FitFirewallToken check!
            with self.assertRaises(RuntimeError):
                aux.compute_auxiliary_losses(
                    z_seq,
                    solar_emb=s_emb,
                    gate_probs=g_probs,
                    dual_latents=z_dual,
                    truth=truth_t,
                    valid_mask=valid_t,
                    boundary_distance_m=bdist_t,
                    boundary_distance_valid=bval_t,
                    response_5x5_bitpack=bpack_t,
                    same_day_adjacent_mask=adj_t,
                    receivers_xyz=r_xyz_t,
                    tile_ids=[val_tile],
                    firewall_token=fw_tok,
                    expected_split_sha256=contract.split_sha256,
                )

            # Fit tile succeeds across all 5 structured mask modes and yields D_X >= 0
            for mask_mode in ("spatial", "temporal", "spacetime_tube", "driver", "boundary"):
                res_j = aux.compute_auxiliary_losses(
                    z_seq,
                    solar_emb=s_emb,
                    gate_probs=g_probs,
                    dual_latents=z_dual,
                    truth=truth_t,
                    valid_mask=valid_t,
                    boundary_distance_m=bdist_t,
                    boundary_distance_valid=bval_t,
                    response_5x5_bitpack=bpack_t,
                    same_day_adjacent_mask=adj_t,
                    receivers_xyz=r_xyz_t,
                    tile_ids=[fit_tiles[0]],
                    firewall_token=fw_tok,
                    expected_split_sha256=contract.split_sha256,
                    mask_mode=mask_mode,
                )
                self.assertTrue(torch.isfinite(res_j["total_aux_loss"]).item())
                self.assertGreater(float(res_j["total_aux_loss"].item()), 0.0)
                self.assertIn("jepa_context_sensitivity_dx", res_j["stats"])

            # Label-free ema_geom teacher works with truth=None and bpack=None (`0` label dependency)
            aux_ema = RindJepaAuxiliaryModule(
                latent_dim=32, solar_dim=28, jepa_mode="ema_geom", reg_mode="sigreg"
            )
            res_ema = aux_ema.compute_auxiliary_losses(
                z_seq,
                solar_emb=s_emb,
                gate_probs=g_probs,
                dual_latents=z_dual,
                truth=None,
                valid_mask=None,
                same_day_adjacent_mask=adj_t,
                receivers_xyz=r_xyz_t,
                mask_mode="spacetime_tube",
            )
            self.assertTrue(torch.isfinite(res_ema["total_aux_loss"]).item())

            # Physical boundary, hard-sunlight, focal event mining, and height monotonicity losses
            logits = torch.randn(1, 8, 4, requires_grad=True)
            cmask = torch.ones(1, 8, 4, 3, dtype=torch.bool)
            l_mb, _ = compute_metric_boundary_loss(
                logits,
                truth_t,
                valid_t,
                {"boundary_distance_m": bdist_t, "boundary_distance_valid": bval_t},
                same_day_adj=adj_t,
            )
            l_hs, _ = compute_hard_sunlight_and_contrastive_loss(logits, truth_t, valid_t, cmask)
            g_probs_8 = torch.sigmoid(torch.randn(1, 8, 4))
            l_ev, _, _ = compute_focal_event_hard_negative_loss(
                torch.sigmoid(logits), g_probs_8, truth_t, valid_t, adj_t
            )
            (l_mb + l_hs + l_ev).backward()
            self.assertIsNotNone(logits.grad)

            model_g2 = build_model("rind_full_geom", hidden_dim=32, max_candidates=3)
            batch_m = {
                "candidate_tokens": torch.randn(1, 6, 4, 3, 37),
                "candidate_mask": torch.ones(1, 6, 4, 3, dtype=torch.bool),
                "local_features": torch.randn(1, 4, 16),
                "receivers_xyz": r_xyz_t,
                "sun_dirs": torch.tensor([[[0.2, 0.7, 0.6]] * 6], dtype=torch.float32),
                "same_day_adjacent_mask": torch.ones(1, 5, dtype=torch.bool),
            }
            out_m = model_g2(batch_m)
            l_mono = compute_counterfactual_height_monotonicity_loss(model_g2, batch_m, out_m["probs"])
            self.assertTrue(torch.isfinite(l_mono).item())

    def test_03_nonlocal_counterexamples_strided_rollout_and_label_free_gap_reset(self) -> None:
        # 1. ZYY Nonlocal Counterexample: 800m remote blocker alters RIND tokens & logits, while local crop is unchanged
        rec = torch.zeros(1, 2, 3)
        suns = torch.tensor([[[0.98, 0.198, 0.0], [0.98, 0.198, 0.0]]], dtype=torch.float32)
        # Near empty vs 800m tall wall (x=800m, y=0..220m)
        rel_near = torch.zeros(2, 2, 2, 3, 3)
        rel_far = torch.zeros(2, 2, 2, 3, 3)
        rel_far[:, :, 0, 0, :] = torch.tensor([800.0, 0.0, -30.0])
        rel_far[:, :, 0, 1, :] = torch.tensor([800.0, 220.0, 0.0])
        rel_far[:, :, 0, 2, :] = torch.tensor([800.0, 0.0, 30.0])
        mk_near = torch.zeros(2, 2, 2, dtype=torch.bool)
        mk_far = torch.zeros(2, 2, 2, dtype=torch.bool)
        mk_far[:, :, 0] = True

        tok_near = build_canonical37_torch(rel_near, mk_near, mk_near, suns.squeeze(0))
        tok_far = build_canonical37_torch(rel_far, mk_far, mk_far, suns.squeeze(0))
        m_rind = build_model("rind_full_geom", hidden_dim=32, max_candidates=2, reanchor_every=2)
        m_rind.eval()
        lf = torch.zeros(1, 2, 16)
        adj = torch.ones(1, 1, dtype=torch.bool)
        lg_near = m_rind(
            {
                "candidate_tokens": tok_near.unsqueeze(0),
                "candidate_mask": mk_near.unsqueeze(0),
                "local_features": lf,
                "receivers_xyz": rec,
                "sun_dirs": suns,
                "same_day_adjacent_mask": adj,
            }
        )["logits"]
        lg_far = m_rind(
            {
                "candidate_tokens": tok_far.unsqueeze(0),
                "candidate_mask": mk_far.unsqueeze(0),
                "local_features": lf,
                "receivers_xyz": rec,
                "sun_dirs": suns,
                "same_day_adjacent_mask": adj,
            }
        )["logits"]
        self.assertGreater(float((lg_near - lg_far).abs().max().item()), 1e-4)

        # 2. Strided rollout (`stride=2`) & label-free `receiver_obs_mask` per-receiver reset (`0` label reads)
        obs_mask = torch.tensor([[True, False]], dtype=torch.bool)
        b_long = {
            "candidate_tokens": tok_far.repeat(3, 1, 1, 1).unsqueeze(0),
            "candidate_mask": mk_far.repeat(3, 1, 1).unsqueeze(0),
            "local_features": lf,
            "receivers_xyz": rec,
            "receiver_obs_mask": obs_mask,
            "sun_dirs": suns.repeat(1, 3, 1),
            "same_day_adjacent_mask": torch.ones(1, 5, dtype=torch.bool),
        }
        out_strided = m_rind.evaluate_strided_rollout(b_long, stride=2)
        self.assertEqual(out_strided["logits"].shape, (1, 6, 2))
        self.assertEqual(int(out_strided["nonfinite_count"]), 0)

    def test_04_closed_box_roof_tmax_exact_kkt_and_active_queries(self) -> None:
        # 1. Closed box interior upward ray (0, 1, 0) MUST be occluded (`sunlit ~ 0`) in BOTH oracle and soft surrogate!
        box_tris = _make_closed_axis_aligned_box(-5.0, 5.0, 0.0, 10.0, -5.0, 5.0)
        rec_inside = np.array([[0.0, 1.0, 0.0], [1.2, 1.0, -0.7]], dtype=np.float64)
        sun_up = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        bvh = ExactSceneBVH(box_tris)
        exact_sun = bvh.query_trajectory_sunlit(rec_inside, sun_up)
        self.assertTrue(np.array_equal(exact_sun, np.zeros((1, 2), dtype=np.int8)))

        soft_no_bid = _differentiable_soft_moller_trumbore_sunlit(
            torch.as_tensor(rec_inside, dtype=torch.float32),
            torch.as_tensor(sun_up, dtype=torch.float32),
            torch.as_tensor(box_tris, dtype=torch.float32),
            building_ids=None,
            temperature_m=0.5,
        )
        soft_with_bid = _differentiable_soft_moller_trumbore_sunlit(
            torch.as_tensor(rec_inside, dtype=torch.float32),
            torch.as_tensor(sun_up, dtype=torch.float32),
            torch.as_tensor(box_tris, dtype=torch.float32),
            building_ids=torch.zeros((box_tris.shape[0],), dtype=torch.long),
            temperature_m=0.5,
        )
        self.assertLess(float(soft_no_bid.max().item()), 1e-3)
        self.assertLess(float(soft_with_bid.max().item()), 1e-3)

        # 2. Unified `DEFAULT_T_MAX_M = 4000.0` horizon cutoff
        self.assertEqual(DEFAULT_T_MAX_M, 4000.0)
        wall_3500 = _make_closed_axis_aligned_box(3490.0, 3510.0, 0.0, 500.0, -100.0, 100.0)
        wall_4500 = _make_closed_axis_aligned_box(4490.0, 4510.0, 0.0, 700.0, -100.0, 100.0)
        low_east_sun = np.array([[0.995, 0.0998, 0.0]], dtype=np.float64)
        origin_pt = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        self.assertEqual(int(ExactSceneBVH(wall_3500).query_trajectory_sunlit(origin_pt, low_east_sun)[0, 0]), 0)
        self.assertEqual(int(ExactSceneBVH(wall_4500).query_trajectory_sunlit(origin_pt, low_east_sun)[0, 0]), 1)

        # 3. Exact O(B log B) breakpoint KKT (< 1e-14 residual) + DifferentiableAreaWeightedVolumeKKT gradcheck
        rng = np.random.default_rng(20261112)
        areas = rng.uniform(25.0, 4500.0, size=(64,))
        heights = rng.uniform(5.0, 80.0, size=(64,))
        raw_dh = rng.normal(3.5, 18.0, size=(64,))
        proj_dh = project_area_weighted_volume_conservation_f64(raw_dh, areas, heights, max_delta_h_m=15.0)
        vol_res = abs(math.fsum((areas * proj_dh).tolist()))
        self.assertLess(vol_res, 1e-12 * float(areas.max()))
        self.assertLess(abs(float(np.dot(areas, proj_dh))) / float(areas.sum()), 1e-14)

        v_chk = torch.tensor([3.2, -1.5, 4.1, -2.0], dtype=torch.float64, requires_grad=True)
        a_chk = torch.tensor([100.0, 250.0, 80.0, 170.0], dtype=torch.float64)
        h_chk = torch.tensor([30.0, 40.0, 25.0, 50.0], dtype=torch.float64)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: DifferentiableAreaWeightedVolumeKKT.apply(x, a_chk, h_chk, 15.0, 0.05),
                (v_chk,),
                eps=1e-6,
                atol=1e-5,
            )
        )

        # 4. Inverse design with RIND checkpoint + AmortizedHeightSolver + mesh volume delta <= 1e-5
        b1 = _make_closed_axis_aligned_box(12.0, 22.0, 0.0, 18.0, -8.0, 8.0)
        b2 = _make_closed_axis_aligned_box(-35.0, -20.0, 0.0, 20.0, -10.0, 10.0)
        scene_2b = np.concatenate([b1, b2], axis=0)
        bids_2b = np.array([0] * 12 + [1] * 12, dtype=np.int64)
        recs_inv = np.array([[0.0, 1.0, 0.0], [2.0, 1.0, 2.0]], dtype=np.float64)
        suns_inv = np.array([[0.75, 0.55, 0.0], [0.80, 0.45, 0.1], [0.70, 0.65, -0.1]], dtype=np.float64)
        m_inv = build_model("rind_full_geom", hidden_dim=24, max_candidates=6)
        am_solver = AmortizedHeightSolver(hidden_dim=16)
        inv_res = optimize_fullday_building_heights(
            recs_inv,
            suns_inv,
            scene_2b,
            building_ids=bids_2b,
            optimizer_arm="rind_checkpoint_grad",
            rind_model=m_inv,
            amortized_solver=am_solver,
            steps=8,
            lr=0.5,
            smoothness_weight=1e-3,
        )
        self.assertTrue(inv_res.building_consistency_verified)
        self.assertEqual(inv_res.instance_provenance_tier, "explicit_instance_ids")
        self.assertLessEqual(abs(inv_res.mesh_signed_volume_delta_m3), 1e-5)

        # 5. Active disagreement query selector
        ens_probs = np.stack(
            [
                np.clip(rng.uniform(0.1, 0.9, size=(6, 8)), 0.0, 1.0),
                np.clip(rng.uniform(0.1, 0.9, size=(6, 8)), 0.0, 1.0),
                np.clip(rng.uniform(0.1, 0.9, size=(6, 8)), 0.0, 1.0),
            ],
            axis=0,
        )
        aq = select_active_disagreement_queries(
            ens_probs,
            receivers_xyz=rng.normal(0.0, 20.0, size=(8, 3)),
            top_k=5,
        )
        self.assertEqual(len(aq["selected_frame_receiver_pairs"]), 5)
        self.assertGreater(aq["mean_acquisition_score"], 0.0)

    def test_05_streaming_shard_chunked_eval_parity_convgru_halo_and_formal_cli_guard(self) -> None:
        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fx", grid_side=14, frames_per_day=8, num_fit_tiles=8
            )
            contract = SplitContract(
                split_path=sp_json, manifest_path=mf_json, enforce_pinned_split_sha256=False
            )
            reader = FullDayTileReader(contract)

            # 1. Stripped record dict missing tile_id/source_core_sha256 MUST be rejected
            with self.assertRaises(ValueError):
                reader._validate_record_identity({}, contract.tiles[contract.get_fit_tiles("fit_8")[0]])

            # 2. Formal CLI MUST reject `--receiver-mode sampled` when `--diagnostic-mode` is omitted
            with self.assertRaises(RuntimeError):
                rind_cli_main(
                    [
                        "train",
                        "--cache-dir",
                        str(root / "c_formal_bad"),
                        "--model-type",
                        "rind_full_geom",
                        "--receiver-mode",
                        "sampled",
                        "--checkpoint-out",
                        str(root / "bad.pt"),
                        "--quiet",
                    ]
                )

            # 3. Streaming shard build verifies mesh_provenance_sha256 & response_5x5_bitpack
            fit_tid = contract.get_fit_tiles("fit_8")[0]
            cal_tid = contract.get_calibration_tiles()[0]
            val_tid = contract.get_eval_tiles("val_spatial_clean")[0]
            cache_dir = root / "mmap"

            sh_fit = build_or_open_single_tile_shard(
                cache_dir, reader, fit_tid, week_indices=[0, 13], receiver_mode="all_valid"
            )
            self.assertTrue(sh_fit.has_response_5x5_bitpack)
            self.assertIn("mesh_provenance_sha256", sh_fit.provenance)
            with self.assertRaises((ValueError, RuntimeError)):
                ContiguousTileMmapShard(
                    sh_fit.shard_dir,
                    expected_mesh_provenance_sha256="0" * 64,
                )

            # 4. Full-tile ConvGRU (`full_tile_512_halo`) trains + calibrates + chunk-evaluates cleanly
            fw_tok = contract.issue_fit_firewall_token([fit_tid], subset_name="fit_8")
            m_conv = build_model(
                "local_convgru_grid",
                hidden_dim=24,
                crop_size=21,
                meters_per_pixel=sh_fit.meters_per_pixel,
                spatial_execution_mode="full_tile_512_halo",
            )
            fit_stats = fit_model_fullday(
                m_conv,
                lambda _ep: sh_fit.iter_day_batches(
                    days_per_batch=1,
                    receiver_chunk_size=16,
                    model_type="local_convgru_grid",
                    convgru_execution_mode="full_tile_512_halo",
                ),
                firewall_token=fw_tok,
                expected_split_sha256=contract.split_sha256,
                epochs=1,
                lr=1e-3,
                device="cpu",
                use_bf16_amp=False,
            )
            self.assertGreater(fit_stats["total_steps"], 0)

            sh_cal = build_or_open_single_tile_shard(
                cache_dir, reader, cal_tid, week_indices=[0, 13], receiver_mode="all_valid"
            )
            cal_res = calibrate_model_on_split(
                m_conv,
                sh_cal.iter_day_batches(
                    days_per_batch=1,
                    receiver_chunk_size=16,
                    model_type="local_convgru_grid",
                    convgru_execution_mode="full_tile_512_halo",
                ),
                contract=contract,
                device="cpu",
            )
            self.assertGreater(cal_res["valid_point_frames"], 0)

            # 5. Chunked `_evaluate_single_tile_streaming` (`chunk=5` vs `chunk=512`) parity!
            sh_val = build_or_open_single_tile_shard(
                cache_dir, reader, val_tid, week_indices=[0, 13], receiver_mode="all_valid"
            )
            guard = HostResourceGuard(device=torch.device("cpu"))
            acc_small, regs_small, bands_small, info_small = _evaluate_single_tile_streaming(
                m_conv,
                sh_val,
                torch.device("cpu"),
                guard=guard,
                days_per_batch=1,
                receiver_chunk_size=5,
                temperature=float(m_conv.calibrated_temperature.item()),
                threshold=float(m_conv.calibrated_threshold.item()),
            )
            acc_same_dpb, _, _, info_same_dpb = _evaluate_single_tile_streaming(
                m_conv,
                sh_val,
                torch.device("cpu"),
                guard=guard,
                days_per_batch=1,
                receiver_chunk_size=512,
                temperature=float(m_conv.calibrated_temperature.item()),
                threshold=float(m_conv.calibrated_threshold.item()),
            )
            acc_full, regs_full, bands_full, info_full = _evaluate_single_tile_streaming(
                m_conv,
                sh_val,
                torch.device("cpu"),
                guard=guard,
                days_per_batch=2,
                receiver_chunk_size=512,
                temperature=float(m_conv.calibrated_temperature.item()),
                threshold=float(m_conv.calibrated_threshold.item()),
            )
            # Matched `days_per_batch=1` (`chunk=5` vs `chunk=512`) executes identical full-tile Conv2d batching
            # (diff ~6e-8 over sum=410.17, i.e. 1.5e-10 relative float32 reduction rounding):
            self.assertAlmostEqual(acc_small.brier_sum, acc_same_dpb.brier_sum, places=6)
            self.assertAlmostEqual(
                acc_small.summarize()["brier_score"], acc_same_dpb.summarize()["brier_score"], places=9
            )
            self.assertEqual(info_small["long_day_diagnostics"], info_same_dpb["long_day_diagnostics"])
            # Cross `days_per_batch=1` vs `days_per_batch=2` (`T=8` vs `T=16` MKL Conv2d float32 SIMD blocking):
            self.assertEqual(acc_small.point_frames, acc_full.point_frames)
            self.assertEqual(acc_small.state_tp, acc_full.state_tp)
            self.assertEqual(acc_small.state_tn, acc_full.state_tn)
            self.assertEqual(acc_small.transition_pairs, acc_full.transition_pairs)
            self.assertAlmostEqual(acc_small.brier_sum, acc_full.brier_sum, places=4)
            self.assertAlmostEqual(
                acc_small.summarize()["brier_score"], acc_full.summarize()["brier_score"], places=7
            )
            self.assertEqual(info_small["long_day_diagnostics"], info_full["long_day_diagnostics"])
            for rk in regs_small:
                self.assertEqual(regs_small[rk].point_frames, regs_full[rk].point_frames)
            for bk in bands_small:
                self.assertEqual(bands_small[bk].point_frames, bands_full[bk].point_frames)

    def test_06_astra_round2_gates_full_verification(self) -> None:
        torch.manual_seed(20261112)
        rng = np.random.default_rng(20261112)

        # Gate 1: Exact 102D FourierSphereTangentEncoder102, 0.0 disabled-slot grad leakage,
        # and detached solar_angular_speed inside FocalEventTransitionGate.
        self.assertEqual(FourierSphereTangentEncoder102.RAW_ENC_DIM, 102)
        enc102 = FourierSphereTangentEncoder102(hidden_dim=32)
        self.assertEqual(enc102.in_dim, 102)
        captured_in_dims: list[int] = []
        hook = enc102.net[0].register_forward_pre_hook(
            lambda _m, inp: captured_in_dims.append(int(inp[0].shape[-1]))
        )
        tok_req = torch.randn(12, 4, 37, requires_grad=True)
        cm_flat = torch.ones(12, 4, dtype=torch.bool)
        _ = enc102(tok_req, tok_req, cm_flat)
        hook.remove()
        self.assertEqual(captured_in_dims, [102])
        cm_102 = torch.ones(1, 3, 4, 4, dtype=torch.bool)

        # Backward disabled-slot gradient isolation across Tier-G1, Tier-G0, and CoarseAABB
        lf_4 = torch.randn(1, 4, 16)
        r_xyz_4 = torch.randn(1, 4, 3) * 8.0
        suns_3 = torch.tensor([[[0.2, 0.8, 0.5], [0.3, 0.7, 0.4], [0.4, 0.6, 0.3]]], dtype=torch.float32)
        adj_2 = torch.ones(1, 2, dtype=torch.bool)
        for m_name, disabled_slices in (
            ("rind_no_edge_margin", (slice(22, 25),)),
            ("rind_zero_margin", (slice(0, 10), slice(12, 13), slice(22, 25), slice(28, 37))),
            ("rind_coarse_aabb", (slice(0, 10), slice(12, 25), slice(34, 37))),
        ):
            m_iso = build_model(m_name, hidden_dim=24, max_candidates=4)
            t_in = torch.randn(1, 3, 4, 4, 37, requires_grad=True)
            out_iso = m_iso(
                {
                    "candidate_tokens": t_in,
                    "candidate_mask": cm_102,
                    "local_features": lf_4,
                    "receivers_xyz": r_xyz_4,
                    "sun_dirs": suns_3,
                    "same_day_adjacent_mask": adj_2,
                }
            )["logits"]
            out_iso.sum().backward()
            assert t_in.grad is not None
            for sl in disabled_slices:
                self.assertEqual(float(t_in.grad[..., sl].abs().max().item()), 0.0)

        fgate = FocalEventTransitionGate(hidden_dim=16)
        z_seq_det = torch.randn(1, 3, 4, 16)
        lbce_det = torch.randn(1, 3, 4)
        lbnd_det = torch.randn(1, 3, 4)
        ang_req = torch.tensor([[0.15, 0.25, 0.20]], dtype=torch.float32, requires_grad=True)
        fused_det, gp = fgate(z_seq_det, lbce_det, lbnd_det, solar_angular_speed=ang_req)
        (fused_det.sum() + gp.sum()).backward()
        self.assertIsNone(ang_req.grad)

        # Gate 2: Zero-label enforcement (`LabelReadForbiddenBatchView`) across forward, normalizers,
        # counterfactual monotonicity, and `ema_geom` ST-JEPA across all 5 mask modes + pure self-supervised fit.
        raw_b = {
            "candidate_tokens": torch.randn(1, 6, 5, 3, 37),
            "candidate_mask": torch.ones(1, 6, 5, 3, dtype=torch.bool),
            "local_features": torch.randn(1, 5, 16),
            "receivers_xyz": torch.randn(1, 5, 3) * 10.0,
            "receiver_obs_mask": torch.tensor([[True, True, True, True, False]], dtype=torch.bool),
            "sun_dirs": torch.tensor([[[0.25, 0.72, 0.55]] * 6], dtype=torch.float32),
            "same_day_adjacent_mask": torch.ones(1, 5, dtype=torch.bool),
            "truth": torch.ones(1, 6, 5),
            "valid_mask": torch.ones(1, 6, 5, dtype=torch.bool),
            "boundary_distance_m": torch.ones(1, 6, 5),
            "boundary_distance_valid": torch.ones(1, 6, 5, dtype=torch.bool),
            "response_5x5_bitpack": torch.ones(1, 6, 5, dtype=torch.long),
        }
        guarded_b = LabelReadForbiddenBatchView(raw_b)
        with self.assertRaises(RuntimeError):
            _ = guarded_b["truth"]
        with self.assertRaises(RuntimeError):
            _ = guarded_b.get("valid_mask")
        self.assertNotIn("truth", guarded_b)

        m_selfsup = build_model("rind_full_geom", hidden_dim=24, max_candidates=3)
        out_ss = m_selfsup(guarded_b)
        l_mono_ss = compute_counterfactual_height_monotonicity_loss(m_selfsup, guarded_b, out_ss["probs"])
        self.assertTrue(torch.isfinite(l_mono_ss).item())

        aux_lf = RindJepaAuxiliaryModule(
            latent_dim=24, solar_dim=24, jepa_mode="ema_geom", reg_mode="vicreg_sigreg"
        )
        for m_mode in ("spatial", "temporal", "spacetime_tube", "driver", "boundary"):
            loss_lf, _ = aux_lf.forward_objectives(out_ss, guarded_b, mask_mode_override=m_mode)
            self.assertTrue(torch.isfinite(loss_lf).item())

        # Gate 3: `"driver"` mask replaces `sol_fut` via `driver_mask_token` while keeping `zc` intact;
        # official Epps-Pulley `SIGReg` detects mean shift AND variance collapse!
        pred_mod = aux_lf.predictor
        assert pred_mod is not None
        pred_mod.eval()
        z_ctx = torch.randn(1, 8, 4, 24)
        z_tgt = torch.randn(1, 8, 4, 24)
        s_drv_a = torch.randn(1, 8, 24)
        drv_mask = pred_mod.build_structured_mask(z_ctx, mode="driver", step_seed=7)
        # Ensure masked window includes at least one target frame `t >= 1`
        self.assertTrue(bool(drv_mask[:, 1:, :].any().item()))
        s_drv_b = s_drv_a.clone()
        s_drv_b[drv_mask[:, :, 0]] += 50.0  # alter solar embeddings strictly inside masked blackout window!
        v_drv = torch.ones(1, 8, 4, dtype=torch.bool)
        adj_drv = torch.ones(1, 7, dtype=torch.bool)
        l_drv_a, _ = pred_mod.forward_multi_horizon(
            z_ctx, z_tgt, s_drv_a, valid_mask=v_drv, same_day_adjacent_mask=adj_drv, mask_mode="driver", step_seed=7
        )
        l_drv_b, _ = pred_mod.forward_multi_horizon(
            z_ctx, z_tgt, s_drv_b, valid_mask=v_drv, same_day_adjacent_mask=adj_drv, mask_mode="driver", step_seed=7
        )
        l_drv_z2, _ = pred_mod.forward_multi_horizon(
            z_ctx + 3.0,
            z_tgt,
            s_drv_a,
            valid_mask=v_drv,
            same_day_adjacent_mask=adj_drv,
            mask_mode="driver",
            step_seed=7,
        )
        self.assertEqual(float((l_drv_a - l_drv_b).abs().item()), 0.0)
        self.assertGreater(float((l_drv_a - l_drv_z2).abs().item()), 1e-4)

        z_std = torch.randn(512, 24)
        sig_base = float(sketched_isotropic_gaussian_reg(z_std, num_slices=64).item())
        sig_shift = float(sketched_isotropic_gaussian_reg(z_std + 1.5, num_slices=64).item())
        sig_shrink = float(sketched_isotropic_gaussian_reg(z_std * 0.15, num_slices=64).item())
        self.assertGreater(sig_shift, sig_base * 5.0)
        self.assertGreater(sig_shrink, sig_base * 5.0)

        # Gate 4: Active `StaticGeometryMemory` & `LocalValidNeighborhoodReadout` at `N = 2500 > 1024`,
        # `0.0` `UNKNOWN` contamination, and exact `receiver_chunk_size` invariance (`5` vs `64` vs `2500`)!
        m_chunk = build_model("rind_full_geom", hidden_dim=24, max_candidates=2)
        torch.nn.init.normal_(m_chunk.spatial_knn_readout.mlp[-1].weight, mean=0.0, std=0.3)
        torch.nn.init.normal_(m_chunk.static_geom_memory.edge_msg[-1].weight, mean=0.0, std=0.3)
        torch.nn.init.normal_(m_chunk.static_geom_memory.solar_film[-1].weight, mean=0.0, std=0.3)
        m_chunk.eval()

        n_large = 2500
        lf_large = torch.randn(1, n_large, 16)
        r_large = torch.randn(1, n_large, 3) * 12.0  # dense enough that k-NN distances are < 15m!
        obs_large = torch.ones(1, n_large, dtype=torch.bool)
        obs_large[0, 2400:] = False  # 100 UNKNOWN receivers
        tok_large = torch.randn(1, 2, n_large, 2, 37)
        cm_large = torch.ones(1, 2, n_large, 2, dtype=torch.bool)
        suns_2 = torch.tensor([[[0.3, 0.7, 0.5], [0.35, 0.65, 0.55]]], dtype=torch.float32)
        adj_1 = torch.ones(1, 1, dtype=torch.bool)

        gknn = compute_tile_global_knn_pack_torch(r_large[0], obs_large[0], lf_large[0], k_neighbors=8)
        full_logits = m_chunk(
            {
                "candidate_tokens": tok_large,
                "candidate_mask": cm_large,
                "local_features": lf_large,
                "receivers_xyz": r_large,
                "receiver_obs_mask": obs_large,
                "global_knn_local_features": gknn["knn_local_features"].unsqueeze(0),
                "global_knn_rel_xyz": gknn["knn_rel_xyz"].unsqueeze(0),
                "global_knn_dist": gknn["knn_dist"].unsqueeze(0),
                "global_knn_obs_valid": gknn["knn_obs_valid"].unsqueeze(0),
                "sun_dirs": suns_2,
                "same_day_adjacent_mask": adj_1,
            }
        )["logits"]

        # Verify N = 2500 actually uses global_knn (NOT bypassed above 1024!)
        perturbed_gknn_lf = gknn["knn_local_features"].clone() + 2.5
        pert_logits = m_chunk(
            {
                "candidate_tokens": tok_large,
                "candidate_mask": cm_large,
                "local_features": lf_large,
                "receivers_xyz": r_large,
                "receiver_obs_mask": obs_large,
                "global_knn_local_features": perturbed_gknn_lf.unsqueeze(0),
                "global_knn_rel_xyz": gknn["knn_rel_xyz"].unsqueeze(0),
                "global_knn_dist": gknn["knn_dist"].unsqueeze(0),
                "global_knn_obs_valid": gknn["knn_obs_valid"].unsqueeze(0),
                "sun_dirs": suns_2,
                "same_day_adjacent_mask": adj_1,
            }
        )["logits"]
        self.assertGreater(float((full_logits - pert_logits).abs().max().item()), 1e-3)

        # Verify poisoning UNKNOWN (`obs_large == False`) slots with extreme values has 0 effect on valid slots
        lf_poison_unk = lf_large.clone()
        lf_poison_unk[0, 2400:] = 1e5
        r_poison_unk = r_large.clone()
        r_poison_unk[0, 2400:] = -1e5
        gknn_poison = compute_tile_global_knn_pack_torch(r_poison_unk[0], obs_large[0], lf_poison_unk[0], k_neighbors=8)
        poison_unk_logits = m_chunk(
            {
                "candidate_tokens": tok_large,
                "candidate_mask": cm_large,
                "local_features": lf_poison_unk,
                "receivers_xyz": r_poison_unk,
                "receiver_obs_mask": obs_large,
                "global_knn_local_features": gknn_poison["knn_local_features"].unsqueeze(0),
                "global_knn_rel_xyz": gknn_poison["knn_rel_xyz"].unsqueeze(0),
                "global_knn_dist": gknn_poison["knn_dist"].unsqueeze(0),
                "global_knn_obs_valid": gknn_poison["knn_obs_valid"].unsqueeze(0),
                "sun_dirs": suns_2,
                "same_day_adjacent_mask": adj_1,
            }
        )["logits"]
        self.assertLess(
            float((full_logits[:, :, :2400] - poison_unk_logits[:, :, :2400]).abs().max().item()),
            1e-5,
        )

        # Verify exact chunk parity (`chunk=5` vs `chunk=64` vs `full N=2500`) on first 20 valid receivers
        chunks_5 = []
        for r0 in range(0, 20, 5):
            r1 = r0 + 5
            chunks_5.append(
                m_chunk(
                    {
                        "candidate_tokens": tok_large[:, :, r0:r1],
                        "candidate_mask": cm_large[:, :, r0:r1],
                        "local_features": lf_large[:, r0:r1],
                        "receivers_xyz": r_large[:, r0:r1],
                        "receiver_obs_mask": obs_large[:, r0:r1],
                        "global_knn_local_features": gknn["knn_local_features"][r0:r1].unsqueeze(0),
                        "global_knn_rel_xyz": gknn["knn_rel_xyz"][r0:r1].unsqueeze(0),
                        "global_knn_dist": gknn["knn_dist"][r0:r1].unsqueeze(0),
                        "global_knn_obs_valid": gknn["knn_obs_valid"][r0:r1].unsqueeze(0),
                        "sun_dirs": suns_2,
                        "same_day_adjacent_mask": adj_1,
                    }
                )["logits"]
            )
        cat_5 = torch.cat(chunks_5, dim=2)
        chunk_64 = m_chunk(
            {
                "candidate_tokens": tok_large[:, :, :64],
                "candidate_mask": cm_large[:, :, :64],
                "local_features": lf_large[:, :64],
                "receivers_xyz": r_large[:, :64],
                "receiver_obs_mask": obs_large[:, :64],
                "global_knn_local_features": gknn["knn_local_features"][:64].unsqueeze(0),
                "global_knn_rel_xyz": gknn["knn_rel_xyz"][:64].unsqueeze(0),
                "global_knn_dist": gknn["knn_dist"][:64].unsqueeze(0),
                "global_knn_obs_valid": gknn["knn_obs_valid"][:64].unsqueeze(0),
                "sun_dirs": suns_2,
                "same_day_adjacent_mask": adj_1,
            }
        )["logits"][:, :, :20]
        self.assertLess(float((cat_5 - full_logits[:, :, :20]).abs().max().item()), 1e-5)
        self.assertLess(float((chunk_64 - full_logits[:, :, :20]).abs().max().item()), 1e-5)

        # Gate 5: `evaluate_strided_rollout` lazy token materialization from `az_rel_triangles`,
        # `0.0` cross-day interpolation, and `reanchor_every` intra-day offset parity.
        m_roll = build_model("rind_full_geom", hidden_dim=24, max_candidates=2, reanchor_every=3)
        m_roll.eval()
        az_tris_lazy = torch.randn(1, 3, 8, 2, 3, 3) * 12.0
        az_mask_lazy = torch.ones(1, 3, 8, 2, dtype=torch.bool)
        az_opq_lazy = torch.ones(1, 3, 8, 2, dtype=torch.bool)
        az_bin_8 = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
        suns_8 = torch.tensor(
            [[[0.2, 0.6, 0.5], [0.25, 0.65, 0.45], [0.3, 0.7, 0.4], [0.35, 0.6, 0.4]] * 2],
            dtype=torch.float32,
        )
        # Boundary between t=3 and t=4 (`same_day_adjacent_mask[0, 3] == False`)
        sdm_2days = torch.tensor([[True, True, True, False, True, True, True]], dtype=torch.bool)
        lf_3 = torch.randn(1, 3, 16)
        r_3 = torch.randn(1, 3, 3) * 5.0
        b_lazy_a = {
            "az_rel_triangles": az_tris_lazy,
            "az_mask": az_mask_lazy,
            "az_opaque": az_opq_lazy,
            "azimuth_bin_idx": az_bin_8,
            "local_features": lf_3,
            "receivers_xyz": r_3,
            "sun_dirs": suns_8,
            "same_day_adjacent_mask": sdm_2days,
        }
        roll_a = m_roll.evaluate_strided_rollout(b_lazy_a, stride=3)
        self.assertIn("has_active_candidates", roll_a)
        self.assertEqual(roll_a["logits"].shape, (1, 8, 3))

        # Poison Day 0 (`t = 0..3`) solar directions & azimuth bins: Day 1 (`t = 4..7`) MUST change by 0.0!
        suns_8_poison_d0 = suns_8.clone()
        suns_8_poison_d0[:, :4, :] = torch.tensor([[-0.7, 0.2, 0.6]] * 4, dtype=torch.float32)
        az_bin_8_poison_d0 = az_bin_8.clone()
        az_bin_8_poison_d0[:, :4] = 0
        b_lazy_b = dict(b_lazy_a)
        b_lazy_b["sun_dirs"] = suns_8_poison_d0
        b_lazy_b["azimuth_bin_idx"] = az_bin_8_poison_d0
        roll_b = m_roll.evaluate_strided_rollout(b_lazy_b, stride=3)
        self.assertLess(float((roll_a["logits"][:, 4:, :] - roll_b["logits"][:, 4:, :]).abs().max().item()), 1e-6)

        # Intra-day `reanchor_every=3` chunk parity (`[0..5]` single pass vs `[0..2]` + `[3..5]` with offset=3)
        tok_6 = torch.randn(1, 6, 3, 2, 37)
        cm_6 = torch.ones(1, 6, 3, 2, dtype=torch.bool)
        suns_6 = suns_8[:, :6, :]
        out_full6 = m_roll(
            {
                "candidate_tokens": tok_6,
                "candidate_mask": cm_6,
                "local_features": lf_3,
                "receivers_xyz": r_3,
                "sun_dirs": suns_6,
                "same_day_adjacent_mask": torch.ones(1, 5, dtype=torch.bool),
                "intra_day_step_offset": 0,
            }
        )["logits"]
        out_p1 = m_roll(
            {
                "candidate_tokens": tok_6[:, :3],
                "candidate_mask": cm_6[:, :3],
                "local_features": lf_3,
                "receivers_xyz": r_3,
                "sun_dirs": suns_6[:, :3],
                "same_day_adjacent_mask": torch.ones(1, 2, dtype=torch.bool),
                "intra_day_step_offset": 0,
            }
        )
        out_p2 = m_roll(
            {
                "candidate_tokens": tok_6[:, 3:],
                "candidate_mask": cm_6[:, 3:],
                "local_features": lf_3,
                "receivers_xyz": r_3,
                "sun_dirs": suns_6[:, 3:],
                "same_day_adjacent_mask": torch.ones(1, 2, dtype=torch.bool),
                "carry_state_from_prev_chunk": True,
                "intra_day_step_offset": 3,
                "initial_latent_state": out_p1["final_latent_state"],
                "prev_frame_sun_dir": suns_6[:, 2, :],
            }
        )
        cat_6 = torch.cat([out_p1["logits"], out_p2["logits"]], dim=1)
        self.assertLess(float((out_full6 - cat_6).abs().max().item()), 1e-5)

        # Gate 6: Exact 37-slot `build_height_raised_counterfactual_tokens_torch` parity vs `build_canonical37_numpy`
        wall_and_roof = _make_closed_axis_aligned_box(10.0, 18.0, 0.0, 16.0, -4.0, 4.0)
        # Keep non-floor faces (`ymax > 0.25m`: 2 roof + 8 wall triangles = 10 triangles)
        non_floor = wall_and_roof[wall_and_roof[:, :, 1].max(axis=1) > 0.25]
        rec_cf = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, -1.0]], dtype=np.float32)
        rel_cf_np = non_floor[None, None, :, :, :].astype(np.float32) - rec_cf[None, :, None, None, :]
        rel_cf_np = np.repeat(np.repeat(rel_cf_np, 2, axis=0), 2, axis=1)[:, :2]  # (2, 2, 10, 3, 3)
        mk_cf_np = np.ones((2, 2, 10), dtype=bool)
        suns_cf_np = np.array([[0.8, 0.45, 0.1], [0.7, 0.6, -0.2]], dtype=np.float32)
        suns_cf_np = suns_cf_np / np.linalg.norm(suns_cf_np, axis=-1, keepdims=True)

        tok_base_t = build_canonical37_torch(
            torch.from_numpy(rel_cf_np),
            torch.from_numpy(mk_cf_np),
            torch.from_numpy(mk_cf_np),
            torch.from_numpy(suns_cf_np),
        ).unsqueeze(0)
        dh_cf = 4.0
        tok_cf_t = build_height_raised_counterfactual_tokens_torch(
            tok_base_t,
            torch.from_numpy(mk_cf_np).unsqueeze(0),
            torch.from_numpy(suns_cf_np).unsqueeze(0),
            delta_height_m=dh_cf,
        )[0].numpy()

        rel_extruded_np = rel_cf_np.copy()
        rel_extruded_np[..., 1] = np.where(
            rel_extruded_np[..., 1] > 0.25, rel_extruded_np[..., 1] + dh_cf, rel_extruded_np[..., 1]
        )
        tok_exact_np = build_canonical37_numpy(
            rel_extruded_np, mk_cf_np, mk_cf_np, suns_cf_np
        )
        self.assertLess(float(np.max(np.abs(tok_cf_t - tok_exact_np))), 1e-4)

        # Gate 7: `< 1e-14 m³` absolute `math.fsum` KKT residual on 20 random 128-building problems
        # + exact `9x9` (`81`-pixel) raster parity of `_build_local_feature_height_couplings`.
        for _ in range(20):
            a_128 = rng.uniform(40.0, 4500.0, size=(128,))
            h_128 = rng.uniform(6.0, 120.0, size=(128,))
            v_128 = rng.normal(5.0, 22.0, size=(128,))
            p_128 = project_area_weighted_volume_conservation_f64(v_128, a_128, h_128, max_delta_h_m=15.0)
            res_128 = abs(math.fsum((a_128 * p_128).tolist()))
            self.assertLess(res_128, 1e-14)

        # Exact 9x9 raster parity check against `extract_receiver_local_features_and_crops`
        h_g, w_g = 32, 32
        mpp = 1.0
        wx = np.arange(w_g, dtype=np.float64) * mpp
        wz = np.arange(h_g, dtype=np.float64) * mpp
        xx, zz = np.meshgrid(wx, wz, indexing="xy")
        # Building 0 covers x in [4.5, 11.5], z in [5.5, 12.5]; Building 1 covers x in [16.5, 24.5], z in [10.5, 19.5]
        b0_tris = _make_closed_axis_aligned_box(4.5, 11.5, 0.0, 20.0, 5.5, 12.5)
        b1_tris = _make_closed_axis_aligned_box(16.5, 24.5, 0.0, 30.0, 10.5, 19.5)
        scene_lf = np.concatenate([b0_tris, b1_tris], axis=0)
        bids_lf = assign_building_ids_from_triangles(scene_lf)
        in_b0 = (xx >= 4.5) & (xx <= 11.5) & (zz >= 5.5) & (zz <= 12.5)
        in_b1 = (xx >= 16.5) & (xx <= 24.5) & (zz >= 10.5) & (zz <= 19.5)

        def _make_fields_8hw(h0: float, h1: float) -> np.ndarray:
            f8 = np.zeros((8, h_g, w_g), dtype=np.float32)
            roof = np.where(in_b0, h0, np.where(in_b1, h1, 0.0)).astype(np.float32)
            f8[1] = roof / 100.0
            f8[2] = roof / 100.0
            return f8

        rc_test = np.array([[0, 0], [6, 6], [5, 4], [14, 18], [8, 13], [31, 31]], dtype=np.int64)
        rec_xyz_hw3 = np.stack([xx, np.zeros_like(xx), zz], axis=-1).astype(np.float32)
        rec_pts = rec_xyz_hw3[rc_test[:, 0], rc_test[:, 1]].astype(np.float64)
        lf_0 = extract_receiver_local_features_and_crops(_make_fields_8hw(20.0, 30.0), rec_xyz_hw3, rc_test, wx, wz)[
            "local_features"
        ]
        dh_test = np.array([4.5, -3.5], dtype=np.float32)
        lf_1_true = extract_receiver_local_features_and_crops(
            _make_fields_8hw(20.0 + float(dh_test[0]), 30.0 + float(dh_test[1])),
            rec_xyz_hw3,
            rc_test,
            wx,
            wz,
        )["local_features"]
        ew, nw = _build_local_feature_height_couplings(
            rec_pts,
            scene_lf,
            bids_lf,
            2,
            meters_per_pixel=mpp,
            stencil_radius_px=4,
            grid_bounds_xz=(float(wx[0]), float(wx[-1]), float(wz[0]), float(wz[-1])),
        )
        lf_1_pred = _differentiably_shift_local_features(
            torch.from_numpy(lf_0),
            torch.from_numpy(dh_test),
            torch.from_numpy(ew),
            torch.from_numpy(nw),
        ).numpy()
        self.assertLess(float(np.max(np.abs(lf_1_pred - lf_1_true))), 1e-6)

        # Gate 8: `SHARD_PROVENANCE.json` tamper/deletion rejection & persistent `.blind_unseal_ledger.jsonl`
        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fx8", grid_side=10, frames_per_day=6, num_fit_tiles=8
            )
            ledger_file = root / "blind_ledger.jsonl"
            c1 = SplitContract(
                split_path=sp_json,
                manifest_path=mf_json,
                enforce_pinned_split_sha256=False,
                blind_ledger_path=ledger_file,
            )
            r1 = FullDayTileReader(c1)
            blind_tid = [t for t, rec in c1.tiles.items() if rec.canonical_split == "test_spatial_unexposed"][0]
            with self.assertRaises(RuntimeError):
                r1.load_tile_fullday(blind_tid)
            with self.assertRaises(RuntimeError):
                r1.prepare_tile_streaming_context(blind_tid)
            self.assertEqual(c1.blind_open_count, 0)

            # Build shard and verify provenance enforcement against optional array deletion or rogue injection
            fit_tid = c1.get_fit_tiles("fit_8")[0]
            fw_tok8 = c1.issue_fit_firewall_token([fit_tid], subset_name="fit_8")
            m_selfsup.update_normalizers(
                guarded_b,
                firewall_token=fw_tok8,
                tile_ids=[fit_tid],
                expected_split_sha256=c1.split_sha256,
            )
            poisoned_ss_batch = dict(raw_b)
            poisoned_ss_batch["tile_ids"] = [fit_tid]
            poisoned_ss_batch["truth"] = torch.full_like(raw_b["truth"], float("nan"))
            ss_summary = fit_model_fullday(
                m_selfsup,
                lambda _ep: [poisoned_ss_batch],
                firewall_token=fw_tok8,
                expected_split_sha256=c1.split_sha256,
                epochs=1,
                jepa_mode="ema_geom",
                reg_mode="sigreg",
                supervise_labels=False,
            )
            self.assertFalse(bool(ss_summary["supervise_labels"]))
            self.assertEqual(ss_summary["jepa_mode"], "ema_geom")

            sh_ok = build_or_open_single_tile_shard(
                root / "sh_cache", r1, fit_tid, week_indices=[0], receiver_mode="all_valid"
            )
            rogue_npy = sh_ok.shard_dir / "undeclared_rogue.npy"
            np.save(rogue_npy, np.zeros((2,), dtype=np.float32))
            with self.assertRaises(RuntimeError):
                ContiguousTileMmapShard(sh_ok.shard_dir)
            rogue_npy.unlink()

            bpack_file = sh_ok.shard_dir / "response_5x5_bitpack.npy"
            bpack_bak = sh_ok.shard_dir / "response_5x5_bitpack.bak"
            bpack_file.rename(bpack_bak)
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                ContiguousTileMmapShard(sh_ok.shard_dir)
            bpack_bak.rename(bpack_file)
            _ = ContiguousTileMmapShard(
                sh_ok.shard_dir, expected_optional_arrays=sh_ok.provenance["optional_arrays"]
            )

            # Unseal on disposable contract `c1` and verify brand-new `c2` reloads `blind_open_count == 1` from disk
            freeze_mf = {
                "benchmark_contract": BENCHMARK_CONTRACT_ID,
                "split_sha256": c1.split_sha256,
                "manifest_sha256": c1.manifest_sha256,
                "selection_rule": "disposable_unit_test_audit",
                "frozen_checkpoints": [{"checkpoint_sha256": "0" * 64}],
                "evaluator_version": "v1_unit_test",
            }
            unsealed_tiles = c1.get_eval_tiles(
                "test_spatial_unexposed",
                model_provenance={
                    "trained_from_scratch_fit_only": True,
                    "fit_tile_ids": [fit_tid],
                    "split_contract_sha256": c1.split_sha256,
                    "manifest_sha256": c1.manifest_sha256,
                },
                unseal_freeze_manifest=freeze_mf,
            )
            self.assertIn(blind_tid, unsealed_tiles)
            self.assertEqual(c1.blind_open_count, 1)
            c2 = SplitContract(
                split_path=sp_json,
                manifest_path=mf_json,
                enforce_pinned_split_sha256=False,
                blind_ledger_path=ledger_file,
            )
            self.assertEqual(c2.blind_open_count, 1)

            # Certified two-level spatial branch-and-bound (`use_spatial_bnb=True`) vs brute-force (`False`)
            # `100%` bit-exact parity + zero-`RssFile` `_read_npy_slice_no_mmap` slice parity
            rng_bnb = np.random.default_rng(20261112)
            rec_bnb = np.column_stack(
                [
                    rng_bnb.uniform(-45.0, 45.0, size=(64,)),
                    rng_bnb.uniform(0.0, 8.0, size=(64,)),
                    rng_bnb.uniform(-45.0, 45.0, size=(64,)),
                ]
            )
            centers_bnb = np.column_stack(
                [
                    rng_bnb.uniform(-2800.0, 2800.0, size=(1500,)),
                    rng_bnb.uniform(2.0, 180.0, size=(1500,)),
                    rng_bnb.uniform(-2800.0, 2800.0, size=(1500,)),
                ]
            )
            offsets_bnb = rng_bnb.uniform(-12.0, 12.0, size=(1500, 3, 3))
            offsets_bnb[:, :, 1] = np.abs(offsets_bnb[:, :, 1])
            tris_bnb = centers_bnb[:, None, :] + offsets_bnb
            t_ref, o_ref, m_ref, s_ref = select_azimuth_sector_candidates(
                rec_bnb,
                tris_bnb,
                num_azimuth_bins=8,
                max_candidates=8,
                return_source_ids=True,
                use_spatial_bnb=False,
            )
            t_fast, o_fast, m_fast, s_fast = select_azimuth_sector_candidates(
                rec_bnb,
                tris_bnb,
                num_azimuth_bins=8,
                max_candidates=8,
                return_source_ids=True,
                use_spatial_bnb=True,
            )
            self.assertTrue(np.array_equal(s_ref, s_fast))
            self.assertTrue(np.array_equal(m_ref, m_fast))
            self.assertTrue(np.array_equal(o_ref, o_fast))
            self.assertEqual(float(np.max(np.abs(t_ref - t_fast))), 0.0)

            truth_npy = sh_ok.shard_dir / "truth.npy"
            ref_truth = np.load(truth_npy, allow_pickle=False)
            self.assertTrue(
                np.array_equal(
                    ref_truth[2:7, 3:15],
                    _read_npy_slice_no_mmap(truth_npy, (slice(2, 7), slice(3, 15))),
                )
            )


if __name__ == "__main__":
    unittest.main()

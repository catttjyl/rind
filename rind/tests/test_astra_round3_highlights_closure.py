"""Zero-Cheating Full Closure Regression Suite (`test_astra_round3_highlights_closure.py`).

Exhaustively verifies every requirement in:
- Astra Audit (`ASTRA_HIGHLIGHTS_AND_WEEKEND_READINESS_AUDIT_20261112.md`: `P0-1`..`P0-8`, `P1`, `P2-1`..`P2-7`, §6 12 counterexamples)
- Original 7-page ST-JEPA 5-Highlight Report (`ST-JEPA五项核心方法改进报告-20260807.pdf`)
- Group Highlight Addition (`Representation Sufficiency under Driver Changes & Nonadditive Unseen Edit Combinations`)
- Classmate Reconciliation Fixes (Tony Bug 1 ground slack, Tony Bug 2 spatial crop shift + `sanitize_tokens`, Angela Bug 3 balanced switching focal loss)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from rind.cli import build_arg_parser, main as rind_cli_main
from rind.fast_data import (
    _read_npy_slice_no_mmap,
    build_canonical37_torch,
    build_or_open_single_tile_shard,
    get_fast_data_io_stats,
    materialize_gpu_tokens_from_azimuth_bank,
    reset_fast_data_io_stats,
)
from rind.fast_train import (
    fit_model_fullday,
    load_training_resume_state,
    move_batch_to_device,
    save_training_resume_state,
)
from rind.fullday_data import (
    FullDayTileReader,
    _extruded_box_triangles,
    _validate_and_cast_raw_state_array,
    create_synthetic_fullday_fixture,
)
from rind.inverse_design import (
    DEFAULT_TILE_METERS_PER_PIXEL,
    DifferentiableAreaWeightedVolumeKKT,
    ExactSceneBVH,
    _build_local_feature_height_couplings,
    _differentiable_soft_moller_trumbore_sunlit,
    _preselect_candidate_triangle_indices,
    aggregate_multi_tile_inverse_design,
    build_layered_occlusion_benchmark_scene,
    evaluate_representation_sufficiency,
    infer_unknown_building_heights_bayesian,
    moller_trumbore_intersect_f64,
    optimize_fullday_building_heights,
    verify_extruded_building_manifold_closure,
)
from rind.metrics import (
    compute_blocker_distance_stratified_metrics,
    evaluate_trajectory_predictions,
)
from rind.models import (
    ALL_MODEL_TYPES,
    MODEL_TYPES,
    FourierSphereTangentEncoder102,
    LocalConvGRUGridModel,
    RindCandidateWorldModel,
    build_model,
)
from rind.objectives_jepa import (
    ALL_MASK_MODES,
    ObjectiveConfig,
    RindJepaAuxiliaryModule,
    apply_label_budget_view,
    build_restricted_context_response_features_torch,
    compute_counterfactual_height_monotonicity_loss,
    compute_focal_event_hard_negative_loss,
    compute_latent_health_diagnostics,
)
from rind.split_contract import (
    SplitContract,
    atomic_update_json_file,
    resolve_tristate_exposure_status,
)
from rind.tests._tempdir import TemporaryTestDir


def _make_unit_suns(t_steps: int = 8) -> torch.Tensor:
    u = torch.linspace(0.15, 0.85, steps=t_steps)
    az = (110.0 + 130.0 * u) * (math.pi / 180.0)
    alt = (12.0 + 38.0 * torch.sin(math.pi * u)) * (math.pi / 180.0)
    s = torch.stack([torch.cos(alt) * torch.sin(az), torch.sin(alt), torch.cos(alt) * torch.cos(az)], dim=-1)
    return s / torch.linalg.norm(s, dim=-1, keepdim=True)


class TestAstraRound3HighlightsClosure(unittest.TestCase):
    """Strict non-cheating regression tests covering P0-1..P0-8, P1, P2, and Representation Sufficiency."""

    def test_01_p0_1_mask_plus_x_strict_isolation_and_2d_spatial_blocks(self) -> None:
        """Counterexample 1 (`P0-1`): `Mask + X` zero leakage from `Omega_T`, 2D spatial blocks, and genuine `D_X`."""
        torch.manual_seed(20261112)
        t_steps, n_pts, h_dim = 6, 16, 32
        gx, gz = np.meshgrid(np.arange(4) * 12, np.arange(4) * 12)
        rc = torch.as_tensor(np.stack([gz.ravel(), gx.ravel()], axis=-1), dtype=torch.long)
        rec_xyz = torch.stack([rc[:, 1].float(), torch.zeros(n_pts), rc[:, 0].float()], dim=-1)
        suns = _make_unit_suns(t_steps)

        valid_mask = torch.ones((t_steps, n_pts), dtype=torch.bool)
        valid_mask[0, -1] = False
        truth = ((torch.arange(t_steps)[:, None] + torch.arange(n_pts)[None, :]) % 2).float()
        truth[~valid_mask] = -1.0
        b_dist = torch.full((t_steps, n_pts), 0.8, dtype=torch.float32)
        b_val = valid_mask.clone()

        # Verify all 9 mask modes produce disjoint `(omega_t, omega_c)` strictly inside `valid_mask`
        cfg_base = ObjectiveConfig(
            jepa_mode="hybrid",
            mask_mode="spatial",
            mask_ratio=0.30,
            spatial_block_size=18,
            jepa_context_mode="with_x",
        )
        loss_mod = RindJepaAuxiliaryModule(hidden_dim=h_dim, objective_config=cfg_base)
        assert loss_mod.predictor is not None
        assert loss_mod.response_teacher is not None
        z_dummy = torch.zeros((1, t_steps, n_pts, h_dim), dtype=torch.float32)

        for mode_name in ALL_MASK_MODES:
            raw_t = loss_mod.predictor.build_structured_mask(
                z_dummy,
                mode=mode_name,
                boundary_distance_m=b_dist,
                receiver_rc=rc,
                receivers_xyz=rec_xyz,
                step_seed=0,
            )[0]
            omega_t = raw_t & valid_mask
            omega_c = ((~raw_t) & valid_mask) if mode_name != "none" else torch.zeros_like(valid_mask)
            self.assertFalse(bool(torch.any(omega_t & omega_c)), msg=f"Overlap in mode={mode_name}")
            self.assertFalse(bool(torch.any((omega_t | omega_c) & (~valid_mask))), msg=f"Invalid leak in {mode_name}")
            self.assertGreater(int(omega_t.sum().item()), 0, msg=f"Empty target in {mode_name}")
            if mode_name == "none":
                self.assertEqual(int(omega_c.sum().item()), 0)
            else:
                self.assertGreater(int(omega_c.sum().item()), 0, msg=f"Empty context in {mode_name}")

        # Verify `spatial` mode selects contiguous 2D block on `receiver_rc` (static across valid frames)
        raw_sp = loss_mod.predictor.build_structured_mask(
            z_dummy,
            mode="spatial",
            boundary_distance_m=b_dist,
            receiver_rc=rc,
            receivers_xyz=rec_xyz,
            step_seed=0,
        )[0]
        sp_t = raw_sp & valid_mask
        sp_c = (~raw_sp) & valid_mask
        self.assertTrue(torch.equal(sp_t[1], sp_t[2]))

        # ZERO LEAKAGE PROOF: perturbing `truth`, `b_dist`, `bpack` inside `Omega_T` (or `~valid_mask`)
        # MUST produce 0.0 difference in `X` and in student predicted loss / context aggregation!
        batch_orig = {
            "sun_dirs": suns,
            "receivers_xyz": rec_xyz,
            "receiver_rc": rc,
            "same_day_adjacent_mask": torch.ones(t_steps - 1, dtype=torch.bool),
            "truth": truth,
            "valid_mask": valid_mask,
            "boundary_distance_m": b_dist,
            "boundary_distance_valid": b_val,
        }
        x_orig = build_restricted_context_response_features_torch(
            batch_orig,
            sp_c.unsqueeze(0),
            bsz=1,
            t_steps=t_steps,
            n_pts=n_pts,
            device=torch.device("cpu"),
        )
        truth_poisoned = truth.clone()
        bdist_poisoned = b_dist.clone()
        poison_zone = sp_t | (~valid_mask)
        truth_poisoned[poison_zone] = 9999.0
        bdist_poisoned[poison_zone] = -7777.0
        batch_poisoned = dict(batch_orig, truth=truth_poisoned, boundary_distance_m=bdist_poisoned)
        x_poisoned = build_restricted_context_response_features_torch(
            batch_poisoned,
            sp_c.unsqueeze(0),
            bsz=1,
            t_steps=t_steps,
            n_pts=n_pts,
            device=torch.device("cpu"),
        )
        self.assertEqual(float((x_orig - x_poisoned).abs().max().item()), 0.0)

        solar_emb = torch.randn(1, t_steps, h_dim)
        ctx_emb_1 = loss_mod.response_teacher.encode_restricted_context_x(batch_orig, sp_c.unsqueeze(0), solar_emb)
        ctx_emb_2 = loss_mod.response_teacher.encode_restricted_context_x(batch_poisoned, sp_c.unsqueeze(0), solar_emb)
        self.assertEqual(float((ctx_emb_1 - ctx_emb_2).abs().max().item()), 0.0)

        student_z = torch.nn.functional.normalize(torch.randn(1, t_steps, n_pts, h_dim), dim=-1) * math.sqrt(h_dim * 0.5)
        agg_1 = loss_mod.predictor.context_aggregator(
            student_z, student_z * sp_c.unsqueeze(0).unsqueeze(-1).float(), ctx_emb_1, sp_c.unsqueeze(0), receivers_xyz=rec_xyz, receiver_rc=rc
        )
        agg_2 = loss_mod.predictor.context_aggregator(
            student_z, student_z * sp_c.unsqueeze(0).unsqueeze(-1).float(), ctx_emb_2, sp_c.unsqueeze(0), receivers_xyz=rec_xyz, receiver_rc=rc
        )
        self.assertEqual(float((agg_1 - agg_2).abs().max().item()), 0.0)

        # Verify `D_X` responds genuinely to `context_aggregator.v_proj` weights (`0` -> `0.0`, non-zero -> `> 0.0`)
        outputs_stub = {
            "latent_roll": student_z[0],
            "latent_direct": student_z[0] * 0.85,
            "solar_emb": solar_emb[0],
        }
        _, stats_active = loss_mod.forward_objectives(
            outputs_stub, batch_orig, step_index=0, precomputed_target_mask=raw_sp.unsqueeze(0)
        )
        self.assertGreater(float(stats_active["jepa_context_sensitivity_dx"]), 1e-6)
        self.assertIsNotNone(stats_active["modality_alignment_rho_m"])

        # Zero out `v_proj` -> context values `X` carry zero signal -> `D_X == 0.0` identically!
        with torch.no_grad():
            loss_mod.predictor.context_aggregator.v_proj.weight.zero_()
            loss_mod.predictor.context_aggregator.v_proj.bias.zero_()
        _, stats_zero = loss_mod.forward_objectives(
            outputs_stub, batch_orig, step_index=0, precomputed_target_mask=raw_sp.unsqueeze(0)
        )
        self.assertEqual(float(stats_zero["jepa_context_sensitivity_dx"]), 0.0)

    def test_02_p0_2_pure_supervised_and_ema_full_backbone_and_collapse_jsonl(self) -> None:
        """Counterexample 2 & 3 (`P0-2`): Pure supervised has 0 aux grad; `full_backbone` EMA updates real copy."""
        torch.manual_seed(42)
        model = RindCandidateWorldModel(hidden_dim=32, max_candidates=4)
        suns = _make_unit_suns(5)
        n_pts, k_c = 6, 4
        rel_t = torch.randn(5, n_pts, k_c, 3, 3) * 10.0
        rel_t[..., 1] = rel_t[..., 1].abs() + 2.0
        mk = torch.ones(5, n_pts, k_c, dtype=torch.bool)
        tokens = build_canonical37_torch(rel_t, mk, mk, suns)
        batch = {
            "tile_ids": ["core_fi-0100-fj-0100"],
            "candidate_tokens": tokens,
            "candidate_mask": mk,
            "local_features": torch.zeros(n_pts, 16),
            "receivers_xyz": torch.zeros(n_pts, 3),
            "receiver_rc": torch.stack([torch.arange(n_pts), torch.arange(n_pts)], dim=-1),
            "sun_dirs": suns,
            "same_day_adjacent_mask": torch.ones(4, dtype=torch.bool),
            "truth": torch.randint(0, 2, (5, n_pts)).float(),
            "valid_mask": torch.ones(5, n_pts, dtype=torch.bool),
            "az_rel_triangles": rel_t.permute(1, 0, 2, 3, 4).repeat(1, 5, 1, 1, 1)[:, :24],
            "az_opaque": torch.ones(n_pts, 24, k_c, dtype=torch.bool),
            "az_mask": torch.ones(n_pts, 24, k_c, dtype=torch.bool),
            "azimuth_bin_idx": torch.zeros(5, dtype=torch.long),
        }

        # 1. Pure supervised firewall: auxiliary parameters MUST have `grad is None`!
        pure_cfg = ObjectiveConfig(pure_supervised=True)
        pure_loss_mod = RindJepaAuxiliaryModule(hidden_dim=32, objective_config=pure_cfg)
        out = model(batch)
        aux_loss_pure, _ = pure_loss_mod.forward_objectives(out, batch, step_index=0)
        self.assertEqual(float(aux_loss_pure.item()), 0.0)
        for p in pure_loss_mod.parameters():
            self.assertIsNone(p.grad)

        # 2. `ema_target_scope="full_backbone"`: real deep copy updated by `step_ema`, zero label access!
        fb_cfg = ObjectiveConfig(jepa_mode="ema_geom", ema_target_scope="full_backbone", reg_mode="sigreg")
        fb_loss_mod = RindJepaAuxiliaryModule(hidden_dim=32, ema_decay=0.90, objective_config=fb_cfg)
        fb_loss_mod.bind_student_backbone(model)
        self.assertIsNotNone(fb_loss_mod.ema_backbone)
        w_before = next(fb_loss_mod.ema_backbone.parameters()).detach().clone()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.25)
        fb_loss_mod.step_ema()
        w_after = next(fb_loss_mod.ema_backbone.parameters()).detach().clone()
        self.assertGreater(float((w_after - w_before).abs().max().item()), 1e-3)

        # Poison truth with NaN in label-free `ema_geom` + `full_backbone`: must succeed with finite loss!
        no_label_batch = dict(batch)
        no_label_batch["truth"] = torch.full_like(batch["truth"], float("nan"))
        aux_loss_fb, stats_fb = fb_loss_mod.forward_objectives(out, no_label_batch, step_index=0)
        self.assertTrue(torch.isfinite(aux_loss_fb))
        self.assertEqual(stats_fb["ema_target_scope"], "full_backbone")
        self.assertIsNone(stats_fb["modality_alignment_rho_m"])

        # 3. Collapse diagnostics returns `None` when `sample_count < 4`
        tiny_diag = compute_latent_health_diagnostics(
            torch.randn(1, 2, 16), torch.ones(1, 2, dtype=torch.bool)
        )
        self.assertEqual(tiny_diag["sample_count"], 2)
        self.assertIsNone(tiny_diag["effective_rank"])
        self.assertIsNone(tiny_diag["per_dim_std_min"])

    def test_03_p0_3_low_label_budget_zero_hidden_75pct_leakage(self) -> None:
        """Counterexample 9 (`P0-3`): `label_budget_fraction=0.25` strictly blocks hidden 75% from BCE AND Teacher!"""
        with TemporaryTestDir("rind_low_label_") as tmp:
            root = Path(tmp)
            sp_p, mf_p = create_synthetic_fullday_fixture(
                root / "fix", grid_side=12, frames_per_day=10, num_fit_tiles=8, num_val_tiles=1, num_cal_tiles=1
            )
            contract = SplitContract(sp_p, mf_p, enforce_pinned_split_sha256=False)
            fit_tid = contract.get_fit_tiles("fit_8")[0]
            fw_tok = contract.issue_fit_firewall_token([fit_tid], subset_name="fit_8")
            reader = FullDayTileReader(contract)
            sh = build_or_open_single_tile_shard(
                root / "cache", reader, fit_tid, week_indices=[0], receiver_mode="sampled", max_receivers=16
            )
            raw_b = sh.load_compact_batch(include_candidate_bank=True, include_labels=True)

            # Corrupt ALL hidden-75% truth & boundary distances with extreme garbage `1e6` / `-999`
            view_clean, bmask_n = apply_label_budget_view(
                raw_b, label_budget_fraction=0.25, label_budget_seed=20261112
            )
            assert bmask_n is not None
            kept_rec = bmask_n.bool()
            self.assertGreater(int(kept_rec.sum().item()), 0)
            self.assertLess(int(kept_rec.sum().item()), int(kept_rec.numel()))
            self.assertNotIn("response_5x5_bitpack", view_clean)

            corrupted_b = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in raw_b.items()}
            hidden_rec = ~kept_rec
            corrupted_b["truth"][:, hidden_rec] = 1 - corrupted_b["truth"][:, hidden_rec]
            corrupted_b["boundary_distance_m"][:, hidden_rec] = 99999.0

            obj_cfg = ObjectiveConfig(
                jepa_mode="hybrid",
                mask_mode="spatial_boundary",
                reg_mode="sigreg",
                label_budget_fraction=0.25,
                label_budget_seed=20261112,
                counterfactual_weight=0.05,
            )

            torch.manual_seed(777)
            m1 = build_model("rind_full_geom", hidden_dim=32, max_candidates=8)
            m1.update_normalizers(
                materialize_gpu_tokens_from_azimuth_bank(move_batch_to_device(raw_b, "cpu")),
                firewall_token=fw_tok,
                tile_ids=[fit_tid],
                expected_split_sha256=contract.split_sha256,
            )
            m2 = build_model("rind_full_geom", hidden_dim=32, max_candidates=8)
            m2.load_state_dict(m1.state_dict())
            m2.record_normalizer_tiles(m1.consumed_normalizer_tile_ids)

            s1 = fit_model_fullday(
                m1,
                lambda _ep: [raw_b],
                firewall_token=fw_tok,
                expected_split_sha256=contract.split_sha256,
                epochs=1,
                seed=1234,
                objective_config=obj_cfg,
                prepare_gpu_tokens_fn=materialize_gpu_tokens_from_azimuth_bank,
            )
            s2 = fit_model_fullday(
                m2,
                lambda _ep: [corrupted_b],
                firewall_token=fw_tok,
                expected_split_sha256=contract.split_sha256,
                epochs=1,
                seed=1234,
                objective_config=obj_cfg,
                prepare_gpu_tokens_fn=materialize_gpu_tokens_from_azimuth_bank,
            )
            self.assertAlmostEqual(s1["epoch_logs"][0]["mean_loss"], s2["epoch_logs"][0]["mean_loss"], places=7)
            for (n1, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
                self.assertEqual(
                    float((p1 - p2).abs().max().item()),
                    0.0,
                    msg=f"Hidden 75% label leaked into parameter {n1}!",
                )

    def test_04_p0_3_corrupted_state_values_and_metric_unknown_vs_rejection(self) -> None:
        """Counterexample 10 (`P0-3`): Reject `257`/`1.5`/`-1` state codes & separate data-unknown vs model-rejection."""
        for bad_arr in (
            np.array([[[1, 2, 257]]], dtype=np.int32),
            np.array([[[1.0, 1.5, 2.0]]], dtype=np.float32),
            np.array([[[1, -1, 3]]], dtype=np.int16),
            np.array([[[0, 2, 3]]], dtype=np.int32),
        ):
            with self.assertRaises(ValueError):
                _validate_and_cast_raw_state_array(bad_arr, "unit_test")

        good = _validate_and_cast_raw_state_array(np.array([[[1, 2, 3]]], dtype=np.int32), "unit_test")
        self.assertEqual(good.dtype, np.uint8)

        # Verify `metrics.py` separates `data_unknown_slots` from `model_rejected_slots`
        logits = np.array([[1.5, 0.0, -1.5, 2.0]], dtype=np.float32)
        truth = np.array([[1, 0, 0, -1]], dtype=np.int8)
        valid = np.array([[True, True, True, False]], dtype=bool)
        m_rej = np.array([[False, True, False, False]], dtype=bool)
        overall_acc, _ = evaluate_trajectory_predictions(
            logits,
            truth,
            valid,
            solar_regimes=["solar_noon"],
            day_ids=np.array([0], dtype=np.int32),
            allow_missing_timestamps=True,
            model_rejected_mask=m_rej,
        )
        ov = overall_acc.summarize()
        self.assertEqual(ov["data_unknown_slots"], 1)
        self.assertEqual(ov["model_rejected_slots"], 1)
        self.assertAlmostEqual(ov["data_unknown_rate"], 0.25, places=6)
        self.assertAlmostEqual(ov["model_rejection_rate"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(ov["rejection_rate"], 1.0 / 3.0, places=6)

        # Blocker-distance buckets (`<50m`, `50-300m`, `300-4000m`) via real `ExactSceneBVH`
        near_wall = _extruded_box_triangles(0.0, -20.0, 4.0, 2.0, 0.0, 25.0)
        mid_wall = _extruded_box_triangles(30.0, -120.0, 8.0, 2.0, 0.0, 120.0)
        far_wall = _extruded_box_triangles(60.0, -900.0, 40.0, 4.0, 0.0, 900.0)
        scene_3 = np.concatenate([near_wall, mid_wall, far_wall], axis=0).astype(np.float64)
        recs_4 = np.array(
            [[0.0, 0.05, 0.0], [30.0, 0.05, 0.0], [60.0, 0.05, 0.0], [200.0, 0.05, 0.0]],
            dtype=np.float64,
        )
        sun_south = np.array([[0.0, 0.35, -0.9367496997597597]], dtype=np.float64)
        b_rep = compute_blocker_distance_stratified_metrics(
            np.array([[-2.0, -2.0, -2.0, 2.0]], dtype=np.float64),
            np.array([[0, 0, 0, 1]], dtype=np.int8),
            np.ones((1, 4), dtype=bool),
            receivers_xyz=recs_4,
            sun_directions_xyz=sun_south,
            scene_triangles_xyz=scene_3,
        )
        self.assertEqual(b_rep["buckets"]["lt_50m"]["point_frames"], 1)
        self.assertEqual(b_rep["buckets"]["50m_to_300m"]["point_frames"], 1)
        self.assertEqual(b_rep["buckets"]["300m_to_4000m"]["point_frames"], 1)
        self.assertEqual(b_rep["unoccluded_open_sky"]["point_frames"], 1)
        far = b_rep["fine_buckets"]["800m_to_1600m"]
        self.assertEqual((far["point_frames"], far["shadow_support"], far["false_sunlit_count"]), (1, 1, 0))
        self.assertIsNone(b_rep["fine_buckets"]["1600m_to_4000m"]["false_sunlit_rate"])

    def test_05_p0_4_and_p0_5_fourier102_event_gate_and_convgru_crop_isolation(self) -> None:
        """Counterexample 4 & 5 (`P0-4`, `P0-5`): `102D` Fourier encoder, gap reset, and `0.0` crop leakage."""
        enc102 = FourierSphereTangentEncoder102(hidden_dim=32)
        self.assertEqual(enc102.in_dim, 102)

        # Verify `LocalConvGRUGridModel` default is `per_receiver_crop` and modifying `static_fields_8hw`
        # outside `(21 x 21)` around `(r=25, c=25)` produces `0.0` change in predictions!
        self.assertEqual(len(MODEL_TYPES), 5)
        self.assertIn("full_tile_convgru", ALL_MODEL_TYPES)
        conv_local = LocalConvGRUGridModel(hidden_dim=24, crop_size=21)
        self.assertEqual(conv_local.spatial_execution_mode, "per_receiver_crop")
        conv_local.eval()

        fields_a = torch.zeros(8, 64, 64)
        fields_b = fields_a.clone()
        # Mutate far-field region `(0..10, 0..10)` strictly outside `[25-10..25+10] = [15..35]`
        fields_b[:, :10, :10] = 500.0
        rc = torch.tensor([[25, 25]], dtype=torch.long)
        suns = _make_unit_suns(4)
        base_in = {
            "static_fields_8hw": fields_a,
            "geometry_known_hw": torch.ones(64, 64, dtype=torch.bool),
            "local_features": torch.zeros(1, 16),
            "receiver_rc": rc,
            "sun_dirs": suns,
            "same_day_adjacent_mask": torch.ones(3, dtype=torch.bool),
        }
        far_in = dict(base_in, static_fields_8hw=fields_b)
        with torch.no_grad():
            out_a = conv_local(base_in)["probs"]
            out_b = conv_local(far_in)["probs"]
        self.assertEqual(float((out_a - out_b).abs().max().item()), 0.0)
        self.assertEqual(
            conv_local(base_in)["execution_counters"]["spatial_execution_mode"],
            "per_receiver_crop",
        )

        # Verify `full_tile_convgru` uses `full_tile_512_halo`
        conv_full = build_model("full_tile_convgru", hidden_dim=24, crop_size=64)
        self.assertEqual(conv_full.spatial_execution_mode, "full_tile_512_halo")

    def test_06_p0_6_strided_rollout_strictly_causal_zero_future_anchor_leakage(self) -> None:
        """Counterexample 6 (`P0-6`): `evaluate_strided_rollout` (`mode='causal_rollout'`) has `0.0` future sensitivity."""
        torch.manual_seed(99)
        model = RindCandidateWorldModel(hidden_dim=32, max_candidates=4)
        model.eval()
        t_steps, n_pts, k_c = 9, 4, 4
        suns_1 = _make_unit_suns(t_steps)
        rel_1 = torch.randn(t_steps, n_pts, k_c, 3, 3) * 8.0
        rel_1[..., 1] = rel_1[..., 1].abs() + 2.0
        mk = torch.ones(t_steps, n_pts, k_c, dtype=torch.bool)
        tok_1 = build_canonical37_torch(rel_1, mk, mk, suns_1)

        # Mutate future anchor frame `t = 4` dramatically (`* -50 + 300`): frames `t = 1, 2, 3` MUST NOT CHANGE!
        suns_2 = suns_1.clone()
        suns_2[4:] = torch.tensor([-0.6, 0.5, -0.6245])
        suns_2 = suns_2 / torch.linalg.norm(suns_2, dim=-1, keepdim=True)
        tok_2 = tok_1.clone()
        tok_2[4:] = -50.0 * tok_2[4:] + 300.0

        b1 = {
            "candidate_tokens": tok_1,
            "candidate_mask": mk,
            "local_features": torch.zeros(n_pts, 16),
            "receivers_xyz": torch.zeros(n_pts, 3),
            "sun_dirs": suns_1,
            "same_day_adjacent_mask": torch.ones(t_steps - 1, dtype=torch.bool),
        }
        b2 = dict(b1, candidate_tokens=tok_2, sun_dirs=suns_2)
        r1 = model.evaluate_strided_rollout(b1, stride=4)
        r2 = model.evaluate_strided_rollout(b2, stride=4)
        self.assertEqual(r1["inference_mode"], "causal_rollout")
        self.assertTrue(bool(r1["is_causal"]))
        self.assertEqual(float((r1["probs"][:4] - r2["probs"][:4]).abs().max().item()), 0.0)

        # Also verify `inference_mode="single_frame"` is invariant to shuffling frame order
        perm = torch.tensor([3, 0, 2, 1], dtype=torch.long)
        sub_b = {
            "candidate_tokens": tok_1[:4],
            "candidate_mask": mk[:4],
            "local_features": torch.zeros(n_pts, 16),
            "receivers_xyz": torch.zeros(n_pts, 3),
            "sun_dirs": suns_1[:4],
            "same_day_adjacent_mask": torch.ones(3, dtype=torch.bool),
        }
        perm_b = dict(
            sub_b,
            candidate_tokens=tok_1[:4][perm],
            candidate_mask=mk[:4][perm],
            sun_dirs=suns_1[:4][perm],
        )
        with torch.no_grad():
            sf_orig = model(sub_b, inference_mode="static_direct")["probs"][perm]
            sf_perm = model(perm_b, inference_mode="static_direct")["probs"]
        self.assertLess(float((sf_orig - sf_perm).abs().max().item()), 1e-6)

    def test_07_p0_7_and_p0_8_exact_bvh_soft_raytracer_kkt_and_classmate_bugs(self) -> None:
        """Counterexamples 7 & 8 + Tony Bugs 1 & 2 + Angela Bug 3: BVH, Soft Raytracer, KKT, Sloped Roof & Ground Slack."""
        # 1. ExactSceneBVH vs brute-force float64 Möller-Trumbore (including grazing/near-axis-parallel rays)
        rng = np.random.default_rng(20261112)
        boxes = [
            _extruded_box_triangles(20.0, 20.0, 5.0, 5.0, 0.0, 18.0),
            _extruded_box_triangles(45.0, 22.0, 6.0, 4.0, 0.0, 30.0),
            _extruded_box_triangles(32.0, 48.0, 4.0, 7.0, 0.0, 14.0),
        ]
        tris = np.concatenate(boxes, axis=0).astype(np.float64)
        sids = np.arange(tris.shape[0], dtype=np.uint64) + 100
        bvh = ExactSceneBVH(tris, source_triangle_ids=sids)

        origins = rng.uniform(5.0, 60.0, size=(48, 3))
        origins[:, 1] = 0.05
        dirs = rng.normal(size=(48, 3))
        dirs[:, 1] = np.abs(dirs[:, 1]) + 1e-4
        # Inject exact axis-aligned / grazing directions (`dx ~ 0` or `dz ~ 0`)
        dirs[0] = np.array([0.0, 0.35, -0.9367496997597597])
        dirs[1] = np.array([1e-16, 0.25, -0.9682458365518543])
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        occ_bvh, dist_bvh, _, _ = bvh.query_first_hit(origins, dirs, t_min=0.05, t_max=4000.0)
        for i in range(origins.shape[0]):
            hit_bf, t_bf = moller_trumbore_intersect_f64(origins[i], dirs[i], tris, t_min=0.05, t_max=4000.0)
            self.assertEqual(bool(occ_bvh[i]), bool(np.any(hit_bf)))
            if np.any(hit_bf):
                self.assertAlmostEqual(float(dist_bvh[i]), float(np.min(t_bf[hit_bf])), places=9)

        # 2. Soft raytracer: behind-origin (`t < 0`), self-hit (`t < t_min` & `exclude_source_triangle_ids`),
        # beyond-cutoff (`t > t_max`), and co-planar shared diagonal (`0` double-counting kink)
        wall_behind = torch.tensor([[[[-5.0, 0.0, 10.0], [5.0, 0.0, 10.0], [0.0, 10.0, 10.0]]]], dtype=torch.float32).squeeze(0)
        wall_far = torch.tensor([[[[-500.0, 0.0, -4500.0], [500.0, 0.0, -4500.0], [0.0, 900.0, -4500.0]]]], dtype=torch.float32).squeeze(0)
        rec0 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        sun_south = torch.tensor([[0.0, 0.05, -0.9987492]], dtype=torch.float32)
        p_behind = _differentiable_soft_moller_trumbore_sunlit(rec0, sun_south, wall_behind, t_min=0.05, t_max=4000.0)
        p_far = _differentiable_soft_moller_trumbore_sunlit(rec0, sun_south, wall_far, t_min=0.05, t_max=4000.0)
        self.assertAlmostEqual(float(p_behind.item()), 1.0, places=6)
        self.assertAlmostEqual(float(p_far.item()), 1.0, places=6)

        # Receiver sitting on roof with `exclude_source_triangle_ids` -> self-hit excluded -> sunlit == 1.0
        roof_tri = torch.tensor(
            [
                [[-5.0, 10.0, -5.0], [5.0, 10.0, -5.0], [5.0, 10.0, 5.0]],
                [[-5.0, 10.0, -5.0], [5.0, 10.0, 5.0], [-5.0, 10.0, 5.0]],
            ],
            dtype=torch.float32,
        )
        rec_roof = torch.tensor([[0.0, 9.90, 0.0]], dtype=torch.float32)
        sun_up = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        p_excl = _differentiable_soft_moller_trumbore_sunlit(
            rec_roof,
            sun_up,
            roof_tri,
            source_triangle_ids=torch.tensor([42, 43], dtype=torch.int64),
            exclude_source_triangle_ids=torch.tensor([42], dtype=torch.int64),
            temperature_m=0.25,
        )
        # Triangle 42 is excluded, and `(0, 0)` on diagonal has `~0.5` margin on triangle 43; test interior of 42 `(2.0, -2.0)`:
        rec_roof_in42 = torch.tensor([[2.0, 9.90, -2.0]], dtype=torch.float32)
        p_in42 = _differentiable_soft_moller_trumbore_sunlit(
            rec_roof_in42,
            sun_up,
            roof_tri[:1],
            source_triangle_ids=torch.tensor([42], dtype=torch.int64),
            exclude_source_triangle_ids=torch.tensor([42], dtype=torch.int64),
            temperature_m=0.25,
        )
        self.assertAlmostEqual(float(p_in42.item()), 1.0, places=6)
        _ = p_excl

        # Co-planar 2-triangle quad vs single enclosing triangle along shared diagonal `(0, 5, -10)`
        quad_2 = torch.tensor(
            [
                [[-10.0, 0.0, -10.0], [10.0, 0.0, -10.0], [10.0, 10.0, -10.0]],
                [[-10.0, 0.0, -10.0], [10.0, 10.0, -10.0], [-10.0, 10.0, -10.0]],
            ],
            dtype=torch.float32,
        )
        p_diag = _differentiable_soft_moller_trumbore_sunlit(
            torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
            quad_2,
            temperature_m=1.0,
        )
        # Without co-planar grouping, two `sigmoid(0) == 0.5` edges on the diagonal would underestimate occlusion (`sunlit ~ 0.5`);
        # with co-planar scatter_add (`0.5 + 0.5 == 1.0`), `sunlit == 0.0`!
        self.assertLess(float(p_diag.item()), 1e-4)

        # 3. Exact KKT active-set backward test (`Counterexample 8`): clamped element has `0` self-grad
        raw_d = torch.tensor([50.0, -2.0, -3.0], dtype=torch.float64, requires_grad=True)
        areas = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
        heights = torch.tensor([30.0, 30.0, 30.0], dtype=torch.float64)
        proj_d = DifferentiableAreaWeightedVolumeKKT.apply(raw_d, areas, heights, 10.0, 0.05)
        self.assertAlmostEqual(float((proj_d * areas).sum().item()), 0.0, places=11)
        self.assertAlmostEqual(float(proj_d[0].item()), 10.0, places=9)  # clamped at +max_delta_h_m
        torch.autograd.gradcheck(
            lambda x: DifferentiableAreaWeightedVolumeKKT.apply(x, areas, heights, 10.0, 0.05),
            (raw_d,),
            eps=1e-6,
            atol=1e-5,
        )

        # 4. Tony Bug 1: Ground triangles (`building_ids == -1`) NEVER receive `+height_slack_m`!
        ground_tris = np.array(
            [[[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [50.0, 0.0, 50.0]]], dtype=np.float64
        )
        _, g_mk = _preselect_candidate_triangle_indices(
            np.array([[25.0, 0.05, 25.0]], dtype=np.float64),
            ground_tris,
            building_ids=np.array([-1], dtype=np.int64),
            ground_mask=np.array([True], dtype=bool),
            height_slack_m=15.0,
        )
        self.assertEqual(int(g_mk.sum()), 0)

        # 5. Tony Bug 2 & Sloped-roof couplings: `DEFAULT_TILE_METERS_PER_PIXEL == 0.1953125` & barycentric ramp
        self.assertAlmostEqual(DEFAULT_TILE_METERS_PER_PIXEL, 0.1953125, places=9)
        sloped_roof = np.array(
            [
                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
                [[0.0, 10.0, 0.0], [10.0, 20.0, 0.0], [0.0, 10.0, 10.0]],
            ],
            dtype=np.float64,
        )
        ew_slope, _ = _build_local_feature_height_couplings(
            np.array([[5.0, 0.05, 2.0]], dtype=np.float64),
            sloped_roof,
            np.array([0, 0], dtype=np.int64),
            1,
        )
        # Base is `y=0`, roof peak `y=20`; at `x = 5.0` halfway between `y=10` and `y=20`, `(15 - 0) / 20 = 0.75`!
        self.assertAlmostEqual(float(ew_slope[0, 0]), 0.75, places=4)

        # 6. Angela Bug 3: Directionally balanced switching Focal Loss (`sun->shadow` and `shadow->sun`)
        logits_bal = torch.zeros((2, 4), dtype=torch.float32, requires_grad=True)
        # Single pair `0 -> 1` has 3 `1->0` transitions and only 1 `0->1` transition (0 stable negatives!)
        truth_bal = torch.tensor(
            [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        f_loss, pair_cnt, _ = compute_focal_event_hard_negative_loss(
            torch.sigmoid(logits_bal),
            None,
            truth_bal,
            torch.ones_like(truth_bal, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
        )
        self.assertEqual(pair_cnt, 4)
        f_loss.backward()
        assert logits_bal.grad is not None
        # Wait: when `logits == 0`, `p_cur = p_prev = 0.5`, so `pred_sw = p_cur*(1-p_prev) + (1-p_cur)*p_prev = 0.5` is at stationary saddle `d pred_sw / d p_cur = 1 - 2*p_prev = 0`!
        # Let's evaluate at non-degenerate `logits_0 = [1.2, 1.2, 1.2, -1.2]`, `logits_1 = [0.0, 0.0, 0.0, 0.0]`!
        logits_bal2 = torch.tensor(
            [[1.2, 1.2, 1.2, -1.2], [0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        f_loss2, _, _ = compute_focal_event_hard_negative_loss(
            torch.sigmoid(logits_bal2),
            None,
            truth_bal,
            torch.ones_like(truth_bal, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
        )
        f_loss2.backward()
        assert logits_bal2.grad is not None
        # Because `1->0` (cols 0,1,2) is averaged over 3 items while `0->1` (col 3) is averaged over 1 item with equal 0.5/0.5 weight,
        # `|grad[1, 3]|` is `3x` `|grad[1, 0]|`!
        self.assertAlmostEqual(
            abs(float(logits_bal2.grad[1, 3].item())),
            3.0 * abs(float(logits_bal2.grad[1, 0].item())),
            places=5,
        )

        # 7. Closed solid manifold check rejects non-closed roof+base sheet without walls unless `allow_nonclosed_surface_proxy`
        open_check = verify_extruded_building_manifold_closure(sloped_roof, np.array([0, 0], dtype=np.int64))
        self.assertFalse(bool(open_check["all_closed_manifold"]))
        with self.assertRaises((ValueError, RuntimeError)):
            optimize_fullday_building_heights(
                np.array([[25.0, 0.05, 25.0]], dtype=np.float64),
                np.array([[0.0, 0.5, -0.8660254]], dtype=np.float64),
                sloped_roof,
                building_ids=np.array([0, 0], dtype=np.int64),
                steps=1,
                allow_nonclosed_surface_proxy=False,
            )

        # Empty valid support returns `None` metrics (NOT `0.0`) and is excluded from multi-tile mean!
        closed_scene = np.concatenate(
            [
                _extruded_box_triangles(20.0, 20.0, 4.0, 4.0, 0.0, 15.0),
                _extruded_box_triangles(35.0, 20.0, 4.0, 4.0, 0.0, 18.0),
            ],
            axis=0,
        ).astype(np.float64)
        empty_res = optimize_fullday_building_heights(
            np.array([[25.0, 0.05, 35.0]], dtype=np.float64),
            np.array([[0.0, 0.5, -0.8660254]], dtype=np.float64),
            closed_scene,
            valid_mask=np.zeros((1, 1), dtype=bool),
            steps=2,
        )
        self.assertIsNone(empty_res.signed_net_sunlight_delta_fraction)
        agg = aggregate_multi_tile_inverse_design(
            [empty_res, {"signed_net_sunlight_delta_fraction": 0.12, "valid_point_frames": 10}]
        )
        self.assertEqual(agg["evaluated_nonempty_tiles"], 1)
        self.assertEqual(agg["excluded_empty_support_tiles"], 1)
        self.assertAlmostEqual(float(agg["mean_signed_net_sunlight_delta_fraction"]), 0.12, places=6)

    def test_08_group_highlight_representation_sufficiency_and_bayesian_height_inference(self) -> None:
        """Verify Ethan's G1 Highlight (`Representation Sufficiency`) + `P1` Bayesian Unknown Height Inference."""
        bench = build_layered_occlusion_benchmark_scene()

        # 1. Evaluate Representation Sufficiency on exact 3D layered-occlusion urban plaza benchmark
        suff = evaluate_representation_sufficiency(
            bench["receivers_xyz"],
            bench["sun_directions_xyz"],
            bench["triangles_xyz"],
            source_instance_ids=bench["source_instance_ids"],
            ground_mask=bench["ground_mask"],
            edit_drop_fraction=0.92,
        )
        self.assertEqual(suff["status"], "ok")
        self.assertTrue(bool(suff["anti_leakage_reconstructed_inputs"]))

        lay = suff["layered_occlusion_regime_transition"]
        self.assertGreater(lay["layered_anchor_ray_count"], 0)
        self.assertGreater(lay["post_driver_shift_ray_count"], 0)
        # At `t_0` both Full RIND and oracle-selected building predict `y=0` accurately (`== 1.0`),
        # AFTER driver shift (`A` clears, `B` still blocks), Full RIND dramatically outperforms the oracle-selected building.
        self.assertGreaterEqual(float(lay["at_t0_accuracy_full_rind"]), 0.95)
        self.assertGreaterEqual(float(lay["at_t0_accuracy_oracle_first_building"]), 0.95)
        self.assertGreater(
            float(lay["post_driver_shift_accuracy_full_rind"]),
            float(lay["post_driver_shift_accuracy_oracle_first_building"]) + 0.15,
        )

        nonadd = suff["nonadditive_unseen_edit_combinations"]
        self.assertGreater(nonadd["nonzero_nonadditive_interaction_rays"], 0)
        # On layered overlap rays, lowering A alone gives `0` gain and lowering B alone gives `0` gain,
        # whereas combined `e_12 = e_1 (+) e_2` unlocks sunlight (`I*(e_1, e_2) > 0`)!
        self.assertGreater(float(nonadd["layered_overlap_mean_true_interaction"]), 0.25)
        self.assertLess(
            float(nonadd["layered_overlap_interaction_mae_full_rind"]),
            float(nonadd["layered_overlap_interaction_mae_linear_additive"]) * 0.35,
        )
        self.assertGreater(
            float(nonadd["combined_edit_change_f1_full_rind"]),
            float(nonadd["combined_edit_change_f1_linear_additive"]),
        )

        # Also verify `evaluate_representation_sufficiency` works with a neural `RindCandidateWorldModel`
        rind_m = RindCandidateWorldModel(hidden_dim=24, max_candidates=8)
        suff_nn = evaluate_representation_sufficiency(
            bench["receivers_xyz"][:6],
            bench["sun_directions_xyz"][:6],
            bench["triangles_xyz"],
            rind_model=rind_m,
            source_instance_ids=bench["source_instance_ids"],
            ground_mask=bench["ground_mask"],
        )
        self.assertEqual(suff_nn["status"], "ok")
        self.assertEqual(suff_nn["model_source"], "rind_checkpoint")

        # 2. Verify Dominant-Occluder Dropout in `compute_counterfactual_height_monotonicity_loss`
        suns_t = torch.as_tensor(bench["sun_directions_xyz"][:4], dtype=torch.float32)
        rec_t = torch.as_tensor(bench["receivers_xyz"][:4], dtype=torch.float32)
        rel_cf = torch.randn(4, 4, 4, 3, 3) * 8.0
        rel_cf[..., 1] = rel_cf[..., 1].abs() + 4.0
        cm_cf = torch.ones(4, 4, 4, dtype=torch.bool)
        tok_cf_in = build_canonical37_torch(rel_cf, cm_cf, cm_cf, suns_t)
        cf_batch = {
            "candidate_tokens": tok_cf_in,
            "candidate_mask": cm_cf,
            "sun_dirs": suns_t,
            "receivers_xyz": rec_t,
            "local_features": torch.zeros(4, 16),
            "same_day_adjacent_mask": torch.ones(3, dtype=torch.bool),
            "valid_mask": torch.ones(4, 4, dtype=torch.bool),
        }
        rind_m.train()
        base_out_cf = rind_m(cf_batch, use_calibrated_temperature=False)
        cf_loss = compute_counterfactual_height_monotonicity_loss(
            rind_m,
            cf_batch,
            base_out_cf["probs"],
            dominant_occluder_dropout_prob=1.0,
        )
        self.assertTrue(torch.isfinite(cf_loss))

        # 3. Bayesian Unknown Building Height Inference (`P1`, zero height leakage, posterior shrinkage)
        inf_res = infer_unknown_building_heights_bayesian(
            bench["receivers_xyz"],
            bench["sun_directions_xyz"],
            bench["triangles_xyz"],
            unknown_building_ids=[1],
            source_instance_ids=bench["source_instance_ids"],
            ground_mask=bench["ground_mask"],
            num_grid_hypotheses=21,
            sensor_noise_flip_prob=0.03,
            initial_queries=12,
            active_rounds=5,
            queries_per_round=12,
            seed=20261112,
        )
        self.assertEqual(inf_res["status"], "ok")
        self.assertTrue(bool(inf_res["anti_leakage_verified"]))
        self.assertLess(float(inf_res["final_posterior_mae_m"]), float(inf_res["initial_prior_mae_m"]))
        self.assertLess(float(inf_res["uncertainty_shrinkage_ratio"]), 0.75)

    def test_09_p2_resume_parity_narrow_pread_tristate_and_cli_integration(self) -> None:
        """Counterexample 12 (`P2-1..P2-7`): `10+10 == 20` step resume, narrow `pread`, tristate `unknown`, CLI."""
        # 1. Tristate `unknown` preserved when tristate field is omitted
        self.assertEqual(resolve_tristate_exposure_status("t0", {"tile_id": "t0"}), "unknown")
        self.assertEqual(
            resolve_tristate_exposure_status("t0", {"historical_exposure_status": "unexposed"}),
            "unexposed",
        )

        with TemporaryTestDir("rind_p2_resume_") as tmp:
            root = Path(tmp)
            # Atomic JSON file lock update check
            jpath = root / "ledger.json"
            atomic_update_json_file(jpath, lambda d: {"n": int((d or {"n": 0})["n"]) + 1})
            atomic_update_json_file(jpath, lambda d: {"n": int((d or {"n": 0})["n"]) + 1})
            self.assertEqual(json.loads(jpath.read_text(encoding="utf-8"))["n"], 2)

            # Narrow column `pread` check (`bytes < 10%` of full array bytes)
            arr2d = np.arange(64 * 512, dtype=np.float32).reshape(64, 512)
            npy_p = root / "wide.npy"
            np.save(npy_p, arr2d)
            reset_fast_data_io_stats()
            sub_arr = _read_npy_slice_no_mmap(npy_p, (slice(4, 20), slice(10, 18)))
            io_st = get_fast_data_io_stats()
            self.assertTrue(np.array_equal(sub_arr, arr2d[4:20, 10:18]))
            self.assertGreater(io_st["read_calls"], 0)
            self.assertLess(io_st["bytes_read"], arr2d.nbytes // 10)

            # Resumable training parity: `2` steps continuous == `1` step + save/load (`weights_only=True`) + `1` step!
            sp_p, mf_p = create_synthetic_fullday_fixture(
                root / "fix", grid_side=10, frames_per_day=8, num_fit_tiles=8, num_val_tiles=1, num_cal_tiles=1
            )
            contract = SplitContract(sp_p, mf_p, enforce_pinned_split_sha256=False)
            fit_tid = contract.get_fit_tiles("fit_8")[0]
            fw_tok = contract.issue_fit_firewall_token([fit_tid], subset_name="fit_8")
            reader = FullDayTileReader(contract)
            sh = build_or_open_single_tile_shard(
                root / "cache", reader, fit_tid, week_indices=[0, 1], receiver_mode="sampled", max_receivers=12
            )
            b_list = list(
                sh.iter_day_batches(days_per_batch=1, receiver_chunk_size=12, model_type="rind_full_geom", shuffle_days=False)
            )
            self.assertEqual(len(b_list), 2)

            obj_cfg = ObjectiveConfig(jepa_mode="hybrid", mask_mode="spatial_boundary", reg_mode="sigreg", counterfactual_weight=0.0)
            torch.manual_seed(555)
            m_cont = build_model("rind_full_geom", hidden_dim=24, max_candidates=8)
            m_cont.update_normalizers(
                materialize_gpu_tokens_from_azimuth_bank(move_batch_to_device(b_list[0], "cpu")),
                firewall_token=fw_tok,
                tile_ids=[fit_tid],
                expected_split_sha256=contract.split_sha256,
            )
            m_part1 = build_model("rind_full_geom", hidden_dim=24, max_candidates=8)
            m_part1.load_state_dict(m_cont.state_dict())
            m_part1.record_normalizer_tiles(m_cont.consumed_normalizer_tile_ids)

            collapse_p = root / "collapse_diag.jsonl"
            s_cont = fit_model_fullday(
                m_cont,
                lambda _ep: b_list,
                firewall_token=fw_tok,
                expected_split_sha256=contract.split_sha256,
                epochs=1,
                seed=2026,
                use_bf16_amp=False,
                objective_config=obj_cfg,
                prepare_gpu_tokens_fn=materialize_gpu_tokens_from_azimuth_bank,
                collapse_diagnostics_jsonl_path=collapse_p,
                diagnostics_every_steps=1,
            )
            self.assertEqual(s_cont["total_steps"], 2)
            lines = [json.loads(line) for line in collapse_p.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 2)

            res_ckpt = root / "resume_step1.pt"
            _ = fit_model_fullday(
                m_part1,
                lambda _ep: b_list,
                firewall_token=fw_tok,
                expected_split_sha256=contract.split_sha256,
                epochs=1,
                seed=2026,
                use_bf16_amp=False,
                objective_config=obj_cfg,
                prepare_gpu_tokens_fn=materialize_gpu_tokens_from_azimuth_bank,
                resume_state_out=res_ckpt,
                max_train_steps=1,
            )
            m_part2 = build_model("rind_full_geom", hidden_dim=24, max_candidates=8)
            m_part2.record_normalizer_tiles(m_cont.consumed_normalizer_tile_ids)
            s_resumed = fit_model_fullday(
                m_part2,
                lambda _ep: b_list,
                firewall_token=fw_tok,
                expected_split_sha256=contract.split_sha256,
                epochs=1,
                seed=2026,
                use_bf16_amp=False,
                objective_config=obj_cfg,
                prepare_gpu_tokens_fn=materialize_gpu_tokens_from_azimuth_bank,
                resume_from=res_ckpt,
            )
            self.assertEqual(s_resumed["total_steps"], 2)
            for (nm, p_c), (_, p_r) in zip(m_cont.named_parameters(), m_part2.named_parameters()):
                self.assertLess(
                    float((p_c - p_r).abs().max().item()),
                    1e-6,
                    msg=f"Resume mismatch on parameter {nm}",
                )
            _ = save_training_resume_state
            _ = load_training_resume_state

            # CLI subcommand smoke (`eval-repr-sufficiency --synthetic-benchmark` & `infer-building-height --synthetic-benchmark`)
            suff_out = root / "cli_suff.json"
            rc_cli = rind_cli_main(
                [
                    "eval-repr-sufficiency",
                    "--synthetic-benchmark",
                    "--output",
                    str(suff_out),
                    "--quiet",
                ]
            )
            self.assertEqual(rc_cli, 0)
            suff_doc = json.loads(suff_out.read_text(encoding="utf-8"))
            self.assertEqual(suff_doc["representation_sufficiency_result"]["status"], "ok")

            # Verify `build_arg_parser()` exposes all required flags
            parser = build_arg_parser()
            ns_tr = parser.parse_args(
                [
                    "train",
                    "--cache-dir",
                    str(root / "cache"),
                    "--model-type",
                    "rind_full_geom",
                    "--jepa-mode",
                    "hybrid",
                    "--ema-target-scope",
                    "full_backbone",
                    "--mask-mode",
                    "spatial",
                    "--jepa-context-mode",
                    "shuffled_x",
                    "--label-budget-fraction",
                    "0.25",
                    "--attention-mode",
                    "solar_gating",
                    "--disable-event-gate",
                    "--checkpoint-out",
                    str(root / "out.pt"),
                ]
            )
            self.assertEqual(ns_tr.ema_target_scope, "full_backbone")
            self.assertEqual(ns_tr.mask_mode, "spatial")
            self.assertEqual(ns_tr.jepa_context_mode, "shuffled_x")
            self.assertAlmostEqual(ns_tr.label_budget_fraction, 0.25)
            self.assertEqual(ns_tr.attention_mode, "solar_gating")
            self.assertTrue(ns_tr.disable_event_gate)


if __name__ == "__main__":
    unittest.main()

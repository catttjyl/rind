"""Regression tests for pure-supervised CLI guard and az0/az24 trajectory reconstruction."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import warnings

import numpy as np
import torch

from rind.cli import (
    _build_train_objective_config,
    build_arg_parser,
    cmd_eval_repr_sufficiency,
    cmd_train,
)
from rind.fast_data import build_or_open_single_tile_shard, materialize_gpu_tokens_from_azimuth_bank
from rind.fast_train import fit_model_fullday
from rind.fullday_data import (
    FullDayTileReader,
    create_synthetic_fullday_fixture,
    select_azimuth_sector_candidates,
)
from rind.inverse_design import (
    _preselect_candidate_triangle_indices,
    _reconstruct_and_predict_scene_trajectory,
    build_layered_occlusion_benchmark_scene,
    evaluate_representation_sufficiency,
    optimize_fullday_building_heights,
)
from rind.models import build_model, load_rind_checkpoint, save_rind_checkpoint
from rind.ray_candidates import RayCandidateScene
from rind.split_contract import SplitContract


class TestPureSupervisedCLIGuard(unittest.TestCase):
    def _base_train_args(self, *extra: str, model_type: str = "rind_full_geom"):
        parser = build_arg_parser()
        return parser.parse_args(
            [
                "train",
                "--cache-dir",
                "/tmp/unused_cache",
                "--model-type",
                model_type,
                "--checkpoint-out",
                "/tmp/unused_ckpt.pt",
                *extra,
            ]
        )

    def test_jepa_none_reg_none_without_pure_supervised_raises(self):
        args = self._base_train_args("--jepa-mode", "none", "--reg-mode", "none")
        with self.assertRaisesRegex(ValueError, "Ambiguous pure-supervised"):
            _build_train_objective_config(args)
        with self.assertRaisesRegex(ValueError, "Ambiguous pure-supervised"):
            cmd_train(args)

    def test_partial_cli_overrides_still_raise(self):
        args_no_cf_zero = self._base_train_args(
            "--jepa-mode", "none", "--reg-mode", "none", "--transition-mode", "none"
        )
        with self.assertRaisesRegex(ValueError, "Ambiguous pure-supervised"):
            _build_train_objective_config(args_no_cf_zero)

        args_no_trans_none = self._base_train_args(
            "--jepa-mode", "none", "--reg-mode", "none", "--counterfactual-weight", "0"
        )
        with self.assertRaisesRegex(ValueError, "Ambiguous pure-supervised"):
            _build_train_objective_config(args_no_trans_none)

    def test_jepa_none_reg_none_with_explicit_zero_transition_and_cf_succeeds_cleanly(self):
        args = self._base_train_args(
            "--jepa-mode",
            "none",
            "--reg-mode",
            "none",
            "--transition-mode",
            "none",
            "--counterfactual-weight",
            "0",
        )
        cfg = _build_train_objective_config(args)
        self.assertEqual(cfg.jepa_mode, "none")
        self.assertEqual(cfg.reg_mode, "none")
        self.assertEqual(cfg.transition_mode, "none")
        self.assertEqual(cfg.dual_path_weight, 0.0)
        self.assertEqual(cfg.counterfactual_weight, 0.0)

    def test_pure_supervised_forces_all_aux_and_cf_weights_to_zero(self):
        args = self._base_train_args("--pure-supervised")
        cfg = _build_train_objective_config(args)
        self.assertTrue(cfg.pure_supervised)
        self.assertEqual(cfg.jepa_mode, "none")
        self.assertEqual(cfg.reg_mode, "none")
        self.assertEqual(cfg.transition_mode, "none")
        self.assertEqual(cfg.dual_path_weight, 0.0)
        self.assertEqual(cfg.counterfactual_weight, 0.0)
        eff_cf_weight = 0.0 if cfg.pure_supervised else float(cfg.counterfactual_weight)
        self.assertEqual(eff_cf_weight, 0.0)

    def test_non_rind_model_has_consistent_pure_supervised_objective_config(self):
        args = self._base_train_args(model_type="local_convgru_grid")
        cfg = _build_train_objective_config(args)
        self.assertTrue(cfg.pure_supervised)
        self.assertEqual(cfg.jepa_mode, "none")
        self.assertEqual(cfg.reg_mode, "none")
        self.assertEqual(cfg.transition_mode, "none")
        self.assertEqual(cfg.dual_path_weight, 0.0)
        self.assertEqual(cfg.counterfactual_weight, 0.0)

    def test_valid_non_pure_supervised_configs_remain_allowed(self):
        cfg_default = _build_train_objective_config(self._base_train_args())
        self.assertFalse(cfg_default.pure_supervised)
        self.assertEqual(cfg_default.jepa_mode, "hybrid")
        self.assertEqual(cfg_default.reg_mode, "sigreg")
        self.assertEqual(cfg_default.dual_path_weight, 0.05)
        self.assertEqual(cfg_default.counterfactual_weight, 0.08)

        cfg_ema = _build_train_objective_config(
            self._base_train_args("--jepa-mode", "ema_geom", "--reg-mode", "none")
        )
        self.assertFalse(cfg_ema.pure_supervised)
        self.assertEqual(cfg_ema.jepa_mode, "ema_geom")

        cfg_matched = _build_train_objective_config(
            self._base_train_args("--matched-objective", "direct")
        )
        self.assertTrue(cfg_matched.pure_supervised)
        self.assertEqual(cfg_matched.dual_path_weight, 0.0)
        self.assertEqual(cfg_matched.counterfactual_weight, 0.0)

    def test_real_fit_step_under_pure_supervised_has_zero_aux_cf_and_no_ema_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split, manifest = create_synthetic_fullday_fixture(
                root / "fixture", grid_side=8, frames_per_day=8
            )
            contract = SplitContract(split, manifest, enforce_pinned_split_sha256=False)
            fit_tiles = contract.get_fit_tiles("fit_8")[:1]
            fw_tok = contract.issue_fit_firewall_token(fit_tiles, subset_name="fit_8")
            shard = build_or_open_single_tile_shard(
                root / "cache",
                FullDayTileReader(contract),
                fit_tiles[0],
                week_indices=[0],
                max_receivers=8,
                max_candidates=8,
            )
            args = self._base_train_args("--pure-supervised")
            obj_cfg = _build_train_objective_config(args)
            model = build_model("rind_full_geom", hidden_dim=16, max_candidates=8)

            with patch("rind.fast_train.compute_candidate_face_perturbation_loss") as mock_cf, \
                 patch("rind.objectives_jepa.RindJepaAuxiliaryModule.step_ema") as mock_ema:
                stats = fit_model_fullday(
                    model,
                    lambda _ep: shard.iter_day_batches(
                        days_per_batch=1,
                        receiver_chunk_size=8,
                        model_type="rind_full_geom",
                    ),
                    firewall_token=fw_tok,
                    expected_split_sha256=contract.split_sha256,
                    epochs=1,
                    max_train_steps=1,
                    lr=1e-3,
                    device="cpu",
                    use_bf16_amp=False,
                    prepare_gpu_tokens_fn=materialize_gpu_tokens_from_azimuth_bank,
                    objective_config=obj_cfg,
                    cf_every_n_steps=1,
                )
            self.assertEqual(stats["total_steps"], 1)
            self.assertTrue(stats["pure_supervised"])
            self.assertEqual(mock_cf.call_count, 0)
            self.assertEqual(mock_ema.call_count, 0)
            self.assertEqual(stats["jepa_and_aux_summary"]["jepa_pred_loss"], 0.0)
            self.assertEqual(stats["jepa_and_aux_summary"]["jepa_trans_vel_loss"], 0.0)
            self.assertEqual(stats["jepa_and_aux_summary"]["dual_path_consistency_loss"], 0.0)
            self.assertEqual(stats["jepa_and_aux_summary"]["anti_collapse_reg_loss"], 0.0)
            self.assertEqual(stats["objective_config"]["dual_path_weight"], 0.0)
            self.assertEqual(stats["objective_config"]["counterfactual_weight"], 0.0)


class TestReconstructTrajectoryAzProtocol(unittest.TestCase):
    def test_az0_and_az24_dispatch_matches_model_protocol(self):
        torch.manual_seed(20261114)
        scene = build_layered_occlusion_benchmark_scene()
        t_steps = int(scene["sun_directions_xyz"].shape[0])
        n_pts = int(scene["receivers_xyz"].shape[0])
        m_tris = int(scene["triangles_xyz"].shape[0])

        model = build_model("rind_full_geom", hidden_dim=16, max_candidates=8).eval()

        # 1. az0 protocol (`num_azimuth_bins=0`) with non-None opaque_mask, source_triangle_ids, exclude_source_triangle_ids
        model.num_azimuth_bins = 0
        opaque_mask = np.ones((m_tris,), dtype=bool)
        opaque_mask[0] = False
        source_ids = np.arange(100, 100 + m_tris, dtype=np.int64)
        exclude_ids = np.full((n_pts,), 100, dtype=np.int64)
        with patch.object(RayCandidateScene, "query", wraps=RayCandidateScene.query, autospec=True) as mock_ray_query, \
             patch("rind.fullday_data.select_azimuth_sector_candidates", wraps=select_azimuth_sector_candidates) as mock_sector:
            probs_az0 = _reconstruct_and_predict_scene_trajectory(
                scene["receivers_xyz"],
                scene["sun_directions_xyz"],
                scene["triangles_xyz"],
                rind_model=model,
                opaque_mask=opaque_mask,
                source_triangle_ids=source_ids,
                exclude_source_triangle_ids=exclude_ids,
                device="cpu",
            )
            self.assertEqual(mock_ray_query.call_count, 1)
            self.assertEqual(mock_sector.call_count, 0)
        self.assertEqual(probs_az0.shape, (t_steps, n_pts))
        self.assertTrue(np.all(np.isfinite(probs_az0)))
        self.assertTrue(np.all((probs_az0 >= 0.0) & (probs_az0 <= 1.0)))

        # 2. az24 protocol (`num_azimuth_bins=24`) must dispatch to select_azimuth_sector_candidates
        model.num_azimuth_bins = 24
        with patch.object(RayCandidateScene, "query", wraps=RayCandidateScene.query, autospec=True) as mock_ray_query, \
             patch("rind.fullday_data.select_azimuth_sector_candidates", wraps=select_azimuth_sector_candidates) as mock_sector:
            probs_az24 = _reconstruct_and_predict_scene_trajectory(
                scene["receivers_xyz"],
                scene["sun_directions_xyz"],
                scene["triangles_xyz"],
                rind_model=model,
                device="cpu",
            )
            self.assertEqual(mock_ray_query.call_count, 0)
            self.assertEqual(mock_sector.call_count, 1)
            self.assertEqual(mock_sector.call_args.kwargs["num_azimuth_bins"], 24)
        self.assertEqual(probs_az24.shape, (t_steps, n_pts))
        self.assertTrue(np.all(np.isfinite(probs_az24)))
        self.assertTrue(np.all((probs_az24 >= 0.0) & (probs_az24 <= 1.0)))

    def test_p1_2_oracle_none_first_k_respects_az0_and_az24(self):
        scene = build_layered_occlusion_benchmark_scene()
        t_steps = int(scene["sun_directions_xyz"].shape[0])
        n_pts = int(scene["receivers_xyz"].shape[0])

        with patch.object(RayCandidateScene, "query", wraps=RayCandidateScene.query, autospec=True) as mock_ray_query, \
             patch("rind.fullday_data.select_azimuth_sector_candidates", wraps=select_azimuth_sector_candidates) as mock_sector:
            probs_az0 = _reconstruct_and_predict_scene_trajectory(
                scene["receivers_xyz"],
                scene["sun_directions_xyz"],
                scene["triangles_xyz"],
                rind_model=None,
                num_azimuth_bins=0,
                first_k_candidates_only=1,
                device="cpu",
            )
            self.assertEqual(mock_ray_query.call_count, 1)
            self.assertEqual(mock_sector.call_count, 0)
        self.assertEqual(probs_az0.shape, (t_steps, n_pts))
        self.assertTrue(np.all(np.isfinite(probs_az0)))
        self.assertTrue(np.all((probs_az0 >= 0.0) & (probs_az0 <= 1.0)))

    def test_p1_1_optimize_building_heights_rind_checkpoint_grad_az0_and_az24(self):
        torch.manual_seed(20261114)
        scene = build_layered_occlusion_benchmark_scene()
        model = build_model("rind_full_geom", hidden_dim=16, max_candidates=8).eval()

        for az_bins in (0, 24):
            model.num_azimuth_bins = az_bins
            with patch.object(RayCandidateScene, "query", wraps=RayCandidateScene.query, autospec=True) as mock_ray_query, \
                 patch("rind.inverse_design._preselect_candidate_triangle_indices", wraps=_preselect_candidate_triangle_indices) as mock_preselect:
                res = optimize_fullday_building_heights(
                    scene["receivers_xyz"],
                    scene["sun_directions_xyz"],
                    scene["triangles_xyz"],
                    optimizer_arm="rind_checkpoint_grad",
                    rind_model=model,
                    source_instance_ids=scene["source_instance_ids"],
                    ground_mask=scene["ground_mask"],
                    max_candidates=8,
                    steps=3,
                    device="cpu",
                )
                if az_bins == 0:
                    self.assertEqual(mock_ray_query.call_count, 1)
                    self.assertEqual(mock_preselect.call_count, 0)
                else:
                    self.assertEqual(mock_ray_query.call_count, 0)
                    self.assertEqual(mock_preselect.call_count, 1)
                    self.assertEqual(mock_preselect.call_args.kwargs["num_azimuth_bins"], 24)
            self.assertEqual(res.optimizer_arm, "rind_checkpoint_grad")
            self.assertTrue(np.all(np.isfinite(res.delta_h_per_building_m)))

    def test_p1_3_cmd_eval_repr_sufficiency_passes_az24_to_load_tile_fullday(self):
        parser = build_arg_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split, manifest = create_synthetic_fullday_fixture(
                root / "fixture", grid_side=8, frames_per_day=8
            )
            contract = SplitContract(split, manifest, enforce_pinned_split_sha256=False)
            model = build_model("rind_full_geom", hidden_dim=16, max_candidates=8).eval()
            ckpt_path = root / "model_az24.pt"
            save_rind_checkpoint(
                ckpt_path,
                model,
                split_contract_sha256=contract.split_sha256,
                manifest_sha256=contract.manifest_sha256,
                fit_subset_name="fit_8",
                fit_tile_ids=contract.get_fit_tiles("fit_8")[:1],
                trained_from_scratch_fit_only=True,
                seed=1,
                extra_metadata={
                    "candidate_protocol": {
                        "num_azimuth_bins": 24,
                        "capacity": 8,
                        "horizon_m": 250.0,
                        "require_complete": False,
                        "retrieval": "sector_range_v4",
                    }
                },
            )
            args = parser.parse_args(
                [
                    "eval-repr-sufficiency",
                    "--split-file",
                    str(split),
                    "--manifest-file",
                    str(manifest),
                    "--allow-synthetic-split",
                    "--diagnostic-mode",
                    "--checkpoint",
                    str(ckpt_path),
                    "--max-receivers",
                    "8",
                    "--max-candidates",
                    "8",
                    "--quiet",
                ]
            )
            orig_load = FullDayTileReader.load_tile_fullday
            with patch.object(
                FullDayTileReader, "load_tile_fullday", autospec=True, side_effect=orig_load
            ) as mock_load:
                cmd_eval_repr_sufficiency(args)
                self.assertEqual(mock_load.call_count, 1)
                self.assertEqual(mock_load.call_args.kwargs["num_azimuth_bins"], 24)

    def test_checkpoint_roundtrip_and_missing_metadata_warning(self):
        torch.manual_seed(20261114)
        scene = build_layered_occlusion_benchmark_scene()
        model = build_model("rind_full_geom", hidden_dim=16, max_candidates=8).eval()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # 1. calibration_candidate_protocol round-trip (takes precedence over candidate_protocol)
            cal_ckpt = tmp_dir / "model_cal_az0.pt"
            save_rind_checkpoint(
                cal_ckpt,
                model,
                split_contract_sha256="a" * 64,
                manifest_sha256="b" * 64,
                fit_subset_name="fit_8",
                fit_tile_ids=["tile_0"],
                trained_from_scratch_fit_only=True,
                seed=1,
                extra_metadata={
                    "calibration_candidate_protocol": {
                        "num_azimuth_bins": 0,
                        "capacity": 8,
                        "horizon_m": 250.0,
                        "require_complete": False,
                        "retrieval": "source_ray_aabb_v1",
                    }
                },
            )
            loaded_cal, _ = load_rind_checkpoint(cal_ckpt, device="cpu")
            self.assertEqual(loaded_cal.num_azimuth_bins, 0)

            # 2. Legacy checkpoint without candidate_protocol emits UserWarning
            legacy_ckpt = tmp_dir / "model_legacy.pt"
            save_rind_checkpoint(
                legacy_ckpt,
                model,
                split_contract_sha256="a" * 64,
                manifest_sha256="b" * 64,
                fit_subset_name="fit_8",
                fit_tile_ids=["tile_0"],
                trained_from_scratch_fit_only=True,
                seed=1,
            )
            with warnings.catch_warnings(record=True) as caught_load:
                warnings.simplefilter("always")
                loaded_legacy, _ = load_rind_checkpoint(legacy_ckpt, device="cpu")
            self.assertTrue(
                any("missing candidate_protocol.num_azimuth_bins" in str(w.message) for w in caught_load)
            )
            with warnings.catch_warnings(record=True) as caught_rec:
                warnings.simplefilter("always")
                _reconstruct_and_predict_scene_trajectory(
                    scene["receivers_xyz"],
                    scene["sun_directions_xyz"],
                    scene["triangles_xyz"],
                    rind_model=loaded_legacy,
                    device="cpu",
                )
            self.assertTrue(
                any("falling back to num_azimuth_bins=24" in str(w.message) for w in caught_rec)
            )

            # 3. az0 & az24 candidate_protocol round-trip in evaluate_representation_sufficiency
            for az_bins in (0, 24):
                ckpt_path = tmp_dir / f"model_az{az_bins}.pt"
                save_rind_checkpoint(
                    ckpt_path,
                    model,
                    split_contract_sha256="a" * 64,
                    manifest_sha256="b" * 64,
                    fit_subset_name="fit_8",
                    fit_tile_ids=["tile_0"],
                    trained_from_scratch_fit_only=True,
                    seed=1,
                    extra_metadata={
                        "candidate_protocol": {
                            "num_azimuth_bins": az_bins,
                            "capacity": 8,
                            "horizon_m": 250.0,
                            "require_complete": False,
                            "retrieval": "source_ray_aabb_v1" if az_bins == 0 else "sector_range_v4",
                        }
                    },
                )
                loaded, _ = load_rind_checkpoint(ckpt_path, device="cpu")
                self.assertEqual(loaded.num_azimuth_bins, az_bins)

                with patch.object(RayCandidateScene, "query", wraps=RayCandidateScene.query, autospec=True) as mock_ray_query, \
                     patch("rind.fullday_data.select_azimuth_sector_candidates", wraps=select_azimuth_sector_candidates) as mock_sector:
                    res = evaluate_representation_sufficiency(
                        scene["receivers_xyz"],
                        scene["sun_directions_xyz"],
                        scene["triangles_xyz"],
                        rind_model=loaded,
                        source_instance_ids=scene["source_instance_ids"],
                        ground_mask=scene["ground_mask"],
                        device="cpu",
                    )
                    if az_bins == 0:
                        self.assertGreater(mock_ray_query.call_count, 0)
                        self.assertEqual(mock_sector.call_count, 0)
                    else:
                        self.assertEqual(mock_ray_query.call_count, 0)
                        self.assertGreater(mock_sector.call_count, 0)
                self.assertEqual(res.get("status"), "ok")
                self.assertIn("layered_occlusion_regime_transition", res)


if __name__ == "__main__":
    unittest.main()

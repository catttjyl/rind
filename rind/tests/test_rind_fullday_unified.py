"""Core Package Verification & Regression Suite (`MANHATTAN_4445_ANNUAL_18808_V1`).

Covers naming hygiene, split contract & tri-state exposure firewall, 5-model checkpoint
round-trip parity, bounded 500-step latent recurrence, `Tier-G0` / `Tier-G1` structural
sanitizers, exact float64 Möller–Trumbore oracle, `canonical37` CPU/Torch parity, and
end-to-end diagnostic vs formal CLI suite guards.
"""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np
import torch

from rind import BENCHMARK_CONTRACT_ID
from rind.cli import main as rind_cli_main
from rind.fast_data import (
    ContiguousTileMmapShard,
    build_canonical37_torch,
    materialize_gpu_tokens_from_azimuth_bank,
)
from rind.fast_train import fit_model_fullday
from rind.fullday_data import (
    CANONICAL_4SEASON_WEEK_INDICES,
    FullDayTileReader,
    _extruded_box_triangles,
    build_canonical37_numpy,
    create_synthetic_fullday_fixture,
)
from rind.inverse_design import (
    assign_building_ids_from_triangles,
    exact_moller_trumbore_occluded_f64,
    exact_moller_trumbore_trajectory_sunlit_f64,
    moller_trumbore_intersect_f64,
    optimize_fullday_building_heights,
    perturb_buildings_consistent_f64,
)
from rind.metrics import (
    SOLAR_REGIME_ORDER,
    classify_solar_regime,
    evaluate_trajectory_predictions,
)
from rind.models import (
    MODEL_TYPES,
    RunningNormalizer,
    build_model,
    compute_head_parameter_norms,
    load_rind_checkpoint,
    save_rind_checkpoint,
)
from rind.split_contract import (
    ADDITIONAL_FIT_HOLDOUT,
    EXPECTED_MANHATTAN_4445_SPLIT_SHA256,
    SplitContract,
)

from rind.tests._tempdir import TemporaryTestDir


class TestRindUnifiedSuite(unittest.TestCase):
    def test_01_zero_legacy_batch_naming_in_rind_package(self) -> None:
        """Verify unified `MANHATTAN_4445_ANNUAL_18808_V1` naming with zero legacy batch-centric terms."""
        self.assertEqual(BENCHMARK_CONTRACT_ID, "MANHATTAN_4445_ANNUAL_18808_V1")
        pkg_root = Path(__file__).resolve().parents[2]
        forbidden_substrings = ("first" + "100", "百" + "区", "point_" + "hours")
        checked_files = 0
        for path in sorted(pkg_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in (".py", ".md", ".toml"):
                continue
            if "__pycache__" in path.parts or ".tmp_unittest" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            lower = text.lower()
            for bad in forbidden_substrings:
                self.assertNotIn(
                    bad,
                    lower,
                    msg=f"Forbidden legacy term {bad!r} found in {path}",
                )
            checked_files += 1
        self.assertGreaterEqual(checked_files, 12)

    def test_02_authoritative_split_contract_and_firewall(self) -> None:
        """Verify real on-disk split SHA-256, nested `fit_8..fit_67`, and runtime firewall."""
        contract = SplitContract()
        self.assertEqual(contract.split_sha256, EXPECTED_MANHATTAN_4445_SPLIT_SHA256)
        report = contract.build_verification_report()
        self.assertTrue(report["pinned_split_sha256_matched"])
        self.assertTrue(report["pinned_snapshot_v1_manifest_sha256_matched"])
        self.assertTrue(report["nested_fit_chain_verified"])
        self.assertTrue(report["spatial_1km_block_disjoint_verified"])
        self.assertEqual(report["cross_split_tile_overlap_count"], 0)
        self.assertEqual(contract.blind_open_count, 0)

        f8 = set(contract.get_fit_tiles("fit_8"))
        f17 = set(contract.get_fit_tiles("fit_17"))
        f34 = set(contract.get_fit_tiles("fit_34"))
        f67 = set(contract.get_fit_tiles("fit_67"))
        fall = set(contract.get_fit_tiles("fit_all"))
        self.assertEqual(len(f8), 8)
        self.assertEqual(len(f17), 17)
        self.assertEqual(len(f34), 34)
        self.assertEqual(len(f67), 67)
        self.assertTrue(f8 < f17 < f34 < f67 <= fall)

        cal_tiles = set(contract.get_calibration_tiles())
        val_tiles = set(contract.get_eval_tiles("val_spatial_clean"))
        legacy_test_tiles = set(
            contract.get_eval_tiles(
                "test_spatial_legacy_reassigned",
                model_provenance={
                    "split_contract_sha256": contract.split_sha256,
                    "manifest_sha256": contract.manifest_sha256,
                    "fit_tile_ids": sorted(f8),
                    "calibration_tile_ids": sorted(cal_tiles),
                    "calibrated": True,
                    "trained_from_scratch_fit_only": True,
                },
            )
        )
        self.assertEqual(len(cal_tiles), 12)
        self.assertEqual(len(val_tiles), 14)
        self.assertEqual(len(legacy_test_tiles), 16)

        self.assertEqual(len(fall & cal_tiles), 0)
        self.assertEqual(len(fall & val_tiles), 0)
        self.assertEqual(len(fall & legacy_test_tiles), 0)
        self.assertEqual(len(fall & set(ADDITIONAL_FIT_HOLDOUT)), 0)

        for bad_tile in [
            next(iter(cal_tiles)),
            next(iter(ADDITIONAL_FIT_HOLDOUT)),
            next(iter(val_tiles)),
            next(iter(legacy_test_tiles)),
        ]:
            with self.assertRaises(RuntimeError):
                contract.assert_fit_allowed([bad_tile])

        token = contract.issue_fit_firewall_token(sorted(f8), subset_name="fit_8")
        norm = RunningNormalizer(8)
        norm.update(
            torch.randn(4, 8),
            firewall_token=token,
            tile_ids=[next(iter(f8))],
            expected_split_sha256=contract.split_sha256,
        )
        with self.assertRaises(RuntimeError):
            norm.update(
                torch.randn(4, 8),
                firewall_token=token,
                tile_ids=[next(iter(val_tiles))],
                expected_split_sha256=contract.split_sha256,
            )

        with self.assertRaises(RuntimeError):
            contract.assert_eval_bucket_allowed(
                "test_spatial_legacy_reassigned",
                sorted(legacy_test_tiles),
                model_provenance={
                    "split_contract_sha256": contract.split_sha256,
                    "fit_tile_ids": sorted(f8),
                    "trained_from_scratch_fit_only": False,
                },
            )

        with self.assertRaises(RuntimeError):
            contract.assert_eval_bucket_allowed(
                "test_spatial_unexposed",
                [next(iter(legacy_test_tiles))],
            )

    def test_03_checkpoint_save_reload_bit_exact_all_models(self) -> None:
        """Verify all 5 architectures serialize `bce_head` > 0 and reload with bit-exact forward parity."""
        self.assertEqual(len(MODEL_TYPES), 5)
        contract = SplitContract()
        fit_t = contract.get_fit_tiles("fit_8")[:1]
        cal_t = contract.get_calibration_tiles()[:1]
        token = contract.issue_fit_firewall_token(fit_t, subset_name="fit_8")

        t_steps, n_pts, k_cands = 12, 10, 4
        sun_dirs = torch.randn(t_steps, 3)
        sun_dirs[:, 1] = sun_dirs[:, 1].abs() + 0.2
        sun_dirs = sun_dirs / torch.linalg.norm(sun_dirs, dim=-1, keepdim=True)
        batch = {
            "tile_ids": fit_t,
            "candidate_tokens": torch.randn(t_steps, n_pts, k_cands, 37),
            "candidate_mask": torch.ones(t_steps, n_pts, k_cands, dtype=torch.bool),
            "local_features": torch.randn(n_pts, 16),
            "sun_dirs": sun_dirs,
            "same_day_adjacent_mask": torch.ones(t_steps - 1, dtype=torch.bool),
            "truth": torch.randint(0, 2, (t_steps, n_pts), dtype=torch.int8),
            "valid_mask": torch.ones(t_steps, n_pts, dtype=torch.bool),
            "receiver_crops_21": torch.randn(n_pts, 9, 21, 21),
            "receiver_crops_64": torch.randn(n_pts, 9, 64, 64),
        }

        with TemporaryTestDir() as tmp:
            tmp_dir = Path(tmp)
            for m_type in MODEL_TYPES:
                crop_variants = [21, 64] if m_type == "local_convgru_grid" else [64]
                for c_sz in crop_variants:
                    kwargs = {"hidden_dim": 32}
                    if m_type == "local_convgru_grid":
                        kwargs["crop_size"] = c_sz
                        kwargs["meters_per_pixel"] = 0.1953125
                    model = build_model(m_type, **kwargs)
                    fit_model_fullday(
                        model,
                        lambda _ep: [batch],
                        firewall_token=token,
                        expected_split_sha256=contract.split_sha256,
                        epochs=1,
                        lr=2e-3,
                        device="cpu",
                        use_bf16_amp=False,
                        pin_and_async=False,
                    )
                    model.set_calibration(temperature=1.25, decision_threshold=0.47)
                    model.eval()

                    ckpt_file = tmp_dir / f"{m_type}_{c_sz}.pt"
                    save_rind_checkpoint(
                        ckpt_file,
                        model,
                        split_contract_sha256=contract.split_sha256,
                        manifest_sha256=contract.manifest_sha256,
                        fit_subset_name="fit_8",
                        fit_tile_ids=fit_t,
                        calibrated=True,
                        calibration_tile_ids=cal_t,
                    )
                    reloaded, prov = load_rind_checkpoint(
                        ckpt_file,
                        expected_split_sha256=contract.split_sha256,
                        expected_manifest_sha256=contract.manifest_sha256,
                        require_calibrated=True,
                    )
                    reloaded.eval()
                    norms = compute_head_parameter_norms(reloaded)
                    self.assertGreater(norms["bce_head"], 0.0)
                    self.assertTrue(prov["trained_from_scratch_fit_only"])

                    with torch.no_grad():
                        o1 = model(batch, use_calibrated_temperature=True)
                        o2 = reloaded(batch, use_calibrated_temperature=True)
                    self.assertTrue(torch.allclose(o1["logits"], o2["logits"], atol=0.0, rtol=0.0))
                    self.assertTrue(
                        torch.allclose(o1["logits_bce"], o2["logits_bce"], atol=0.0, rtol=0.0)
                    )

    def test_04_bounded_latent_norm_500_steps_extreme_spikes(self) -> None:
        """Verify 0 non-finite values and strict mathematical norm bound over T=500 rollout steps."""
        t_steps, n_pts, k_cands = 500, 6, 4
        suns = torch.randn(t_steps, 3)
        suns[:, 1] = suns[:, 1].abs() + 0.05
        suns = suns / torch.linalg.norm(suns, dim=-1, keepdim=True)
        extreme_tokens = torch.randn(t_steps, n_pts, k_cands, 37) * 1e6
        batch = {
            "candidate_tokens": extreme_tokens,
            "candidate_mask": torch.ones(t_steps, n_pts, k_cands, dtype=torch.bool),
            "local_features": torch.randn(n_pts, 16) * 1e4,
            "sun_dirs": suns,
            "same_day_adjacent_mask": torch.ones(t_steps - 1, dtype=torch.bool),
            "receiver_crops": torch.randn(n_pts, 9, 21, 21) * 1e4,
        }

        for m_type in ("rind_full_geom", "rind_no_edge_margin", "rind_zero_margin", "local_convgru_grid"):
            kw = {"hidden_dim": 32, "crop_size": 21} if m_type == "local_convgru_grid" else {"hidden_dim": 32}
            model = build_model(m_type, **kw)
            model.eval()
            with torch.no_grad():
                out = model(batch)
            self.assertEqual(out["nonfinite_count"], 0, msg=f"Nonfinite output in {m_type}")
            self.assertTrue(torch.isfinite(out["logits"]).all())
            theoretical_max = math.sqrt(32.0) + 1e-4
            self.assertLessEqual(out["max_latent_norm"], theoretical_max)

    def test_05_zero_margin_and_no_edge_margin_nan_immunity(self) -> None:
        """Verify `rind_zero_margin` (`Tier-G0`) & `rind_no_edge_margin` (`Tier-G1`) structural isolation."""
        t_steps, n_pts, k_cands = 8, 4, 3
        base_tokens = torch.randn(t_steps, n_pts, k_cands, 37, requires_grad=True)
        cmask = torch.ones(t_steps, n_pts, k_cands, dtype=torch.bool)
        local_f = torch.randn(n_pts, 16)
        suns = torch.tensor([[0.5, 0.7, -0.5]] * t_steps, dtype=torch.float32)
        suns = suns / torch.linalg.norm(suns, dim=-1, keepdim=True)
        sdm = torch.ones(t_steps - 1, dtype=torch.bool)

        g0 = build_model("rind_zero_margin", hidden_dim=32)
        g0.eval()
        out_a = g0(
            {
                "candidate_tokens": base_tokens,
                "candidate_mask": cmask,
                "local_features": local_f,
                "sun_dirs": suns,
                "same_day_adjacent_mask": sdm,
            }
        )
        out_a["logits"].sum().backward()
        grad = base_tokens.grad.detach()
        self.assertEqual(float(grad[..., 22:25].abs().max().item()), 0.0)
        self.assertEqual(float(grad[..., 0:10].abs().max().item()), 0.0)
        self.assertEqual(float(grad[..., 28:37].abs().max().item()), 0.0)
        self.assertGreater(float(grad[..., 13:22].abs().max().item()), 0.0)

        corrupted = base_tokens.detach().clone()
        corrupted[..., 22] = 1e9
        corrupted[..., 23] = -1e9
        corrupted[..., 24] = float("nan")
        nan_local = torch.full_like(local_f, float("nan"))
        with torch.no_grad():
            out_b = g0(
                {
                    "candidate_tokens": corrupted,
                    "candidate_mask": cmask,
                    "local_features": nan_local,
                    "sun_dirs": suns,
                    "same_day_adjacent_mask": sdm,
                }
            )
        self.assertEqual(out_b["nonfinite_count"], 0)
        self.assertTrue(torch.allclose(out_a["logits"].detach(), out_b["logits"], atol=0.0, rtol=0.0))

    def test_06_moller_trumbore_astra_counterexample_and_signed_inverse_design(self) -> None:
        """Verify `t = -1 < 0` counterexample and building-ID consistent signed inverse design."""
        origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        direction = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        counter_tri = np.array(
            [[[-1.0, -2.0, -1.0], [1.0, -2.0, -1.0], [0.0, 1.0, 2.0]]], dtype=np.float64
        )
        hit, t_mat = moller_trumbore_intersect_f64(origin, direction, counter_tri)
        self.assertFalse(bool(hit[0, 0]))
        self.assertTrue(math.isinf(float(t_mat[0, 0])))
        occ = exact_moller_trumbore_occluded_f64(origin, direction, counter_tri)
        self.assertFalse(bool(occ[0]))

        forward_tri = counter_tri.copy()
        forward_tri[:, :, 1] += 3.0
        hit_f, t_f = moller_trumbore_intersect_f64(origin, direction, forward_tri)
        self.assertTrue(bool(hit_f[0, 0]))
        self.assertAlmostEqual(float(t_f[0, 0]), 2.0, places=9)

        b_south = _extruded_box_triangles(0.0, -15.0, 8.0, 4.0, 0.0, 22.0)
        b_north = _extruded_box_triangles(0.0, 20.0, 6.0, 4.0, 0.0, 10.0)
        scene = np.concatenate([b_south, b_north], axis=0)
        b_ids = assign_building_ids_from_triangles(scene)
        self.assertEqual(int(b_ids.max()) + 1, 2)

        rec = np.array([[-3.0, 0.05, 0.0], [0.0, 0.05, 0.0], [3.0, 0.05, 0.0]], dtype=np.float64)
        az = np.radians(np.linspace(110.0, 250.0, 25))
        alt = np.radians(12.0 + 38.0 * np.sin(np.linspace(0.0, math.pi, 25)))
        suns = np.stack([np.cos(alt) * np.sin(az), np.sin(alt), np.cos(alt) * np.cos(az)], axis=-1)
        regimes = [
            classify_solar_regime(float(math.degrees(a)), float(math.degrees(z)))
            for a, z in zip(alt, az)
        ]

        res = optimize_fullday_building_heights(
            rec, suns, scene, building_ids=b_ids, solar_regimes=regimes, max_delta_h_m=15.0, steps=25
        )
        self.assertTrue(res.building_consistency_verified)
        self.assertGreater(res.signed_net_sunlight_delta_fraction, 0.0)

        raised = perturb_buildings_consistent_f64(scene, b_ids, np.array([+15.0, 0.0]))
        s_init = exact_moller_trumbore_trajectory_sunlit_f64(rec, suns, scene).mean()
        s_raised = exact_moller_trumbore_trajectory_sunlit_f64(rec, suns, raised).mean()
        self.assertLessEqual(float(s_raised - s_init), 0.0)

    def test_07_canonical37_torch_numpy_parity_and_no_cross_day_bridging(self) -> None:
        """Verify Torch `canonical37` parity with NumPy and strict `dt == 120s` same-day masking."""
        rng = np.random.default_rng(42)
        rel_tri = rng.normal(0.0, 35.0, size=(6, 5, 4, 3, 3)).astype(np.float32)
        opaque = np.ones((6, 5, 4), dtype=bool)
        cmask = rng.random((6, 5, 4)) > 0.2
        suns = rng.normal(0.0, 1.0, size=(6, 3)).astype(np.float32)
        suns[:, 1] = np.abs(suns[:, 1]) + 0.15
        suns /= np.linalg.norm(suns, axis=-1, keepdims=True)

        tok_np = build_canonical37_numpy(rel_tri, opaque, cmask, suns)
        tok_th = build_canonical37_torch(
            torch.from_numpy(rel_tri),
            torch.from_numpy(opaque),
            torch.from_numpy(cmask),
            torch.from_numpy(suns),
        ).numpy()
        self.assertLess(float(np.max(np.abs(tok_np - tok_th))), 1e-5)

        truth = np.array([[1, 1], [1, 1], [0, 0], [0, 0]], dtype=np.int8)
        truth[0, 1] = -1
        valid = truth >= 0
        logits = np.array([[5.0, 5.0], [5.0, 5.0], [-5.0, -5.0], [-5.0, -5.0]], dtype=np.float32)
        day_ids = np.array([0, 0, 13, 13], dtype=np.int32)
        ts = np.array([1000, 1120, 8000000, 8000120], dtype=np.int64)
        regs = ["sunrise_low", "solar_noon", "afternoon_west", "sunset_low"]

        overall, _ = evaluate_trajectory_predictions(
            logits, truth, valid, solar_regimes=regs, day_ids=day_ids, timestamps_unix_s=ts
        )
        summary = overall.summarize()
        self.assertEqual(summary["point_frames"], 7)
        self.assertEqual(summary["transition_pairs"], 3)
        self.assertEqual(summary["transition_confusion"]["tp"], 0)
        self.assertEqual(summary["transition_confusion"]["fn"], 0)
        self.assertEqual(summary["transition_confusion"]["tn"], 3)

    def test_08_end_to_end_synthetic_fixture_and_dev_cli_suite(self) -> None:
        """Exercise 18,808-frame reader, mmap shard parity, formal guard, and `run-dev-suite` CLI."""
        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fixture", grid_side=21, frames_per_day=20, num_fit_tiles=8
            )
            contract = SplitContract(sp_json, mf_json, enforce_pinned_split_sha256=False)
            reader = FullDayTileReader(contract)
            fit_t = contract.get_fit_tiles("fit_8")[0]

            annual_info = reader.inspect_tile_annual_summary(fit_t, decode_week_sample=[0, 26])
            self.assertEqual(annual_info["annual_weeks"], 52)
            self.assertEqual(annual_info["annual_frames"], 18808)
            self.assertEqual(annual_info["unique_sampling_dates"], 52)
            self.assertEqual(annual_info["same_day_adjacent_edges"], 18756)

            traj = reader.load_tile_fullday(
                fit_t, week_indices=CANONICAL_4SEASON_WEEK_INDICES, max_receivers=24, max_candidates=4
            )
            self.assertEqual(set(traj.solar_regimes), set(SOLAR_REGIME_ORDER))
            shard = ContiguousTileMmapShard.write_from_trajectory(
                root / "shard_test", traj, contract=contract
            )
            compact = materialize_gpu_tokens_from_azimuth_bank(shard.load_compact_batch())
            baseline = traj.materialize_canonical37_batch()
            self.assertLess(
                float((compact["candidate_tokens"] - baseline["candidate_tokens"]).abs().max().item()),
                1e-5,
            )

            work_dir = root / "dev_out"
            # Formal mode MUST reject synthetic/truncated run when `--diagnostic-mode` is absent
            with self.assertRaises(RuntimeError):
                rind_cli_main(
                    [
                        "run-dev-suite",
                        "--split-file",
                        str(sp_json),
                        "--manifest-file",
                        str(mf_json),
                        "--allow-synthetic-split",
                        "--work-dir",
                        str(work_dir),
                        "--fit-subset",
                        "fit_8",
                        "--temporal-protocol",
                        "canonical_4season",
                        "--max-tiles",
                        "2",
                        "--device",
                        "cpu",
                        "--quiet",
                    ]
                )

            rc = rind_cli_main(
                [
                    "run-dev-suite",
                    "--split-file",
                    str(sp_json),
                    "--manifest-file",
                    str(mf_json),
                    "--allow-synthetic-split",
                    "--diagnostic-mode",
                    "--work-dir",
                    str(work_dir),
                    "--fit-subset",
                    "fit_8",
                    "--temporal-protocol",
                    "canonical_4season",
                    "--seeds",
                    "20261111",
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
                    "--quiet",
                ]
            )
            self.assertEqual(rc, 0)
            diag_receipt = work_dir / "DIAGNOSTIC_DEV_SUITE_RECEIPT.json"
            self.assertTrue(diag_receipt.exists())


if __name__ == "__main__":
    unittest.main()

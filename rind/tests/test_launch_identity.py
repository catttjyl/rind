"""G2 Regression Module 1: Launch Identity, Provenance Firewalls & Blind-Test Lock.

Verifies that every identity/provenance defect is rejected BEFORE any weight update,
normalizer statistics write, or blind-test label access (`P0-2`, `P0-3`, `G2`).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import unittest

import torch

from rind.fast_data import ContiguousTileMmapShard, build_or_open_single_tile_shard
from rind.fast_train import fit_model_fullday
from rind.fullday_data import (
    CANONICAL_4SEASON_WEEK_INDICES,
    FullDayTileReader,
    create_synthetic_fullday_fixture,
)
from rind.models import build_model
from rind.split_contract import (
    EXPECTED_MANHATTAN_4445_SPLIT_SHA256,
    EXPECTED_SNAPSHOT_V1_CSV_SHA256,
    EXPECTED_SNAPSHOT_V1_MANIFEST_SHA256,
    SplitContract,
)

from rind.tests._tempdir import TemporaryTestDir


def _snapshot_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class TestLaunchIdentity(unittest.TestCase):
    def test_01_pinned_snapshot_v1_digests_and_exact_partition_counts(self) -> None:
        contract = SplitContract()
        report = contract.build_verification_report()
        self.assertEqual(contract.split_sha256, EXPECTED_MANHATTAN_4445_SPLIT_SHA256)
        self.assertEqual(contract.manifest_sha256, EXPECTED_SNAPSHOT_V1_MANIFEST_SHA256)
        self.assertTrue(report["pinned_split_sha256_matched"])
        self.assertTrue(report["pinned_snapshot_v1_manifest_sha256_matched"])

        csv_ver = report["catalog_csv_verification"]
        self.assertTrue(csv_ver["csv_found"])
        self.assertEqual(csv_ver["csv_sha256"], EXPECTED_SNAPSHOT_V1_CSV_SHA256)
        self.assertTrue(csv_ver["pinned_snapshot_v1_csv_sha256_matched"])
        self.assertTrue(csv_ver["manifest_csv_row_parity_verified"])
        self.assertEqual(csv_ver["discrepancies"], [])

        self.assertEqual(report["universe_tile_count"], 4445)
        self.assertEqual(len(set(contract.universe_spatial_groups.values())), 81)
        self.assertEqual(report["ready_annual_tile_count"], 110)
        counts = report["counts_by_canonical_split"]
        self.assertEqual(counts["fit"], 67)
        self.assertEqual(counts["student_holdout"], 1)
        self.assertEqual(counts["calibration"], 12)
        self.assertEqual(counts["val_spatial_clean"], 14)
        self.assertEqual(counts["test_spatial_unexposed"], 0)
        self.assertEqual(counts["test_spatial_legacy_reassigned"], 16)
        self.assertEqual(contract.blind_open_count, 0)

    def test_02_mispointed_fit_path_carrying_val_payload_rejected_before_updates(self) -> None:
        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fx", grid_side=17, frames_per_day=12, num_fit_tiles=8
            )
            mf_raw = json.loads(mf_json.read_text(encoding="utf-8"))
            fit_entry = next(t for t in mf_raw["tiles"] if t["canonical_role"] == "fit")
            val_entry = next(t for t in mf_raw["tiles"] if t["canonical_role"] == "val_spatial_clean")
            # Point fit_entry's directory & reader_entry to val_entry's on-disk payload while keeping fit's expected SHA
            fit_entry["win_data_dir"] = val_entry["win_data_dir"]
            fit_entry["wsl_data_dir"] = val_entry["wsl_data_dir"]
            fit_entry["win_reader_entry"] = val_entry["win_reader_entry"]
            fit_entry["wsl_reader_entry"] = val_entry["wsl_reader_entry"]
            tampered_mf = root / "tampered_mispointed_manifest.json"
            tampered_mf.write_text(json.dumps(mf_raw, indent=2), encoding="utf-8")

            contract = SplitContract(sp_json, tampered_mf, enforce_pinned_split_sha256=False)
            reader = FullDayTileReader(contract)
            fit_tid = fit_entry["tile_id"]
            token = contract.issue_fit_firewall_token([fit_tid], subset_name="fit_8")

            model = build_model("rind_full_geom", hidden_dim=24)
            before = _snapshot_params(model)
            self.assertEqual(int(model.token_normalizer.count.item()), 0)

            def _bad_epoch_iter(_ep: int):
                traj = reader.load_tile_fullday(
                    fit_tid, week_indices=CANONICAL_4SEASON_WEEK_INDICES, max_receivers=8, max_candidates=4
                )
                yield traj.materialize_canonical37_batch()

            with self.assertRaises((ValueError, RuntimeError)):
                fit_model_fullday(
                    model,
                    _bad_epoch_iter,
                    firewall_token=token,
                    expected_split_sha256=contract.split_sha256,
                    epochs=1,
                    device="cpu",
                )

            after = _snapshot_params(model)
            for k in before:
                self.assertTrue(torch.equal(before[k], after[k]), msg=f"Parameter {k} mutated before rejection!")
            self.assertEqual(int(model.token_normalizer.count.item()), 0)
            self.assertEqual(len(model.consumed_normalizer_tile_ids), 0)
            self.assertEqual(contract.blind_open_count, 0)

    def test_03_shard_identity_tampering_and_tmp_recovery_rejected_before_updates(self) -> None:
        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fx", grid_side=17, frames_per_day=12, num_fit_tiles=8
            )
            contract = SplitContract(sp_json, mf_json, enforce_pinned_split_sha256=False)
            reader = FullDayTileReader(contract)
            fit_a, fit_b = contract.get_fit_tiles("fit_8")[:2]
            weeks_a = list(CANONICAL_4SEASON_WEEK_INDICES)  # [0, 13, 25, 38]
            weeks_perm = [13, 0, 25, 38]

            traj_a = reader.load_tile_fullday(fit_a, week_indices=weeks_a, max_receivers=8, max_candidates=4)
            traj_b = reader.load_tile_fullday(fit_b, week_indices=weeks_a, max_receivers=8, max_candidates=4)

            shard_a_dir = root / "shards" / fit_a
            shard_b_dir = root / "shards" / fit_b
            ContiguousTileMmapShard.write_from_trajectory(shard_a_dir, traj_a, contract=contract)
            ContiguousTileMmapShard.write_from_trajectory(shard_b_dir, traj_b, contract=contract)

            model = build_model("rind_full_geom", hidden_dim=24)
            before = _snapshot_params(model)
            token = contract.issue_fit_firewall_token([fit_a, fit_b], subset_name="fit_8")

            # (a) Wrong-tile shard (`shard_b_dir` opened expecting `fit_a`)
            with self.assertRaises(RuntimeError):
                ContiguousTileMmapShard(
                    shard_b_dir,
                    expected_split_sha256=contract.split_sha256,
                    expected_manifest_sha256=contract.manifest_sha256,
                    expected_tile_id=fit_a,
                )

            # (b) Same length, different week schedule (`[13, 0, 25, 38]` vs `[0, 13, 25, 38]`)
            with self.assertRaises(RuntimeError):
                ContiguousTileMmapShard(
                    shard_a_dir,
                    expected_split_sha256=contract.split_sha256,
                    expected_manifest_sha256=contract.manifest_sha256,
                    expected_tile_id=fit_a,
                    expected_week_indices=weeks_perm,
                )

            # (c) Stale manifest SHA
            stale_dir = root / "shards_stale"
            shutil.copytree(shard_a_dir, stale_dir)
            p_json = stale_dir / "SHARD_PROVENANCE.json"
            prov_stale = json.loads(p_json.read_text(encoding="utf-8"))
            prov_stale["manifest_sha256"] = "0" * 64
            p_json.write_text(json.dumps(prov_stale), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                ContiguousTileMmapShard(
                    stale_dir,
                    expected_split_sha256=contract.split_sha256,
                    expected_manifest_sha256=contract.manifest_sha256,
                    expected_tile_id=fit_a,
                )

            # (d) Missing `array_sha256` entry
            missing_hash_dir = root / "shards_nohash"
            shutil.copytree(shard_a_dir, missing_hash_dir)
            p_nohash = missing_hash_dir / "SHARD_PROVENANCE.json"
            prov_nh = json.loads(p_nohash.read_text(encoding="utf-8"))
            del prov_nh["array_sha256"]["truth"]
            p_nohash.write_text(json.dumps(prov_nh), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                ContiguousTileMmapShard(
                    missing_hash_dir,
                    expected_split_sha256=contract.split_sha256,
                    expected_manifest_sha256=contract.manifest_sha256,
                    expected_tile_id=fit_a,
                )

            # (e) Corrupted payload byte in `.npy` array -> caught before training step!
            corrupt_dir = root / "shards_corrupt"
            shutil.copytree(shard_a_dir, corrupt_dir)
            truth_npy = corrupt_dir / "truth.npy"
            raw_b = bytearray(truth_npy.read_bytes())
            raw_b[-1] ^= 0x01
            truth_npy.write_bytes(bytes(raw_b))

            def _corrupt_iter(_ep: int):
                sh = ContiguousTileMmapShard(
                    corrupt_dir,
                    expected_split_sha256=contract.split_sha256,
                    expected_manifest_sha256=contract.manifest_sha256,
                    expected_tile_id=fit_a,
                )
                yield sh.load_compact_batch()

            with self.assertRaises(RuntimeError):
                fit_model_fullday(
                    model,
                    _corrupt_iter,
                    firewall_token=token,
                    expected_split_sha256=contract.split_sha256,
                    epochs=1,
                    device="cpu",
                )

            after = _snapshot_params(model)
            for k in before:
                self.assertTrue(torch.equal(before[k], after[k]))
            self.assertEqual(int(model.token_normalizer.count.item()), 0)

            # (f) Interrupted `.tmp_shard_*` concurrent directory recovery
            cache_root = root / "auto_cache"
            orphan_tmp = cache_root / f".tmp_shard_{fit_a}_99999_deadbeef"
            orphan_tmp.mkdir(parents=True, exist_ok=True)
            (orphan_tmp / "partial.npy").write_bytes(b"incomplete")
            recovered = build_or_open_single_tile_shard(
                cache_root,
                reader,
                fit_a,
                week_indices=weeks_a,
                max_receivers=8,
                max_candidates=4,
            )
            self.assertEqual(recovered.tile_id, fit_a)

    def test_04_tristate_exposure_and_legacy_lineage_and_blind_lock(self) -> None:
        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fx", grid_side=17, frames_per_day=12, num_fit_tiles=8
            )
            base_mf = json.loads(mf_json.read_text(encoding="utf-8"))

            # 1. Missing exposure metadata on test tile -> rejected (never defaults to unexposed)
            mf_missing = copy.deepcopy(base_mf)
            t0 = next(t for t in mf_missing["tiles"] if t["canonical_role"] == "test_spatial_unexposed")
            t0.pop("historical_exposure_status", None)
            t0.pop("has_historical_exposure", None)
            t0.pop("source_batch", None)
            p_missing = root / "mf_missing_exp.json"
            p_missing.write_text(json.dumps(mf_missing), encoding="utf-8")
            with self.assertRaises((ValueError, RuntimeError)):
                SplitContract(sp_json, p_missing, enforce_pinned_split_sha256=False)

            # 2. `unknown` exposure routed into `test_spatial_unexposed` -> rejected
            mf_unk = copy.deepcopy(base_mf)
            tu = next(t for t in mf_unk["tiles"] if t["canonical_role"] == "test_spatial_unexposed")
            tu["historical_exposure_status"] = "unknown"
            tu.pop("has_historical_exposure", None)
            p_unk = root / "mf_unk_exp.json"
            p_unk.write_text(json.dumps(mf_unk), encoding="utf-8")
            with self.assertRaises((ValueError, RuntimeError)):
                SplitContract(sp_json, p_unk, enforce_pinned_split_sha256=False)

            # 3. Conflicting `historical_exposure_status="unexposed"` + `has_historical_exposure=True` -> rejected
            mf_conf = copy.deepcopy(base_mf)
            tc = next(t for t in mf_conf["tiles"] if t["canonical_role"] == "test_spatial_unexposed")
            tc["historical_exposure_status"] = "unexposed"
            tc["has_historical_exposure"] = True
            p_conf = root / "mf_conf_exp.json"
            p_conf.write_text(json.dumps(mf_conf), encoding="utf-8")
            with self.assertRaises((ValueError, RuntimeError)):
                SplitContract(sp_json, p_conf, enforce_pinned_split_sha256=False)

            # 4. Legacy lineage firewall & blind-test lock on valid contract
            contract = SplitContract(sp_json, mf_json, enforce_pinned_split_sha256=False)
            self.assertEqual(contract.blind_open_count, 0)
            fit_t = contract.get_fit_tiles("fit_8")
            cal_t = contract.get_calibration_tiles()
            val_t = contract.get_eval_tiles("val_spatial_clean")

            # Empty fit_tile_ids on legacy test -> rejected
            with self.assertRaises(RuntimeError):
                contract.get_eval_tiles(
                    "test_spatial_legacy_reassigned",
                    model_provenance={
                        "split_contract_sha256": contract.split_sha256,
                        "manifest_sha256": contract.manifest_sha256,
                        "fit_tile_ids": [],
                        "calibration_tile_ids": cal_t,
                        "calibrated": True,
                        "trained_from_scratch_fit_only": True,
                    },
                )

            # Val tile contaminating fit_tile_ids -> rejected
            with self.assertRaises(RuntimeError):
                contract.get_eval_tiles(
                    "test_spatial_legacy_reassigned",
                    model_provenance={
                        "split_contract_sha256": contract.split_sha256,
                        "manifest_sha256": contract.manifest_sha256,
                        "fit_tile_ids": [fit_t[0], val_t[0]],
                        "calibration_tile_ids": cal_t,
                        "calibrated": True,
                        "trained_from_scratch_fit_only": True,
                    },
                )

            # Blind-test access without freeze manifest -> rejected & blind_open_count stays 0
            with self.assertRaises(RuntimeError):
                contract.get_eval_tiles("test_spatial_unexposed")
            self.assertEqual(contract.blind_open_count, 0)

            # Valid one-shot freeze manifest -> opens once (`blind_open_count == 1`), rejects second attempt!
            freeze_manifest = {
                "benchmark_contract": "MANHATTAN_4445_ANNUAL_18808_V1",
                "split_sha256": contract.split_sha256,
                "manifest_sha256": contract.manifest_sha256,
                "selection_rule": "fixed_final_epoch_calibrated",
                "evaluator_version": "2.0.0",
                "frozen_checkpoints": [{"model_type": "rind_full_geom", "checkpoint_sha256": "a" * 64}],
            }
            blind_tiles = contract.get_eval_tiles(
                "test_spatial_unexposed",
                unseal_freeze_manifest=freeze_manifest,
                model_provenance={
                    "split_contract_sha256": contract.split_sha256,
                    "manifest_sha256": contract.manifest_sha256,
                    "fit_tile_ids": fit_t,
                    "calibration_tile_ids": cal_t,
                    "calibrated": True,
                    "trained_from_scratch_fit_only": True,
                },
            )
            self.assertGreaterEqual(len(blind_tiles), 1)
            self.assertEqual(contract.blind_open_count, 1)

            with self.assertRaises(RuntimeError):
                contract.get_eval_tiles(
                    "test_spatial_unexposed",
                    unseal_freeze_manifest=freeze_manifest,
                )
            self.assertEqual(contract.blind_open_count, 1)


if __name__ == "__main__":
    unittest.main()

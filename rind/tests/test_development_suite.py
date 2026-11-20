"""G5 Regression Module 1: Development-Only Suite, Pre-GPU Plan, Val Ledger & Publication Guards.

Covers Astra `P0-1`, `P0-3`, `P1-10`, and `G5`:
1. Formal CLI commands (`train`, `calibrate`, `eval`, `run-dev-suite`) MUST refuse `canonical_4season`,
   `--allow-synthetic-split`, and `--max-tiles > 0` unless `--diagnostic-mode` is explicitly passed.
2. `run-dev-suite` writes `00_pre_gpu_execution_plan.json` (split counts, weeks, receiver policy,
   5 configurations, seeds, `selection_rule`, and estimated updates), keeps `blind_open_count == 0`
   (`blind_test_status.result == "NA"`), logs `val_exposure_ledger.json`, emits frozen checkpoint
   SHAs in `BLIND_UNSEAL_FREEZE_MANIFEST_TEMPLATE.json`, and stamps `DIAGNOSTIC_NON_PUBLICATION`
   on synthetic/truncated diagnostic runs.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from rind.cli import main as rind_cli_main
from rind.fullday_data import create_synthetic_fullday_fixture
from rind.models import MODEL_TYPES

from rind.tests._tempdir import TemporaryTestDir


class TestDevelopmentSuite(unittest.TestCase):
    def test_01_formal_mode_rejects_seasonal_4day_synthetic_and_max_tiles(self) -> None:
        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fx", grid_side=15, frames_per_day=10, num_fit_tiles=8
            )
            # 1. Formal `train` rejects `canonical_4season` on real contract without `--diagnostic-mode`
            with self.assertRaises(RuntimeError):
                rind_cli_main(
                    [
                        "train",
                        "--cache-dir",
                        str(root / "c1"),
                        "--model-type",
                        "rind_full_geom",
                        "--temporal-protocol",
                        "canonical_4season",
                        "--checkpoint-out",
                        str(root / "m1.pt"),
                        "--quiet",
                    ]
                )

            # 2. Formal `train` rejects `--max-tiles 1` without `--diagnostic-mode`
            with self.assertRaises(RuntimeError):
                rind_cli_main(
                    [
                        "train",
                        "--cache-dir",
                        str(root / "c2"),
                        "--model-type",
                        "rind_full_geom",
                        "--max-tiles",
                        "1",
                        "--checkpoint-out",
                        str(root / "m2.pt"),
                        "--quiet",
                    ]
                )

            # 3. Formal `run-dev-suite` rejects `--allow-synthetic-split` without `--diagnostic-mode`
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
                        str(root / "w_bad"),
                        "--quiet",
                    ]
                )

    def test_02_end_to_end_dev_suite_plan_blind_lock_ledger_and_multiseed_freeze_manifest(self) -> None:
        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fx", grid_side=17, frames_per_day=12, num_fit_tiles=8
            )
            work_dir = root / "dev_suite_run"
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
                    "20261111,20261112",
                    "--selection-rule",
                    "fixed_final_epoch",
                    "--hidden-dim",
                    "24",
                    "--epochs",
                    "1",
                    "--max-tiles",
                    "2",
                    "--max-receivers",
                    "12",
                    "--max-candidates",
                    "4",
                    "--device",
                    "cpu",
                    "--quiet",
                ]
            )
            self.assertEqual(rc, 0)

            plan_file = work_dir / "00_pre_gpu_execution_plan.json"
            self.assertTrue(plan_file.exists())
            plan = json.loads(plan_file.read_text(encoding="utf-8"))
            self.assertEqual(plan["receipt_class"], "DIAGNOSTIC_NON_PUBLICATION")
            self.assertEqual(plan["model_configurations"], list(MODEL_TYPES))
            self.assertEqual(plan["seeds"], [20261111, 20261112])
            self.assertEqual(plan["selection_rule"], "fixed_final_epoch")
            self.assertGreater(plan["estimated_updates_per_run"], 0)

            # Formal publication receipt MUST NOT exist when diagnostic-mode was used
            self.assertFalse((work_dir / "DEV_SUITE_EXPERIMENT_RECEIPT_V2.json").exists())
            diag_file = work_dir / "DIAGNOSTIC_DEV_SUITE_RECEIPT.json"
            self.assertTrue(diag_file.exists())
            diag = json.loads(diag_file.read_text(encoding="utf-8"))
            self.assertEqual(diag["receipt_class"], "DIAGNOSTIC_NON_PUBLICATION")
            self.assertEqual(diag["blind_open_count"], 0)
            self.assertEqual(diag["blind_test_status"]["result"], "NA")
            self.assertEqual(len(diag["model_runs"]), 10)  # 5 configs * 2 seeds

            # Verify val exposure ledger recorded all 10 validation evaluations
            ledger_file = work_dir / "val_exposure_ledger.json"
            self.assertTrue(ledger_file.exists())
            ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
            self.assertEqual(len(ledger["entries"]), 10)

            # Verify freeze manifest template contains all 10 calibrated checkpoint SHAs
            freeze_file = work_dir / "BLIND_UNSEAL_FREEZE_MANIFEST_TEMPLATE.json"
            self.assertTrue(freeze_file.exists())
            freeze_doc = json.loads(freeze_file.read_text(encoding="utf-8"))
            self.assertEqual(freeze_doc["selection_rule"], "fixed_final_epoch")
            self.assertEqual(len(freeze_doc["frozen_checkpoints"]), 10)
            for entry in freeze_doc["frozen_checkpoints"]:
                self.assertEqual(len(entry["checkpoint_sha256"]), 64)

            # Verify multi-seed summary has n == 2 and paired comparisons exist against all 4 baselines/ablations
            for m_type in MODEL_TYPES:
                ms = diag["multiseed_val_summary"][m_type]["metrics"]["state_accuracy"]
                self.assertEqual(ms["n"], 2)
                self.assertIn("±", ms["formatted"])
            self.assertEqual(len(diag["paired_val_comparisons"]), 4)


if __name__ == "__main__":
    unittest.main()

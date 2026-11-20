"""G2 Regression Module 3: 24-Sector Azimuth Parity, Disabled-Slot Isolation & Subprocess Checkpoint Reload.

Covers Astra `P0-3`, `P0-4`, `P0-5`, and `G2`.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import torch

from rind.fast_data import (
    compute_canonical_azimuth_bins_torch,
    materialize_gpu_tokens_from_azimuth_bank,
)
from rind.fast_train import fit_model_fullday
from rind.fullday_data import (
    _extruded_box_triangles,
    build_canonical37_numpy,
    select_azimuth_sector_candidates,
    sun_dirs_to_azimuth_bins,
)
from rind.models import MODEL_TYPES, build_model, compute_head_parameter_norms, save_rind_checkpoint
from rind.split_contract import SplitContract

from rind.tests._tempdir import TemporaryTestDir


class TestFeatureAndCheckpointParity(unittest.TestCase):
    def test_01_all_24_sector_boundaries_and_sides_exact_candidate_and_feature_parity(self) -> None:
        boxes = []
        for idx in range(36):
            ang = (2.0 * math.pi * idx) / 36.0
            dist = 28.0 + 9.0 * (idx % 3)
            cx = dist * math.sin(ang)
            cz = dist * math.cos(ang)
            h_top = 14.0 + 4.0 * (idx % 5)
            boxes.append(_extruded_box_triangles(cx, cz, 4.0, 4.0, 0.0, h_top))
        tris = np.concatenate(boxes, axis=0)
        opaque = np.ones((tris.shape[0],), dtype=bool)
        sids = np.arange(1000, 1000 + tris.shape[0], dtype=np.uint64)

        receivers = np.array(
            [[-4.0, 0.05, -3.0], [0.0, 0.05, 0.0], [5.0, 0.05, 2.5], [1.5, 0.05, -6.0]],
            dtype=np.float32,
        )
        az_rel, az_op, az_mk, az_sids = select_azimuth_sector_candidates(
            receivers,
            tris,
            scene_opaque=opaque,
            scene_source_triangle_ids=sids,
            num_azimuth_bins=24,
            max_candidates=8,
            horizon_m=4000.0,
            return_source_ids=True,
        )

        sector_w = (2.0 * math.pi) / 24.0
        sun_list = []
        for b in range(24):
            base_ang = b * sector_w
            for eps in (-1e-7, 0.0, +1e-7):
                a = base_ang + eps
                alt = math.radians(18.0 + (b % 6) * 7.0)
                sun_list.append(
                    [math.cos(alt) * math.sin(a), math.sin(alt), math.cos(alt) * math.cos(a)]
                )
        suns_f64 = np.asarray(sun_list, dtype=np.float64)
        suns_f32 = suns_f64.astype(np.float32)

        bins_np = sun_dirs_to_azimuth_bins(suns_f32, num_azimuth_bins=24)
        bins_th = compute_canonical_azimuth_bins_torch(
            torch.from_numpy(suns_f32), num_azimuth_bins=24
        ).numpy()
        self.assertTrue(np.array_equal(bins_np, bins_th))

        rel_np = np.transpose(az_rel, (1, 0, 2, 3, 4))[bins_np]
        op_np = np.transpose(az_op, (1, 0, 2))[bins_np]
        mk_np = np.transpose(az_mk, (1, 0, 2))[bins_np]
        sid_np = np.transpose(az_sids, (1, 0, 2))[bins_np]
        tok_np = build_canonical37_numpy(rel_np, op_np, mk_np, suns_f32)

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for dev_str in devices:
            dev = torch.device(dev_str)
            batch = {
                "az_rel_triangles": torch.from_numpy(az_rel).to(dev),
                "az_opaque": torch.from_numpy(az_op).to(dev),
                "az_mask": torch.from_numpy(az_mk).to(dev),
                "az_source_ids": torch.from_numpy(az_sids.view(np.int64)).to(dev),
                "sun_dirs": torch.from_numpy(suns_f32).to(dev),
                "azimuth_bin_idx": torch.from_numpy(bins_np.astype(np.int64)).to(dev),
            }
            mat = materialize_gpu_tokens_from_azimuth_bank(batch)
            sid_th = mat["candidate_source_ids"].detach().cpu().numpy().view(np.uint64)
            mk_th = mat["candidate_mask"].detach().cpu().numpy()
            tok_th = mat["candidate_tokens"].detach().cpu().numpy()

            self.assertTrue(np.array_equal(mk_np, mk_th), msg=f"Mask mismatch on {dev_str}")
            self.assertTrue(
                np.array_equal(sid_np[mk_np], sid_th[mk_th]),
                msg=f"Candidate ID / order mismatch at sector boundary on {dev_str}",
            )
            max_slot_err = float(np.max(np.abs(tok_np - tok_th)))
            self.assertLess(max_slot_err, 1e-5, msg=f"Slot error {max_slot_err} >= 1e-5 on {dev_str}")

            batch_live = dict(batch)
            batch_live.pop("azimuth_bin_idx")
            mat_live = materialize_gpu_tokens_from_azimuth_bank(batch_live)
            sid_live = mat_live["candidate_source_ids"].detach().cpu().numpy().view(np.uint64)
            self.assertTrue(np.array_equal(sid_np[mk_np], sid_live[mk_np]))

    def test_02_disabled_slot_nan_inf_extreme_immunity_for_g0_and_g1(self) -> None:
        torch.manual_seed(42)
        t_steps, n_pts, k_cands = 10, 5, 4
        clean_tokens = torch.randn(t_steps, n_pts, k_cands, 37)
        cmask = torch.ones(t_steps, n_pts, k_cands, dtype=torch.bool)
        cmask[:, -1, -1] = False
        clean_lf = torch.randn(n_pts, 16)
        suns = torch.randn(t_steps, 3)
        suns[:, 1] = suns[:, 1].abs() + 0.15
        suns = suns / torch.linalg.norm(suns, dim=-1, keepdim=True)
        sdm = torch.ones(t_steps - 1, dtype=torch.bool)

        g0 = build_model("rind_zero_margin", hidden_dim=24)
        g0.eval()
        poison_g0 = clean_tokens.clone()
        poison_g0[..., 0:10] = float("nan")
        poison_g0[..., 22] = float("inf")
        poison_g0[..., 23] = float("-inf")
        poison_g0[..., 24] = 1e30
        poison_g0[..., 28:37] = -1e30
        poison_lf = torch.full_like(clean_lf, float("nan"))

        with torch.no_grad():
            ref_g0 = g0(
                {
                    "candidate_tokens": clean_tokens,
                    "candidate_mask": cmask,
                    "local_features": clean_lf,
                    "sun_dirs": suns,
                    "same_day_adjacent_mask": sdm,
                }
            )
            tst_g0 = g0(
                {
                    "candidate_tokens": poison_g0,
                    "candidate_mask": cmask,
                    "local_features": poison_lf,
                    "sun_dirs": suns,
                    "same_day_adjacent_mask": sdm,
                }
            )
        self.assertEqual(tst_g0["nonfinite_count"], 0)
        self.assertEqual(float((ref_g0["logits"] - tst_g0["logits"]).abs().max().item()), 0.0)

        g1 = build_model("rind_no_edge_margin", hidden_dim=24)
        g1.eval()
        poison_g1 = clean_tokens.clone()
        poison_g1[..., 22] = float("nan")
        poison_g1[..., 23] = float("inf")
        poison_g1[..., 24] = -1e30
        with torch.no_grad():
            ref_g1 = g1(
                {
                    "candidate_tokens": clean_tokens,
                    "candidate_mask": cmask,
                    "local_features": clean_lf,
                    "sun_dirs": suns,
                    "same_day_adjacent_mask": sdm,
                }
            )
            tst_g1 = g1(
                {
                    "candidate_tokens": poison_g1,
                    "candidate_mask": cmask,
                    "local_features": clean_lf,
                    "sun_dirs": suns,
                    "same_day_adjacent_mask": sdm,
                }
            )
        self.assertEqual(tst_g1["nonfinite_count"], 0)
        self.assertEqual(float((ref_g1["logits"] - tst_g1["logits"]).abs().max().item()), 0.0)

    def test_03_calibrated_checkpoint_reload_in_independent_subprocess_on_real_disk(self) -> None:
        contract = SplitContract()
        fit_t = contract.get_fit_tiles("fit_8")[:1]
        cal_t = contract.get_calibration_tiles()[:1]
        token = contract.issue_fit_firewall_token(fit_t, subset_name="fit_8")

        torch.manual_seed(20261111)
        t_steps, n_pts, k_cands = 10, 8, 4
        suns = torch.randn(t_steps, 3)
        suns[:, 1] = suns[:, 1].abs() + 0.2
        suns = suns / torch.linalg.norm(suns, dim=-1, keepdim=True)
        batch = {
            "tile_ids": fit_t,
            "candidate_tokens": torch.randn(t_steps, n_pts, k_cands, 37),
            "candidate_mask": torch.ones(t_steps, n_pts, k_cands, dtype=torch.bool),
            "local_features": torch.randn(n_pts, 16),
            "sun_dirs": suns,
            "same_day_adjacent_mask": torch.ones(t_steps - 1, dtype=torch.bool),
            "truth": torch.randint(0, 2, (t_steps, n_pts), dtype=torch.int8),
            "valid_mask": torch.ones(t_steps, n_pts, dtype=torch.bool),
            "receiver_crops": torch.randn(n_pts, 9, 21, 21),
        }

        pkg_root = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{pkg_root}{os.pathsep}{existing_pp}" if existing_pp else pkg_root

        child_code = (
            "import sys, torch\n"
            "from rind.models import compute_head_parameter_norms, load_rind_checkpoint\n"
            "ckpt_p, batch_p, out_p, sp_sha, mf_sha = sys.argv[1:6]\n"
            "model, prov = load_rind_checkpoint(ckpt_p, expected_split_sha256=sp_sha, expected_manifest_sha256=mf_sha, require_calibrated=True)\n"
            "model.eval()\n"
            "batch = torch.load(batch_p, map_location='cpu', weights_only=False)\n"
            "with torch.no_grad():\n"
            "    out = model(batch, use_calibrated_temperature=True)\n"
            "probs = out['probs']\n"
            "thr = float(model.calibrated_threshold.item())\n"
            "preds = (probs >= thr).to(torch.int8)\n"
            "norms = compute_head_parameter_norms(model)\n"
            "norm_stats = {}\n"
            "for nm in ('token_normalizer', 'local_normalizer', 'crop_normalizer'):\n"
            "    sub = getattr(model, nm, None)\n"
            "    if sub is not None:\n"
            "        norm_stats[nm] = {'count': float(sub.count.item()), 'mean_sum': float(sub.mean.sum().item()), 'var_sum': float(sub.var.sum().item())}\n"
            "torch.save({'probs': probs, 'preds': preds, 'logits': out['logits'], 'norms': norms, 'norm_stats': norm_stats, 'temp': float(model.calibrated_temperature.item()), 'thr': thr, 'prov': prov}, out_p)\n"
        )

        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            batch_path = root / "eval_batch.pt"
            torch.save(batch, batch_path)

            for m_type in MODEL_TYPES:
                kw = {"hidden_dim": 24, "crop_size": 21} if m_type == "local_convgru_grid" else {"hidden_dim": 24}
                model = build_model(m_type, **kw)
                fit_model_fullday(
                    model,
                    lambda _ep: [batch],
                    firewall_token=token,
                    expected_split_sha256=contract.split_sha256,
                    epochs=2,
                    lr=3e-3,
                    device="cpu",
                )
                model.set_calibration(temperature=1.37, decision_threshold=0.43)
                model.eval()
                with torch.no_grad():
                    ref_out = model(batch, use_calibrated_temperature=True)
                ref_probs = ref_out["probs"]
                ref_preds = (ref_probs >= 0.43).to(torch.int8)
                ref_norms = compute_head_parameter_norms(model)

                ckpt_path = root / f"calibrated_{m_type}.pt"
                out_path = root / f"child_out_{m_type}.pt"
                save_rind_checkpoint(
                    ckpt_path,
                    model,
                    split_contract_sha256=contract.split_sha256,
                    manifest_sha256=contract.manifest_sha256,
                    fit_subset_name="fit_8",
                    fit_tile_ids=fit_t,
                    calibrated=True,
                    calibration_tile_ids=cal_t,
                )

                proc = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-c",
                        child_code,
                        str(ckpt_path),
                        str(batch_path),
                        str(out_path),
                        contract.split_sha256,
                        contract.manifest_sha256,
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    proc.returncode,
                    0,
                    msg=f"Child reload failed for {m_type}: stderr={proc.stderr}",
                )
                child = torch.load(out_path, map_location="cpu", weights_only=False)
                self.assertAlmostEqual(child["temp"], 1.37, places=6)
                self.assertAlmostEqual(child["thr"], 0.43, places=6)
                for h_key, h_val in ref_norms.items():
                    self.assertAlmostEqual(child["norms"][h_key], h_val, places=6)
                for nm in ("token_normalizer", "local_normalizer", "crop_normalizer"):
                    sub = getattr(model, nm, None)
                    if sub is not None:
                        self.assertAlmostEqual(
                            child["norm_stats"][nm]["mean_sum"],
                            float(sub.mean.sum().item()),
                            places=6,
                        )
                        self.assertAlmostEqual(
                            child["norm_stats"][nm]["var_sum"],
                            float(sub.var.sum().item()),
                            places=6,
                        )
                self.assertTrue(torch.allclose(ref_probs, child["probs"], atol=0.0, rtol=0.0))
                self.assertTrue(torch.equal(ref_preds, child["preds"]))


if __name__ == "__main__":
    unittest.main()

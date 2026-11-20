"""G2 Regression Module 2: Annual Time-Axis Integrity, Intra-Day State Threading & Centered Crops.

Covers Astra `P0-1`, `P1-1`, `P1-3`, `P1-8`, and `G2`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import unittest
import zlib

import numpy as np
import torch

from rind.fast_data import ContiguousTileMmapShard, materialize_gpu_tokens_from_azimuth_bank
from rind.fast_train import compute_fullday_loss
from rind.fullday_data import (
    CANONICAL_4SEASON_WEEK_INDICES,
    FullDayTileReader,
    create_synthetic_fullday_fixture,
    decode_dsfch2_container_cpu,
    extract_exact_centered_receiver_crops,
    translate_storage_path,
)
from rind.metrics import build_same_day_adjacent_mask, evaluate_trajectory_predictions
from rind.models import build_model
from rind.split_contract import SplitContract, compute_file_sha256

from rind.tests._tempdir import TemporaryTestDir


def _pack_raw_dsfch2_unchecked(states_u8: np.ndarray, reasons_u16: np.ndarray) -> bytes:
    """Test helper packing raw uint8 codes (including illegal 0 or 4) to test decoder rejection."""
    f_cnt, h, w = states_u8.shape
    block_bytes = 65536
    fields_meta = []
    payload_parts = []
    for prefix, series, dt_s in (("states", states_u8, "uint8"), ("reasons", reasons_u16, "uint16")):
        for idx in range(f_cnt):
            cur_raw = np.ascontiguousarray(series[idx]).view(np.uint8).reshape(-1)
            n_b = int(cur_raw.size)
            bc = (n_b + block_bytes - 1) // block_bytes
            padded = np.zeros((bc * block_bytes,), dtype=np.uint8)
            padded[:n_b] = cur_raw
            blocks = padded.reshape(bc, block_bytes)
            lit_mask = np.any(blocks != 0, axis=1)
            lit_blocks = int(np.sum(lit_mask))
            payload_parts.append(np.packbits(lit_mask.astype(np.uint8), bitorder="big").tobytes())
            if lit_blocks > 0:
                payload_parts.append(blocks[lit_mask].tobytes())
            fields_meta.append(
                {
                    "name": f"{prefix}_{idx:03d}",
                    "dtype": dt_s,
                    "shape": [h * w],
                    "bytes": n_b,
                    "block_count": bc,
                    "literal_blocks": lit_blocks,
                    "encoding": "base",
                }
            )
    payload = b"".join(payload_parts)
    comp = zlib.compressobj(level=6, wbits=-zlib.MAX_WBITS)
    compressed = comp.compress(payload) + comp.flush()
    header = {
        "schema": "dsfch2.v2",
        "codec": {"algorithm": "Deflate", "container_compression": "single_raw_deflate_stream"},
        "payload_bytes": len(payload),
        "compressed_bytes": len(compressed),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "fields": fields_meta,
    }
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<8sQ", b"DSFCH2\x00\x00", len(hb)) + hb + compressed


class TestAnnualTimeAndReceivers(unittest.TestCase):
    def test_01_annual_52_dates_18808_frames_18756_edges_and_real_fit_if_present(self) -> None:
        auth_contract = SplitContract()
        real_fit_tid = auth_contract.get_fit_tiles("fit_8")[0]
        rec = auth_contract.tiles[real_fit_tid]
        probe_dir = translate_storage_path(rec.win_data_dir)
        if probe_dir.exists() or Path(rec.win_data_dir).exists():
            real_reader = FullDayTileReader(auth_contract)
            real_summary = real_reader.inspect_tile_annual_summary(
                real_fit_tid, decode_week_sample=[0, 26]
            )
            self.assertEqual(real_summary["annual_weeks"], 52)
            self.assertEqual(real_summary["unique_sampling_dates"], 52)
            self.assertEqual(real_summary["annual_frames"], 18808)
            self.assertEqual(real_summary["unique_window_timestamp_frames"], 18808)
            self.assertEqual(real_summary["same_day_adjacent_edges"], 18756)
            self.assertTrue(real_summary["all_5_solar_regimes_present"])

        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fx", grid_side=15, frames_per_day=16, num_fit_tiles=8
            )
            contract = SplitContract(sp_json, mf_json, enforce_pinned_split_sha256=False)
            reader = FullDayTileReader(contract)
            tid = contract.get_fit_tiles("fit_8")[0]
            summary = reader.inspect_tile_annual_summary(tid, decode_week_sample=[0, 13, 25, 38])
            self.assertEqual(summary["annual_weeks"], 52)
            self.assertEqual(summary["unique_sampling_dates"], 52)
            self.assertEqual(summary["annual_frames"], 18808)
            self.assertEqual(summary["unique_window_timestamp_frames"], 18808)
            self.assertEqual(summary["same_day_adjacent_edges"], 18756)
            self.assertGreater(summary["decoded_sample_verification"]["unknown_masked_slots"], 0)

    def test_02_cadence_anomalies_and_state_domain_rejections(self) -> None:
        day_ids = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int32)
        ts_anom = np.array([1000, 1120, 1239, 1360, 1600, 1600, 90000, 90120], dtype=np.int64)
        adj = build_same_day_adjacent_mask(day_ids, ts_anom)
        self.assertEqual(adj.tolist(), [True, False, False, False, False, False, True])

        with self.assertRaises(ValueError):
            build_same_day_adjacent_mask(day_ids, None)

        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            for bad_code in (0, 4):
                st_bad = np.ones((4, 8, 8), dtype=np.uint8) * 2
                st_bad[1, 0, 0] = bad_code
                rs_bad = np.zeros((4, 8, 8), dtype=np.uint16)
                bad_file = root / f"bad_state_{bad_code}.dsfch2"
                bad_file.write_bytes(_pack_raw_dsfch2_unchecked(st_bad, rs_bad))
                with self.assertRaises(ValueError):
                    decode_dsfch2_container_cpu(bad_file, expected_frames=4, grid_shape=(8, 8))

            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fx", grid_side=13, frames_per_day=12, num_fit_tiles=8
            )
            contract = SplitContract(sp_json, mf_json, enforce_pinned_split_sha256=False)
            tid = contract.get_fit_tiles("fit_8")[0]
            tile_dir = Path(contract.tiles[tid].win_data_dir)
            tile_mf_path = tile_dir / "MANIFEST.json"
            orig_mf_bytes = tile_mf_path.read_bytes()
            orig_doc = json.loads(orig_mf_bytes.decode("utf-8"))
            chunks_root = Path(orig_doc["object_roots"]["chunks"])
            cond_root = Path(orig_doc["object_roots"]["condition"])

            # Non-UNKNOWN state on ineligible pixel (0, 0) -> rejected!
            m_doc = json.loads(orig_mf_bytes.decode("utf-8"))
            r0 = m_doc["records"][0]
            ch0_path = chunks_root / r0["chunks"][0]["file"]
            orig_ch0_bytes = ch0_path.read_bytes()
            st_elig_viol = np.ones((int(r0["frames"]), 13, 13), dtype=np.uint8) * 2
            rs_elig_viol = np.zeros_like(st_elig_viol, dtype=np.uint16)
            viol_bytes = _pack_raw_dsfch2_unchecked(st_elig_viol, rs_elig_viol)
            ch0_path.write_bytes(viol_bytes)
            r0["chunks"][0]["sha256"] = hashlib.sha256(viol_bytes).hexdigest()
            tile_mf_path.write_text(json.dumps(m_doc), encoding="utf-8")
            with self.assertRaises((ValueError, RuntimeError)):
                FullDayTileReader(contract).load_tile_fullday(tid, week_indices=[0], max_receivers=8)
            ch0_path.write_bytes(orig_ch0_bytes)
            tile_mf_path.write_bytes(orig_mf_bytes)

            # Cadence jitter (119s / 121s / 240s / duplicate / negative) inside condition JSON -> rejected!
            m_doc = json.loads(orig_mf_bytes.decode("utf-8"))
            r0 = m_doc["records"][0]
            cond0_path = cond_root / r0["condition"]["file"]
            orig_cond_bytes = cond0_path.read_bytes()
            for delta_override in (119, 121, 240, 0, -120):
                c_doc = json.loads(orig_cond_bytes.decode("utf-8"))
                t_base = int(c_doc["frames"][1]["timestamp_unix_seconds"])
                c_doc["frames"][2]["timestamp_unix_seconds"] = t_base + delta_override
                cond0_path.write_text(json.dumps(c_doc), encoding="utf-8")
                r0["condition"]["sha256"] = compute_file_sha256(cond0_path)
                tile_mf_path.write_text(json.dumps(m_doc), encoding="utf-8")
                with self.assertRaises((ValueError, RuntimeError)):
                    FullDayTileReader(contract).load_tile_fullday(tid, week_indices=[0], max_receivers=8)
            cond0_path.write_bytes(orig_cond_bytes)
            tile_mf_path.write_bytes(orig_mf_bytes)

            # Missing chunk (51 records) & overlapping/duplicate window_id -> rejected!
            m_51 = json.loads(orig_mf_bytes.decode("utf-8"))
            m_51["records"] = m_51["records"][:51]
            tile_mf_path.write_text(json.dumps(m_51), encoding="utf-8")
            with self.assertRaises((ValueError, RuntimeError)):
                FullDayTileReader(contract).inspect_tile_annual_summary(tid)

            m_dup = json.loads(orig_mf_bytes.decode("utf-8"))
            m_dup["records"][1] = dict(m_dup["records"][0])
            tile_mf_path.write_text(json.dumps(m_dup), encoding="utf-8")
            with self.assertRaises((ValueError, RuntimeError)):
                FullDayTileReader(contract).inspect_tile_annual_summary(tid)
            tile_mf_path.write_bytes(orig_mf_bytes)

            # Overlapping frame sub-chunks inside a single day (`[0..8]` and `[6..12]`) -> rejected!
            m_ov = json.loads(orig_mf_bytes.decode("utf-8"))
            r0_ov = m_ov["records"][0]
            c0 = dict(r0_ov["chunks"][0])
            c1 = dict(r0_ov["chunks"][0])
            c0["start"], c0["stop"] = 0, 8
            c1["start"], c1["stop"] = 6, int(r0_ov["frames"])
            r0_ov["chunks"] = [c0, c1]
            tile_mf_path.write_text(json.dumps(m_ov), encoding="utf-8")
            with self.assertRaises((ValueError, RuntimeError)):
                FullDayTileReader(contract).load_tile_fullday(tid, week_indices=[0], max_receivers=8)
            tile_mf_path.write_bytes(orig_mf_bytes)

    def test_03_day0_perturbation_invariance_and_intraday_chunk_parity(self) -> None:
        torch.manual_seed(20261111)
        t_total, n_pts, k_cands = 16, 6, 4
        day_ids = np.array([0] * 8 + [1] * 8, dtype=np.int32)
        ts = np.array(
            [1000 + 120 * i for i in range(8)] + [90000 + 120 * i for i in range(8)],
            dtype=np.int64,
        )
        sdm = torch.from_numpy(build_same_day_adjacent_mask(day_ids, ts))
        suns = torch.randn(t_total, 3)
        suns[:, 1] = suns[:, 1].abs() + 0.2
        suns = suns / torch.linalg.norm(suns, dim=-1, keepdim=True)
        tokens_a = torch.randn(t_total, n_pts, k_cands, 37)
        tokens_b = tokens_a.clone()
        tokens_b[:8] = tokens_b[:8] * -7.5 + 12.0
        suns_b = suns.clone()
        suns_b[:8] = torch.tensor([0.1, 0.98, -0.17])
        suns_b[:8] = suns_b[:8] / torch.linalg.norm(suns_b[:8], dim=-1, keepdim=True)

        cmask = torch.ones(t_total, n_pts, k_cands, dtype=torch.bool)
        lf = torch.randn(n_pts, 16)
        crops = torch.randn(n_pts, 9, 21, 21)

        for m_type in ("rind_full_geom", "local_convgru_grid"):
            kw = {"hidden_dim": 24, "crop_size": 21} if m_type == "local_convgru_grid" else {"hidden_dim": 24}
            model = build_model(m_type, **kw)
            model.eval()
            with torch.no_grad():
                oa = model(
                    {
                        "candidate_tokens": tokens_a,
                        "candidate_mask": cmask,
                        "local_features": lf,
                        "sun_dirs": suns,
                        "same_day_adjacent_mask": sdm,
                        "receiver_crops": crops,
                    }
                )
                ob = model(
                    {
                        "candidate_tokens": tokens_b,
                        "candidate_mask": cmask,
                        "local_features": lf,
                        "sun_dirs": suns_b,
                        "same_day_adjacent_mask": sdm,
                        "receiver_crops": crops,
                    }
                )
            diff_day1 = float((oa["logits"][8:] - ob["logits"][8:]).abs().max().item())
            self.assertEqual(diff_day1, 0.0, msg=f"Cross-day leakage detected in {m_type}: {diff_day1}")

        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fx", grid_side=15, frames_per_day=16, num_fit_tiles=8
            )
            contract = SplitContract(sp_json, mf_json, enforce_pinned_split_sha256=False)
            reader = FullDayTileReader(contract)
            tid = contract.get_fit_tiles("fit_8")[0]
            traj = reader.load_tile_fullday(tid, week_indices=[0], max_receivers=6, max_candidates=4)

            traj.truth[:, :] = 0
            traj.truth[7:, 0] = 1
            traj.valid_mask[:, :] = True

            shard = ContiguousTileMmapShard.write_from_trajectory(
                root / "shard0", traj, contract=contract
            )
            full_batch = materialize_gpu_tokens_from_azimuth_bank(shard.load_compact_batch())

            for m_type in ("rind_full_geom", "local_convgru_grid"):
                kw = {"hidden_dim": 24, "crop_size": 21} if m_type == "local_convgru_grid" else {"hidden_dim": 24}
                model = build_model(m_type, **kw)
                model.eval()
                with torch.no_grad():
                    out_full = model(full_batch)
                    _, st_full = compute_fullday_loss(out_full, full_batch)

                    chunk_logits = []
                    prev_lat = None
                    prev_conv = None
                    prev_prob = None
                    prev_truth = None
                    prev_valid = None
                    total_trans_pairs = 0
                    for ch in shard.iter_day_batches(
                        days_per_batch=1,
                        receiver_chunk_size=6,
                        model_type=m_type,
                        crop_size=21,
                        max_intra_day_frames=7,
                    ):
                        ch = materialize_gpu_tokens_from_azimuth_bank(ch)
                        if ch.get("carry_state_from_prev_chunk"):
                            if prev_lat is not None:
                                ch["initial_latent_state"] = prev_lat
                            if prev_conv is not None:
                                ch["initial_conv_state"] = prev_conv
                            if prev_prob is not None:
                                ch["prev_chunk_last_prob"] = prev_prob
                                ch["prev_chunk_last_truth"] = prev_truth
                                ch["prev_chunk_last_valid"] = prev_valid
                        else:
                            prev_lat = None
                            prev_conv = None
                            prev_prob = None
                            prev_truth = None
                            prev_valid = None
                        out_ch = model(ch)
                        _, st_ch = compute_fullday_loss(out_ch, ch)
                        total_trans_pairs += int(st_ch["same_day_transition_pairs"])
                        chunk_logits.append(out_ch["logits"])
                        prev_lat = out_ch.get("final_latent_state")
                        prev_conv = out_ch.get("final_conv_state")
                        prev_prob = out_ch["probs"][-1]
                        prev_truth = ch["truth"][-1]
                        prev_valid = (ch["valid_mask"].to(torch.bool) & (ch["truth"] >= 0))[-1]

                    cat_logits = torch.cat(chunk_logits, dim=0)
                max_diff = float((out_full["logits"] - cat_logits).abs().max().item())
                self.assertLess(max_diff, 1e-6, msg=f"Intra-day chunk mismatch in {m_type}: {max_diff}")
                self.assertEqual(total_trans_pairs, int(st_full["same_day_transition_pairs"]))

                pred_perfect = np.where(traj.truth == 1, 8.0, -8.0).astype(np.float32)
                ov, _ = evaluate_trajectory_predictions(
                    pred_perfect,
                    traj.truth,
                    traj.valid_mask,
                    solar_regimes=traj.solar_regimes,
                    day_ids=traj.day_ids,
                    timestamps_unix_s=traj.timestamps_unix_s,
                )
                s = ov.summarize()
                self.assertEqual(s["transition_confusion"]["tp"], 1)
                self.assertEqual(s["transition_confusion"]["fp"], 0)
                self.assertEqual(s["transition_confusion"]["fn"], 0)
                self.assertEqual(s["directed_transitions"]["shadow_to_sun"]["tp"], 1)

    def test_04_exact_centered_receiver_crops_and_sampling_disclosure(self) -> None:
        h, w = 512, 512
        fields = np.zeros((8, h, w), dtype=np.float32)
        rr, cc = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        fields[0] = rr
        fields[1] = cc
        known = np.ones((h, w), dtype=bool)

        test_rc = np.array(
            [
                [0, 0],
                [0, 511],
                [511, 0],
                [511, 511],
                [0, 256],
                [256, 0],
                [17, 499],
                [256, 256],
            ],
            dtype=np.int64,
        )
        for crop_sz in (21, 64):
            mid = crop_sz // 2
            crops = extract_exact_centered_receiver_crops(fields, known, test_rc, crop_size=crop_sz)
            self.assertEqual(crops.shape, (len(test_rc), 9, crop_sz, crop_sz))
            for idx, (r_true, c_true) in enumerate(test_rc.tolist()):
                self.assertEqual(float(crops[idx, 0, mid, mid]), float(r_true))
                self.assertEqual(float(crops[idx, 1, mid, mid]), float(c_true))
                self.assertEqual(float(crops[idx, 8, mid, mid]), 1.0)
            self.assertEqual(float(crops[0, 8, 0, 0]), 0.0)
            self.assertEqual(float(crops[0, 0, 0, 0]), 0.0)
            self.assertEqual(float(crops[3, 8, crop_sz - 1, crop_sz - 1]), 0.0)

        with TemporaryTestDir() as tmp:
            root = Path(tmp)
            sp_json, mf_json = create_synthetic_fullday_fixture(
                root / "fx", grid_side=21, frames_per_day=12, num_fit_tiles=8
            )
            contract = SplitContract(sp_json, mf_json, enforce_pinned_split_sha256=False)
            reader = FullDayTileReader(contract)
            tid = contract.get_fit_tiles("fit_8")[0]
            t_samp = reader.load_tile_fullday(
                tid, week_indices=CANONICAL_4SEASON_WEEK_INDICES, receiver_mode="sampled", max_receivers=16
            )
            t_all = reader.load_tile_fullday(
                tid, week_indices=CANONICAL_4SEASON_WEEK_INDICES, receiver_mode="all_valid", max_receivers=0
            )
            d_samp = t_samp.receiver_sampling_disclosure()
            d_all = t_all.receiver_sampling_disclosure()
            self.assertEqual(d_samp["evaluated_receiver_count"], 16)
            self.assertEqual(d_samp["eligible_receiver_count"], d_all["eligible_receiver_count"])
            self.assertEqual(d_all["evaluated_receiver_count"], d_all["eligible_receiver_count"])
            self.assertEqual(len(d_samp["receiver_support_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

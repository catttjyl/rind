#!/usr/bin/env python3
"""Strict mathematical parity and blind-seal test suite for `SceneCell-az24` GPU/CPU router."""
from __future__ import annotations

import io
import math
import os
import sys
import tarfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
PORTABLE_PKG = Path(os.environ.get("RIND_PORTABLE_PKG", str(BUNDLE_ROOT)))
PHASE_ROOT = Path(os.environ.get("RIND_PHASE_ROOT", str(BUNDLE_ROOT)))
P3_DIR = Path(os.environ.get("RIND_P3_DIR", str(PHASE_ROOT)))
P4_ROOT = Path(os.environ.get("RIND_P4_ROOT", str(PHASE_ROOT)))
FROZEN_META = Path(os.environ.get("RIND_FROZEN_META", str(BUNDLE_ROOT / "meta")))
BUNDLE_DATA_DIR = Path(os.environ.get("RIND_BUNDLE_DIR", str(BUNDLE_ROOT / "data" / "bundles_gz")))
SCRIPT_DIR = Path(__file__).resolve().parent

for _p in (str(PORTABLE_PKG), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rind import fast_data, fullday_data
from rind.models import load_rind_checkpoint
from rind.split_contract import SplitContract
from rind_scene_cell_az24_gpu import (
    build_scene_cell_az24_index,
    gather_azimuth_bank_chunk,
    query_scene_cell_az24,
)


def _load_manhattan_tile(tile_id: str = "core_fi-0018-fj-0012"):
    tar_path = BUNDLE_DATA_DIR / f"{tile_id}.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(f"Missing {tar_path}; set RIND_BUNDLE_DIR to the downloaded tile bundle directory.")
    arrays = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            for key in (
                "scene_triangles_xyz",
                "scene_opaque",
                "scene_source_ids",
                "receivers_xyz",
                "receiver_source_triangle_ids",
            ):
                if m.name.endswith(f"/{key}.npy"):
                    arrays[key] = np.load(io.BytesIO(tf.extractfile(m).read()), allow_pickle=False)
    return arrays


class SceneCellAz24ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sc = SplitContract(
            split_path=FROZEN_META / "split.json",
            manifest_path=FROZEN_META / "manifest.json",
            blind_ledger_path=FROZEN_META / ".blind_unseal_ledger.jsonl",
        )
        assert sc.blind_open_count == 0, f"Blind seal violated: {sc.blind_open_count}"
        cls.tile_id = "core_fi-0018-fj-0012"
        cls.mesh = _load_manhattan_tile(cls.tile_id)
        cls.all_rec = cls.mesh["receivers_xyz"]
        cls.bmin = cls.all_rec.min(axis=0)
        cls.bmax = cls.all_rec.max(axis=0)
        cls.dev = "mps" if torch.backends.mps.is_available() else "cpu"

        t0 = time.perf_counter()
        cls.idx_gpu = build_scene_cell_az24_index(
            cls.mesh["scene_triangles_xyz"],
            cls.mesh["scene_opaque"],
            cls.mesh["scene_source_ids"],
            cls.bmin,
            cls.bmax,
            num_azimuth_bins=24,
            max_candidates=32,
            horizon_m=4000.0,
            grid_nx=8,
            grid_nz=8,
            device=cls.dev,
        )
        if cls.dev == "mps":
            torch.mps.synchronize()
        cls.build_ms_gpu = (time.perf_counter() - t0) * 1000.0

    def test_01_superset_lossless_across_all_densities(self) -> None:
        surv_sids = self.idx_gpu.active_source_ids.cpu().numpy().astype(np.uint64)
        for tag, rel_p in (
            ("N256", P4_ROOT / "cache" / "az24_r256" / f"{self.tile_id}.npz"),
            ("M1024", P4_ROOT / "cache" / "val_dense_m1024" / f"{self.tile_id}.npz"),
            ("M4096", P4_ROOT / "cache" / "val_ultra_m4096" / f"{self.tile_id}.npz"),
        ):
            with np.load(rel_p) as z:
                ref_sids = z["az_source_ids"][z["az_mask"]].astype(np.uint64)
            u_ref = np.unique(ref_sids)
            missed = np.setdiff1d(u_ref, surv_sids)
            print(
                f"[Density {tag}] active={self.idx_gpu.num_active_triangles} "
                f"storage={self.idx_gpu.serialized_storage_bytes() / (1024*1024):.2f} MB "
                f"unique_ref={len(u_ref)} missed={len(missed)}"
            )
            self.assertEqual(len(missed), 0, f"Missed {len(missed)} ref triangles on {tag}")

    def test_02_exact_slot_parity_and_mps_model_accuracy(self) -> None:
        # 1) Exact float64 query on CPU index built from exact scene vs cached reference N=256
        idx_cpu = build_scene_cell_az24_index(
            self.mesh["scene_triangles_xyz"],
            self.mesh["scene_opaque"],
            self.mesh["scene_source_ids"],
            self.bmin,
            self.bmax,
            num_azimuth_bins=24,
            max_candidates=32,
            horizon_m=4000.0,
            grid_nx=8,
            grid_nz=8,
            device="cpu",
        )
        with np.load(P4_ROOT / "cache" / "az24_r256" / f"{self.tile_id}.npz") as z:
            ref_rec = z["receivers_xyz"]
            ref_sids = z["az_source_ids"]
            ref_mask = z["az_mask"]
            td = {k: z[k] for k in z.files}

        # Exact-fp64 parity check using true float64 vertex coordinates for active triangles
        # Note: idx_cpu stores float32 centered coords; restore world float64 from original scene for exact fp64 comparison
        sid_to_row = {int(s): r for r, s in enumerate(self.mesh["scene_source_ids"])}
        act_rows = np.array([sid_to_row[int(s)] for s in idx_cpu.active_source_ids.numpy()], dtype=np.int64)
        tris_act_64 = self.mesh["scene_triangles_xyz"][act_rows].astype(np.float64) - idx_cpu.origin64[None, None, :]
        idx_cpu_64_compatible = idx_cpu
        # Patch active geometry in float64 on CPU via direct test of query_scene_cell_az24(exact_fp64=True)
        # First check slot match on standard float32->fp64 query
        cand_idx_cpu, cand_msk_cpu = query_scene_cell_az24(idx_cpu_64_compatible, ref_rec, exact_fp64=True)
        got_sids_cpu = np.where(
            cand_msk_cpu.numpy(),
            idx_cpu.active_source_ids.numpy()[np.maximum(cand_idx_cpu.numpy(), 0)],
            0,
        )
        ref_sids_masked = np.where(ref_mask, ref_sids, 0)
        slot_match_cpu = float(np.mean((got_sids_cpu == ref_sids_masked) & (cand_msk_cpu.numpy() == ref_mask)))

        # Also check exact match when using original uncentered tris_act_64 in exact_fp64 mode
        idx_cpu_exact = idx_cpu
        idx_cpu_exact.active_tris_rel = torch.from_numpy(tris_act_64)  # temporarily float64 for exact check
        idx_cpu_exact.active_cent_rel = idx_cpu_exact.active_tris_rel.mean(dim=1)
        idx_cpu_exact.active_ymax_rel = idx_cpu_exact.active_tris_rel[:, :, 1].amax(dim=1)
        idx_cpu_exact.active_rxz = torch.linalg.norm(
            idx_cpu_exact.active_tris_rel[:, :, [0, 2]] - idx_cpu_exact.active_cent_rel[:, None, [0, 2]], dim=-1
        ).amax(dim=-1)
        n64 = torch.cross(
            idx_cpu_exact.active_tris_rel[:, 1] - idx_cpu_exact.active_tris_rel[:, 0],
            idx_cpu_exact.active_tris_rel[:, 2] - idx_cpu_exact.active_tris_rel[:, 0],
            dim=-1,
        )
        idx_cpu_exact.active_normals = n64 / torch.linalg.norm(n64, dim=-1, keepdim=True).clamp_min(1e-20)
        cand_idx_64, cand_msk_64 = query_scene_cell_az24(idx_cpu_exact, ref_rec, exact_fp64=True)
        got_sids_64 = np.where(
            cand_msk_64.numpy(),
            idx_cpu_exact.active_source_ids.numpy()[np.maximum(cand_idx_64.numpy(), 0)],
            0,
        )
        exact_match = float(np.mean((got_sids_64 == ref_sids_masked) & (cand_msk_64.numpy() == ref_mask)))
        print(f"[Parity N256] float32->fp64 slot_match={slot_match_cpu*100:.6f}% | exact_fp64 slot_match={exact_match*100:.6f}%")
        self.assertGreaterEqual(slot_match_cpu, 0.999)
        self.assertEqual(exact_match, 1.0, f"Expected 100.0000% exact fp64 slot match, got {exact_match*100:.6f}%")

        # 2) End-to-end model accuracy parity: MPS SceneCell-az24 GPU router vs reference cached CPU IEEE FP32
        cand_idx_gpu, cand_msk_gpu = query_scene_cell_az24(self.idx_gpu, ref_rec, exact_fp64=False)
        rec_t_gpu = torch.from_numpy(ref_rec).to(self.idx_gpu.device)
        gathered_gpu = gather_azimuth_bank_chunk(self.idx_gpu, rec_t_gpu, cand_idx_gpu, cand_msk_gpu)
        if self.dev == "mps":
            torch.mps.synchronize()

        ckpt = P3_DIR / "results" / "out_p3_n4_S0_F" / "arm_F_anchor_hybrid" / "calibrated.pt"
        model_cpu, _ = load_rind_checkpoint(ckpt, device="cpu")
        model_cpu.eval()
        model_gpu, _ = load_rind_checkpoint(ckpt, device="cpu")
        model_gpu = model_gpu.float().to(self.idx_gpu.device)
        model_gpu.eval()

        # Run first full day (275 frames)
        day_ids = td["day_ids"]
        ts = td["timestamps_unix_s"]
        f1 = 1
        while f1 < len(day_ids) and day_ids[f1] == day_ids[0] and (ts[f1] - ts[f1 - 1]) == 120:
            f1 += 1
        truth = td["truth"][:f1] > 0.5
        vmask = td["valid_mask"][:f1].astype(bool)

        b_ref = {
            "receivers_xyz": torch.from_numpy(td["receivers_xyz"]),
            "local_features": torch.from_numpy(td["local_features"]),
            "az_rel_triangles": torch.from_numpy(td["az_rel_triangles"]),
            "az_opaque": torch.from_numpy(td["az_opaque"]),
            "az_mask": torch.from_numpy(td["az_mask"]),
            "az_source_ids": torch.from_numpy(td["az_source_ids"].astype(np.int64)),
            "azimuth_bin_idx": torch.from_numpy(td["azimuth_bin_idx"][:f1].astype(np.int64)),
            "sun_dirs": torch.from_numpy(td["sun_dirs"][:f1]),
            "same_day_adjacent_mask": torch.ones((f1 - 1,), dtype=torch.bool),
        }
        b_gpu = {
            "receivers_xyz": rec_t_gpu,
            "local_features": torch.from_numpy(td["local_features"]).to(self.idx_gpu.device),
            **gathered_gpu,
            "azimuth_bin_idx": torch.from_numpy(td["azimuth_bin_idx"][:f1].astype(np.int64)).to(self.idx_gpu.device),
            "sun_dirs": torch.from_numpy(td["sun_dirs"][:f1]).to(self.idx_gpu.device),
            "same_day_adjacent_mask": torch.ones((f1 - 1,), dtype=torch.bool, device=self.idx_gpu.device),
        }
        with torch.no_grad():
            p_ref = model_cpu(fast_data.materialize_gpu_tokens_from_azimuth_bank(b_ref), use_calibrated_temperature=True)["probs"].numpy()
            p_gpu = model_gpu(fast_data.materialize_gpu_tokens_from_azimuth_bank(b_gpu), use_calibrated_temperature=True)["probs"].cpu().numpy()

        acc_ref = float(np.mean((p_ref[vmask] >= 0.5) == truth[vmask])) * 100.0
        acc_gpu = float(np.mean((p_gpu[vmask] >= 0.5) == truth[vmask])) * 100.0
        max_p_diff = float(np.max(np.abs(p_ref - p_gpu)))
        print(f"[Model Parity Day0] acc_ref={acc_ref:.4f}% acc_gpu={acc_gpu:.4f}% delta={abs(acc_gpu-acc_ref):.4f} pp max_prob_diff={max_p_diff:.6f}")
        self.assertLess(abs(acc_gpu - acc_ref), 0.05)

    def test_03_sloped_terrain_self_exclusion_parity(self) -> None:
        # Synthetic sloped terrain mesh + 6 buildings where every receiver sits on an inclined terrain triangle
        rng = np.random.default_rng(20261117)
        gx = np.linspace(-50.0, 50.0, 11)
        gz = np.linspace(-50.0, 50.0, 11)
        tris = []
        sids = []
        slope = math.tan(math.radians(15.0))
        tid = 1
        rec_list = []
        own_list = []
        for ix in range(10):
            for iz in range(10):
                x0, x1 = gx[ix], gx[ix + 1]
                z0, z1 = gz[iz], gz[iz + 1]
                y00 = z0 * slope
                y10 = z0 * slope
                y01 = z1 * slope
                y11 = z1 * slope
                t1 = np.array([[x0, y00, z0], [x1, y10, z0], [x1, y11, z1]], dtype=np.float32)
                t2 = np.array([[x0, y00, z0], [x1, y11, z1], [x0, y01, z1]], dtype=np.float32)
                tris.extend([t1, t2])
                sids.extend([tid, tid + 1])
                if ix % 2 == 0 and iz % 2 == 0:
                    c1 = t1.mean(axis=0)
                    rec_list.append(c1)
                    own_list.append(tid)
                tid += 2
        # Add 6 tall building blocks on the slope
        for bx, bz, h in [(-35, 30, 45), (25, 35, 60), (-20, -30, 35), (30, -25, 50), (0, 80, 90), (-80, 10, 75)]:
            base_y = bz * slope
            for dx0, dx1, dz0, dz1 in [(-6, 6, -6, 6)]:
                p0 = [bx + dx0, base_y + h, bz + dz0]
                p1 = [bx + dx1, base_y + h, bz + dz0]
                p2 = [bx + dx1, base_y + h, bz + dz1]
                p3 = [bx + dx0, base_y + h, bz + dz1]
                b0 = [bx + dx0, base_y, bz + dz0]
                b1 = [bx + dx1, base_y, bz + dz0]
                tris.append(np.array([p0, p1, p2], dtype=np.float32))
                tris.append(np.array([p0, p2, p3], dtype=np.float32))
                tris.append(np.array([b0, b1, p1], dtype=np.float32))
                tris.append(np.array([b0, p1, p0], dtype=np.float32))
                sids.extend([tid, tid + 1, tid + 2, tid + 3])
                tid += 4
        # Add random extra tall triangles to trigger >32 competition
        for _ in range(120):
            cx, cz = rng.uniform(-300, 300, size=2)
            base_y = cz * slope
            h = rng.uniform(20, 120)
            tris.append(np.array([[cx - 5, base_y, cz], [cx + 5, base_y, cz], [cx, base_y + h, cz + 3]], dtype=np.float32))
            sids.append(tid)
            tid += 1

        tris_arr = np.stack(tris, axis=0)
        sop_arr = np.ones((len(tris_arr),), dtype=bool)
        sids_arr = np.array(sids, dtype=np.uint64)
        rec_arr = np.stack(rec_list, axis=0).astype(np.float32)
        own_arr = np.array(own_list, dtype=np.uint64)

        _, _, ref_mask, ref_sids = fullday_data.select_azimuth_sector_candidates(
            rec_arr,
            tris_arr,
            scene_opaque=sop_arr,
            scene_source_triangle_ids=sids_arr,
            exclude_source_triangle_ids=own_arr,
            num_azimuth_bins=24,
            max_candidates=32,
            horizon_m=4000.0,
            return_source_ids=True,
        )
        idx_slope = build_scene_cell_az24_index(
            tris_arr,
            sop_arr,
            sids_arr,
            rec_arr.min(axis=0),
            rec_arr.max(axis=0),
            num_azimuth_bins=24,
            max_candidates=32,
            horizon_m=4000.0,
            grid_nx=4,
            grid_nz=4,
            device="cpu",
        )
        sid_to_row = {int(s): r for r, s in enumerate(sids_arr)}
        act_rows = np.array([sid_to_row[int(s)] for s in idx_slope.active_source_ids.numpy()], dtype=np.int64)
        idx_slope.active_tris_rel = torch.from_numpy(tris_arr[act_rows].astype(np.float64) - idx_slope.origin64[None, None, :])
        idx_slope.active_cent_rel = idx_slope.active_tris_rel.mean(dim=1)
        idx_slope.active_ymax_rel = idx_slope.active_tris_rel[:, :, 1].amax(dim=1)
        idx_slope.active_rxz = torch.linalg.norm(
            idx_slope.active_tris_rel[:, :, [0, 2]] - idx_slope.active_cent_rel[:, None, [0, 2]], dim=-1
        ).amax(dim=-1)
        n64 = torch.cross(
            idx_slope.active_tris_rel[:, 1] - idx_slope.active_tris_rel[:, 0],
            idx_slope.active_tris_rel[:, 2] - idx_slope.active_tris_rel[:, 0],
            dim=-1,
        )
        idx_slope.active_normals = n64 / torch.linalg.norm(n64, dim=-1, keepdim=True).clamp_min(1e-20)
        c_idx, c_msk = query_scene_cell_az24(
            idx_slope,
            rec_arr,
            exclude_source_triangle_ids=own_arr,
            exact_fp64=True,
        )
        got_sids = np.where(c_msk.numpy(), idx_slope.active_source_ids.numpy()[np.maximum(c_idx.numpy(), 0)], 0)
        ref_sids_m = np.where(ref_mask, ref_sids, 0)
        match = float(np.mean((got_sids == ref_sids_m) & (c_msk.numpy() == ref_mask)))
        # Ensure own ground triangle never enters any receiver's candidates
        self_leaks = int(np.sum((got_sids == own_arr[:, None, None]) & c_msk.numpy()))
        print(f"[Sloped Terrain] exact_slot_match={match*100:.6f}% self_leaks={self_leaks}")
        self.assertEqual(self_leaks, 0)
        self.assertEqual(match, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

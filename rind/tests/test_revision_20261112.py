"""Regression cases for the November 12 independent revision."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rind.inverse_design import (
    assign_building_ids_from_triangles, build_layered_occlusion_benchmark_scene,
    compute_building_vertical_anchors, infer_unknown_building_heights_bayesian,
    perturb_buildings_consistent_f64,
)
from rind.label_budget import TileLabelDomain
from rind.objectives_jepa import compute_label_budget_mask


class TestRevision(unittest.TestCase):
    def test_prior_domain_independent_of_hidden_height(self):
        scene = build_layered_occlusion_benchmark_scene()
        tri = scene["triangles_xyz"]
        ids = assign_building_ids_from_triangles(
            tri, source_instance_ids=scene["source_instance_ids"], ground_mask=scene["ground_mask"]
        )
        _, heights, n = compute_building_vertical_anchors(tri, ids)
        for hidden in (20., 60.):
            delta = np.zeros(n)
            delta[0] = hidden - heights[0]
            result = infer_unknown_building_heights_bayesian(
                scene["receivers_xyz"][:4], scene["sun_directions_xyz"][:4],
                perturb_buildings_consistent_f64(tri, ids, delta),
                source_instance_ids=scene["source_instance_ids"], ground_mask=scene["ground_mask"],
                unknown_building_ids=[0], height_grid_min_m=2, height_grid_max_m=50,
                num_grid_hypotheses=5, initial_queries=1, active_rounds=0,
            )
            self.assertEqual(result["height_grid_m"], [2, 14, 26, 38, 50])
            self.assertEqual(result["active_query_curve"][0]["num_queries"], 0)
            self.assertEqual(result["prior_credible_95_width_m"], 48)
            self.assertEqual(result["active_query_curve"][0]["credible_95_coverage"], float(hidden <= 50))

    def test_exact_nested_label_ids_independent_of_partition(self):
        rc = torch.tensor([(i // 8, i % 8) for i in range(64)])
        domain = TileLabelDomain("tile", rc.flip([0]).numpy())
        previous = torch.zeros(64, dtype=torch.bool)
        for fraction in (.25, .5, .75):
            def select(coords):
                return compute_label_budget_mask(len(coords), receiver_rc=coords, tile_id="tile",
                                                 label_budget_fraction=fraction, domain=domain)
            expected = select(rc)
            self.assertEqual(expected.sum().item(), int(64 * fraction))
            self.assertTrue(torch.all(~previous | expected))
            for width in (1, 3, 8):
                self.assertTrue(torch.equal(expected, torch.cat([select(x) for x in rc.split(width)])))
            self.assertTrue(torch.equal(expected.flip([0]), select(rc.flip([0]))))
            previous = expected


    def test_ordinary_candidate_predictions_never_receive_oracle_building_ids(self):
        from rind import inverse_design as inv
        scene = build_layered_occlusion_benchmark_scene()
        with patch.object(inv, "_reconstruct_and_predict_scene_trajectory",
                          wraps=inv._reconstruct_and_predict_scene_trajectory) as calls:
            report = inv.evaluate_representation_sufficiency(
                scene["receivers_xyz"], scene["sun_directions_xyz"], scene["triangles_xyz"],
                source_instance_ids=scene["source_instance_ids"], ground_mask=scene["ground_mask"])
        ranked = [c.kwargs for c in calls.call_args_list if c.kwargs.get("first_k_candidates_only") == 1]
        self.assertGreaterEqual(len(ranked), 2)
        for kwargs in ranked:
            self.assertIsNone(kwargs.get("dominant_building_per_receiver"))
            self.assertIsNone(kwargs.get("building_ids_for_bottleneck"))
        rows = report["layered_occlusion_regime_transition"]
        for arm in ("ranked_candidate_k1", "oracle_first_building"):
            self.assertIn("post_driver_shift_accuracy_" + arm, rows)
        self.assertFalse(any("greedy_k1" in key for key in rows))

    def test_driver_loss_and_legacy_increment(self):
        from rind.objectives_jepa import transition_distance
        student = torch.tensor([[3., 4.]], requires_grad=True)
        teacher = student.detach().clone().requires_grad_()
        self.assertEqual(transition_distance(student, teacher, "driver_smooth_l2").item(), 25)
        for arm in ("driver_smooth", "unweighted_increment", "event_aware"):
            self.assertEqual(transition_distance(student, teacher, arm).item(), 0)
        transition_distance(student, teacher, "driver_smooth_l2").sum().backward()
        torch.testing.assert_close(student.grad, torch.tensor([[6., 8.]]))
        self.assertIsNone(teacher.grad)

    def test_even_crop_matches_reader_center_and_zero_padding(self):
        from rind.inverse_design import _build_spatial_crop_height_couplings
        roof = np.array([[[-.05,10,0],[.05,10,0],[.05,10,1]],
                         [[-.05,10,0],[.05,10,1],[-.05,10,1]]])
        base = roof.copy()
        base[:,:,1] = 0
        tri = np.concatenate([roof, base])
        for size in (3, 4, 64):
            weights = _build_spatial_crop_height_couplings(
                np.array([[0., 0., .5]]), tri, np.zeros(4, dtype=int), 1,
                crop_h=size, crop_w=size, meters_per_pixel=.1953125,
                grid_bounds_xz=(0, 1, .5, 1))
            self.assertEqual(weights[0,size//2,size//2,0], 1.)
            self.assertEqual(weights[0,size//2-1,size//2,0], 0.)
            reverse = _build_spatial_crop_height_couplings(
                np.array([[0., 0., .5]]), tri, np.zeros(4, dtype=int), 1,
                crop_h=size, crop_w=size, grid_step_xz=(.2, -.15),
                grid_bounds_xz=(0, 1, .5, 1))
            self.assertEqual(reverse[0,size//2+1,size//2,0], 0.)
            self.assertEqual(reverse[0,size//2-1,size//2,0], 1.)

    def test_noncontiguous_source_ids_exclude_before_candidate_cap(self):
        from rind.fullday_data import select_azimuth_sector_candidates
        triangle = np.array([[-1.,1.,5.],[1.,1.,5.],[0.,4.,5.]])
        faces = np.stack([triangle, triangle, triangle])
        ids = np.array([4000, 999, 31], dtype=np.uint64)
        for order in ([0,1,2],[2,0,1]):
            _, _, mask, chosen = select_azimuth_sector_candidates(
                np.zeros((1,3)), faces[order], scene_source_triangle_ids=ids[order],
                exclude_source_triangle_ids=np.array([31]), max_candidates=1, return_source_ids=True)
            self.assertTrue(mask.any())
            self.assertTrue(np.all(chosen[mask] == 999))

    def test_shard_teacher_carry_label_firewall_and_fixed_loss_windows(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from rind.fast_data import build_or_open_single_tile_shard, iter_objective_windows
        from rind.fast_train import LabelReadForbiddenBatchView
        from rind.fullday_data import FullDayTileReader, create_synthetic_fullday_fixture
        from rind.objectives_jepa import (apply_label_budget_view, compute_focal_event_hard_negative_loss,
            sketched_isotropic_gaussian_reg, unpack_response_5x5_features_torch)
        from rind.split_contract import SplitContract
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            split, manifest = create_synthetic_fullday_fixture(root/"fixture", grid_side=12, frames_per_day=10)
            contract = SplitContract(split, manifest, enforce_pinned_split_sha256=False)
            shard = build_or_open_single_tile_shard(root/"cache", FullDayTileReader(contract),
                contract.get_fit_tiles("fit_8")[0], week_indices=[0], receiver_mode="sampled", max_receivers=16)
            def read(width):
                return shard.iter_day_batches(days_per_batch=1, receiver_chunk_size=16,
                    model_type="pointwise_mlp", max_intra_day_frames=width)
            all_frames = next(read(10))
            blocks = list(read(2))
            b = next(block for block in blocks if block["carry_state_from_prev_chunk"])
            def descriptor(batch):
                t, n = batch["truth"].shape
                return unpack_response_5x5_features_torch(batch, bsz=1,t_steps=t,n_pts=n,device=torch.device("cpu"))
            start, end = b["frame_range"]
            torch.testing.assert_close(descriptor(b)[0,:,:,52], descriptor(all_frames)[0,start:end,:,52])
            # Force a known edge in the actual builder's carry fields.
            b["truth"] = torch.ones_like(b["truth"])
            b["valid_mask"] = torch.ones_like(b["valid_mask"])
            b["prev_frame_truth"] = torch.zeros_like(b["prev_frame_truth"])
            b["prev_frame_valid_mask"] = torch.ones_like(b["prev_frame_valid_mask"])
            self.assertTrue((descriptor(b)[0,0,:,52] == 1).all())
            legacy = dict(b)
            legacy["prev_frame_valid"] = legacy.pop("prev_frame_valid_mask")
            torch.testing.assert_close(descriptor(b), descriptor(legacy))
            masked, keep = apply_label_budget_view(legacy, label_budget_fraction=.25)
            self.assertTrue(torch.equal(masked["prev_frame_valid_mask"], keep))
            for key in ("prev_frame_valid", "prev_frame_valid_mask"):
                with self.assertRaises(RuntimeError):
                    LabelReadForbiddenBatchView(legacy)[key]
            with self.assertRaises(ValueError):
                descriptor(dict(legacy, prev_frame_valid_mask=~legacy["prev_frame_valid"]))
            self.assertTrue((descriptor(dict(b, prev_frame_same_day=False))[0,0,:,52] == 0).all())
            self.assertTrue((descriptor(dict(b, prev_frame_truth=-torch.ones_like(b["prev_frame_truth"])))[0,0,:,52] == 0).all())
            # Different read widths must yield the same differentiable statistics on the fixed window.
            def values(width):
                win = next(iter_objective_windows(read(width), 10))
                signal = (win["sun_dirs"][:, :1] + win["local_features"][:,0]).requires_grad_()
                hn = compute_focal_event_hard_negative_loss(signal.sigmoid(), None, win["truth"].clamp_min(0),
                        win["valid_mask"], win["same_day_adjacent_mask"])[0]
                z = torch.stack([signal, signal.square()], dim=-1)[win["valid_mask"]]
                reg = sketched_isotropic_gaussian_reg(z, step_seed=11)
                return hn, reg, torch.autograd.grad(hn + reg, signal)[0]
            reference = values(10)
            for width in (1, 2, 5):
                for expected, observed in zip(reference, values(width)):
                    torch.testing.assert_close(observed, expected, rtol=0, atol=0)

    def test_event_gate_init_full_jacobian_and_learnable_residual(self):
        from rind.models import FocalEventTransitionGate
        for shape in ((3,2,8),(1,3,2,8)):
            gate = FocalEventTransitionGate(8)
            z = torch.randn(shape, requires_grad=True)
            main = torch.nn.Linear(8,1)
            def direct(x):
                return main(x).squeeze(-1)
            def fused(x):
                return gate(x, direct(x), x[...,0], same_day_adjacent_mask=torch.tensor([True,False]))[0]
            self.assertTrue(torch.equal(fused(z), direct(z)))
            self.assertTrue(torch.equal(torch.autograd.functional.jacobian(fused,z),
                                        torch.autograd.functional.jacobian(direct,z)))
            logits = direct(z).detach().requires_grad_()
            def from_logits(l):
                return gate(z, l, z[...,0], same_day_adjacent_mask=torch.tensor([True,False]))[0]
            jac = torch.autograd.functional.jacobian(from_logits, logits).reshape(logits.numel(), -1)
            self.assertTrue(torch.equal(jac, torch.eye(logits.numel())))
            with torch.no_grad():
                gate.res_linear.bias.fill_(.1)
            out = from_logits(logits)
            self.assertFalse(torch.equal(out, logits))
            out.sum().backward()
            self.assertIsNone(z.grad)
            self.assertGreater(gate.res_linear.bias.grad.abs().sum().item(), 0)

    def test_unknown_inside_lead_span_rejects_both_endpoints(self):
        from rind.objectives_jepa import MultiHorizonLatentPredictor
        predictor = MultiHorizonLatentPredictor(8, leads=[2])
        z = torch.randn(1,3,2,8, requires_grad=True)
        v = torch.ones(1,3,2,dtype=torch.bool)
        v[:,1] = False
        args = dict(same_day_adjacent_mask=torch.ones(2,dtype=torch.bool),
                    mask_mode="random", precomputed_target_mask=torch.ones_like(v))
        loss, info = predictor.forward_multi_horizon(z,z.detach(),z[:,:,0], valid_mask=v, **args)
        self.assertEqual(info["jepa_active_leads"], 0)
        self.assertEqual(loss.item(), 0)
        _, info = predictor.forward_multi_horizon(z,z.detach(),z[:,:,0], valid_mask=torch.ones_like(v), **args)
        self.assertEqual(info["jepa_active_leads"], 1)

    def test_layered_timing_support_stops_at_first_invalid_or_day_boundary(self):
        from rind.inverse_design import _binary_f1_np, _layered_segments
        np_bool = lambda items: np.array(items, dtype=bool)
        runs = list(_layered_segments(np_bool([1,0,0,0,1,0,1]), np_bool([1,1,0,1,1,1,1]),
                                      np_bool([1,1,1,1,0,1])))
        self.assertEqual(runs, [(0,2),(4,5),(6,7)])
        self.assertIsNone(_binary_f1_np(np_bool([0]),np_bool([0])))
        self.assertEqual(_binary_f1_np(np_bool([1]),np_bool([0])),0.)

    def test_metric_contours_at_submeter_spacing(self):
        from rind.metrics import compute_shadow_contour_f1
        rc = np.indices((5, 10)).reshape(2,-1).T
        truth = (rc[:,1] >= 4)[None, :]
        for shift in (1, 2):
            pred = (rc[:,1] >= 4+shift)[None,:]
            args = (np.where(pred, 4., -4.), truth)
            axes = (rc, np.arange(10)*.1953125, -np.arange(5)*.1953125)
            out = compute_shadow_contour_f1(*args, np.ones_like(truth), *axes)
            self.assertEqual(out["true_contour_points"], 5)
            self.assertEqual(out["scores"]["one_pixel"]["f1"], float(shift == 1))
            self.assertEqual(out["scores"]["0.195m"]["f1"], 0.)
            self.assertEqual(out["scores"]["0.5m"]["f1"], 1.)
            empty = compute_shadow_contour_f1(*args, np.zeros_like(truth), *axes)
            self.assertIsNone(empty["scores"]["1.0m"]["f1"])

    def test_far_perspective_is_nested_and_trainable(self):
        from rind.models import build_model
        for model_type, kwargs in (("rind_coarse_aabb", {}), ("rind_full_geom", {"use_fourier102":False})):
            with self.assertRaises(ValueError):
                build_model(model_type, far_perspective=True, **kwargs)
        for tier in ("rind_zero_margin", "rind_no_edge_margin", "rind_full_geom"):
            torch.manual_seed(41)
            base = build_model(tier, hidden_dim=16)
            torch.manual_seed(41)
            augmented = build_model(tier, hidden_dim=16, far_perspective=True)
            for key, value in base.state_dict().items():
                torch.testing.assert_close(value, augmented.state_dict()[key], rtol=0, atol=0)
            tokens = torch.randn(2, 3, 37)
            tokens[...,25:28] = torch.tensor([0.,1.,0.])
            tokens[...,13:22:3] = 1.
            tokens[...,14:22:3] = 2.
            mask = torch.ones(2,3,dtype=torch.bool)
            old, _ = base.candidate_encoder(tokens, tokens, mask)
            new, _ = augmented.candidate_encoder(tokens, tokens, mask)
            torch.testing.assert_close(old,new,rtol=0,atol=0)
            new.sum().backward()
            self.assertGreater(augmented.candidate_encoder.angle_projection.grad.abs().sum().item(), 0.)

    def test_window_trainer_resume_and_stream_attention(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from rind.cli import _evaluate_single_tile_streaming
        from rind.fast_data import (build_or_open_single_tile_shard, iter_objective_windows,
                                    materialize_gpu_tokens_from_azimuth_bank)
        from rind.fast_train import fit_model_fullday
        from rind.fullday_data import FullDayTileReader, create_synthetic_fullday_fixture
        from rind.models import build_model
        from rind.objectives_jepa import ObjectiveConfig
        from rind.split_contract import SplitContract
        from itertools import islice
        from unittest.mock import Mock
        with TemporaryDirectory() as folder:
            root = Path(folder)
            split, manifest = create_synthetic_fullday_fixture(root/"fixture", grid_side=10, frames_per_day=10)
            contract = SplitContract(split, manifest, enforce_pinned_split_sha256=False)
            tile = contract.get_fit_tiles("fit_8")[0]
            token = contract.issue_fit_firewall_token([tile], subset_name="fit_8")
            shard = build_or_open_single_tile_shard(root/"cache", FullDayTileReader(contract),
                tile, week_indices=[0,1], receiver_mode="sampled", max_receivers=12)
            def read(width):
                return shard.iter_day_batches(days_per_batch=1, receiver_chunk_size=12,
                    model_type="rind_full_geom", shuffle_days=False, max_intra_day_frames=width)
            def train(width, **kwargs):
                torch.manual_seed(33)
                model = build_model("rind_full_geom", hidden_dim=16, far_perspective=True)
                model.update_normalizers(materialize_gpu_tokens_from_azimuth_bank(next(read(10))),
                    firewall_token=token, tile_ids=[tile], expected_split_sha256=contract.split_sha256)
                stats = fit_model_fullday(model, lambda epoch: islice(iter_objective_windows(read(width), 10), 4),
                    firewall_token=token, expected_split_sha256=contract.split_sha256, epochs=1,
                    seed=25, use_bf16_amp=False, objective_config=ObjectiveConfig(reg_mode="sigreg"),
                    prepare_gpu_tokens_fn=materialize_gpu_tokens_from_azimuth_bank,
                    batch_schedule={"objective_window_frames":10}, **kwargs)
                return model, stats["total_steps"]
            base, steps = train(5)
            self.assertEqual(steps, 4)
            checkpoint = root/"resume.pt"
            train(1, resume_state_out=checkpoint, max_train_steps=1)
            resumed, resumed_steps = train(2, resume_from=checkpoint)
            self.assertEqual(resumed_steps, steps)
            for name, value in base.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
            self.assertGreater(base.candidate_encoder.angle_projection.detach().abs().sum().item(), 0.)
            base.eval()
            _, _, _, receipt = _evaluate_single_tile_streaming(base, shard, torch.device("cpu"),
                guard=Mock(), days_per_batch=1, receiver_chunk_size=6, intra_day_frames=5,
                collect_logits_for_stratified=True, collect_attention=True)
            weights = receipt["collected_attention"]
            self.assertEqual(weights.shape[:2], receipt["collected_logits"].shape)
            self.assertTrue(np.isfinite(weights).all())

    def test_first_blocker_attention_retrieval_miss_stays_in_denominator(self):
        from rind.metrics import compute_blocker_distance_stratified_metrics
        tri = np.array([[[-1,10,-1],[1,10,-1],[0,10,1]]],float)
        out = compute_blocker_distance_stratified_metrics(
            np.ones((1,2)),np.zeros((1,2)),np.ones((1,2),bool),
            receivers_xyz=np.zeros((2,3)),sun_directions_xyz=np.array([[0,1.,0]]),
            scene_triangles_xyz=tri,scene_source_triangle_ids=np.array([41]),
            candidate_source_triangle_ids=np.array([[[41],[51]]]),
            candidate_attn_weights=np.full((1,2,1),.8),candidate_layout="time",max_ray_slots=1)
        near = out["fine_buckets"]["0m_to_25m"]
        self.assertEqual(near["attention_ray_support"], 2)
        self.assertEqual(near["candidate_recalled_ray_count"], 1)
        self.assertEqual(near["top1_attention_hit_rate"], .5)
        self.assertEqual(near["conditional_on_candidate_recalled"]["top1_attention_hit_rate"], 1.)
        self.assertAlmostEqual(near["mean_true_blocker_attention_mass"], .4)

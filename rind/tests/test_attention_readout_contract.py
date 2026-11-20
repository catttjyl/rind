"""CPU contracts for bounded attention and set-valued candidate readout."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest
import tempfile

import torch

from rind.fast_data import build_canonical37_torch
from rind.models import RindCandidateWorldModel, StaticGeometryMemory, load_rind_checkpoint, save_rind_checkpoint
from rind.tests._tempdir import TemporaryTestDir


def make_batch(frames=7):
    torch.manual_seed(731)
    sun = torch.tensor([[0.3, 0.8, 0.4]], dtype=torch.float32).expand(frames, -1).clone()
    sun = torch.nn.functional.normalize(sun, dim=-1)
    tri = torch.tensor([[[10., 0., 12.], [10., 14., 12.], [18., 0., 12.]],
                        [[30., 0., 20.], [30., 16., 20.], [38., 0., 20.]]])
    tri = tri[None, None].expand(frames, 3, -1, -1, -1).clone()
    mask = torch.ones(frames, 3, 2, dtype=torch.bool)
    return {"candidate_tokens": build_canonical37_torch(tri, mask, mask, sun),
            "candidate_mask": mask, "sun_dirs": sun,
            "local_features": torch.randn(3, 16),
            "same_day_adjacent_mask": torch.ones(frames - 1, dtype=torch.bool)}


def make_model(**kwargs):
    torch.manual_seed(812)
    return RindCandidateWorldModel(hidden_dim=16, num_heads=2, max_candidates=4,
                                   enable_spatial_knn_readout=False, use_static_mpnn=False, **kwargs).eval()


class TestAttentionReadoutContract(unittest.TestCase):
    def test_solar_beta_bound_smooth_and_legacy_identity(self):
        legacy = StaticGeometryMemory(16, 16)
        bounded = StaticGeometryMemory(16, 16, film_beta_bound=0.7)
        bounded.load_state_dict(legacy.state_dict(), strict=True)
        z = torch.randn(1, 3, 16)
        sun = torch.randn(1, 2, 16)
        self.assertTrue(torch.equal(legacy.modulate_solar_film(z, sun), bounded.modulate_solar_film(z, sun)))
        with torch.no_grad():
            legacy.solar_film[-1].bias[16:] = 25.
            bounded.solar_film[-1].bias[16:] = 25.
        self.assertEqual(float(legacy.modulate_solar_film(torch.zeros_like(z), sun).detach().max()), 25.)
        out = bounded.modulate_solar_film(torch.zeros_like(z), sun)
        self.assertLessEqual(float(out.detach().abs().max()), 0.700001)
        self.assertTrue(torch.isfinite(out).all())
        with self.assertRaises(ValueError):
            StaticGeometryMemory(16, 16, film_beta_bound=-1)

    def test_set_readout_permutation_duplicate_padding_and_empty(self):
        model = make_model(cand_readout="attn_plus_lse")
        with torch.no_grad():
            model.cand_face_score[-1].weight.fill_(0.3)
            model.cand_face_score[-1].bias.fill_(0.4)
        b = make_batch(3)
        baseline = model(b)
        tok, mask = b["candidate_tokens"], b["candidate_mask"]
        variants = [
            (tok.flip(-2), mask.flip(-1)),
            (torch.cat([tok, tok[..., :1, :]], -2), torch.cat([mask, mask[..., :1]], -1)),
            (torch.cat([tok, torch.randn_like(tok[..., :1, :]) * 1000], -2),
             torch.cat([mask, torch.zeros_like(mask[..., :1])], -1)),
        ]
        for nonfinite in (float("nan"), float("inf"), float("-inf")):
            variants.append((torch.cat([tok, torch.full_like(tok[..., :1, :], nonfinite)], -2),
                             torch.cat([mask, torch.zeros_like(mask[..., :1])], -1)))
        for tokens, valid in variants:
            out = model(dict(b, candidate_tokens=tokens, candidate_mask=valid))
            torch.testing.assert_close(out["logits"], baseline["logits"], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(out["latent_roll"], baseline["latent_roll"], rtol=1e-5, atol=1e-6)
        tokens = torch.full_like(tok.unsqueeze(0), float("nan")).requires_grad_()
        q = torch.randn(1, 3, 3, 32)
        _, _, residual = model._encode_candidate_cross_attention(tokens, torch.zeros_like(mask).unsqueeze(0), q,
                                                                  return_readout=True)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        residual.sum().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertEqual(float(tokens.grad.abs().max()), 0.)
        empty_batch = dict(b, candidate_tokens=tok[..., :0, :], candidate_mask=mask[..., :0])
        self.assertTrue(torch.isfinite(model(empty_batch)["logits"]).all())
        _, _, empty_residual = model._encode_candidate_cross_attention(
            tok[None, ..., :0, :], mask[None, ..., :0], q, return_readout=True)
        self.assertTrue(torch.equal(empty_residual, torch.zeros_like(empty_residual)))

    def test_readout_uses_sanitized_representation_only(self):
        model = make_model(ablation_mode="no_edge_margin", cand_readout="attn_plus_lse")
        with torch.no_grad():
            model.cand_face_score[-1].weight.fill_(0.3)
        batch = make_batch(3)
        changed = copy.deepcopy(batch)
        changed["candidate_tokens"][..., 22:25] += 5000
        self.assertTrue(torch.equal(model(batch)["logits"], model(changed)["logits"]))

    def test_gate_receives_complete_base_and_detaches_events(self):
        model = make_model(cand_readout="attn_plus_lse")
        with torch.no_grad():
            model.cand_face_score[-1].bias.fill_(0.6)
        captured = []
        handle = model.focal_event_gate.register_forward_pre_hook(lambda _m, args: captured.append(args))
        out = model(make_batch(3))
        handle.remove()
        self.assertTrue(torch.equal(out["logits"], out["logits_bce"]))
        torch.testing.assert_close(captured[0][1].squeeze(0), out["logits_bce"], rtol=0, atol=0)
        grad = torch.autograd.grad(out["logits"].sum(), captured[0][1], retain_graph=True)[0]
        self.assertTrue(torch.equal(grad, torch.ones_like(grad)))
        # Nonzero gate parameters still cannot backpropagate through event features.
        gate = model.focal_event_gate
        with torch.no_grad():
            gate.res_linear.weight.fill_(0.2)
        z = torch.randn(1, 3, 2, 16, requires_grad=True)
        base = torch.randn(1, 3, 2, requires_grad=True)
        boundary = torch.randn_like(base, requires_grad=True)
        fused, _ = gate(z, base, boundary)
        grads = torch.autograd.grad(fused.sum(), [base, z, boundary], allow_unused=True)
        self.assertTrue(torch.equal(grads[0], torch.ones_like(base)))
        self.assertIsNone(grads[1]); self.assertIsNone(grads[2])

    def test_qk_norm_keeps_bounded_ray_bias_observable_at_large_content_scale(self):
        model = make_model(attn_qk_norm=True, film_beta_bound=0.5)
        b = make_batch(2)
        q = torch.full((1, 2, 3, 32), 1e6)
        with torch.no_grad():
            model.cand_attn.in_proj_weight.mul_(1e5)
            model.ray_attn_bias.mlp[-1].bias.fill_(-100.)
        _, neg = model._encode_candidate_cross_attention(b["candidate_tokens"][None], b["candidate_mask"][None], q,
                                                          return_attn_weights=True)
        with torch.no_grad():
            model.ray_attn_bias.mlp[-1].bias.fill_(100.)
        _, pos = model._encode_candidate_cross_attention(b["candidate_tokens"][None], b["candidate_mask"][None], q,
                                                          return_attn_weights=True)
        self.assertTrue(torch.isfinite(pos).all())
        self.assertGreater(float((pos.sum(-1) - neg.sum(-1)).detach().min()), 0.3)

    def test_checkpoint_roundtrip_and_legacy_defaults(self):
        with TemporaryTestDir() as tmp:
            path = Path(tmp) / "attention.pt"
            for kwargs in ({}, {"attn_qk_norm": True, "film_beta_bound": 0.8, "cand_readout": "attn_plus_lse"}):
                model = make_model(**kwargs)
                save_rind_checkpoint(path, model, split_contract_sha256="s", manifest_sha256="m",
                                     fit_subset_name="fit", fit_tile_ids=["t"])
                loaded, _ = load_rind_checkpoint(path)
                self.assertEqual(loaded.get_model_config(), model.get_model_config())
                if not kwargs:
                    payload = torch.load(path, weights_only=False)
                    for key in ("attn_qk_norm", "film_beta_bound", "cand_readout"):
                        del payload["model_config"][key]
                    torch.save(payload, path)
                    legacy, _ = load_rind_checkpoint(path)
                    self.assertEqual(legacy.get_model_config(), model.get_model_config())
                else:
                    original = torch.load(path, weights_only=False)
                    for key, wrong in (("cand_readout", "attn_only"), ("attn_qk_norm", False), ("film_beta_bound", 0.0)):
                        payload = copy.deepcopy(original)
                        payload["model_config"][key] = wrong
                        torch.save(payload, path)
                        with self.assertRaisesRegex(RuntimeError, "model_config mismatch"):
                            load_rind_checkpoint(path)

    def test_lse_cached_across_chunks_including_no_new_anchor(self):
        model = make_model(cand_readout="attn_plus_lse", attn_qk_norm=True, film_beta_bound=0.6)
        with torch.no_grad():
            model.cand_face_score[-1].weight.fill_(0.2)
            model.cand_face_score[-1].bias.fill_(0.5)
        b = make_batch(9)
        whole = model.evaluate_strided_rollout(b, stride=4, use_calibrated_temperature=False)
        outputs = []
        prior = latent = prev_sun = None
        for start, end in ((0, 2), (2, 3), (3, 6), (6, 9)):
            sub = {k: v[start:end] if k in ("candidate_tokens", "candidate_mask", "sun_dirs") else v for k, v in b.items()}
            sub["same_day_adjacent_mask"] = torch.ones(end - start - 1, dtype=torch.bool)
            if prior is not None:
                sub.update(carry_state_from_prev_chunk=True, prev_frame_same_day=True, stride_anchor_state=prior,
                           initial_latent_state=latent, prev_frame_sun_dir=prev_sun)
            out = model.evaluate_strided_rollout(sub, stride=4, use_calibrated_temperature=False)
            outputs.append(out["logits"])
            prior, latent, prev_sun = out["stride_anchor_state"], out["final_latent_state"], b["sun_dirs"][end-1]
            self.assertIn("readout", prior)
        torch.testing.assert_close(torch.cat(outputs), whole["logits"], rtol=1e-5, atol=1e-6)
        # Stride 1 and full forward use exactly the same LSE base readout.
        stride1 = model.evaluate_strided_rollout(b, stride=1, use_calibrated_temperature=False)
        torch.testing.assert_close(stride1["logits"], model(b)["logits"], rtol=0, atol=0)

    def test_real_provider_compact_anchor_bank_matches_chunked_rollout(self):
        from rind.tests.test_bvh_provider_contract import SpyCandidateProvider
        from rind.fast_data import build_or_open_single_tile_shard
        from rind.fullday_data import create_synthetic_fullday_fixture, FullDayTileReader
        from rind.ray_candidates import BVHCandidateProvider
        from rind.split_contract import SplitContract
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, manifest = create_synthetic_fullday_fixture(root / "fixture", grid_side=10, frames_per_day=11)
            contract = SplitContract(split, manifest, enforce_pinned_split_sha256=False)
            shard = build_or_open_single_tile_shard(root / "cache", FullDayTileReader(contract),
                contract.get_fit_tiles("fit_8")[0], week_indices=[0], max_receivers=8, max_candidates=32)
            provider = BVHCandidateProvider(*[shard._mmap_slice_copy(key) for key in
                ("scene_triangles_xyz", "scene_opaque", "scene_source_ids")])
            model = make_model(cand_readout="attn_plus_lse", attn_qk_norm=True, film_beta_bound=0.7)
            with torch.no_grad():
                model.cand_face_score[-1].bias.fill_(0.4)
                model.cand_face_score[-1].weight.fill_(0.2)

            def run(width):
                spy = SpyCandidateProvider(provider)
                logits, active, anchors = [], [], 0
                prior = latent = None
                for batch in shard.iter_day_batches(days_per_batch=1, receiver_chunk_size=8,
                        max_intra_day_frames=width, include_labels=False,
                        candidate_stride=4, candidate_provider=spy):
                    if batch.get("carry_state_from_prev_chunk", False):
                        batch["stride_anchor_state"] = prior
                        batch["initial_latent_state"] = latent
                    out = model.evaluate_strided_rollout(batch, stride=4, use_calibrated_temperature=False)
                    prior, latent = out["stride_anchor_state"], out["final_latent_state"]
                    logits.append(out["logits"]); active.append(out["has_active_candidates"])
                    anchors += out["execution_counters"]["anchor_frames"]
                self.assertEqual(anchors, len(spy.suns))
                return torch.cat(logits), torch.cat(active)

            full, full_active = run(0)
            for width in (1, 3):
                chunked, chunked_active = run(width)
                torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-6)
                self.assertTrue(torch.equal(full_active, chunked_active))

    def test_geometry_edit_rejects_cached_anchor(self):
        model = make_model(cand_readout="attn_plus_lse")
        b = make_batch(2)
        b["candidate_geometry_identity"] = "geometry-v1"
        out = model.evaluate_strided_rollout(b, stride=4)
        changed = dict(b, carry_state_from_prev_chunk=True, prev_frame_same_day=True,
                       stride_anchor_state=out["stride_anchor_state"], initial_latent_state=out["final_latent_state"],
                       candidate_geometry_identity="geometry-v2")
        with self.assertRaisesRegex(ValueError, "geometry changed"):
            model.evaluate_strided_rollout(changed, stride=4)
        no_anchor = dict(b, candidate_anchor_mask=torch.zeros(2, dtype=torch.bool),
                         candidate_anchor_next_phase=3, candidate_anchor_stride=4)
        with self.assertRaisesRegex(ValueError, "requires previous"):
            model.evaluate_strided_rollout(no_anchor, stride=4)
        with self.assertRaisesRegex(ValueError, "stride differs"):
            model.evaluate_strided_rollout(dict(b, candidate_anchor_stride=2), stride=4)


if __name__ == "__main__":
    unittest.main()

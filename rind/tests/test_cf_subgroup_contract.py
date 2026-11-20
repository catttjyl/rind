"""Counterfactual physical-response gradients and receiver subgroup weighting."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from rind.fast_data import build_canonical37_torch
from rind.fast_train import _validate_training_resume_configuration
from rind.objectives_jepa import (
    compute_candidate_face_perturbation_loss,
    exact_candidate_ray_blocked_torch,
)


class ConstantResponse(torch.nn.Module):
    model_type = "rind_full_geom"

    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch, **kwargs):
        shape = batch["candidate_mask"].shape[:-1]
        logits = self.logit.expand(shape)
        return {"probs": logits.sigmoid(), "logits_bce": logits}


def flip_batch(n=1, height=1.0):
    sun = torch.tensor([[1.0, 1.0, 0.0]])
    roof = torch.tensor([[5.0, height, -2.0], [9.0, height, -2.0], [7.0, height, 2.0]])
    tri = roof.reshape(1, 1, 1, 3, 3).expand(1, n, 1, 3, 3).clone()
    mask = torch.ones((1, n, 1), dtype=torch.bool)
    return dict(candidate_tokens=build_canonical37_torch(tri, mask, mask, sun),
                candidate_mask=mask, sun_dirs=sun,
                local_features=torch.arange(n, dtype=torch.float32).view(n, 1))


class TestCFSubgroupContract(unittest.TestCase):
    def test_physical_flip_constant_prediction_has_loss_and_directed_gradient(self):
        for height, seed, expected_gradient_sign in ((1.0, 0, 1), (7.0, 2, -1)):
            with self.subTest(height=height):
                model = ConstantResponse()
                batch = flip_batch(height=height)
                counts = {}
                base = model(batch)["probs"]
                loss = compute_candidate_face_perturbation_loss(
                    model, batch, base, step_seed=seed, dominant_occluder_dropout_prob=0,
                    edit_counts=counts,
                )
                self.assertAlmostEqual(float(loss.detach()), 1.0)
                loss.backward()
                self.assertGreater(expected_gradient_sign * float(model.logit.grad), 0.1)
                key = "sun_to_shadow" if seed == 0 else "shadow_to_sun"
                self.assertEqual(counts[key], 1)

    def test_exact_counterfactual_oracle_has_inclusive_4km_limit(self):
        heights = torch.tensor([3999.0, 4000.0, 4000.01])
        tri = torch.tensor([[-1., 0., -1.], [1., 0., -1.], [0., 0., 1.]])
        tris = tri.reshape(1, 1, 1, 3, 3).expand(1, 3, 1, 3, 3).clone()
        tris[..., 1] = heights.reshape(1, 3, 1, 1)
        hits = exact_candidate_ray_blocked_torch(
            tris, torch.ones(1, 3, 1, dtype=torch.bool), torch.tensor([[0., 1., 0.]])
        )
        self.assertEqual(hits.tolist(), [[True, True, False]])

    def test_subgroups_restore_samples_without_multiplying_loss_weight(self):
        batch = flip_batch(n=130)
        model = ConstantResponse()
        counts = {}
        loss = compute_candidate_face_perturbation_loss(
            model, batch, model(batch)["probs"], cf_subgroup_size=64,
            dominant_occluder_dropout_prob=0, edit_counts=counts,
        )
        # Independent bounded samples: 32 + 32 + 2, versus only 32 in legacy mode.
        self.assertEqual(counts["sun_to_shadow"], 66)
        self.assertEqual(counts["raised_calls"], 3)
        self.assertAlmostEqual(float(loss.detach()), 1.0)
        loss.backward()
        self.assertAlmostEqual(float(model.logit.grad), 0.5, places=6)

    def test_short_subgroup_uses_actual_count_and_all_context_is_sliced(self):
        batch = flip_batch(n=130)
        batch.update(receiver_obs_mask=torch.ones(1, 130, dtype=torch.bool),
                     receivers_xyz=torch.arange(390.).view(130, 3),
                     global_knn_dist=torch.zeros(130, 4),
                     global_knn_local_features=torch.zeros(130, 4, 16))
        parameter = torch.tensor(2.0, requires_grad=True)
        seen = []

        def worker(model, sub, base_probs, **kwargs):
            n = sub["candidate_tokens"].shape[-3]
            seen.append((n, kwargs["step_seed"]))
            self.assertEqual(sub["receivers_xyz"].shape[0], n)
            self.assertEqual(sub["global_knn_dist"].shape[0], n)
            self.assertEqual(sub["global_knn_local_features"].shape[0], n)
            self.assertEqual(sub["receiver_obs_mask"].shape[-1], n)
            return parameter * sub["local_features"].mean()

        with patch("rind.objectives_jepa._compute_candidate_face_perturbation_loss", worker):
            loss = compute_candidate_face_perturbation_loss(
                ConstantResponse(), batch, torch.zeros(1, 130), cf_subgroup_size=64, step_seed=7
            )
        self.assertEqual(seen, [(64, 7), (64, 104737), (2, 209467)])
        self.assertAlmostEqual(float(loss.detach()), 129.0, places=5)
        loss.backward()
        self.assertAlmostEqual(float(parameter.grad), 64.5, places=5)

    def test_legacy_default_and_invalid_subgroup(self):
        batch = flip_batch(n=130)
        counts = {}
        model = ConstantResponse()
        compute_candidate_face_perturbation_loss(
            model, batch, model(batch)["probs"], edit_counts=counts,
            dominant_occluder_dropout_prob=0,
        )
        self.assertEqual(counts["sun_to_shadow"], 32)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            compute_candidate_face_perturbation_loss(model, batch, torch.zeros(1, 130), cf_subgroup_size=-1)

    def test_resume_optional_model_switches_and_cf_schedule_are_strict(self):
        class ConfigModel:
            def __init__(self, **overrides):
                self.config = dict(hidden_dim=24, film_beta_bound=0.0,
                                   attn_qk_norm=False, cand_readout="attn_only", **overrides)

            def get_model_config(self):
                return self.config

        model = ConfigModel()
        legacy = {"model_config": {"hidden_dim": 24}}
        _validate_training_resume_configuration(legacy, model)
        for key, value in (("film_beta_bound", 1.0), ("attn_qk_norm", True),
                           ("cand_readout", "attn_plus_lse")):
            changed = ConfigModel()
            changed.config[key] = value
            with self.assertRaisesRegex(ValueError, key):
                _validate_training_resume_configuration(legacy, changed)
        for options in ({"cf_subgroup_size": 64}, {"cf_every_n_steps": 1}):
            with self.assertRaisesRegex(ValueError, "CF configuration"):
                _validate_training_resume_configuration(legacy, model, **options)
        state = {"model_config": model.config,
                 "cf_config": {"cf_subgroup_size": 64, "cf_every_n_steps": 1}}
        _validate_training_resume_configuration(state, model, cf_subgroup_size=64, cf_every_n_steps=1)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rind.fast_data import ContiguousTileMmapShard, materialize_gpu_tokens_from_azimuth_bank
from rind.fast_train import compute_long_day_lag_diagnostics
from rind.fullday_data import FullDayTileReader, create_synthetic_fullday_fixture
from rind.inverse_design import _reconstruct_and_predict_scene_trajectory
from rind.models import build_model
from rind.native_scene import native_scene_batch
from rind.objectives_jepa import MultiHorizonLatentPredictor
from rind.split_contract import SplitContract


class NativeViews(unittest.TestCase):
    def test_native_zero_edit_checkpoint_and_crops(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split, manifest = create_synthetic_fullday_fixture(root/"fixture",grid_side=10,frames_per_day=8)
            contract = SplitContract(split, manifest, enforce_pinned_split_sha256=False)
            traj = FullDayTileReader(contract).load_tile_fullday(contract.get_fit_tiles("fit_8")[0],
                week_indices=[0], max_receivers=12, max_candidates=4)
            model = build_model("rind_full_geom", hidden_dim=16, max_candidates=4).eval()
            model.calibrated_temperature.fill_(1.7)
            shard = ContiguousTileMmapShard.write_from_trajectory(root/"shard",traj,contract=contract)
            normal = materialize_gpu_tokens_from_azimuth_bank(next(shard.iter_day_batches(
                days_per_batch=1,receiver_chunk_size=12,model_type=model.model_type,include_labels=False)))
            batch = native_scene_batch(traj,model,traj.scene_triangles_xyz,crop_size=4)
            for key in ("candidate_source_ids","candidate_tokens","candidate_mask","local_features","receiver_rc",
                        "receivers_xyz","receiver_obs_mask","global_knn_local_features","global_knn_obs_valid",
                        "global_knn_rel_xyz","global_knn_dist","sun_dirs","same_day_adjacent_mask"):
                torch.testing.assert_close(batch[key],normal[key].reshape_as(batch[key]),rtol=0,atol=0)
            np.testing.assert_array_equal(batch["receiver_crops"],traj.extract_centered_crops(4))
            pred = _reconstruct_and_predict_scene_trajectory(traj.receivers_xyz,traj.sun_dirs,traj.scene_triangles_xyz,
                rind_model=model,reference_trajectory=traj)
            with torch.no_grad():
                expected = model(normal,use_calibrated_temperature=True)["probs"].reshape(pred.shape)
            np.testing.assert_allclose(pred,expected.numpy(),rtol=0,atol=1e-7)

    def test_driver_uses_only_public_geometry(self):
        torch.manual_seed(4)
        predictor = MultiHorizonLatentPredictor(8,leads=(1,))
        z = torch.randn(1,3,2,8)
        target = torch.randn_like(z)
        solar = torch.randn(1,3,8)
        geom = torch.randn(1,2,8)
        def run(latent,sun,static):
            return predictor.forward_multi_horizon(latent,target,sun,valid_mask=torch.ones(1,3,2,dtype=torch.bool),
                same_day_adjacent_mask=None,precomputed_target_mask=torch.ones(1,3,2,dtype=torch.bool),
                mask_mode="driver",static_geom_memory=static)[0]
        baseline = run(z,solar,geom)
        torch.testing.assert_close(run(-z,-solar,geom),baseline,atol=0,rtol=0)
        self.assertGreater(abs(float(run(z,solar,-geom).detach()-baseline.detach())),1e-5)

    def test_signed_lag_and_missing(self):
        truth = np.array([[0,0],[1,0],[1,1],[1,1],[1,1]])
        pred = np.array([[0,0],[0,1],[1,1],[1,1],[1,1]])
        out = compute_long_day_lag_diagnostics(pred,truth,np.ones_like(truth,bool),[0]*5)
        self.assertEqual(out["signed_transition_lag_seconds_histogram"],{"-120":1,"120":1})
        self.assertEqual(out["mean_transition_lag_seconds"],120)

    def test_visible_envelope_derivatives(self):
        from types import SimpleNamespace
        from rind.fullday_data import select_azimuth_sector_candidates
        tri = np.array([[[0,h,0],[5,h,0],[0,h,5]] for h in (2.,5.,4.)])
        tri[2,:,0] += 5
        rec = np.array([[1,0,1],[6,0,1]],dtype=np.float32)
        ids = np.arange(3,dtype=np.uint64)
        opaque = np.ones(3,bool)
        az = select_azimuth_sector_candidates(rec,tri,scene_opaque=opaque,scene_source_triangle_ids=ids,
                                              max_candidates=3,return_source_ids=True)
        tr = SimpleNamespace(scene_triangles_xyz=tri,receivers_xyz=rec,world_x_m=np.arange(11),
             world_z_m=np.arange(6),scene_opaque=opaque,scene_source_ids=ids,bbox_margin_m=4000.,
             receiver_rc=np.array([[1,1],[1,6]]),static_fields_8hw=np.zeros((8,6,11),np.float32),
             geometry_known_hw=np.ones((6,11),bool),az_rel_triangles=az[0],az_opaque=az[1],az_mask=az[2],
             az_source_ids=az[3],azimuth_bin_idx=np.array([0]),sun_dirs=np.array([[0,1,0]],np.float32),
             same_day_adjacent_mask=np.array([],bool))
        def response(d, grad):
            geom = torch.tensor(tri,dtype=d.dtype) + d[:,None,None]*d.new_tensor([0,1,0])
            batch = native_scene_batch(tr,SimpleNamespace(crop_size=3),geom.detach().numpy(),
                                       differentiable_triangles=geom if grad else None)
            local = batch["local_features"]
            return torch.cat((local[:,2],local[:,10],batch["receiver_crops"][:,2].mean((1,2)),
                              batch["global_knn_local_features"][:,0,2]))
        zero = torch.zeros(3,requires_grad=True)
        jac = torch.autograd.functional.jacobian(lambda d: response(d,True),zero)
        jac64 = torch.autograd.functional.jacobian(lambda d: response(d,True),zero.double())
        torch.testing.assert_close(jac64.float(),jac)
        for i in range(3):
            direction = torch.zeros(3)
            direction[i] = .01
            fd = (response(direction,False)-response(-direction,False))/.02
            torch.testing.assert_close(jac[:,i],fd,rtol=1e-4,atol=1e-6)
        torch.testing.assert_close(jac[:,0],torch.zeros(8))
        self.assertGreater(float(jac[0,1]),0)
        self.assertGreater(float(jac[1,2]),0)
        self.assertAlmostEqual(float(jac[6,2]),.01,places=6)
        self.assertAlmostEqual(float(jac[7,1]),.01,places=6)

    def test_authorized_diagonal_two_pixel_far_and_hidden(self):
        from rind.objectives_jepa import _authorized_grid_boundaries
        axis=torch.arange(11)
        rc=torch.cartesian_prod(axis,axis)
        truth=torch.zeros(4,121,dtype=torch.int8)
        visible=torch.ones_like(truth,dtype=torch.bool)
        truth[0,6*11+6]=1
        truth[1,5*11+7]=1
        truth[3,5*11+7]=1
        visible[3,5*11+6]=False
        batch={"receiver_rc":rc,"meters_per_pixel":.5}
        distance,valid=_authorized_grid_boundaries(batch,truth,visible)
        torch.testing.assert_close(distance[:3,60],torch.tensor([2**.5*.5,1.,2.25]))
        self.assertEqual(valid[:,60].tolist(),[True,True,True,False])
        truth[3,5*11+6]=1
        dist2,v2=_authorized_grid_boundaries(batch,truth,visible)
        torch.testing.assert_close(distance,dist2)
        torch.testing.assert_close(valid,v2)

    def test_blind_missing_and_sentinel_sources(self):
        from rind.inverse_design import infer_unknown_building_heights_bayesian
        tri=np.array([[[0,0,0],[2,0,0],[0,0,2]],[[0,8,0],[2,8,0],[0,8,2]]],float)
        rec=np.array([[.5,8.,.5],[5.,0.,5.]])
        for ids in (None,np.full(2,np.iinfo(np.uint64).max,dtype=np.uint64)):
            report=infer_unknown_building_heights_bayesian(rec,np.array([[0,1,0]],float),tri,
                building_ids=np.array([0,0]),exclude_source_triangle_ids=ids,
                num_grid_hypotheses=5,initial_queries=2,active_rounds=0)
            self.assertEqual(report["uncertified_receivers_excluded"],[0])
            self.assertEqual(report["query_domain_point_frames"],1)
            self.assertEqual([v[1] for v in report["active_observations"]],[1])

    def test_dropout_deletes_first_actual_hit(self):
        from rind.fast_data import build_canonical37_torch
        from rind.objectives_jepa import compute_candidate_face_perturbation_loss
        class Probe(torch.nn.Module):
            model_type="rind_full_geom"
            def forward(self,b,**kwargs):
                if "candidate_keep_mask" in b:
                    self.keep=b["candidate_keep_mask"].flatten().tolist()
                return {"probs":torch.full((1,1),.5),"logits_bce":torch.zeros(1,1)}
        sun=torch.tensor([[0.,1.,0.]])
        face=torch.tensor([[-1.,1.,-1.],[1.,1.,-1.],[0.,1.,1.]])
        mask=torch.ones((1,1,2),dtype=torch.bool)
        for offset,height,count,tie in [(5.,10.,0,False),(0.,10.,1,False),(0.,1.,0,True)]:
            near=face.clone();near[:,0]+=offset
            far=face.clone();far[:,1]=height
            tri=torch.stack((near,far)).reshape(1,1,2,3,3)
            tok=build_canonical37_torch(tri,mask,mask,sun)
            obs={};model=Probe()
            compute_candidate_face_perturbation_loss(model,dict(candidate_tokens=tok,candidate_mask=mask,
                sun_dirs=sun),torch.zeros(1,1),dominant_occluder_dropout_prob=1,edit_counts=obs)
            self.assertEqual(obs["secondary_hit_after_dropout"],count)
            self.assertEqual(obs["single_hit_rays"],int(bool(offset)))
            self.assertEqual(obs["tied_front_depth"],int(tie))
            self.assertEqual(model.keep,[bool(offset),bool(count)])

    def test_inverse_cli_rejects_mislabeled_tile_before_read(self):
        from argparse import Namespace
        from unittest.mock import Mock, patch
        from rind import cli
        contract=Mock()
        contract.get_fit_tiles.return_value=["fit_a"]
        contract.get_eval_tiles.return_value=["val_a"]
        contract.get_calibration_tiles.return_value=["cal_a"]
        with patch.object(cli,"_load_contract_from_args",return_value=contract),patch.object(cli,"FullDayTileReader") as reader:
            for bucket,tile in [("val_spatial_clean","fit_a"),("fit_8","fit_outside"),("calibration","val_a")]:
                for checkpoint in (None,"nonexistent.pt"):
                    args=Namespace(bucket=bucket,tile_id=tile,device="cpu",checkpoint=checkpoint)
                    for cmd in (cli.cmd_inverse_design,cli.cmd_eval_repr_sufficiency,cli.cmd_infer_building_height):
                        with self.assertRaisesRegex(ValueError,"outside requested bucket"):
                            cmd(args)
            reader.assert_not_called()

    def test_roof_without_base_rejected_before_hypothesis(self):
        from unittest.mock import patch
        from rind.inverse_design import infer_unknown_building_heights_bayesian
        for elevation in (8.,18.):
            roof=np.array([[[0,elevation,0],[2,elevation,0],[0,elevation,2]]])
            with patch("rind.inverse_design._reconstruct_and_predict_scene_trajectory") as predict:
                with self.assertRaisesRegex(ValueError,"public base"):
                    infer_unknown_building_heights_bayesian(np.array([[4.,0.,4.]]),np.array([[0,1,0]]),
                        roof,building_ids=np.array([0]),unknown_building_ids=[0],initial_queries=0)
                predict.assert_not_called()

"""Optional fixed-pose scope through real Torch FK/LBS and native tracking.

Optimiser probes avoid a long calibration, but evaluate actual losses/gradients
and deliberately corrupt disabled latent entries to test the effective mask.
"""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vsmpl.calibration as calibration
from vsmpl.body_model import BodyModel, MarkerRig
from vsmpl.rotations import exp_so3
from vsmpl.solver import RealtimeSolver, SolverConfig


def fixture():
    vertices = np.array([[0.,0.,0.],[0.,1.,0.],[0.,2.,0.],[.2,2.1,.1],[.4,1.2,.1],[-.2,1.4,-.1]])
    regressor = np.zeros((3,6));regressor[np.arange(3),np.arange(3)] = 1.
    weights = np.zeros((6,3));weights[0,0] = 1.;weights[[1,4,5],1] = 1.;weights[[2,3],2] = 1.
    posedirs = np.random.default_rng(5).normal(size=(18,6,3))*.001
    model = BodyModel(vertices,np.zeros((6,3,1)),posedirs,regressor,weights,np.array([-1,0,1]),
                      np.array([[0,1,3],[0,3,5],[1,4,3],[1,3,2]]),"smplx")
    rig = MarkerRig(model,list("ABCDEF"),np.arange(6),np.zeros((6,3)),np.zeros(1))
    source_aa = np.array([[[.05,-.1,.02],[.15,-.1,.2],[.6,-.3,.4]],
                          [[.02,-.08,.04],[.2,-.15,.3],[.5,-.2,.3]],
                          [[.01,-.04,.03],[.1,-.2,.25],[.7,-.1,.5]]])
    R = exp_so3(source_aa)
    trans = np.tile([.1,.2,.3],(3,1))
    target = R.copy();target[:,1] = exp_so3([.35,-.3,.4]);target[:,2] = exp_so3([-.4,.2,-.2])
    observed = np.stack([rig.forward(rotation,t) for rotation,t in zip(target,trans)])
    return model,rig,R,trans,observed,np.ones((3,6),bool)


class ProbeAdam:
    gradients = []
    corrupt_fixed = False

    def __init__(self,params,**kwargs): self.params = list(params)
    def zero_grad(self):
        for parameter in self.params: parameter.grad = None
    def step(self):
        for parameter in self.params:
            if parameter.shape == (2,3):
                self.gradients.append(parameter.grad.detach().numpy().copy())
                if self.corrupt_fixed:
                    with torch.no_grad(): parameter[1] = torch.tensor([.4,-.2,.3])


class ProbeLBFGS:
    initial = gradients = None
    corrupt_fixed = False

    def __init__(self,params,**kwargs):
        self.params = list(params)
        type(self).initial = self.params[2].detach().numpy().copy()
    def zero_grad(self):
        for parameter in self.params: parameter.grad = None
    def step(self,closure):
        if self.corrupt_fixed:
            with torch.no_grad(): self.params[2][:,2] = torch.tensor([.4,-.2,.3])
        closure()
        type(self).gradients = self.params[2].grad.detach().numpy().copy()


class FixedPoseScopeTests(unittest.TestCase):
    def setUp(self):
        self.model,self.rig,self.R,self.trans,self.observed,self.valid = fixture()
        ProbeAdam.gradients=[];ProbeAdam.corrupt_fixed=False
        ProbeLBFGS.initial=ProbeLBFGS.gradients=None;ProbeLBFGS.corrupt_fixed=False

    def static_fit(self,fixed_joints=None):
        reference = np.array([[0.,0.,0.],[.15,-.1,.2],[.7,.2,-.4]])
        original = reference.copy()
        with patch.object(calibration.torch.optim,"Adam",ProbeAdam):
            result = calibration.fit_tpose(self.model,self.observed[0],self.valid[0],.2,iters=3,
                vids=self.rig.vertex_ids,marker_radius=0.,reference_pose=reference,fixed_joints=fixed_joints)
        np.testing.assert_array_equal(reference,original)
        return result

    def refine(self,fixed_joints=None):
        anchor = np.array([[0.,0.,0.],[.1,-.2,.15],[.7,.2,-.4]])
        original = anchor.copy();original_R = self.R.copy()
        with patch.object(calibration.torch.optim,"LBFGS",ProbeLBFGS):
            result = calibration.refine(self.model,self.rig.vertex_ids,self.rig.offsets,self.rig.betas,
                self.observed,self.valid,self.R,self.trans,0.,anchor=0,anchor_pose=anchor,
                fixed_joints=fixed_joints,verbose=False)
        np.testing.assert_array_equal(anchor,original)
        np.testing.assert_array_equal(self.R,original_R)
        return result

    def test_static_reference_and_effective_pose_are_fixed_with_zero_gradient(self):
        ProbeAdam.corrupt_fixed=True
        result = self.static_fit([2])
        np.testing.assert_array_equal(result["body"][1],np.zeros(3))
        self.assertTrue(ProbeAdam.gradients)
        for gradient in ProbeAdam.gradients:
            np.testing.assert_array_equal(gradient[1],np.zeros(3))
            self.assertGreater(np.linalg.norm(gradient[0]),1e-8)
        np.testing.assert_array_equal(result["body"][0],[.15,-.1,.2])

    def test_refine_masks_nonzero_initial_and_latent_pose_and_reports_actual_fixed_fk(self):
        ProbeLBFGS.corrupt_fixed=True
        betas,offsets,R,trans,rms = self.refine([2])
        np.testing.assert_array_equal(ProbeLBFGS.initial[:,2],np.zeros((3,3)))
        np.testing.assert_array_equal(ProbeLBFGS.gradients[:,2],np.zeros((3,3)))
        self.assertGreater(np.linalg.norm(ProbeLBFGS.gradients[:,1]),1e-8)
        np.testing.assert_array_equal(R[:,2],np.tile(np.eye(3),(3,1,1)))
        np.testing.assert_allclose(R[:,1],self.R[:,1],atol=2e-16)
        final_rig = MarkerRig(self.model,self.rig.names,self.rig.vertex_ids,offsets,betas)
        prediction = np.stack([final_rig.forward(rotation,t) for rotation,t in zip(R,trans)])
        expected = np.sqrt(np.mean(np.sum((prediction-self.observed)**2,axis=-1),axis=0))
        np.testing.assert_allclose(rms,expected,atol=5e-16)

    def test_native_tracking_cfg_freezes_requested_joints_without_mutating_caller_cfg(self):
        cfg=SolverConfig(iters=1,init_iters=2,min_markers=3,active_joints=[0,1,2],fixed_joints=[1])
        built=[]
        def solver(rig,config):
            instance=RealtimeSolver(rig,config);built.append(instance);return instance
        with patch.object(calibration,"RealtimeSolver",side_effect=solver):
            R,_,_,_=calibration.track_take(self.rig,self.observed,self.valid,cfg,fixed_joints=[2])
        self.assertEqual(cfg.fixed_joints,[1])
        self.assertEqual(built[0].cfg.fixed_joints,[1,2])
        self.assertEqual(built[0].active.tolist(),[0])
        np.testing.assert_array_equal(R[:,1:],np.tile(np.eye(3),(3,2,1,1)))

    def test_default_scope_keeps_nonzero_free_pose_and_empty_scope_is_identical(self):
        no_scope=self.static_fit()
        no_scope_gradients=[g.copy() for g in ProbeAdam.gradients]
        ProbeAdam.gradients=[]
        empty_scope=self.static_fit([])
        for key in no_scope:
            np.testing.assert_array_equal(no_scope[key],empty_scope[key])
        np.testing.assert_array_equal(no_scope["body"][1],[.7,.2,-.4])
        self.assertGreater(np.linalg.norm(no_scope_gradients[-1][1]),1e-8)
        no_scope=self.refine()
        np.testing.assert_allclose(no_scope[2],self.R,atol=5e-16)
        gradients=ProbeLBFGS.gradients.copy()
        empty_scope=self.refine([])
        for before,after in zip(no_scope,empty_scope): np.testing.assert_array_equal(before,after)
        self.assertGreater(np.linalg.norm(gradients[:,2]),1e-8)

    def test_invalid_indices_fail_in_all_four_entry_points(self):
        rec=dict(names=self.rig.names,pos=self.observed,valid=self.valid)
        calls=(lambda fixed:self.static_fit(fixed),lambda fixed:self.refine(fixed),
               lambda fixed:calibration.track_take(self.rig,self.observed,self.valid,fixed_joints=fixed),
               lambda fixed:calibration.calibrate(self.model,rec,fixed_joints=fixed))
        for invalid in ([0],[-1],[3],[1,1],[1.0],[True,2],[[1,2]]):
            for call in calls:
                with self.subTest(invalid=invalid,call=call),self.assertRaises(ValueError): call(invalid)

    def test_driver_forwards_one_scope_and_masks_its_reference_and_saved_contract(self):
        self.model.model_type="soma"
        # Six clean source frames allow the driver to select/refine a take.
        pos=np.tile(self.observed[:1],(6,1,1));valid=np.ones((6,6),bool)
        rec=dict(names=self.rig.names,pos=pos,valid=valid,fps=100.)
        selected=np.array([0,2,4]);R=np.tile(np.eye(3),(6,3,1,1))
        fit=dict(loss=0.,betas=np.zeros(1),root=np.zeros(3),body=np.zeros((2,3)),trans=np.zeros(3))
        reference=np.array([[0.,0.,0.],[.1,.2,.3],[.4,.5,.6]])
        refined_R=R[selected].copy();refined_t=np.zeros((3,3))
        with patch.object(calibration,"model_static_reference",return_value=reference), \
                patch.object(calibration,"fit_tpose",return_value=fit) as static_fit, \
                patch.object(calibration,"track_take",return_value=(R,np.zeros((6,3)),np.zeros(6),np.ones(6,bool))) as track, \
                patch.object(calibration,"farthest_point_frames",return_value=selected), \
                patch.object(calibration,"refine",return_value=(np.zeros(1),np.zeros((6,3)),refined_R,refined_t,np.zeros(6))) as refine, \
                redirect_stdout(io.StringIO()):
            result=calibration.calibrate(self.model,rec,tpose=(0,3),facing="+z",rounds=1,
                marker_map=dict(zip(self.rig.names,self.rig.vertex_ids)),fixed_joints=[2],verbose=False)
        for function in (static_fit,track,refine):
            np.testing.assert_array_equal(function.call_args.kwargs["fixed_joints"],[2])
        np.testing.assert_array_equal(result["static_reference_pose"][2],np.zeros(3))
        np.testing.assert_array_equal(reference[2],[.4,.5,.6])
        np.testing.assert_array_equal(result["fixed_joints"],[2])
        self.assertEqual(result["fixed_pose_scope"],"native_bind_identity")
        # Final SOMA metrics still use the refined, fixed pose through native FK.
        prediction=np.stack([self.rig.forward(rotation,t) for rotation,t in zip(refined_R,refined_t)])
        expected=np.sqrt(np.mean(np.sum((prediction-pos[selected])**2,axis=-1),axis=0))
        np.testing.assert_allclose(result["per_marker_rms"],expected,atol=2e-16)

    def test_refine_kw_cannot_replace_the_calibration_wide_scope(self):
        rec=dict(names=self.rig.names,pos=self.observed,valid=self.valid)
        with self.assertRaisesRegex(ValueError,"not inside refine_kw"):
            calibration.calibrate(self.model,rec,fixed_joints=[2],refine_kw=dict(fixed_joints=[1]))


if __name__ == "__main__":
    unittest.main()

"""Core checks: numpy vs torch forward, analytic Jacobian vs finite differences, solver with oracle rig."""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.synthetic import make_body, pick_markers, make_motion, simulate
from vsmpl.body_model import MarkerRig
from vsmpl.rotations import exp_so3, log_so3
from vsmpl.solver import RealtimeSolver, SolverConfig


def test_forward_and_jacobian():
    import torch
    from vsmpl.torch_model import TorchBody
    model, _ = make_body()
    rng = np.random.default_rng(0)
    vids = pick_markers(model)
    betas = rng.standard_normal(10) * 0.5
    v, _ = model.shaped(betas)
    rig = MarkerRig(model, [str(i) for i in vids], vids, 0.01 * model.vertex_normals(v)[vids], betas)
    R = exp_so3(rng.standard_normal((24, 3)) * 0.5)
    t = rng.standard_normal(3)
    # numpy marker forward == numpy full-mesh forward at marker vertices (offset 0 check)
    rig0 = MarkerRig(model, rig.names, vids, np.zeros((len(vids), 3)), betas)
    full = model.vertices(R, t, *model.shaped(betas))
    assert np.abs(rig0.forward(R, t) - full[vids]).max() < 1e-10
    # torch == numpy
    tb = TorchBody(model)
    tv, _, _ = tb.vertices(torch.tensor(R)[None], torch.tensor(t)[None], torch.tensor(betas))
    assert np.abs(tv[0].numpy() - full).max() < 1e-9
    # Jacobian vs finite differences (pose blendshapes off: the analytic J ignores them)
    model.posedirs[:] = 0
    rig = MarkerRig(model, rig.names, vids, rig.offsets, betas)
    act = np.arange(24)
    out, cache = rig.forward(R, t, with_cache=True)
    Ja = rig.jacobian(cache, act)
    eps = 1e-6
    Jn = np.zeros_like(Ja)
    for k, j in enumerate(act):
        for a in range(3):
            d = np.zeros(3); d[a] = eps
            R2 = R.copy(); R2[j] = R[j] @ exp_so3(d)
            Jn[:, :, 3 * k + a] = (rig.forward(R2, t) - out) / eps
    for a in range(3):
        t2 = t.copy(); t2[a] += eps
        Jn[:, :, -3 + a] = (rig.forward(R, t2) - out) / eps
    err = np.abs(Ja - Jn).max()
    print(f"[jacobian] max |analytic - numeric| = {err:.2e}")
    assert err < 1e-4


def test_so3_kernels():
    from vsmpl import rotations as rot
    rng = np.random.default_rng(1)
    r = rng.standard_normal((500, 3))
    r[:50] *= 1e-6  # near identity
    r[50:100] = r[50:100] / np.linalg.norm(r[50:100], axis=1, keepdims=True) * (np.pi - 1e-5)  # near pi
    R = rot.exp_so3(r)
    assert np.abs(rot.exp_batch(r) - R).max() < 1e-10
    assert np.abs(rot.exp_so3(rot.log_batch(R)) - R).max() < 1e-6
    assert np.abs(rot.exp_so3(rot.log_so3(R)) - R).max() < 1e-6
    ok = np.linalg.norm(r, axis=1) < 3.0
    assert np.abs(rot.rjinv_batch(r[ok]) - rot.right_jacobian_inv(r[ok])).max() < 1e-8
    # rjinv is the derivative of log(R exp(w)) at w = 0
    phi = r[200]; eps = 1e-6
    num = np.stack([(rot.log_so3(rot.exp_so3(phi) @ rot.exp_so3(eps * e)) - phi) / eps for e in np.eye(3)], 1)
    assert np.abs(num - rot.right_jacobian_inv(phi)).max() < 1e-4
    print(f"[so3] kernels consistent (numba={rot.HAVE_NUMBA})")


def test_solver_oracle(T=1500, verbose=True):
    model, _ = make_body()
    vids = pick_markers(model)
    betas = np.array([0.8, -0.5, 0.6, -0.4, 0.3, 0, 0, 0, 0, 0])
    R, trans = make_motion(T, seed=3)
    obs, valid, rig = simulate(model, vids, betas, R, trans, seed=4)
    solver = RealtimeSolver(rig, SolverConfig())
    if verbose:
        print(f"[oracle] markers={rig.num_markers} active joints={len(solver.active)}: {solver.active.tolist()}")
    errs, ms, mk = [], [], []
    for i in range(T):
        res = solver.step(obs[i], valid[i], frame_no=i)
        jt = model.joints(R[i], trans[i], rig.J_rest)
        je = model.joints(res.R, res.trans, rig.J_rest)
        errs.append(np.linalg.norm(jt - je, axis=-1)[:22].mean())
        ms.append(res.ms); mk.append(res.rmse)
    errs, ms = np.array(errs) * 1000, np.array(ms)
    print(f"[oracle] MPJPE mean {errs[50:].mean():.1f} mm, p95 {np.percentile(errs[50:], 95):.1f} mm, "
          f"marker rmse {np.nanmean(mk[50:]) * 1000:.1f} mm, solve {np.median(ms):.2f} ms median / {np.percentile(ms, 99):.2f} ms p99")
    return errs, ms


if __name__ == "__main__":
    test_so3_kernels()
    test_forward_and_jacobian()
    test_solver_oracle()

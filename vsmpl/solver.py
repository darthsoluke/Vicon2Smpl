"""Real-time per-frame SMPL fitting to labelled markers.

Levenberg-Marquardt on SO(3)^K x R^3 with an analytic Jacobian (see MarkerRig.jacobian):
  E = sum_m rho(|| marker_m(theta, t) - obs_m ||)       data term, robust (Huber via IRLS), occlusion-masked
    + lam_temp  * sum_j || log(R_pred_j^T R_j) ||^2    temporal term toward a constant-velocity prediction
    + lam_trans * || t - t_pred ||^2
    + lam_prior * sum_j w_j || log(R_j) ||^2           weak pose prior (joint-dependent weights)
Warm-started from the previous frame; re-initialised (Kabsch on torso markers + many iterations) when
tracking is lost.
"""
from dataclasses import dataclass, field
import time
import numpy as np

from .rotations import exp_so3, log_so3, kabsch, exp_batch, log_batch, rjinv_batch
from .body_model import MarkerRig, TORSO_JOINTS


@dataclass
class SolverConfig:
    iters: int = 4  # LM iterations per frame (tracking)
    init_iters: int = 40  # iterations on (re-)initialisation
    huber_delta: float = 0.02  # m; residuals beyond this are down-weighted
    reject_dist: float = 0.10  # m; markers further than this from the prediction are ignored (label swaps, ghosts)
    lam_temp: float = 1e-3  # temporal weight on joint rotations (per rad^2, relative to m^2 data)
    lam_trans: float = 1e-2
    lam_prior: float = 1e-4  # weak pull toward rest pose
    vel_gain: float = 0.8  # constant-velocity prediction gain (0 = hold previous pose)
    min_markers: int = 6  # below this, hold the previous pose
    lost_rmse: float = 0.05  # m; above this for `lost_frames` frames -> re-initialise
    lost_frames: int = 10
    obs_thresh: float = 0.02  # m/rad; joints with weaker marker influence are 'unobserved' (fixed)
    active_joints: list = None  # override automatic joint selection (list of indices)
    fixed_joints: list = field(default_factory=list)  # joints always kept at rest (e.g. hands w/o markers)


@dataclass
class SolverResult:
    R: np.ndarray  # (J,3,3) local rotations
    trans: np.ndarray  # (3,)
    rmse: float  # m, over valid inlier markers
    n_valid: int
    n_inliers: int
    iters: int
    ms: float
    status: str  # 'track' | 'init' | 'hold'

    @property
    def pose_aa(self):
        """(J,3) axis-angle, smplx convention (pose[0] = global_orient)."""
        return log_so3(self.R)


class RealtimeSolver:
    def __init__(self, rig: MarkerRig, cfg: SolverConfig = None):
        self.rig, self.cfg = rig, cfg or SolverConfig()
        J = rig.model.num_joints
        obs = rig.joint_observability()
        if self.cfg.active_joints is not None:
            active = np.array(sorted(self.cfg.active_joints))
        else:
            active = np.nonzero(obs >= self.cfg.obs_thresh)[0]
        active = np.array([j for j in active if j not in set(self.cfg.fixed_joints)], dtype=np.int64)
        if 0 not in active:
            active = np.concatenate([[0], active])
        self.active = active
        self.observability = obs
        # prior weights: root free, others weak
        self.prior_w = np.full(len(active), self.cfg.lam_prior)
        self.prior_w[active == 0] = 0.0
        # torso markers for Kabsch (dominant joint in torso set)
        dom = np.argmax(rig.W, axis=1)
        self.torso_markers = np.isin(dom, TORSO_JOINTS)
        self.reset()

    # -------------------------------------------------------------- state
    def reset(self):
        J = self.rig.model.num_joints
        self.R = np.broadcast_to(np.eye(3), (J, 3, 3)).copy()
        self.trans = np.zeros(3)
        self.hist = []  # [(R, trans, frame_no)] last two accepted states
        self.initialised = False
        self.bad_count = 0

    def _predict(self, frame_no):
        if len(self.hist) < 2 or self.cfg.vel_gain <= 0:
            return self.R.copy(), self.trans.copy()
        (R0, t0, f0), (R1, t1, f1) = self.hist[-2], self.hist[-1]
        gain = self.cfg.vel_gain
        if frame_no is not None and f1 is not None and f0 is not None and f1 > f0:
            gain *= np.clip((frame_no - f1) / (f1 - f0), 0.0, 3.0)
        dR = log_so3(np.einsum("jba,jbc->jac", R0, R1))  # R0^T R1 (local increment)
        Rp = R1 @ exp_so3(gain * dR)
        return Rp, t1 + gain * (t1 - t0)

    # -------------------------------------------------------------- main entry
    def step(self, obs, valid, frame_no=None):
        """obs: (M,3) metres, Y-up, rig marker order. valid: (M,) bool. Returns SolverResult."""
        t_start = time.perf_counter()
        cfg = self.cfg
        valid = np.asarray(valid, bool) & np.all(np.isfinite(obs), -1)
        nv = int(valid.sum())
        if nv < cfg.min_markers:
            return self._result(np.nan, nv, 0, 0, t_start, "hold")

        status = "track"
        if not self.initialised:
            self._kabsch_init(obs, valid)
            R_pred, t_pred = self.R.copy(), self.trans.copy()
            iters, lam_t, lam_tr, reject = cfg.init_iters, 0.0, 0.0, np.inf
            status = "init"
        else:
            R_pred, t_pred = self._predict(frame_no)
            self.R, self.trans = R_pred.copy(), t_pred.copy()
            iters, lam_t, lam_tr, reject = cfg.iters, cfg.lam_temp, cfg.lam_trans, cfg.reject_dist

        rmse, n_in, it = self._lm(obs, valid, R_pred, t_pred, iters, lam_t, lam_tr, reject)

        if status == "init":
            self.initialised = True
            self.hist = []
        # tracking-loss detection
        if rmse > cfg.lost_rmse or n_in < cfg.min_markers:
            self.bad_count += 1
            if self.bad_count >= cfg.lost_frames:
                self.initialised = False
                self.bad_count = 0
        else:
            self.bad_count = 0
        self.hist = (self.hist + [(self.R.copy(), self.trans.copy(), frame_no)])[-2:]
        return self._result(rmse, nv, n_in, it, t_start, status)

    def _result(self, rmse, nv, n_in, it, t_start, status):
        return SolverResult(self.R.copy(), self.trans.copy(), float(rmse), nv, n_in, it,
                            (time.perf_counter() - t_start) * 1e3, status)

    # -------------------------------------------------------------- init
    def _kabsch_init(self, obs, valid):
        rig = self.rig
        J = rig.model.num_joints
        R0 = np.broadcast_to(np.eye(3), (J, 3, 3)).copy()
        P0 = rig.forward(R0, np.zeros(3))
        sel = valid & self.torso_markers
        if sel.sum() < 3:
            sel = valid
        Rg, t = kabsch(P0[sel], obs[sel])
        # Kabsch rotates about the origin; SMPL root rotates about J_rest[0]
        j0 = rig.J_rest[0]
        R0[0] = Rg
        self.R = R0
        self.trans = t + Rg @ j0 - j0

    # -------------------------------------------------------------- LM
    def _cost_terms(self, obs, valid, R, t, R_pred, t_pred, lam_t, lam_tr, reject, w_rob=None, with_jac=False):
        rig, act = self.rig, self.active
        pred, cache = rig.forward(R, t, with_cache=True)
        e = np.where(valid[:, None], pred - obs, 0.0)  # (M,3); occluded markers may be NaN
        dist = np.linalg.norm(e, axis=-1)
        inl = valid & (dist < reject)
        if w_rob is None:
            d = self.cfg.huber_delta
            w_rob = np.where(dist <= d, 1.0, d / np.maximum(dist, 1e-12))
        w = np.where(inl, w_rob, 0.0)
        # regularisers on active joints
        Ra = R[act]
        phi_p = log_batch(Ra)  # prior residual
        phi_t = log_batch(np.transpose(R_pred[act], (0, 2, 1)) @ Ra)  # temporal residual
        lt = np.full(len(act), lam_t)
        # unobserved-but-active joints get no special treatment; fixed joints are excluded from `act`
        cost = (w * dist ** 2).sum() + (self.prior_w * (phi_p ** 2).sum(-1)).sum() \
            + (lt * (phi_t ** 2).sum(-1)).sum() + lam_tr * ((t - t_pred) ** 2).sum()
        if not with_jac:
            return cost, inl, dist
        Jd = rig.jacobian(cache, act)  # (M,3,N)
        sw = np.sqrt(w)
        Jw = (Jd * sw[:, None, None]).reshape(-1, Jd.shape[-1])
        ew = (e * sw[:, None]).reshape(-1)
        H = Jw.T @ Jw
        g = Jw.T @ ew
        K = len(act)
        # block-diagonal regulariser terms, vectorised over joints
        Jp = rjinv_batch(phi_p)  # (K,3,3)
        Jt = rjinv_batch(phi_t)
        JpT, JtT = np.transpose(Jp, (0, 2, 1)), np.transpose(Jt, (0, 2, 1))
        Hb = self.prior_w[:, None, None] * (JpT @ Jp) + lt[:, None, None] * (JtT @ Jt)
        gb = self.prior_w[:, None] * (JpT @ phi_p[..., None])[..., 0] + lt[:, None] * (JtT @ phi_t[..., None])[..., 0]
        idx = 3 * np.arange(K)[:, None] + np.arange(3)[None]  # (K,3)
        H[idx[:, :, None], idx[:, None, :]] += Hb
        g[:-3] += gb.reshape(-1)
        H[-3:, -3:] += lam_tr * np.eye(3)
        g[-3:] += lam_tr * (t - t_pred)
        return cost, inl, dist, H, g, w_rob

    def _apply(self, R, t, delta):
        R = R.copy()
        act = self.active
        R[act] = R[act] @ exp_batch(delta[:-3])
        return R, t + delta[-3:]

    def _lm(self, obs, valid, R_pred, t_pred, iters, lam_t, lam_tr, reject):
        mu = 1e-3
        R, t = self.R, self.trans
        it = 0
        diag_idx = None
        for it in range(1, iters + 1):
            cost, inl, dist, H, g, w_rob = self._cost_terms(obs, valid, R, t, R_pred, t_pred, lam_t, lam_tr, reject, with_jac=True)
            if diag_idx is None:
                diag_idx = np.diag_indices(len(H))
            accepted = False
            delta = np.zeros(len(g))
            Hd = H[diag_idx].copy()
            for _ in range(6):
                A = H.copy()
                A[diag_idx] += mu * Hd + 1e-9
                try:
                    delta = -np.linalg.solve(A, g)
                except np.linalg.LinAlgError:
                    mu *= 10
                    continue
                Rn, tn = self._apply(R, t, delta)
                cn, inl_n, dist_n = self._cost_terms(obs, valid, Rn, tn, R_pred, t_pred, lam_t, lam_tr, reject, w_rob=w_rob)
                if cn < cost:
                    R, t, mu = Rn, tn, max(mu / 3, 1e-7)
                    inl, dist = inl_n, dist_n
                    accepted = True
                    break
                mu *= 10
            if not accepted or np.abs(delta).max() < 1e-5:
                break
        self.R, self.trans = R, t
        n_in = int(inl.sum())
        rmse = float(np.sqrt((dist[inl] ** 2).mean())) if n_in else np.inf
        return rmse, n_in, it

"""Optional numba kernels for the real-time hot path (forward kinematics, marker skinning, Jacobian, SO(3)).
If numba is not installed, HAVE_NUMBA is False and the numpy implementations are used instead.
Disable explicitly with the environment variable VSMPL_NO_NUMBA=1."""
import os
import numpy as np

HAVE_NUMBA = False
if not os.environ.get("VSMPL_NO_NUMBA"):
    try:
        from numba import njit

        HAVE_NUMBA = True
    except ImportError:  # pragma: no cover
        pass

if HAVE_NUMBA:

    @njit(cache=True, fastmath=False)
    def fk_nb(R, J_rest, parents):
        J = R.shape[0]
        Rw = np.empty((J, 3, 3))
        c = np.empty((J, 3))
        Rw[0] = R[0]
        c[0] = J_rest[0]
        for j in range(1, J):
            p = parents[j]
            for a in range(3):
                for b in range(3):
                    s = 0.0
                    for k in range(3):
                        s += Rw[p, a, k] * R[j, k, b]
                    Rw[j, a, b] = s
                s = 0.0
                for k in range(3):
                    s += Rw[p, a, k] * (J_rest[j, k] - J_rest[p, k])
                c[j, a] = s + c[p, a]
        return Rw, c

    @njit(cache=True)
    def skin_nb(Rw, c, J_rest, p, W, trans, want_q):
        """out (M,3) = sum_j W[m,j] (Rw_j (p_m - J_rest_j) + c_j) + trans; q (M,J,3) per-joint transformed points."""
        M, J = p.shape[0], Rw.shape[0]
        At = np.empty((J, 3))
        for j in range(J):
            for a in range(3):
                At[j, a] = c[j, a] - (Rw[j, a, 0] * J_rest[j, 0] + Rw[j, a, 1] * J_rest[j, 1] + Rw[j, a, 2] * J_rest[j, 2])
        out = np.zeros((M, 3))
        q = np.zeros((M, J, 3)) if want_q else np.zeros((1, 1, 3))
        for m in range(M):
            for j in range(J):
                w = W[m, j]
                if w == 0.0:
                    continue
                for a in range(3):
                    v = Rw[j, a, 0] * p[m, 0] + Rw[j, a, 1] * p[m, 1] + Rw[j, a, 2] * p[m, 2] + At[j, a]
                    if want_q:
                        q[m, j, a] = v
                    out[m, a] += w * v
            for a in range(3):
                out[m, a] += trans[a]
        return out, q

    @njit(cache=True)
    def jacobian_nb(Rw, c, q, W, subtree, joints):
        M, J = q.shape[0], q.shape[1]
        K = joints.shape[0]
        out = np.zeros((M, 3, 3 * K + 3))
        for m in range(M):
            for kk in range(K):
                jj = joints[kk]
                ux = 0.0
                uy = 0.0
                uz = 0.0
                S = 0.0
                for k in range(J):
                    if subtree[jj, k] == 0.0:
                        continue
                    w = W[m, k]
                    if w == 0.0:
                        continue
                    ux += w * q[m, k, 0]
                    uy += w * q[m, k, 1]
                    uz += w * q[m, k, 2]
                    S += w
                if S == 0.0:
                    continue
                ux -= S * c[jj, 0]
                uy -= S * c[jj, 1]
                uz -= S * c[jj, 2]
                # -[u]x Rw
                for b in range(3):
                    r0 = Rw[jj, 0, b]
                    r1 = Rw[jj, 1, b]
                    r2 = Rw[jj, 2, b]
                    out[m, 0, 3 * kk + b] = uz * r1 - uy * r2
                    out[m, 1, 3 * kk + b] = -uz * r0 + ux * r2
                    out[m, 2, 3 * kk + b] = uy * r0 - ux * r1
            out[m, 0, 3 * K] = 1.0
            out[m, 1, 3 * K + 1] = 1.0
            out[m, 2, 3 * K + 2] = 1.0
        return out

    @njit(cache=True)
    def exp_nb(r):
        n = r.shape[0]
        R = np.empty((n, 3, 3))
        for i in range(n):
            x, y, z = r[i, 0], r[i, 1], r[i, 2]
            t2 = x * x + y * y + z * z
            if t2 < 1e-8:
                a = 1.0 - t2 / 6.0
                b = 0.5 - t2 / 24.0
            else:
                t = np.sqrt(t2)
                a = np.sin(t) / t
                b = (1.0 - np.cos(t)) / t2
            R[i, 0, 0] = 1.0 - b * (y * y + z * z)
            R[i, 0, 1] = -a * z + b * x * y
            R[i, 0, 2] = a * y + b * x * z
            R[i, 1, 0] = a * z + b * x * y
            R[i, 1, 1] = 1.0 - b * (x * x + z * z)
            R[i, 1, 2] = -a * x + b * y * z
            R[i, 2, 0] = -a * y + b * x * z
            R[i, 2, 1] = a * x + b * y * z
            R[i, 2, 2] = 1.0 - b * (x * x + y * y)
        return R

    @njit(cache=True)
    def log_nb(R):
        n = R.shape[0]
        out = np.empty((n, 3))
        for i in range(n):
            tr = (R[i, 0, 0] + R[i, 1, 1] + R[i, 2, 2] - 1.0) / 2.0
            tr = min(1.0, max(-1.0, tr))
            th = np.arccos(tr)
            wx = R[i, 2, 1] - R[i, 1, 2]
            wy = R[i, 0, 2] - R[i, 2, 0]
            wz = R[i, 1, 0] - R[i, 0, 1]
            if th < 1e-4:
                f = 0.5 + th * th / 12.0
                out[i, 0], out[i, 1], out[i, 2] = f * wx, f * wy, f * wz
            elif th > np.pi - 1e-3:
                # (R + R^T)/2 - cos(th) I = (1 - cos(th)) a a^T
                cth = np.cos(th)
                S = np.empty((3, 3))
                for a in range(3):
                    for b in range(3):
                        S[a, b] = ((R[i, a, b] + R[i, b, a]) / 2 - (cth if a == b else 0.0)) / (1.0 - cth)
                k = 0
                if S[1, 1] > S[0, 0] and S[1, 1] >= S[2, 2]:
                    k = 1
                elif S[2, 2] > S[0, 0] and S[2, 2] > S[1, 1]:
                    k = 2
                nrm = np.sqrt(S[0, k] ** 2 + S[1, k] ** 2 + S[2, k] ** 2)
                s = 1.0 if S[0, k] * wx + S[1, k] * wy + S[2, k] * wz >= 0 else -1.0
                for a in range(3):
                    out[i, a] = S[a, k] / nrm * th * s
            else:
                f = th / (2.0 * np.sin(th))
                out[i, 0], out[i, 1], out[i, 2] = f * wx, f * wy, f * wz
        return out

    @njit(cache=True)
    def rjinv_nb(phi):
        n = phi.shape[0]
        out = np.empty((n, 3, 3))
        for i in range(n):
            x, y, z = phi[i, 0], phi[i, 1], phi[i, 2]
            t2 = x * x + y * y + z * z
            if t2 < 1e-6:
                cc = 1.0 / 12.0
            else:
                t = np.sqrt(t2)
                cc = 1.0 / t2 - (1.0 + np.cos(t)) / (2.0 * t * np.sin(t) + 1e-12)
            # K = skew(phi); K^2 = phi phi^T - t2 I
            K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            for a in range(3):
                for b in range(3):
                    K2 = (phi[i, a] * phi[i, b]) - (t2 if a == b else 0.0)
                    out[i, a, b] = (1.0 if a == b else 0.0) + 0.5 * K[a, b] + cc * K2
        return out

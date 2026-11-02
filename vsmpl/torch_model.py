"""Differentiable (torch) version of the forward pass, batched over frames. Used by offline calibration."""
import numpy as np
import torch


def batch_rodrigues(r):
    t2 = (r * r).sum(-1, keepdim=True)[..., None]
    small = t2 < 1e-8
    t2s = torch.where(small, torch.ones_like(t2), t2)
    t = t2s.sqrt()
    a = torch.where(small, 1 - t2 / 6, torch.sin(t) / t)
    b = torch.where(small, 0.5 - t2 / 24, (1 - torch.cos(t)) / t2s)
    x, y, z = r.unbind(-1)
    o = torch.zeros_like(x)
    K = torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*r.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    return eye + a * K + b * (K @ K)


class TorchBody:
    def __init__(self, model, device="cpu", dtype=torch.float64):
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
        self.m = model
        self.v_template = f(model.v_template)
        self.shapedirs = f(model.shapedirs)
        self.posedirs = f(model.posedirs)
        self.J_regressor = f(model.J_regressor)
        self.weights = f(model.weights)
        self.parents = model.parents.tolist()
        self.faces = torch.as_tensor(model.faces, device=device)
        self.dtype, self.device = dtype, device

    def shaped(self, betas):
        v = self.v_template + torch.einsum("vcb,b->vc", self.shapedirs, betas)
        return v, self.J_regressor @ v

    def vertex_normals(self, v):
        f = self.faces
        fn = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=-1)
        n = torch.zeros_like(v).index_add(0, f[:, 0], fn).index_add(0, f[:, 1], fn).index_add(0, f[:, 2], fn)
        return n / (n.norm(dim=-1, keepdim=True) + 1e-12)

    def fk(self, R, J_rest):
        """R: (F,J,3,3). Returns Rw (F,J,3,3), c (F,J,3)."""
        Rws, cs = [R[:, 0]], [J_rest[0].expand(R.shape[0], 3)]
        for j in range(1, len(self.parents)):
            p = self.parents[j]
            Rws.append(Rws[p] @ R[:, j])
            cs.append((Rws[p] @ (J_rest[j] - J_rest[p]))[..., ] + cs[p])
        return torch.stack(Rws, 1), torch.stack(cs, 1)

    def skin(self, R, trans, pts_rest, W, PD, J_rest):
        """Skin arbitrary rest-space points. R (F,J,3,3), trans (F,3), pts_rest (N,3), W (N,J), PD (P,N,3)."""
        Rw, c = self.fk(R, J_rest)
        eye = torch.eye(3, dtype=R.dtype, device=R.device)
        pf = (R[:, 1:] - eye).reshape(R.shape[0], -1)
        p = pts_rest[None] + torch.einsum("fp,pnc->fnc", pf, PD)
        At = c - torch.einsum("fjab,jb->fja", Rw, J_rest)
        TR = torch.einsum("nj,fjab->fnab", W, Rw)
        Tt = torch.einsum("nj,fja->fna", W, At)
        return torch.einsum("fnab,fnb->fna", TR, p) + Tt + trans[:, None]

    def vertices(self, R, trans, betas):
        v, J = self.shaped(betas)
        return self.skin(R, trans, v, self.weights, self.posedirs, J), v, J

"""Numerical validation of every claim made in docs/harmonic_sheaves.md.

    .venv/bin/python scripts/verify_harmonic_sheaves.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.data.dataset_2d import SinogramCompletionDataset

cfg = DBTGeometryConfig(); geom = DBTGeometry.from_config(cfg)
th = geom.angles; V = len(th); acq = geom.acquired_view_mask

def R(a): return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def sheaf_laplacian(m, edges, w=None):
    """L_F = delta^T W delta with F_{v <| e} = R(-m theta_v); stalks R^2."""
    E = len(edges); d = np.zeros((2 * E, 2 * V))
    for e, (u, v) in enumerate(edges):
        d[2*e:2*e+2, 2*u:2*u+2] = -R(-m * th[u]); d[2*e:2*e+2, 2*v:2*v+2] = R(-m * th[v])
    W = np.eye(2 * E) if w is None else np.kron(np.diag(w), np.eye(2))
    return d.T @ W @ d

band = [(i, j) for i in range(V) for j in range(i + 1, min(i + 7, V))]   # kNN k = 12 graph of the models
for m in range(4):
    L = sheaf_laplacian(m, band); ev, U = np.linalg.eigh(L)
    ker = U[:, ev < 1e-9]
    # predicted global sections: x_v = R(m theta_v) c
    basis = np.stack([np.concatenate([R(m * t) @ c for t in th]) for c in (np.array([1., 0]), np.array([0, 1.]))], 1)
    resid = np.linalg.norm(basis - ker @ (ker.T @ basis))
    print(f"m={m}: dim ker L_F = {ker.shape[1]}, ||sections - proj_ker|| = {resid:.1e}, lambda_3 = {ev[2]:.3e}")

# Block of L_F and link with the SNN operator: off-diagonal block (u,v) must be -R(m(theta_u - theta_v))
L1 = sheaf_laplacian(1, band); u, v = 90, 93
print("offdiag block == -R(theta_u - theta_v):", np.allclose(L1[2*u:2*u+2, 2*v:2*v+2], -R(th[u] - th[v])))

# Dirichlet energy == sum over edges ||x_u - R(m(theta_u - theta_v)) x_v||^2
x = np.random.default_rng(0).normal(size=2 * V)
direct = sum(np.linalg.norm(x[2*a:2*a+2] - R(th[a] - th[b]) @ x[2*b:2*b+2])**2 for a, b in band)
print("x^T L x == edge-wise energy:", np.isclose(x @ L1 @ x, direct))

# ---- HLCC moments of real ground-truth sinograms are harmonic; harmonic extension from +-25 deg ----
ds_ = SinogramCompletionDataset(20, split="val", noise_level=0.0, geometry_config=cfg)
s = (np.arange(cfg.det_col_count) - cfg.det_col_count / 2 + 0.5) * cfg.det_pixel_size_mm
S0 = 100.0  # length scale to keep the moments O(1)
def moments(sino, k): return (sino * (s / S0) ** k).sum(1) * cfg.det_pixel_size_mm

def design(k, t):  # trigonometric polynomials of degree <= k with the parity of k
    cols = []
    for m in range(k % 2, k + 1, 2):
        cols += [np.ones_like(t)] if m == 0 else [np.cos(m * t), np.sin(m * t)]
    return np.stack(cols, 1)

print("\n k | rel. residual of GT moment outside the HL space | harmonic-extension error on the wedge | cond")
for k in range(0, 9):
    res, ext = [], []
    for i in range(20):
        full = ds_[i][1][0].numpy() * cfg.sino_norm
        M = moments(full, k); A = design(k, th)
        res.append(np.linalg.norm(M - A @ np.linalg.lstsq(A, M, rcond=None)[0]) / np.linalg.norm(M))
        coef = np.linalg.lstsq(A[acq], M[acq], rcond=None)[0]            # fit on acquired views only
        ext.append(np.linalg.norm((A @ coef - M)[~acq]) / np.linalg.norm(M[~acq]))
    print(f" {k} | {np.median(res):.2e} | {np.median(ext):.2e} | {np.linalg.cond(design(k, th)[acq]):.1e}")

# Harmonic extension through the sheaf Laplacian == least squares on sections (k = 1, m = 1)
full = ds_[0][1][0].numpy() * cfg.sino_norm; M1 = moments(full, 1)
# lift M1 to a section of the m=1 sheaf: x_v = (M1(theta_v), quadrature) = R(theta_v) c  with M1 = c . (cos, sin)...
c = np.linalg.lstsq(design(1, th), M1, rcond=None)[0]                     # M1 = a cos + b sin
xsec = np.concatenate([R(t) @ np.array([c[0], -c[1]]) for t in th])       # first component = a cos t + b sin t
print("\nfirst component of the section reproduces M1:", np.allclose(xsec[0::2], design(1, th) @ c))
idxA = np.flatnonzero(np.repeat(acq, 2)); idxU = np.flatnonzero(np.repeat(~acq, 2))
xU = np.linalg.solve(L1[np.ix_(idxU, idxU)], -L1[np.ix_(idxU, idxA)] @ xsec[idxA])
print("harmonic extension (solve L_UU x_U = -L_UA x_A) recovers the section on the wedge: rel err =",
      np.linalg.norm(xU - xsec[idxU]) / np.linalg.norm(xsec[idxU]))
print("same with R = I (constant sheaf / GCN smoothing): rel err =", end=" ")
L0 = sheaf_laplacian(0, band)
xU0 = np.linalg.solve(L0[np.ix_(idxU, idxU)], -L0[np.ix_(idxU, idxA)] @ xsec[idxA])
print(np.linalg.norm(xU0 - xsec[idxU]) / np.linalg.norm(xsec[idxU]))

# ---- Closing the circle: the HL parity rule (m = k mod 2) is a holonomy condition ----
# theta -> theta + pi flips the detector: p(theta + pi, s) = p(theta, -s), hence M_k(theta + pi) = (-1)^k M_k(theta).
# On the closing edge (view V-1 at 89 deg -> view 0 seen at -90 + 180 deg) the fibre of the frequency-m sheaf
# turns by R(m pi) = (-1)^m and the order-k moment picks up (-1)^k. Non-zero global sections need (-1)^(k+m) = +1.
print("\nClosing the view path into a circle: dim H^0 of the frequency-m sheaf carrying an order-k moment")
path = [(i, i + 1) for i in range(V - 1)]
for k_parity, label in ((0, "k even"), (1, "k odd ")):
    dims = []
    for m in range(4):
        E = len(path) + 1; d = np.zeros((2 * E, 2 * V))
        for e, (u, v) in enumerate(path):
            d[2*e:2*e+2, 2*u:2*u+2] = -R(-m * th[u]); d[2*e:2*e+2, 2*v:2*v+2] = R(-m * th[v])
        d[-2:, 2*(V-1):2*V] = -R(-m * th[V-1])
        d[-2:, 0:2] = (-1) ** k_parity * R(-m * (th[0] + np.pi))
        dims.append(int((np.linalg.eigvalsh(d.T @ d) < 1e-9).sum()))
    print(f"  {label}: " + ", ".join(f"m={m}: {dim}" for m, dim in enumerate(dims)))

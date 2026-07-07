#!/usr/bin/env python3
"""E3 -- analytic Beltrami mode-sum: exact transversality and the N^-1/2 residual.

u(x) = sum_j a_j ( e1_j cos(phi_j) - s_j e2_j sin(phi_j) ),  phi_j = k_j.x + theta_j,
each term analytically transverse to its own k_j (so div u = 0 pointwise) and Beltrami.
Measures:
  (i)   analytic transversality max_j |k_j . e_{1,2}| and the O(h^2) FD-divergence decay
  (ii)  rho(p) for a sweep, N=40 modes, measured on a grid
  (iii) the p=0 net-helicity residual and its ~1/sqrt(N) scaling
"""
import numpy as np


def frame(khat):
    ref = np.array([0.0, 0.0, 1.0]) if abs(khat[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, khat); e1 /= np.linalg.norm(e1)
    e2 = np.cross(khat, e1)
    return e1, e2


def make_modes(N, p, slope, rng):
    modes = []
    for _ in range(N):
        z = 2 * rng.random() - 1
        th = 2 * np.pi * rng.random()
        d = np.array([np.sqrt(1 - z*z)*np.cos(th), np.sqrt(1 - z*z)*np.sin(th), z])
        km = 1.0 + 5.2 * rng.random()
        k = km * d
        e1, e2 = frame(d)
        s = 1.0 if rng.random() < (1 + p) / 2 else -1.0
        a = km ** (-slope)
        ph = 2 * np.pi * rng.random()
        modes.append((k, km, e1, e2, s, a, ph))
    return modes


def eval_uw(modes, X):
    """velocity u and vorticity w (curl u) at points X (M,3)."""
    u = np.zeros_like(X); w = np.zeros_like(X)
    for k, km, e1, e2, s, a, ph in modes:
        phi = X @ k + ph
        c, sn = np.cos(phi), np.sin(phi)
        term = a * (np.outer(c, e1) - s * np.outer(sn, e2))
        u += term
        w += s * km * term          # curl of a Beltrami mode = s|k| * mode
    return u, w


def rho_on_grid(modes, ng=16):
    g = np.linspace(0, 2*np.pi, ng, endpoint=False)
    X = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T
    u, w = eval_uw(modes, X)
    H = np.sum(u * w)
    return H / (np.sqrt(np.sum(u*u)) * np.sqrt(np.sum(w*w)))


def transversality(modes):
    return max(max(abs(k @ e1), abs(k @ e2)) for k, km, e1, e2, s, a, ph in modes)


def fd_div_rms(modes, h, M=4000, seed=7):
    """Central-difference divergence at random points with explicit step h.
    The modes are analytically div-free, so this residual is pure O(h^2) truncation."""
    rng = np.random.default_rng(seed)
    X = rng.random((M, 3)) * 2 * np.pi
    div = np.zeros(M)
    for c in range(3):
        e = np.zeros(3); e[c] = 1.0
        up, _ = eval_uw(modes, X + h * e)
        um, _ = eval_uw(modes, X - h * e)
        div += (up[:, c] - um[:, c]) / (2 * h)
    return np.sqrt(np.mean(div ** 2))


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    print("E3 analytic Beltrami mode-sum  (seed 0)")

    modes = make_modes(40, 0.5, 1.0, np.random.default_rng(1))
    print(f"  (i) analytic transversality max|k.e| = {transversality(modes):.2e}")
    print("      FD divergence rms vs step h (O(h^2) decay):")
    prev = None
    for h in (0.1, 0.05, 0.025, 0.0125):
        d = fd_div_rms(modes, h)
        ratio = f"  ratio {prev/d:5.2f}" if prev else ""
        print(f"        h={h:6.4f}: rms div = {d:.3e}{ratio}")
        prev = d

    print("  (ii) rho(p), N=40 modes (grid-measured):")
    for p in (-1.0, -0.5, 0.0, 0.5, 1.0):
        modes = make_modes(40, p, 1.0, np.random.default_rng(100 + int(10*p)))
        print(f"        p={p:+.1f}: rho = {rho_on_grid(modes):+.3f}")

    print("  (iii) p=0 net-helicity residual vs N (|k|-weighted, 200-draw rms):")
    for Nm in (20, 40, 80, 160, 320):
        vals = []
        for e in range(200):
            r = np.random.default_rng(1000 + e)
            modes = make_modes(Nm, 0.0, 1.0, r)
            num = sum(s * km * a * a for k, km, e1, e2, s, a, ph in modes)
            den = sum(km * a * a for k, km, e1, e2, s, a, ph in modes)
            vals.append(num / den)
        rms = np.sqrt(np.mean(np.array(vals)**2))
        print(f"        N={Nm:3d}: rms rho = {rms:.3f},  rms*sqrt(N) = {rms*np.sqrt(Nm):.3f}")
    Ns = np.array([20, 40, 80, 160, 320])
    rmss = []
    for Nm in Ns:
        vals = []
        for e in range(200):
            r = np.random.default_rng(1000 + e)
            modes = make_modes(Nm, 0.0, 1.0, r)
            num = sum(s*km*a*a for k, km, e1, e2, s, a, ph in modes)
            den = sum(km*a*a for k, km, e1, e2, s, a, ph in modes)
            vals.append(num/den)
        rmss.append(np.sqrt(np.mean(np.array(vals)**2)))
    b = np.polyfit(np.log(Ns), np.log(rmss), 1)[0]
    print(f"        fitted exponent = {b:.3f}  (theory -0.5)")

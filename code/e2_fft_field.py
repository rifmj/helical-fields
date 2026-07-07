#!/usr/bin/env python3
"""E2 -- FFT helical field: real, divergence-free, helicity linear in p.

Builds a divergence-free helical velocity field on an N^3 periodic grid,
enforcing Hermitian symmetry so the inverse FFT is exactly real, and measures
for a sweep of the helicity parameter p:
  (i)  spectral divergence   max |k . u_hat| / max |u_hat|
  (ii) reality               max |Im u| after the inverse FFT
  (iii) relative helicity     rho = <u.w> / (||u|| ||w||),  w = curl u (spectral)
Prints the p-sweep table and a linear-fit residual for rho(p).
"""
import numpy as np


def frame(khat):
    ref = np.array([0.0, 0.0, 1.0]) if abs(khat[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, khat); e1 /= np.linalg.norm(e1)
    e2 = np.cross(khat, e1)
    return e1, e2


def build(N, p, slope, seed):
    rng = np.random.default_rng(seed)
    freqs = np.fft.fftfreq(N, d=1.0 / N).astype(int)          # 0,1,...,N/2-1,-N/2,...,-1
    kc2 = (0.30 * N) ** 2
    uhat = np.zeros((3, N, N, N), dtype=complex)
    nyq = -N // 2

    def mirror(i): return (-i) % N

    for iz in range(N):
        for iy in range(N):
            for ix in range(N):
                kx, ky, kz = freqs[ix], freqs[iy], freqs[iz]
                if kx == nyq or ky == nyq or kz == nyq:       # drop Nyquist planes
                    continue
                k2 = kx * kx + ky * ky + kz * kz
                if k2 == 0:
                    continue
                mi = (mirror(ix), mirror(iy), mirror(iz))
                # canonical half-space: assign only when this index precedes its mirror
                if (iz, iy, ix) >= (mi[2], mi[1], mi[0]):
                    continue
                k = np.sqrt(k2)
                khat = np.array([kx, ky, kz]) / k
                e1, e2 = frame(khat)
                hp = (e1 + 1j * e2) / np.sqrt(2)
                hm = (e1 - 1j * e2) / np.sqrt(2)
                env = k ** (-slope) * np.exp(-k2 / kc2)
                ap = np.sqrt((1 + p) / 2) * env * np.exp(1j * 2 * np.pi * rng.random())
                am = np.sqrt((1 - p) / 2) * env * np.exp(1j * 2 * np.pi * rng.random())
                u = ap * hp + am * hm
                uhat[:, iz, iy, ix] = u
                uhat[:, mi[2], mi[1], mi[0]] = np.conj(u)      # Hermitian mirror
    return uhat, freqs


def measure(uhat, freqs, N):
    KX = freqs[None, None, :]
    KY = freqs[None, :, None]
    KZ = freqs[:, None, None]
    # spectral divergence
    div = 1j * (KX * uhat[0] + KY * uhat[1] + KZ * uhat[2])
    umag = np.sqrt(np.abs(uhat[0])**2 + np.abs(uhat[1])**2 + np.abs(uhat[2])**2)
    reldiv = np.max(np.abs(div)) / max(np.max(umag), 1e-300)
    # vorticity in spectral space: w_hat = i k x u_hat
    what = np.empty_like(uhat)
    what[0] = 1j * (KY * uhat[2] - KZ * uhat[1])
    what[1] = 1j * (KZ * uhat[0] - KX * uhat[2])
    what[2] = 1j * (KX * uhat[1] - KY * uhat[0])
    u = np.array([np.fft.ifftn(uhat[c]) for c in range(3)])
    w = np.array([np.fft.ifftn(what[c]) for c in range(3)])
    im = np.max(np.abs(u.imag))
    ur, wr = u.real, w.real
    H = np.sum(ur * wr)
    nu = np.sqrt(np.sum(ur * ur)); nw = np.sqrt(np.sum(wr * wr))
    rho = H / (nu * nw)
    return reldiv, im, rho


if __name__ == '__main__':
    N, slope, seed = 32, 1.6, 0
    ps = [-1.0, -0.5, 0.0, 0.5, 1.0]
    print(f"E2 FFT helical field  N={N}^3  slope={slope}  seed={seed}")
    print(f"  {'p':>5}  {'max|k.u|/|u|':>14}  {'max|Im u|':>11}  {'rho':>9}")
    rows = []
    for p in ps:
        uhat, freqs = build(N, p, slope, seed)
        reldiv, im, rho = measure(uhat, freqs, N)
        rows.append((p, rho))
        print(f"  {p:+5.1f}  {reldiv:14.2e}  {im:11.2e}  {rho:+9.4f}")
    P = np.array([r[0] for r in rows]); R = np.array([r[1] for r in rows])
    a, b = np.polyfit(P, R, 1)
    print(f"  linear fit rho = {a:.5f}*p + {b:.1e};  max deviation from line = "
          f"{np.max(np.abs(R - (a*P+b))):.1e}")
    print("  slope-dependence of rho(+1):")
    for s in (1.2, 1.6, 2.0):
        uhat, freqs = build(N, 1.0, s, seed)
        _, _, rho = measure(uhat, freqs, N)
        print(f"    slope={s}: rho(+1) = {rho:+.4f}")

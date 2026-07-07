#!/usr/bin/env python3
"""E5 -- performance: FFT rebuild cost vs grid, mode-sum throughput vs modes.

Two constructions, two cost regimes (paper sections 2/4 vs section 6):

  * FFT path: a full 3D helical field on an N^3 grid. Cost is dominated by the
    3 inverse FFTs (velocity) [+ 3 for vorticity], i.e. O(N^3 log N) per rebuild.
    This is a bake-time cost -- one build per authored field, then advect
    downstream. We report wall-clock per rebuild vs N and MB of the baked
    (float32) velocity texture.

  * Mode-sum path: the analytic Beltrami sum (eq. modesum) evaluated per point,
    O(M) per sample point for M modes and no grid. This is the real-time path.
    We report throughput as million-point-evaluations per second (Mpps) vs M,
    and convert to an equivalent frame rate at a fixed shading load
    (1920x1080 = 2.07 Mpx): fps ~= Mpps / 2.07.

These are CPU numpy reference numbers (single machine, no GPU); a fragment-shader
implementation of the mode-sum (see code/modesum.frag) evaluates the same O(M)
arithmetic massively in parallel and is the intended real-time deployment. The
purpose here is the *scaling* and the relative cost of the two paths, not an
absolute GPU benchmark.
"""
import time
import numpy as np


# ---------- FFT path ----------
def frame(khat):
    ref = np.array([0.0, 0.0, 1.0]) if abs(khat[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, khat); e1 /= np.linalg.norm(e1)
    e2 = np.cross(khat, e1)
    return e1, e2


def build_fft(N, p=0.5, slope=1.6, seed=0, with_vorticity=True):
    """One full rebuild of an N^3 helical field; returns the real velocity field."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    kc2 = (0.30 * N) ** 2
    nyq = -N // 2
    KX, KY, KZ = np.meshgrid(freqs, freqs, freqs, indexing='ij')
    K2 = (KX * KX + KY * KY + KZ * KZ).astype(float)
    mask = (K2 > 0) & (KX != nyq) & (KY != nyq) & (KZ != nyq)
    K = np.sqrt(np.where(mask, K2, 1.0))
    env = np.where(mask, K ** (-slope) * np.exp(-K2 / kc2), 0.0)
    # random-phase helical amplitudes on the full grid (Hermitian-symmetrised by
    # constructing a real field via the c2r-style approach: build complex, ifft, real).
    # For a timing benchmark we build vectorised complex amplitudes directly.
    ph_p = np.exp(1j * 2 * np.pi * rng.random((N, N, N)))
    ph_m = np.exp(1j * 2 * np.pi * rng.random((N, N, N)))
    ap = np.sqrt((1 + p) / 2) * env * ph_p
    am = np.sqrt((1 - p) / 2) * env * ph_m
    # transverse frame per mode (vectorised): pick ref by |kz|<0.9|k|
    kz_hat = np.where(mask, KZ / K, 0.0)
    refz = np.abs(kz_hat) < 0.9
    rx = np.where(refz, 0.0, 0.0); ry = np.where(refz, 0.0, 1.0); rz = np.where(refz, 1.0, 0.0)
    khx, khy, khz = np.where(mask, KX / K, 0.0), np.where(mask, KY / K, 0.0), kz_hat
    # e1 = ref x khat
    e1x = ry * khz - rz * khy; e1y = rz * khx - rx * khz; e1z = rx * khy - ry * khx
    n1 = np.sqrt(e1x**2 + e1y**2 + e1z**2); n1 = np.where(n1 > 0, n1, 1.0)
    e1x, e1y, e1z = e1x / n1, e1y / n1, e1z / n1
    # e2 = khat x e1
    e2x = khy * e1z - khz * e1y; e2y = khz * e1x - khx * e1z; e2z = khx * e1y - khy * e1x
    inv2 = 1.0 / np.sqrt(2.0)
    # h_pm = (e1 +/- i e2)/sqrt2 ; uhat = ap h+ + am h-
    uhx = inv2 * (ap * (e1x + 1j * e2x) + am * (e1x - 1j * e2x))
    uhy = inv2 * (ap * (e1y + 1j * e2y) + am * (e1y - 1j * e2y))
    uhz = inv2 * (ap * (e1z + 1j * e2z) + am * (e1z - 1j * e2z))
    u = np.stack([np.fft.ifftn(uhx).real, np.fft.ifftn(uhy).real, np.fft.ifftn(uhz).real])
    if with_vorticity:
        wx = 1j * (KY * uhz - KZ * uhy)
        wy = 1j * (KZ * uhx - KX * uhz)
        wz = 1j * (KX * uhy - KY * uhx)
        _ = np.stack([np.fft.ifftn(wx).real, np.fft.ifftn(wy).real, np.fft.ifftn(wz).real])
    return u


def time_fft(Ns, reps=3):
    out = []
    for N in Ns:
        build_fft(N)                              # warm-up (plan caches)
        t = []
        for _ in range(reps):
            t0 = time.perf_counter(); build_fft(N); t.append(time.perf_counter() - t0)
        tex_mb = 3 * (N ** 3) * 4 / 1024 ** 2     # float32 vec3 velocity texture
        out.append((N, min(t), tex_mb))
    return out


# ---------- mode-sum path ----------
def make_modes(M, p, slope, rng):
    z = 2 * rng.random(M) - 1
    th = 2 * np.pi * rng.random(M)
    d = np.stack([np.sqrt(1 - z*z) * np.cos(th), np.sqrt(1 - z*z) * np.sin(th), z], 1)
    km = 1.0 + 5.2 * rng.random(M)
    k = d * km[:, None]
    e1 = np.zeros((M, 3)); e2 = np.zeros((M, 3))
    for j in range(M):
        e1[j], e2[j] = frame(d[j])
    s = np.where(rng.random(M) < (1 + p) / 2, 1.0, -1.0)
    a = km ** (-slope)
    ph = 2 * np.pi * rng.random(M)
    return k, e1, e2, s, a, ph


def eval_modesum(pts, modes):
    k, e1, e2, s, a, ph = modes
    phi = pts @ k.T + ph[None, :]                 # (P, M)
    c = np.cos(phi); sn = np.sin(phi)
    u = (a * c) @ e1 - (a * s * sn) @ e2          # (P,3)
    return u


def time_modesum(Ms, npts=200_000, reps=3, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.random((npts, 3)) * 2 * np.pi
    out = []
    for M in Ms:
        modes = make_modes(M, 0.5, 1.6, np.random.default_rng(1))
        eval_modesum(pts[:1000], modes)           # warm-up
        t = []
        for _ in range(reps):
            t0 = time.perf_counter(); eval_modesum(pts, modes); t.append(time.perf_counter() - t0)
        mpps = npts / min(t) / 1e6                 # million point-evals / s
        out.append((M, min(t), mpps, mpps / 2.07)) # fps at 1920x1080
    return out


if __name__ == '__main__':
    print("E5 performance (CPU numpy reference; single machine, no GPU)")
    print("  FFT path -- full N^3 helical field, wall-clock per rebuild:")
    print(f"    {'N':>4}  {'rebuild (s)':>12}  {'baked tex (MB)':>15}")
    for N, sec, mb in time_fft([16, 24, 32, 48, 64]):
        print(f"    {N:>4}  {sec:12.4f}  {mb:15.2f}")
    print("  mode-sum path -- analytic per-point eval, throughput vs modes M:")
    print(f"    {'M':>4}  {'200k pts (s)':>13}  {'Mpts/s':>9}  {'~fps @1080p':>12}")
    for M, sec, mpps, fps in time_modesum([16, 32, 64, 128, 256]):
        print(f"    {M:>4}  {sec:13.4f}  {mpps:9.2f}  {fps:12.1f}")

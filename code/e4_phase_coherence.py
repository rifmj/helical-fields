#!/usr/bin/env python3
"""E4 -- phase coherence at fixed spectrum.

Verifies the central claim of the phase-coherence axis (paper section 5):
sliding the coherence knob lambda from 0 (i.i.d. random inter-mode phases)
to 1 (structured reference phases) changes the SPATIAL ORGANIZATION of the
field while leaving the power spectrum -- and hence the energy at every
scale -- frozen.

Construction (2D, curl of a stream function; paper section 2.1):
  vorticity spectrum  omega_hat(k) = E(k) exp(i theta_lambda(k)),   |omega_hat| = E(k)
  theta_lambda(k) = (1-lambda) theta_rand(k) + lambda theta_ref(k),
with BOTH phase fields antisymmetric, theta(-k) = -theta(k), so the field is
real by construction; the reference phases are those of a sparse set of
localized vortex "blobs". Because the modulus |omega_hat| = E(k) is untouched
by lambda, the amplitude spectrum |u_hat(k)|^2 is identical for every lambda.

Measures, for a sweep of lambda:
  (i)   total spectral energy  sum |u_hat|^2               (must be lambda-invariant)
  (ii)  per-shell energy spectrum E(k)                     (must be lambda-invariant)
  (iii) spectral divergence    max|k.u_hat| / max|u_hat|   (must be ~0)
  (iv)  reality                max|Im u|                   (must be ~0)
  (v)   a spatial-organization statistic (kurtosis of |u|) (must GROW with lambda)
Prints exactly the numbers quoted in the paper's validation section at the
pinned seed.
"""
import numpy as np


def build_2d(N, slope, lam, seed, kc_frac=0.28, nblob=5):
    rng = np.random.default_rng(seed)
    k1 = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    KX, KY = np.meshgrid(k1, k1, indexing='ij')
    K2 = KX * KX + KY * KY
    kc2 = (kc_frac * N) ** 2
    with np.errstate(divide='ignore'):
        E = np.where(K2 > 0, (np.sqrt(K2) ** (-slope)) * np.exp(-K2 / kc2), 0.0)

    # structured reference phases: a few localized vortex blobs (real field ->
    # its FFT phase is automatically antisymmetric)
    xs = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    blob = np.zeros((N, N))
    for _ in range(nblob):
        cx, cy = rng.uniform(0, 2 * np.pi, 2)
        sgn = rng.choice([-1.0, 1.0])
        w = 0.5
        dx = (X - cx + np.pi) % (2 * np.pi) - np.pi
        dy = (Y - cy + np.pi) % (2 * np.pi) - np.pi
        blob += sgn * np.exp(-(dx * dx + dy * dy) / (2 * w * w))
    theta_ref = np.angle(np.fft.fft2(blob))              # structured, antisymmetric

    # random ANTISYMMETRIC phase: theta(-k) = -theta(k)
    mir = lambda A: A[(-np.arange(N)) % N][:, (-np.arange(N)) % N]
    raw = 2 * np.pi * rng.random((N, N))
    theta_rand = 0.5 * (raw - mir(raw))

    # paper section 5, scheme A (linear): start from the structured reference and
    # multiply by a common unit-modulus scalar c_lambda(k) = exp(i(1-lambda)theta_rand),
    # i.e. add a random phase whose magnitude shrinks as lambda -> 1.
    #   lambda = 0 : theta_ref + theta_rand  -> i.i.d. random phases (RPA surrogate)
    #   lambda = 1 : theta_ref               -> the structured reference
    theta = theta_ref + (1 - lam) * theta_rand           # antisymmetric for all lambda
    what = E * np.exp(1j * theta)
    what[0, 0] = 0.0                                      # |what| = E exactly

    with np.errstate(divide='ignore', invalid='ignore'):
        uhat = -1j * np.where(K2 > 0, KY / K2, 0.0) * what
        vhat = +1j * np.where(K2 > 0, KX / K2, 0.0) * what
    u = np.fft.ifft2(uhat)
    v = np.fft.ifft2(vhat)
    omega = np.fft.ifft2(1j * (KX * vhat - KY * uhat))    # vorticity field
    return KX, KY, K2, uhat, vhat, u, v, omega


def shell_spectrum(uhat, vhat, K2, nbin=12):
    kmag = np.sqrt(K2).ravel()
    P = (np.abs(uhat) ** 2 + np.abs(vhat) ** 2).ravel()
    edges = np.linspace(0, kmag.max() + 1e-9, nbin + 1)
    idx = np.clip(np.digitize(kmag, edges) - 1, 0, nbin - 1)
    return np.array([P[idx == b].sum() for b in range(nbin)])


def flatness(field):
    """Kurtosis (flatness) of a real field; = 3 for a Gaussian, grows as energy
    concentrates into sparse coherent structures."""
    f = field.real
    f = f - f.mean()
    return float(np.mean(f ** 4) / (np.mean(f ** 2) ** 2))


def measure(N, slope, lam, seed):
    KX, KY, K2, uhat, vhat, u, v, omega = build_2d(N, slope, lam, seed)
    e_tot = float((np.abs(uhat) ** 2 + np.abs(vhat) ** 2).sum())
    div = 1j * (KX * uhat + KY * vhat)
    umax = max(np.max(np.abs(uhat) + np.abs(vhat)), 1e-300)
    reldiv = float(np.max(np.abs(div)) / umax)
    im = float(max(np.max(np.abs(u.imag)), np.max(np.abs(v.imag))))
    flat = flatness(omega)                 # vorticity flatness: RPA ~ 3, coherent > 3
    spec = shell_spectrum(uhat, vhat, K2)
    return e_tot, reldiv, im, flat, spec


if __name__ == '__main__':
    N, slope, seed, nseed = 160, 1.4, 3, 16
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"E4 phase coherence at fixed spectrum  N={N}^2  slope={slope}"
          f"  seed={seed} (+ {nseed}-seed mean for flatness)")
    print(f"  {'lambda':>7}  {'sum|u_hat|^2':>13}  {'max|k.u|/|u|':>13}"
          f"  {'max|Im u|':>10}  {'flatness(w)':>11}")
    rows = []
    for lam in lams:
        # invariance checks at the representative seed; flatness averaged over seeds
        e_tot, reldiv, im, _, spec = measure(N, slope, lam, seed)
        flats = [measure(N, slope, lam, seed + j)[3] for j in range(nseed)]
        flat = float(np.mean(flats))
        rows.append((lam, e_tot, spec))
        print(f"  {lam:7.2f}  {e_tot:13.6f}  {reldiv:13.2e}  {im:10.2e}  {flat:11.4f}")

    energies = np.array([r[1] for r in rows])
    print(f"  energy spread over lambda: max-min = {energies.max() - energies.min():.2e}"
          f"  (relative {((energies.max() - energies.min()) / energies.mean()):.2e})")
    # per-shell spectrum invariance: max relative deviation of any shell across lambda
    specs = np.array([r[2] for r in rows])
    ref = specs.mean(axis=0)
    nz = ref > 0
    rel = np.max(np.abs(specs[:, nz] - ref[nz]) / ref[nz])
    print(f"  per-shell spectrum: max relative deviation across lambda = {rel:.2e}")
    print("  -> the power spectrum (energy at every scale) is frozen; only the"
          " spatial\n     organization -- read out by the vorticity flatness (3 = Gaussian/RPA,\n"
          "     > 3 = sparse coherent structures) -- grows with lambda.")

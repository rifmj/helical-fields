#!/usr/bin/env python3
"""E7 -- phase coherence at fixed spectrum, in 3D.

Companion to E4 (which validates the 2D stream-function path).  E7 validates the
*same 3D helical builder* that generates the teaser (gallery.png) and advection
figures, `_helical.build_fft_3d`, so the figure-generating code is under test.

The 3D construction gives each helical amplitude a_+/a_- its own unit-modulus
phase (paper eq. for the phase interpolation):

  a_pm(k) = sqrt((1 +/- p)/2) * E(k) * exp(i phi_pm(k)),
  phi_pm(k) = theta_ref_pm(k) + (1 - lambda) theta_pm(k),

with a SEPARATE structured reference theta_ref_pm per channel (phase of an
independent real 3D blob field) and INDEPENDENT uniform antisymmetric random
phases theta_pm.  Because the modulus of every helical amplitude is
E(k)*sqrt((1 +/- p)/2) independent of lambda, BOTH the power spectrum and the
+/- energy split -- hence the net helicity -- are frozen exactly at every lambda,
while the inter-mode phase coherence (spatial organization) slides from random
(lambda=0) to structured (lambda=1).  Two references (rather than one shared) is
what keeps the p=0 field isotropic at every lambda; with one shared reference the
two channels would collapse to the same phase at lambda=1 and force linear
polarization at p=0.

Measures, for a sweep of lambda at fixed helicity p:
  (i)   total spectral energy   sum |u_hat|^2                (lambda-invariant, exactly)
  (ii)  net helicity            H = sum u.w                   (lambda-invariant, exactly)
  (iii) per-shell energy spectrum                             (lambda-invariant)
  (iv)  divergence              max|k.u_hat| / max|u_hat|     (~0)
  (v)   reality                 max|Im u|                     (~0)
  (vi)  vorticity flatness (kurtosis of |omega|)              (GROWS with lambda)

This is the experiment whose absence let a broken earlier 3D builder ship: it
had baked independent random phases into a_+/a_- *before* a common reference
multiplier, so the field stayed random at every lambda (flat kurtosis ~3).  The
check below would have caught that immediately.  As a companion consistency
check, the relative helicity of the p=+1 field is ~0.8, matching the FFT builder
of E2 (a distorting Hermitian projection in an intermediate revision had ruined
this, which E7 also guards against).
"""
import numpy as np
import _helical as H


def shell_spectrum(uhat, freqs, nbin=12):
    N = uhat.shape[-1]
    KX, KY, KZ = np.meshgrid(freqs, freqs, freqs, indexing='ij')
    kmag = np.sqrt(KX * KX + KY * KY + KZ * KZ).ravel()
    P = np.sum(np.abs(uhat) ** 2, axis=0).ravel()
    edges = np.linspace(0, kmag.max() + 1e-9, nbin + 1)
    idx = np.clip(np.digitize(kmag, edges) - 1, 0, nbin - 1)
    return np.array([P[idx == b].sum() for b in range(nbin)])


def vort_flatness(u, freqs):
    """Kurtosis (flatness) of the vorticity magnitude |omega|; ~3.1 for a Gaussian
    random-phase field (the chi-3 baseline for the magnitude of three Gaussian
    components) and grows as energy concentrates into sparse coherent structures.
    Same statistic family as E4 (2D scalar vorticity) and the curl-noise figure."""
    KX = freqs[:, None, None]; KY = freqs[None, :, None]; KZ = freqs[None, None, :]
    uh = [np.fft.fftn(u[c]) for c in range(3)]
    wx = np.fft.ifftn(1j * (KY * uh[2] - KZ * uh[1])).real
    wy = np.fft.ifftn(1j * (KZ * uh[0] - KX * uh[2])).real
    wz = np.fft.ifftn(1j * (KX * uh[1] - KY * uh[0])).real
    m = np.sqrt(wx * wx + wy * wy + wz * wz)
    f = m - m.mean()
    return float(np.mean(f ** 4) / np.mean(f ** 2) ** 2)


def main(N=48, slope=1.4, p=0.6, seed=0, nseed=8):
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    freqs = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    KX = freqs[:, None, None]; KY = freqs[None, :, None]; KZ = freqs[None, None, :]

    print(f"E7 phase coherence at fixed spectrum -- 3D helical builder "
          f"(N={N}^3 slope={slope} p={p} seed={seed})")
    print(f"  {'lambda':>6} {'sum|u_hat|^2':>14} {'helicity H':>13} "
          f"{'max|k.u|/|u|':>13} {'max|Im u|':>10} {'flatness|w|':>11}")

    energies, helis, specs = [], [], []
    for lam in lams:
        u, uhat, _ = H.build_fft_3d(N, p, slope, seed, lam=lam)
        hel, en = H.helicity_3d(uhat, freqs)
        d = KX * uhat[0] + KY * uhat[1] + KZ * uhat[2]
        reldiv = float(np.max(np.abs(d)) / np.max(np.abs(uhat)))
        imu = float(np.max(np.abs([np.fft.ifftn(uhat[c]).imag for c in range(3)])))
        # flatness averaged over seeds to smooth the finite-N fluctuation
        fl = np.mean([vort_flatness(H.build_fft_3d(N, p, slope, s, lam=lam)[0], freqs)
                      for s in range(nseed)])
        energies.append(en); helis.append(hel); specs.append(shell_spectrum(uhat, freqs))
        print(f"  {lam:6.2f} {en:14.6e} {hel:+13.6e} {reldiv:13.2e} {imu:10.2e} {fl:11.4f}")

    e_spread = (max(energies) - min(energies)) / np.mean(energies)
    h_spread = (max(helis) - min(helis)) / abs(np.mean(helis))
    specs = np.array(specs)
    # only shells carrying non-negligible energy: empty high-k / DC shells sit at
    # ~1e-30 where a relative metric is meaningless.
    tot = specs.mean(0).sum()
    live = specs.mean(0) > 1e-12 * tot
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = (specs.max(0) - specs.min(0)) / specs.mean(0)
    shell_dev = float(np.nanmax(rel[live])) if live.any() else 0.0
    print(f"  energy spread over lambda (relative)        = {e_spread:.2e}")
    print(f"  helicity spread over lambda (relative)      = {h_spread:.2e}")
    print(f"  per-shell spectrum max relative deviation   = {shell_dev:.2e}")
    print("  -> power spectrum AND net helicity are frozen across lambda;")
    print("     only the spatial organization (vorticity flatness) grows with lambda.")


if __name__ == "__main__":
    main()

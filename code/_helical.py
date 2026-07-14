#!/usr/bin/env python3
"""Shared field constructions for the figure and validation scripts.

Every figure in the paper is generated from one of the builders here, which are
the *same* constructions measured in the validation experiments e1..e6:

  * frame(khat)              -- Craya-Herring transverse frame (as in e2, e3, e5)
  * build_fft_3d(...)        -- 3D divergence-free helical field on an N^3 grid,
                                with the helicity split p (e2) AND the
                                phase-coherence multiplier lambda (paper sec. 5).
                                Used by gallery.png and advection.png.
  * build_2d(...)            -- 2D stream-function field with the phase-coherence
                                knob lambda (identical to e4). Used by slope.png,
                                curlnoise.png.
  * make_modes / eval_uw     -- analytic Beltrami mode-sum (identical to e3).
  * quintic ramp alpha       -- the Bridson free-slip quintic (as in e6).

Kept dependency-free apart from numpy so `make figures` needs nothing beyond
the reproduction requirements (numpy, matplotlib).
"""
import numpy as np


# ----------------------------------------------------------------------------
# transverse frame -- identical to e2/e3/e5
# ----------------------------------------------------------------------------
def frame(khat):
    ref = np.array([0.0, 0.0, 1.0]) if abs(khat[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, khat); e1 /= np.linalg.norm(e1)
    e2 = np.cross(khat, e1)
    return e1, e2


# ----------------------------------------------------------------------------
# 3D FFT helical field with helicity p and phase coherence lambda.
# The helicity split (e2) and the common-scalar phase-coherence multiplier
# (paper sec. 5) combined in one vectorised build.  The phase multiplier is
# common to the three vector components, so it cancels in the power spectrum and
# in the +/- energy split: energy and helicity bias are frozen across lambda.
# ----------------------------------------------------------------------------
def _ref_phase_3d(N, seed, nblob=10, width=1.2):
    """Structured reference phase from a few 3D localized vortex blobs; its FFT
    phase is automatically antisymmetric (the blob field is real)."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    blob = np.zeros((N, N, N))
    for _ in range(nblob):
        c = rng.uniform(0, 2 * np.pi, 3)
        sgn = rng.choice([-1.0, 1.0])
        dx = (X - c[0] + np.pi) % (2 * np.pi) - np.pi
        dy = (Y - c[1] + np.pi) % (2 * np.pi) - np.pi
        dz = (Z - c[2] + np.pi) % (2 * np.pi) - np.pi
        blob += sgn * np.exp(-(dx * dx + dy * dy + dz * dz) / (2 * width * width))
    return np.angle(np.fft.fftn(blob))


def build_fft_3d(N, p, slope, seed, lam=0.0, kc_frac=0.30, ref_seed=17):
    """Real, divergence-free 3D helical field. Returns (u, uhat, freqs) with u
    the real velocity of shape (3,N,N,N).  lam in [0,1] slides the inter-mode
    phases from random (lam=0) to the structured reference (lam=1) at fixed
    spectrum and fixed helicity bias p."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    kc2 = (kc_frac * N) ** 2
    nyq = -N // 2
    KX, KY, KZ = np.meshgrid(freqs, freqs, freqs, indexing='ij')
    K2 = (KX * KX + KY * KY + KZ * KZ).astype(float)
    mask = (K2 > 0) & (KX != nyq) & (KY != nyq) & (KZ != nyq)
    K = np.sqrt(np.where(mask, K2, 1.0))
    env = np.where(mask, K ** (-slope) * np.exp(-K2 / kc2), 0.0)

    # vectorised transverse frame per mode
    khx = np.where(mask, KX / K, 0.0); khy = np.where(mask, KY / K, 0.0)
    khz = np.where(mask, KZ / K, 0.0)
    refz = np.abs(khz) < 0.9
    rx = np.zeros_like(khx); ry = np.where(refz, 0.0, 1.0); rz = np.where(refz, 1.0, 0.0)
    e1x = ry * khz - rz * khy; e1y = rz * khx - rx * khz; e1z = rx * khy - ry * khx
    n1 = np.sqrt(e1x**2 + e1y**2 + e1z**2); n1 = np.where(n1 > 0, n1, 1.0)
    e1x, e1y, e1z = e1x / n1, e1y / n1, e1z / n1
    e2x = khy * e1z - khz * e1y; e2y = khz * e1x - khx * e1z; e2z = khx * e1y - khy * e1x
    inv2 = 1.0 / np.sqrt(2.0)

    # Per-channel phase: each helical amplitude a_+/a_- carries its own phase
    #   phase_pm(k) = theta_ref_pm(k) + (1 - lam) * theta_rand_pm(k),
    # with a SEPARATE structured reference per channel and INDEPENDENT random
    # parts, so at lam=0 the two channels are independent random phases (an
    # isotropic Gaussian haze) and at lam=1 they lock onto their (distinct)
    # structured references.  Because only unit-modulus phases are applied,
    # |a_pm| = sqrt((1 +/- p)/2)*E is untouched: the power spectrum AND the +/-
    # energy split (hence the net helicity bias p) are frozen exactly at every
    # lam -- only the inter-mode phase coherence, i.e. the spatial organization,
    # varies.
    #
    # Two references (not one shared) is what keeps p=0 ISOTROPIC at every lam:
    # with a single shared reference, at lam=1 both channels collapse to the same
    # phase and a_+ = a_- forces the p=0 field to be linearly polarized along the
    # frame vector e1 (u_z washes out).  Distinct references keep the two channels
    # out of phase, so the balanced field stays elliptically polarized / isotropic.
    #
    # (An earlier revision baked *independent* random phases into a_+/a_- and then
    # multiplied by a common reference phase; that never removed the random phase,
    # so lam=1 stayed a haze and coherence did not respond to lam in 3D at all.
    # Reality here is enforced by copy symmetrization below, not an averaging
    # projection, which would mix the +/- channels and distort rho(p).)
    idx = (-np.arange(N)) % N
    lin = np.arange(N ** 3).reshape(N, N, N)
    upper = lin < lin[np.ix_(idx, idx, idx)]               # strict spectral half-space
    theta_ref_p = _ref_phase_3d(N, ref_seed)               # structured reference, + channel
    theta_ref_m = _ref_phase_3d(N, ref_seed + 101)         # structured reference, - channel

    def _rand_phase():
        # uniform on the half-space, theta(-k) = -theta(k) on the other half
        # (0 on self-conjugate modes): a genuine i.i.d.-uniform antisymmetric
        # random phase, not the triangular difference of two uniforms.
        raw = np.where(upper, 2 * np.pi * rng.random((N, N, N)), 0.0)
        return raw - raw[np.ix_(idx, idx, idx)]

    php = np.where(mask, np.exp(1j * (theta_ref_p + (1.0 - lam) * _rand_phase())), 0.0)
    phm = np.where(mask, np.exp(1j * (theta_ref_m + (1.0 - lam) * _rand_phase())), 0.0)
    ap = np.sqrt((1 + p) / 2) * env * php
    am = np.sqrt((1 - p) / 2) * env * phm
    uhx = inv2 * (ap * (e1x + 1j * e2x) + am * (e1x - 1j * e2x))
    uhy = inv2 * (ap * (e1y + 1j * e2y) + am * (e1y - 1j * e2y))
    uhz = inv2 * (ap * (e1z + 1j * e2z) + am * (e1z - 1j * e2z))
    uhat = np.stack([uhx, uhy, uhz])

    # Reality by COPY (not averaging): keep uhat(k) on the spectral half-space and
    # set uhat(-k) = conj(uhat(k)) on the other half; self-conjugate modes (DC and
    # any Nyquist-plane mode) are excluded by `mask` and are zero.  Unlike the
    # averaging projection 0.5(uhat + conj(uhat(-k))), copying leaves |a_pm| on the
    # half-space untouched, so it does not mix the +/- channels or distort helicity.
    mir = lambda A: A[:, idx][:, :, idx][:, :, :, idx]
    uhat = np.where(upper[None, :, :, :], uhat, np.conj(mir(uhat)))

    u = np.stack([np.fft.ifftn(uhat[c]).real for c in range(3)])
    return u, uhat, freqs


def helicity_3d(uhat, freqs):
    """Net helicity H = sum u.w over the grid, w = i k x u (spectral).
    Uses the 'ij' axis convention (KX on axis 0) matching build_fft_3d."""
    KX = freqs[:, None, None]; KY = freqs[None, :, None]; KZ = freqs[None, None, :]
    what = np.empty_like(uhat)
    what[0] = 1j * (KY * uhat[2] - KZ * uhat[1])
    what[1] = 1j * (KZ * uhat[0] - KX * uhat[2])
    what[2] = 1j * (KX * uhat[1] - KY * uhat[0])
    u = np.stack([np.fft.ifftn(uhat[c]).real for c in range(3)])
    w = np.stack([np.fft.ifftn(what[c]).real for c in range(3)])
    return float(np.sum(u * w)), float(np.sum(uhat.real**2 + uhat.imag**2))


# ----------------------------------------------------------------------------
# 2D stream-function field with phase coherence lambda -- identical to e4.
# ----------------------------------------------------------------------------
def build_2d(N, slope, lam, seed, kc_frac=0.28, nblob=5):
    rng = np.random.default_rng(seed)
    k1 = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    KX, KY = np.meshgrid(k1, k1, indexing='ij')
    K2 = KX * KX + KY * KY
    kc2 = (kc_frac * N) ** 2
    with np.errstate(divide='ignore'):
        E = np.where(K2 > 0, (np.sqrt(K2) ** (-slope)) * np.exp(-K2 / kc2), 0.0)
    xs = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    blob = np.zeros((N, N))
    for _ in range(nblob):
        cx, cy = rng.uniform(0, 2 * np.pi, 2)
        sgn = rng.choice([-1.0, 1.0]); w = 0.5
        dx = (X - cx + np.pi) % (2 * np.pi) - np.pi
        dy = (Y - cy + np.pi) % (2 * np.pi) - np.pi
        blob += sgn * np.exp(-(dx * dx + dy * dy) / (2 * w * w))
    theta_ref = np.angle(np.fft.fft2(blob))
    idx = (-np.arange(N)) % N
    mir = lambda A: A[idx][:, idx]
    # uniform antisymmetric random phase: uniform on a spectral half-space, then
    # theta(-k) = -theta(k) (a genuine i.i.d.-uniform RPA surrogate, not the
    # triangular difference of two uniforms).
    lin = np.arange(N * N).reshape(N, N)
    upper = lin < mir(lin)
    raw = np.where(upper, 2 * np.pi * rng.random((N, N)), 0.0)
    theta_rand = raw - mir(raw)
    theta = theta_ref + (1 - lam) * theta_rand
    what = E * np.exp(1j * theta); what[0, 0] = 0.0
    # u = (d_y psi, -d_x psi), psi_hat = omega_hat/|k|^2, omega = d_x v - d_y u (eq. 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        uhat = +1j * np.where(K2 > 0, KY / K2, 0.0) * what
        vhat = -1j * np.where(K2 > 0, KX / K2, 0.0) * what
    u = np.fft.ifft2(uhat).real
    v = np.fft.ifft2(vhat).real
    omega = np.fft.ifft2(1j * (KX * vhat - KY * uhat)).real
    return u, v, omega, (uhat, vhat, K2)


def flatness(field):
    f = field.real - field.real.mean()
    return float(np.mean(f ** 4) / (np.mean(f ** 2) ** 2))


# ----------------------------------------------------------------------------
# analytic Beltrami mode-sum -- identical to e3.
# ----------------------------------------------------------------------------
def make_modes(N, p, slope, rng):
    modes = []
    for _ in range(N):
        z = 2 * rng.random() - 1
        th = 2 * np.pi * rng.random()
        d = np.array([np.sqrt(1 - z*z)*np.cos(th), np.sqrt(1 - z*z)*np.sin(th), z])
        km = 1.0 + 5.2 * rng.random()
        e1, e2 = frame(d)
        s = 1.0 if rng.random() < (1 + p) / 2 else -1.0
        a = km ** (-slope)
        ph = 2 * np.pi * rng.random()
        modes.append((km * d, km, e1, e2, s, a, ph))
    return modes


def eval_uw(modes, X):
    u = np.zeros_like(X); w = np.zeros_like(X)
    for k, km, e1, e2, s, a, ph in modes:
        phi = X @ k + ph
        c, sn = np.cos(phi), np.sin(phi)
        term = a * (np.outer(c, e1) - s * np.outer(sn, e2))
        u += term; w += s * km * term
    return u, w


# ----------------------------------------------------------------------------
# Bridson free-slip quintic ramp -- identical to e6.
# ----------------------------------------------------------------------------
def ramp_alpha(q):
    q = np.asarray(q, dtype=float)
    a = (1.0 / 8.0) * q * (15 - 10 * q**2 + 3 * q**4)
    return np.clip(np.where(q <= 0, 0.0, np.where(q >= 1, 1.0, a)), 0.0, 1.0)


def ramp_dalpha(q):
    q = np.asarray(q, dtype=float)
    d = (15.0 / 8.0) * (1 - q**2) ** 2
    return np.where((q <= 0) | (q >= 1), 0.0, d)


# ----------------------------------------------------------------------------
# shared matplotlib styling
# ----------------------------------------------------------------------------
def set_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 0.8,
        "mathtext.fontset": "cm",
    })
    return plt

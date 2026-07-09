#!/usr/bin/env python3
"""E6 -- solid-obstacle handling: divergence-free flow around a sphere.

Base field: the same analytic Beltrami mode-sum as e3_mode_sum.py,
    u_j(x) = a_j ( e1_j cos(phi_j) - s_j e2_j sin(phi_j) ),  phi_j = k_j.x + theta_j,
with khat_j uniform on the sphere, amplitude a_j = |k_j|^-slope, transverse frame
(e1_j, e2_j) perp khat_j, and a per-mode helicity sign s_j in {+1,-1}.  Each mode
is a curl-eigenfunction (Beltrami): curl u_j = s_j |k_j| u_j.  It therefore has an
EXACT analytic vector potential

    A_j(x) = u_j(x) / (s_j |k_j|),      A(x) = sum_j A_j(x),    curl A = u.

To keep a rigid sphere obstacle exactly divergence-free we do NOT clip or project
the velocity -- we damp the POTENTIAL with a smooth ramp of the signed-distance
function (SDF) and only then take the curl, so the bounded field is a curl of
something by construction and cannot pick up spurious divergence at the wall:

    u_b(x) = curl( ramp(d(x)/th) * A(x) )
           = (ramp'(d/th)/th) * ( grad d(x) x A(x) )  +  ramp(d/th) * u(x)

Sphere SDF (positive outside):  d(x) = |x - c| - R,  grad d = (x - c)/|x - c| = n.

The ramp is the quintic Bridson free-slip / curl-noise boundary blend, in the
scalar argument q = d/th (th = influence-band thickness):

    ramp(q)  = 0                          q <= 0
             = q(15 - 10 q^2 + 3 q^4)/8    0 < q < 1
             = 1                          q >= 1
    ramp'(q) = 0                          q < 0 or q >= 1
             = (15/8)(1 - q^2)^2           0 <= q < 1

so ramp(0)=0 (field vanishes at the wall), ramp'(0)=15/8>0 (pure tangential slip,
not a dead no-slip zone), and ramp(1)=1, ramp'(1)=ramp''(1)=0 (C^2 blend into the
unconstrained field at the outer edge of the band).  Inside the sphere (d<=0),
u_b = 0 exactly.

Choices (documented, fixed seed everywhere):
  R  = 1.0            sphere radius
  th = 0.6            influence-band thickness (~0.6 R)
  N  = 40 modes       same order as e3_mode_sum.py
  kmin, kmax = 1.5, 4.0,  slope = 1.0     (amplitude a = |k|^-slope)
  box (figure) = 3 R across, z=0 slice through the sphere's center

Measurements (all printed):
  (a) surface tangency: at d=0 the normal component of u_b vanishes identically
      because grad d x A is perpendicular to grad d=n; measured as
      max|u_b.n| / rms|u_b| over many surface points -> machine precision.
  (b) particle containment: RK4-advect a fixed shell of tracers through the
      STEADY field, with the ramp (bounded u_b) and without it (raw u); report
      the fraction ending up inside the sphere (d<0) in both cases.
  (c) divergence is machine zero by construction: (1) sympy identically proves
      curl(f*A) = grad(f) x A + f*curl(A) and div(curl(anything)) = 0 for the
      abstract damped potential f*A; a complex-step evaluation of the analytic
      u_b confirms the residual is ~1e-15 near the surface; (2) for contrast, a
      naive central finite-difference divergence at the same points shows the
      expected O(h^2) truncation error that vanishes only in the h->0 limit.
"""
import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# base field: analytic Beltrami mode-sum with exact vector potential
# ---------------------------------------------------------------------------

def frame(khat):
    """Right-handed transverse frame (e1, e2) with e1,e2 ⊥ khat, e1 x e2 = khat."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(khat[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, khat); e1 /= np.linalg.norm(e1)
    e2 = np.cross(khat, e1)
    return e1, e2


def make_modes(N, slope, kmin, kmax, rng):
    modes = []
    for _ in range(N):
        z = 2 * rng.random() - 1
        th_ = 2 * np.pi * rng.random()
        khat = np.array([np.sqrt(1 - z * z) * np.cos(th_),
                          np.sqrt(1 - z * z) * np.sin(th_), z])
        km = kmin + (kmax - kmin) * rng.random()
        k = km * khat
        e1, e2 = frame(khat)
        s = 1.0 if rng.random() < 0.5 else -1.0
        a = km ** (-slope)
        ph = 2 * np.pi * rng.random()
        modes.append((k, km, e1, e2, s, a, ph))
    return modes


def eval_uA(modes, X):
    """Velocity u and its exact analytic vector potential A (curl A = u) at
    points X (...,3), real-valued path used for the bulk of the script."""
    u = np.zeros_like(X, dtype=float)
    A = np.zeros_like(X, dtype=float)
    for k, km, e1, e2, s, a, ph in modes:
        phi = X @ k + ph
        c, sn = np.cos(phi), np.sin(phi)
        term = a * (np.multiply.outer(c, e1) - s * np.multiply.outer(sn, e2))
        u += term
        A += term / (s * km)          # curl(u_j/(s_j|k_j|)) = u_j  (Beltrami)
    return u, A


def eval_uA_complex(modes, X):
    """Same as eval_uA but supports complex X, for complex-step differentiation."""
    u = np.zeros(X.shape, dtype=complex)
    A = np.zeros(X.shape, dtype=complex)
    for k, km, e1, e2, s, a, ph in modes:
        phi = X @ k + ph
        c, sn = np.cos(phi), np.sin(phi)
        term = a * (np.multiply.outer(c, e1) - s * np.multiply.outer(sn, e2))
        u += term
        A += term / (s * km)
    return u, A


def verify_potential_identity(modes, rng, h=1e-5):
    """Numeric check that curl A = u at a random point (central-difference curl
    of the analytic potential, compared against the analytic velocity)."""
    x0 = rng.normal(size=3) * 0.3

    def Afun(x):
        _, A = eval_uA(modes, x[None, :])
        return A[0]

    def curl_of(x):
        def d(i, j):
            e = np.zeros(3); e[j] = 1.0
            return (Afun(x + h * e)[i] - Afun(x - h * e)[i]) / (2 * h)
        return np.array([d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)])

    u_true, _ = eval_uA(modes, x0[None, :])
    u_curl = curl_of(x0)
    return np.linalg.norm(u_true[0] - u_curl)


# ---------------------------------------------------------------------------
# sphere SDF + Bridson quintic free-slip ramp
# ---------------------------------------------------------------------------

def sdf_sphere(X, c, R):
    """Signed distance to a sphere (positive outside) and unit outward normal."""
    r = X - c
    rn = np.linalg.norm(r, axis=-1)
    d = rn - R
    n = r / rn[..., None]
    return d, n


def ramp(q):
    return np.where(q <= 0, 0.0, np.where(q >= 1, 1.0, q * (15 - 10 * q ** 2 + 3 * q ** 4) / 8))


def ramp_prime(q):
    return np.where((q < 0) | (q >= 1), 0.0, (15.0 / 8.0) * (1 - q ** 2) ** 2)


def eval_bounded(modes, X, c, R, th):
    """Bounded (ramp-damped) field u_b = curl(ramp(d/th) * A), expanded exactly:
    u_b = (ramp'(d/th)/th) * (n x A) + ramp(d/th) * u.  Also returns the raw
    (unbounded) field u, the SDF d, and the outward normal n."""
    u, A = eval_uA(modes, X)
    d, n = sdf_sphere(X, c, R)
    q = d / th
    r_ = ramp(q)
    rp_ = ramp_prime(q)
    cross_nA = np.cross(n, A)
    u_b = (rp_ / th)[..., None] * cross_nA + r_[..., None] * u
    return u_b, u, d, n


# smooth-band-only variant (unbranched polynomial ramp; valid strictly for 0<q<1),
# used only for the complex-step divergence check where we deliberately stay away
# from the C^0/C^2 junctions at q=0,1 so complex-step differentiation sees a single
# analytic branch.
def ramp_smooth(q):
    return q * (15 - 10 * q ** 2 + 3 * q ** 4) / 8


def ramp_prime_smooth(q):
    return (15.0 / 8.0) * (1 - q ** 2) ** 2


def eval_bounded_complex(modes, X, c, R, th):
    u, A = eval_uA_complex(modes, X)
    r = X - c
    rr = np.sum(r * r, axis=-1)
    rn = np.sqrt(rr)                      # complex sqrt; fine away from r=0
    d = rn - R
    n = r / rn[..., None]
    q = d / th
    rp_ = ramp_prime_smooth(q)
    r_ = ramp_smooth(q)
    cross_nA = np.cross(n, A)
    return (rp_ / th)[..., None] * cross_nA + r_[..., None] * u


def complex_step_div(modes, X, c, R, th, h=1e-30):
    """Complex-step divergence of u_b in the smooth band interior: derivative
    accurate to full machine precision, no subtractive-cancellation error."""
    div = np.zeros(X.shape[0])
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        Xc = X.astype(complex) + 1j * h * e
        u_b = eval_bounded_complex(modes, Xc, c, R, th)
        div += u_b[:, k].imag / h
    return div


def fd_div_bounded(modes, X, c, R, th, h):
    """Naive real central-difference divergence of u_b at the same points, for
    contrast with the complex-step / analytic result: this one carries the
    expected O(h^2) truncation error (plus float round-off at very small h)."""
    div = np.zeros(X.shape[0])
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        up, _, _, _ = eval_bounded(modes, X + h * e, c, R, th)
        um, _, _, _ = eval_bounded(modes, X - h * e, c, R, th)
        div += (up[:, k] - um[:, k]) / (2 * h)
    return div


# ---------------------------------------------------------------------------
# symbolic proof: curl(f*A) product rule + div(curl(.))=0, abstract f,A
# ---------------------------------------------------------------------------

def symbolic_divcurl_identity():
    x, y, z = sp.symbols('x y z', real=True)
    f = sp.Function('f')(x, y, z)                       # stands for ramp(d/th)
    Ax, Ay, Az = [sp.Function(n)(x, y, z) for n in ('Ax', 'Ay', 'Az')]
    A = sp.Matrix([Ax, Ay, Az])

    def grad(g):
        return sp.Matrix([sp.diff(g, x), sp.diff(g, y), sp.diff(g, z)])

    def curl(V):
        Vx, Vy, Vz = V
        return sp.Matrix([sp.diff(Vz, y) - sp.diff(Vy, z),
                           sp.diff(Vx, z) - sp.diff(Vz, x),
                           sp.diff(Vy, x) - sp.diff(Vx, y)])

    def div(V):
        Vx, Vy, Vz = V
        return sp.diff(Vx, x) + sp.diff(Vy, y) + sp.diff(Vz, z)

    lhs1 = curl(f * A)
    rhs1 = grad(f).cross(A) + f * curl(A)
    resid_product_rule = sp.simplify(sp.Matrix(lhs1) - sp.Matrix(rhs1))
    resid_divcurl = sp.simplify(div(curl(f * A)))
    return resid_product_rule, resid_divcurl


# ---------------------------------------------------------------------------
# (a) surface tangency
# ---------------------------------------------------------------------------

def surface_tangency(modes, c, R, th, M, rng):
    z = 2 * rng.random(M) - 1
    t = 2 * np.pi * rng.random(M)
    dirs = np.stack([np.sqrt(1 - z ** 2) * np.cos(t), np.sqrt(1 - z ** 2) * np.sin(t), z], axis=1)
    Xsurf = c + R * dirs
    u_b, u, d, n = eval_bounded(modes, Xsurf, c, R, th)
    normal_comp = np.sum(u_b * n, axis=1)
    rms_ub = np.sqrt(np.mean(np.sum(u_b ** 2, axis=1)))
    max_d = np.max(np.abs(d))
    max_normal = np.max(np.abs(normal_comp))
    return max_normal, rms_ub, max_normal / rms_ub, max_d


# ---------------------------------------------------------------------------
# (b) particle containment: RK4 advection with / without the ramp
# ---------------------------------------------------------------------------

def rk4_advect(X0, field_func, dt, nsteps, *args):
    X = X0.copy()
    for _ in range(nsteps):
        k1 = field_func(X, *args)
        k2 = field_func(X + 0.5 * dt * k1, *args)
        k3 = field_func(X + 0.5 * dt * k2, *args)
        k4 = field_func(X + dt * k3, *args)
        X = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return X


def field_bounded(X, modes, c, R, th):
    u_b, _, _, _ = eval_bounded(modes, X, c, R, th)
    return u_b


def field_raw(X, modes, c, R, th):
    u, _ = eval_uA(modes, X)
    return u


def seed_shell(Np, c, R, th, offset_lo, offset_hi, rng):
    z = 2 * rng.random(Np) - 1
    t = 2 * np.pi * rng.random(Np)
    dirs = np.stack([np.sqrt(1 - z ** 2) * np.cos(t), np.sqrt(1 - z ** 2) * np.sin(t), z], axis=1)
    offsets = rng.uniform(offset_lo, offset_hi, size=Np)
    return c + (R + offsets)[:, None] * dirs


def containment_fractions(modes, c, R, th, Np, dt, nsteps, offset_lo, offset_hi, seed):
    rng = np.random.default_rng(seed)
    X0 = seed_shell(Np, c, R, th, offset_lo, offset_hi, rng)
    Xb = rk4_advect(X0, field_bounded, dt, nsteps, modes, c, R, th)
    Xr = rk4_advect(X0, field_raw, dt, nsteps, modes, c, R, th)
    d_b, _ = sdf_sphere(Xb, c, R)
    d_r, _ = sdf_sphere(Xr, c, R)
    return np.mean(d_b < 0), np.mean(d_r < 0), Np


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def make_figure(modes, c, R, th, meas_a, meas_b, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_normal, rms_ub, ratio_a, _ = meas_a
    frac_ramp, frac_raw, Np = meas_b

    box = 3.0 * R
    ng = 70
    gx = np.linspace(-box, box, ng)
    gy = np.linspace(-box, box, ng)
    GX, GY = np.meshgrid(gx, gy, indexing='xy')
    Xg = np.stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)], axis=1)
    u_b_g, u_g, d_g, _ = eval_bounded(modes, Xg, c, R, th)
    Ub = u_b_g[:, :2].reshape(ng, ng, 2)
    Ur = u_g[:, :2].reshape(ng, ng, 2)

    # surface-angle sweep for panel (c): max|u_b.n| relative to local |u_b|
    rng_s = np.random.default_rng(21)
    Ms = 720
    theta_s = np.linspace(0, 2 * np.pi, Ms, endpoint=False)
    dirs_s = np.stack([np.cos(theta_s), np.sin(theta_s), np.zeros(Ms)], axis=1)
    Xs = c + R * dirs_s
    u_b_s, _, _, n_s = eval_bounded(modes, Xs, c, R, th)
    normal_s = np.abs(np.sum(u_b_s * n_s, axis=1))
    mag_s = np.linalg.norm(u_b_s, axis=1)
    rel_s = normal_s / np.maximum(mag_s, 1e-300)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

    ax = axes[0]
    ax.streamplot(GX, GY, Ur[..., 0], Ur[..., 1], density=1.2, color='C0', linewidth=0.7, arrowsize=0.8)
    ax.add_patch(plt.Circle((0, 0), R, color='0.35', alpha=0.35, zorder=3))
    ax.set_aspect('equal'); ax.set_xlim(-box, box); ax.set_ylim(-box, box)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('unbounded base field u\n(passes through sphere)')

    ax = axes[1]
    ax.streamplot(GX, GY, Ub[..., 0], Ub[..., 1], density=1.2, color='C1', linewidth=0.7, arrowsize=0.8)
    ax.add_patch(plt.Circle((0, 0), R, color='0.35', alpha=0.35, zorder=3))
    ax.set_aspect('equal'); ax.set_xlim(-box, box); ax.set_ylim(-box, box)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(r'bounded field $u_b$=curl(ramp$(d/\theta)$A)' + '\n(tangent, u=0 inside)')

    ax = axes[2]
    ax.semilogy(np.degrees(theta_s), np.maximum(rel_s, 1e-18), color='C2', lw=1.0)
    ax.set_xlabel('surface angle (deg, z=0 great circle)')
    ax.set_ylabel(r'$|u_b\!\cdot\!n| \,/\, |u_b|$  (log scale)')
    ax.set_ylim(1e-18, 1e-12)
    ax.set_title('surface normal-flux ratio\n(machine-zero tangency)')
    ax.axhline(ratio_a, color='k', ls='--', lw=0.8, label=f'reported max/rms = {ratio_a:.1e}')
    ax.legend(loc='upper right', fontsize=8)

    caption = (f"E6 obstacle: N=40 Beltrami modes, R={R:g}, "
               r"$\theta$" + f"={th:g}. "
               f"Surface tangency max|u_b.n|/rms|u_b| = {ratio_a:.2e}. "
               f"Particle containment ({Np} tracers, RK4): "
               f"{100*frac_ramp:.1f}% inside WITH ramp vs {100*frac_raw:.1f}% inside WITHOUT ramp.")
    fig.suptitle(caption, fontsize=9.5, y=0.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    SEED = 0
    print("E6 solid-obstacle handling: sphere SDF ramp on the vector potential (seed 0)")
    print()

    # base field: same construction/order as e3_mode_sum.py
    N, SLOPE, KMIN, KMAX = 40, 1.0, 1.5, 4.0
    R, TH = 1.0, 0.6
    C = np.array([0.0, 0.0, 0.0])
    modes = make_modes(N, SLOPE, KMIN, KMAX, np.random.default_rng(SEED + 1))

    print(f"Setup: N={N} modes, slope={SLOPE}, |k| in [{KMIN},{KMAX}], sphere R={R}, band th={TH}")
    print()

    # sanity: curl A = u (numeric, central-difference)
    resid_potential = verify_potential_identity(modes, np.random.default_rng(SEED + 2))
    print(f"[check] analytic-potential identity |curl A - u| (central-diff, h=1e-5) = {resid_potential:.3e}")
    print()

    # (a) surface tangency
    max_normal, rms_ub, ratio_a, max_d = surface_tangency(
        modes, C, R, TH, M=20000, rng=np.random.default_rng(SEED + 3))
    print("(a) SURFACE TANGENCY  (free-slip guarantee at d=0)")
    print(f"    surface points M=20000, max|d| on surface = {max_d:.2e} (should be ~0)")
    print(f"    max|u_b.n|            = {max_normal:.3e}")
    print(f"    rms|u_b| on surface   = {rms_ub:.3e}")
    print(f"    max|u_b.n| / rms|u_b| = {ratio_a:.3e}   <-- headline number (a)")
    print()

    # (b) particle containment
    Np, dt, nsteps = 5000, 0.002, 60
    offset_lo, offset_hi = 0.02 * TH, 1.2 * TH
    PARTICLE_SEED = 3          # fixed seed for the tracer shell (documented, deterministic)
    frac_ramp, frac_raw, Np_used = containment_fractions(
        modes, C, R, TH, Np, dt, nsteps, offset_lo, offset_hi, seed=PARTICLE_SEED)
    print("(b) PARTICLE CONTAINMENT  (RK4 advection through the STEADY field)")
    print(f"    particles Np={Np_used}, dt={dt}, steps={nsteps} (T={dt*nsteps:.3f}), "
          f"seed shell offset in [{offset_lo:.3f},{offset_hi:.3f}]")
    print(f"    fraction inside sphere WITH ramp (bounded u_b)   = {100*frac_ramp:.3f}%")
    print(f"    fraction inside sphere WITHOUT ramp (raw u)      = {100*frac_raw:.3f}%")
    print(f"    reference (GPU particle readback): ~0% vs ~2.9%")
    print()

    # (c) divergence: symbolic identity + complex-step (near machine zero) + FD contrast
    resid_product_rule, resid_divcurl = symbolic_divcurl_identity()
    print("(c) DIVERGENCE IS MACHINE ZERO BY CONSTRUCTION")
    print(f"    sympy identity  curl(f*A) - [grad(f) x A + f*curl(A)]  = {list(resid_product_rule)}")
    print(f"    sympy identity  div(curl(f*A))                        = {resid_divcurl}")

    rng_c = np.random.default_rng(SEED + 5)
    Mc = 5000
    z = 2 * rng_c.random(Mc) - 1
    t = 2 * np.pi * rng_c.random(Mc)
    dirs = np.stack([np.sqrt(1 - z ** 2) * np.cos(t), np.sqrt(1 - z ** 2) * np.sin(t), z], axis=1)
    q_band = rng_c.uniform(0.25, 0.75, size=Mc)          # smooth interior of the ramp band
    Xc = C + (R + q_band * TH)[:, None] * dirs

    div_cs = complex_step_div(modes, Xc, C, R, TH)
    max_div_cs = np.max(np.abs(div_cs))
    rms_div_cs = np.sqrt(np.mean(div_cs ** 2))
    print(f"    complex-step div(u_b), M={Mc} points in band interior (q in [0.25,0.75]):")
    print(f"      max|div|  = {max_div_cs:.3e}")
    print(f"      rms  div  = {rms_div_cs:.3e}   <-- headline number (c), analytic residual")

    H_FD = 0.01
    div_fd = fd_div_bounded(modes, Xc, C, R, TH, H_FD)
    rms_div_fd = np.sqrt(np.mean(div_fd ** 2))
    print(f"    naive central-FD div(u_b) at same points, h={H_FD}: rms = {rms_div_fd:.3e}  "
          f"(O(h^2) truncation, NOT the analytic residual)")
    print()

    fig = make_figure(modes, C, R, TH,
                       (max_normal, rms_ub, ratio_a, max_d),
                       (frac_ramp, frac_raw, Np_used),
                       "figures/obstacle.png")
    print("Saved figure: figures/obstacle.png")
    print()
    print("Summary of headline numbers for the paper:")
    print(f"  (a) max|u_b.n|/rms|u_b|            = {ratio_a:.3e}")
    print(f"  (b) inside fraction: ramp={100*frac_ramp:.2f}%  no-ramp={100*frac_raw:.2f}%  (Np={Np_used})")
    print(f"  (c) analytic (complex-step) rms div = {rms_div_cs:.3e};  FD rms div (h={H_FD}) = {rms_div_fd:.3e}")

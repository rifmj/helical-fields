#!/usr/bin/env python3
"""Figure: advected dye blob driven by the two authoring axes.

A compact dye blob is transported by semi-Lagrangian advection through the
steady field. Top row: phase coherence lambda (0->1) at fixed spectrum -- the
blob disperses at lambda=0 and winds into a tight coherent spiral at lambda=1.
Bottom row: helicity polarization p. The transport needs no pressure projection
because the field is divergence-free by construction.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _helical as H

plt = H.set_style()


def advect_dye(u, v, blob, nsteps=30, dt=0.9):
    """Semi-Lagrangian back-trace of a scalar through a steady periodic 2D field
    on the [0,2pi)^2 grid. u,v,blob are (N,N)."""
    N = blob.shape[0]
    xs = np.arange(N)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    scale = N / (2 * np.pi)                       # grid cells per unit length
    dye = blob.copy()
    for _ in range(nsteps):
        # back-trace departure point (grid units), wrap periodically
        px = (X - dt * u * scale) % N
        py = (Y - dt * v * scale) % N
        i0 = np.floor(px).astype(int); j0 = np.floor(py).astype(int)
        fx = px - i0; fy = py - j0
        i1 = (i0 + 1) % N; j1 = (j0 + 1) % N
        dye = ((1 - fx) * (1 - fy) * dye[i0, j0] + fx * (1 - fy) * dye[i1, j0]
               + (1 - fx) * fy * dye[i0, j1] + fx * fy * dye[i1, j1])
    return dye


def advect_dye_3d(u3, blob3, nsteps=45, dt=1.1):
    """Semi-Lagrangian back-trace of a 3D scalar through a steady periodic 3D
    field u3 of shape (3,N,N,N). The 3D field is exactly divergence-free, so the
    dye stays coherent (a z-slice of it does not, which is why we advect in 3D
    and slice afterwards). Returns the advected 3D dye."""
    N = blob3.shape[0]
    a = np.arange(N)
    X, Y, Z = np.meshgrid(a, a, a, indexing='ij')
    scale = N / (2 * np.pi)
    dye = blob3.copy()
    for _ in range(nsteps):
        px = (X - dt * u3[0] * scale) % N
        py = (Y - dt * u3[1] * scale) % N
        pz = (Z - dt * u3[2] * scale) % N
        i0 = np.floor(px).astype(int); j0 = np.floor(py).astype(int); k0 = np.floor(pz).astype(int)
        fx = px - i0; fy = py - j0; fz = pz - k0
        i1 = (i0 + 1) % N; j1 = (j0 + 1) % N; k1 = (k0 + 1) % N
        dye = (
            (1-fx)*(1-fy)*(1-fz)*dye[i0, j0, k0] + fx*(1-fy)*(1-fz)*dye[i1, j0, k0]
            + (1-fx)*fy*(1-fz)*dye[i0, j1, k0] + (1-fx)*(1-fy)*fz*dye[i0, j0, k1]
            + fx*fy*(1-fz)*dye[i1, j1, k0] + fx*(1-fy)*fz*dye[i1, j0, k1]
            + (1-fx)*fy*fz*dye[i0, j1, k1] + fx*fy*fz*dye[i1, j1, k1])
    return dye


def compact_blob(N, cx=np.pi, cy=np.pi, w=0.35):
    xs = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    dx = (X - cx + np.pi) % (2 * np.pi) - np.pi
    dy = (Y - cy + np.pi) % (2 * np.pi) - np.pi
    return np.exp(-(dx * dx + dy * dy) / (2 * w * w))


def main(out="figures/advection.png", N=256, seed=3):
    blob = compact_blob(N, w=0.28)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 10.0))

    # top row: phase coherence (2D field, fixed spectrum, fixed seed).
    # Short winding time keeps the blob coherent: random phases (lam=0) shear it
    # apart, structured phases (lam=1) wind it into a tight spiral.
    for j, lam in enumerate([0.0, 0.5, 1.0]):
        u, v, _, _ = H.build_2d(N, 1.0, lam=lam, seed=seed, nblob=8)
        u, v = u / np.sqrt((u**2 + v**2).mean()), v / np.sqrt((u**2 + v**2).mean())
        dye = advect_dye(u, v, blob, nsteps=10, dt=0.35)
        ax = axes[0, j]
        ax.imshow(dye.T, origin="lower", cmap="magma", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(rf"$\lambda={lam}$", fontsize=13, loc="left")
        ax.text(-0.02, 1.06, "abc"[j], transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="bottom", ha="right")

    # bottom row: helicity polarization -- advect a 3D blob through the 3D
    # (divergence-free) field, then z-project.  A single z-slice of the advected
    # 3D dye is patchy (a slice of a 3D-incompressible flow is not 2D-solenoidal);
    # the depth integral recovers the coherent transported filament.
    N3 = 96
    a3 = np.linspace(0, 2 * np.pi, N3, endpoint=False)
    XX, YY, ZZ = np.meshgrid(a3, a3, a3, indexing='ij')
    dx = (XX - np.pi + np.pi) % (2*np.pi) - np.pi
    dy = (YY - np.pi + np.pi) % (2*np.pi) - np.pi
    dz = (ZZ - np.pi + np.pi) % (2*np.pi) - np.pi
    blob3 = np.exp(-(dx*dx + dy*dy + dz*dz) / (2 * 0.5**2))
    for j, (p, lbl) in enumerate([(-1.0, r"$p=-1$ (left Beltrami)"),
                                  (0.0, r"$p=0$ (mirror-sym.)"),
                                  (+1.0, r"$p=+1$ (right Beltrami)")]):
        u3, uhat, freqs = H.build_fft_3d(N3, p, 1.2, seed, lam=1.0)
        u3 = u3 / np.sqrt((u3**2).sum(0).mean())     # unit rms speed
        dye3 = advect_dye_3d(u3, blob3, nsteps=9, dt=0.5)
        ax = axes[1, j]
        ax.imshow(dye3.sum(2).T, origin="lower", cmap="magma", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lbl, fontsize=13, loc="left")
        ax.text(-0.02, 1.06, "def"[j], transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="bottom", ha="right")

    axes[0, 0].set_ylabel("Phase coherence\n(fixed spectrum)", fontsize=12)
    axes[1, 0].set_ylabel("Helicity polarization\n(3D field, $z$-projection)", fontsize=12)
    fig.suptitle("Advected dye blob: the two authoring axes in a procedural pipeline", y=0.99,
                 fontsize=14)
    fig.text(0.5, 0.005, "random phases disperse the blob; coherent phases wind it into a tight "
             "spiral, at equal energy per scale", ha="center", fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

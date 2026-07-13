#!/usr/bin/env python3
"""Figure: chirality made concrete on a single Beltrami mode.

u(x) = a(cos kz, -/+ sin kz, 0), which satisfies curl u = +/- k u. A straight
dye line advected by the field winds into a one-turn helix; reversing p mirrors
the handedness. p=+1 is right-handed (net u.w > 0), p=-1 left-handed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _helical as H

plt = H.set_style()
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def advect_line(sign, k=1.0, a=1.0, nseg=12, nt=9, dt=0.16):
    """Advect a straight vertical dye line through u=a(cos kz, -sign*sin kz,0).
    Returns list of (T, line) snapshots. sign=+1 -> p=+1 branch (right-handed)."""
    z = np.linspace(-np.pi, np.pi, 220)
    # seed a straight line along z at x=y=0
    pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], 1)
    snaps = [pts.copy()]
    for _ in range(nt):
        ux = a * np.cos(k * pts[:, 2])
        uy = -sign * a * np.sin(k * pts[:, 2])
        pts = pts.copy()
        pts[:, 0] += dt * ux
        pts[:, 1] += dt * uy
        snaps.append(pts.copy())
    return snaps


def main(out="figures/helix.png"):
    fig = plt.figure(figsize=(13.0, 7.5))
    for col, (sign, lbl) in enumerate([(+1, r"$p=+1$  (right-handed)"),
                                       (-1, r"$p=-1$  (left-handed)")]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        snaps = advect_line(sign)
        nt = len(snaps)
        for i, line in enumerate(snaps):
            frac = i / (nt - 1)
            col_i = plt.cm.plasma(0.15 + 0.8 * frac)
            lw = 1.0 + 3.0 * (frac ** 2)
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=col_i, lw=lw)
        ax.set_title(lbl, fontsize=14, pad=0)
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_zlim(-np.pi, np.pi)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=12, azim=-60)
        ax.grid(False)
    fig.suptitle("A straight dye line advected by a single Beltrami mode winds into a helix;\n"
                 r"reversing $p$ mirrors the handedness (bright = latest time; "
                 r"$u_\pm=a(\cos kz,\ \mp\sin kz,0)$, $\mathrm{curl}\,u=\pm k\,u$)",
                 y=0.98, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    # multi-shell net helicity on a 32^3 grid (the number quoted in the caption):
    # a single-shell mode has |rho|=1; a multi-shell spectrum superposes windings
    # and the net int u.w tracks p in sign.
    for p in (+1.0, 0.0, -1.0):
        _, uh, fr = H.build_fft_3d(32, p, 1.6, 0, lam=0.0)
        Hn, _ = H.helicity_3d(uh, fr)
        print(f"  32^3 multi-shell net helicity at p={p:+.0f}: {Hn:+.2e}")


if __name__ == "__main__":
    main()

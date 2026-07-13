#!/usr/bin/env python3
"""Teaser figure: two independent authoring axes on one divergence-free field.

Columns: phase coherence lambda = 0, 0.5, 1 (random -> organized) at fixed
spectrum. Rows: helicity p = +1, 0, -1 (right -> left handed). Colour is the
out-of-plane velocity u_z on a z-slice of a 48^3 field; black curves are
in-plane streamlines. The two axes are orthogonal: across each row the total
energy is identical and the net helicity is set by p alone (unchanged by
lambda). Prints the per-row net helicity that the caption quotes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _helical as H

plt = H.set_style()


def main(out="figures/gallery.png", N=48, slope=1.4, seed=0):
    lams = [0.0, 0.5, 1.0]
    ps = [+1.0, 0.0, -1.0]
    xs = np.linspace(0, 2 * np.pi, N, endpoint=False)
    sl = N // 2

    fig, axes = plt.subplots(3, 3, figsize=(11.0, 11.3))
    row_H = []
    for i, p in enumerate(ps):
        # symmetric colour scale per row (helicity sets the u_z bias)
        fields = []
        for lam in lams:
            u, uhat, freqs = H.build_fft_3d(N, p, slope, seed, lam=lam)
            fields.append(u)
        Hn, _ = H.helicity_3d(H.build_fft_3d(N, p, slope, seed, lam=0.0)[1], freqs)
        row_H.append(Hn)
        vmax = max(np.abs(f[2][:, :, sl]).max() for f in fields)
        for j, (lam, u) in enumerate(zip(lams, fields)):
            ax = axes[i, j]
            uz = u[2][:, :, sl]
            ux = u[0][:, :, sl]; uy = u[1][:, :, sl]
            ax.imshow(uz.T, origin="lower", extent=[0, 2*np.pi, 0, 2*np.pi],
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.streamplot(xs, xs, ux.T, uy.T, color="k", density=1.0,
                          linewidth=0.5, arrowsize=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2*np.pi)
            if i == 0:
                ax.set_title(rf"$\lambda={lam}$", fontsize=13)
            if j == 0:
                ax.set_ylabel(rf"$p={p:+.0f}$", fontsize=13)

    fig.text(0.015, 0.5, "helicity $p$:  right-handed $\\leftrightarrow$ left-handed",
             rotation=90, ha="center", va="center", fontsize=12)
    fig.suptitle("Two independent authoring axes on one divergence-free field\n"
                 r"(colour: out-of-plane velocity $u_z$; black: in-plane streamlines; "
                 r"$48^3$ field, $z$-slice)", y=0.98, fontsize=13)
    fig.text(0.5, 0.02, r"phase coherence $\lambda$:  random $\longrightarrow$ organized"
             r"  (energy spectrum fixed)", ha="center", fontsize=12)
    fig.tight_layout(rect=[0.05, 0.035, 1, 0.94])
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    print("  per-row net helicity (p=+1,0,-1): " +
          ", ".join(f"{h:+.2e}" for h in row_H))


if __name__ == "__main__":
    main()

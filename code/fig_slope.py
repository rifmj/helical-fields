#!/usr/bin/env python3
"""Figure: the spectral-slope axis s on one 2D field at fixed seed and phase.

Three slopes re-weight the SAME modes (same seed, same phases), so the
large-scale skeleton persists while fine detail is added or removed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _helical as H

plt = H.set_style()


def main(out="figures/slope.png", N=256, seed=5):
    slopes = [0.8, 1.4, 2.2]
    fig, axes = plt.subplots(1, 3, figsize=(14.1, 5.0))
    xs = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    for ax, s in zip(axes, slopes):
        # lambda=1 (structured) at fixed seed so the skeleton is identical
        u, v, omega, _ = H.build_2d(N, s, lam=1.0, seed=seed)
        spd = np.sqrt(u * u + v * v)
        ax.imshow(spd.T, origin="lower", extent=[0, 2*np.pi, 0, 2*np.pi],
                  cmap="viridis", aspect="equal")
        ax.streamplot(xs, xs, u.T, v.T, color="white", density=1.1,
                      linewidth=0.6, arrowsize=0.7)
        ax.set_title(rf"$s={s}$", fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_ylabel("speed $|\\mathbf{u}|$ (colour)\nstreamlines white", fontsize=11)
    fig.suptitle("Spectral-slope axis $s$: the same field, re-weighted across scales at fixed seed",
                 y=1.0, fontsize=14)
    fig.text(0.16, 0.02, "shallow $s$: energy across scales (fine, ragged)", ha="center", fontsize=11)
    fig.text(0.84, 0.02, "steep $s$: few large scales (smooth, silky)", ha="center", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Figure: curl-noise (random phases) vs our construction at lambda=1
(structured phases), on the SAME power spectrum and seed.

Only the inter-mode phases differ. Random phases give a homogeneous (Gaussian)
field with speed flatness ~2; structured phases concentrate the same energy
into distinct coherent vortices (flatness ~4). Prints the two flatness values
that the figure and its caption quote.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _helical as H

plt = H.set_style()


def speed_flatness(u, v):
    spd = np.sqrt(u * u + v * v)
    f = spd - spd.mean()
    return float(np.mean(f ** 4) / (np.mean(f ** 2) ** 2))


def main(out="figures/curlnoise.png", N=256, slope=1.0, seed=7, nblob=8):
    # lambda=0 : random phases (curl-noise surrogate); lambda=1 : structured
    u0, v0, w0, _ = H.build_2d(N, slope, lam=0.0, seed=seed, nblob=nblob)
    u1, v1, w1, _ = H.build_2d(N, slope, lam=1.0, seed=seed, nblob=nblob)
    f0 = speed_flatness(u0, v0)
    f1 = speed_flatness(u1, v1)
    xs = np.linspace(0, 2 * np.pi, N, endpoint=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.2))
    for ax, (u, v, ttl, sub) in zip(axes, [
            (u0, v0, "Curl-noise", "(random inter-mode phases)"),
            (u1, v1, r"Our method, $\lambda=1$", "(structured phases, same spectrum)")]):
        spd = np.sqrt(u * u + v * v)
        ax.imshow(spd.T, origin="lower", extent=[0, 2*np.pi, 0, 2*np.pi],
                  cmap="magma", aspect="equal")
        ax.streamplot(xs, xs, u.T, v.T, color="white", density=1.3,
                      linewidth=0.5, arrowsize=0.6)
        ax.set_title(f"{ttl}\n{sub}", fontsize=13, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].text(0.03, 0.03, f"speed flatness {f0:.1f}", transform=axes[0].transAxes,
                 color="white", fontsize=12)
    axes[1].text(0.03, 0.03, f"speed flatness {f1:.1f}", transform=axes[1].transAxes,
                 color="white", fontsize=12)
    fig.suptitle("Curl-noise vs. our construction at identical power spectrum and random seed\n"
                 r"(only the inter-mode phases differ; colour: speed $|\mathbf{u}|$, "
                 r"white: streamlines; $256^2$, $s=1.0$)", y=1.0, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    print(f"  speed flatness: curl-noise (lam=0) = {f0:.2f}   ours (lam=1) = {f1:.2f}")


if __name__ == "__main__":
    main()

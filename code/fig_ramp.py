#!/usr/bin/env python3
"""Figure: the Bridson free-slip quintic ramp used for solid obstacles (sec. 7).

alpha(0)=0 (no penetration), alpha'(0)=15/8 (tangential slip, not a dead zone),
alpha(1)=1 with alpha'(1)=alpha''(1)=0 (C^2 blend back to the free field).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _helical as H

plt = H.set_style()


def main(out="figures/ramp.png"):
    q = np.linspace(-0.25, 1.5, 400)
    a = H.ramp_alpha(q)
    da = H.ramp_dalpha(q)
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.axvspan(-0.25, 0.0, color="0.85", zorder=0)
    ax.text(-0.12, 1.0, "inside\nsolid", ha="center", va="center", color="0.4", fontsize=10)
    ax.plot(q, a, color="#1f77b4", lw=2.6, label=r"$\alpha(q)$")
    ax.plot(q, da, color="#ff7f0e", lw=2.2, ls="--", label=r"$\alpha'(q)$")
    ax.plot(0, 0, "o", color="#1f77b4", ms=7, zorder=5)
    ax.plot(1, 1, "o", color="#1f77b4", ms=7, zorder=5)
    ax.annotate(r"$\alpha(0)=0$ (no penetration)", xy=(0, 0), xytext=(0.28, 0.16),
                fontsize=10, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1))
    ax.annotate(r"$\alpha'(0)=\frac{15}{8}$ (free slip)", xy=(0, 15/8), xytext=(0.33, 1.9),
                fontsize=10, color="#ff7f0e",
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1))
    ax.annotate(r"$\alpha(1)=1,\ \alpha'=\alpha''=0$", xy=(1, 1), xytext=(0.55, 0.45),
                fontsize=10, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1))
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel(r"signed distance  $q=d/\theta$")
    ax.set_ylabel("ramp value")
    ax.set_xlim(-0.25, 1.5); ax.set_ylim(-0.1, 2.05)
    ax.legend(loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    # numeric receipts (match e6 / caption)
    print(f"  alpha(0)={H.ramp_alpha(0):.4f}  alpha'(0)={H.ramp_dalpha(1e-9):.4f} (=15/8={15/8})")
    print(f"  alpha(1)={H.ramp_alpha(1):.4f}  alpha'(1)={H.ramp_dalpha(1-1e-9):.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Figure: the two cost regimes (E5).

Left:  FFT rebuild time vs grid N, O(N^3 log N) bake cost.
Right: analytic mode-sum throughput vs mode count M, O(1/M) per-point.
Reuses the e5 timing routines verbatim (absolute times are machine-dependent;
the scaling is the claim).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _helical as H
import e5_performance as e5
from matplotlib.ticker import NullLocator, NullFormatter

plt = H.set_style()


def main(out="figures/performance.png"):
    Ns = [16, 24, 32, 48, 64]
    Ms = [16, 32, 64, 128, 256]
    fft = e5.time_fft(Ns)                       # (N, sec, tex_mb)
    ms = e5.time_modesum(Ms)                    # (M, sec, mpps, fps)
    Nv = np.array([r[0] for r in fft]); tms = np.array([r[1] for r in fft]) * 1e3
    Mv = np.array([r[0] for r in ms]); mpps = np.array([r[2] for r in ms])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    axL.loglog(Nv, tms, "o-", color="#1f77b4", ms=9, lw=1.8)
    cN = tms[0] / (Nv[0] ** 3 * np.log2(Nv[0]))
    axL.loglog(Nv, cN * Nv ** 3 * np.log2(Nv), "--", color="0.5")
    axL.text(Nv.mean(), tms.max() * 1.1, r"$\mathcal{O}(N^3\log N)$", color="0.4", fontsize=11)
    for x, y in zip(Nv, tms):
        axL.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9)
    axL.set_xlabel(r"grid $N$ (field is $N^3$)")
    axL.set_ylabel("rebuild time (ms)")
    axL.set_title("FFT path: one bake per authored field")
    axL.xaxis.set_minor_locator(NullLocator())
    axL.set_xticks(Nv); axL.set_xticklabels(Nv)

    axR.loglog(Mv, mpps, "s-", color="#d62728", ms=9, lw=1.8)
    cM = mpps[0] * Mv[0]
    axR.loglog(Mv, cM / Mv, "--", color="0.5")
    axR.text(Mv.mean(), mpps.min() * 1.3, r"$\mathcal{O}(1/M)$", color="0.4", fontsize=11)
    for x, y in zip(Mv, mpps):
        axR.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9)
    axR.set_xlabel(r"number of modes $M$")
    axR.set_ylabel("throughput (Mpts / s)")
    axR.set_title("Mode-sum path: analytic, per-point")
    axR.xaxis.set_minor_locator(NullLocator())
    axR.set_xticks(Mv); axR.set_xticklabels(Mv)

    fig.suptitle("Two cost regimes: bake-time FFT vs real-time mode-sum (CPU numpy reference)",
                 y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

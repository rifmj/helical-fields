#!/usr/bin/env python3
"""Figure: the two quantitative validation results (E2 and E3).

Left:  FFT relative helicity rho is linear in the knob p (fit rho = 0.796 p),
       with a ceiling |rho(+-1)| ~ 0.80 < 1 for a multi-shell spectrum.
Right: the analytic mode-sum p=0 net-helicity residual decays as N^{-1/2}.

Reuses the e2 (FFT) and e3 (mode-sum) constructions verbatim so the plotted
numbers are the numbers those experiments print.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _helical as H
from matplotlib.ticker import NullLocator

# import the exact measurement routines from the validation scripts
import e2_fft_field as e2
import e3_mode_sum as e3

plt = H.set_style()


def rho_sweep_fft(N=32, slope=1.6, seed=0):
    ps = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    rho = []
    for p in ps:
        uhat, freqs = e2.build(N, p, slope, seed)
        _, _, r = e2.measure(uhat, freqs, N)
        rho.append(r)
    return ps, np.array(rho)


def residual_vs_N(Ns=(20, 40, 80, 160, 320), draws=200):
    rms = []
    for Nm in Ns:
        vals = []
        for e in range(draws):
            r = np.random.default_rng(1000 + e)
            modes = e3.make_modes(Nm, 0.0, 1.0, r)
            num = sum(s * km * a * a for k, km, e1, e2_, s, a, ph in modes)
            den = sum(km * a * a for k, km, e1, e2_, s, a, ph in modes)
            vals.append(num / den)
        rms.append(np.sqrt(np.mean(np.array(vals) ** 2)))
    return np.array(Ns), np.array(rms)


def main(out="figures/validation.png"):
    ps, rho = rho_sweep_fft()
    a, b = np.polyfit(ps, rho, 1)
    Ns, rms = residual_vs_N()
    expo = np.polyfit(np.log(Ns), np.log(rms), 1)[0]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    axL.axhline(0, color="0.7", lw=0.6); axL.axvline(0, color="0.7", lw=0.6)
    xx = np.linspace(-1, 1, 50)
    axL.plot(xx, a * xx + b, color="#1f77b4", lw=1.8, label=rf"fit $\rho={a:.3f}\,p$")
    axL.plot(ps, rho, "o", color="#1f77b4", ms=9, label=r"measured $\rho$ (E2, $s=1.6$)")
    axL.text(-0.95, 0.55, rf"ceiling $|\rho(\pm1)|\approx{abs(a):.2f}<1$"
             "\n(spectrum spans many shells)", fontsize=10, color="0.35")
    axL.set_xlabel(r"helicity parameter $p$")
    axL.set_ylabel(r"relative helicity $\rho$")
    axL.set_title("Helicity control is linear (E2)")
    axL.legend(loc="lower right", frameon=False)

    axR.loglog(Ns, rms, "s", color="#d62728", ms=10, label=r"rms $h_N$ at $p=0$ (E3)")
    c = rms[0] * Ns[0] ** 0.5
    axR.loglog(Ns, c * Ns ** -0.5, "--", color="0.4", label=r"$\mathcal{O}(N^{-1/2})$")
    axR.text(22, 0.075, rf"fitted exponent ${expo:.3f}$" "\n(theory $-1/2$)",
             fontsize=10, color="0.35")
    axR.set_xlabel(r"number of modes $N$")
    axR.set_ylabel(r"rms net helicity residual $h_N$")
    axR.set_title(r"Finite-mode residual decays as $N^{-1/2}$ (E3)")
    axR.xaxis.set_minor_locator(NullLocator())
    axR.set_xticks(Ns); axR.set_xticklabels(Ns)
    axR.legend(loc="upper right", frameon=False)

    fig.suptitle("Numerical validation of the two controls", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    print(f"  fit rho = {a:.5f} p + {b:.1e};  ceiling |rho| = {abs(a):.4f}")
    print(f"  residual exponent = {expo:.3f} (theory -0.5)")


if __name__ == "__main__":
    main()

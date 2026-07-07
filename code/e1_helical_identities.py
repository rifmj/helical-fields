#!/usr/bin/env python3
"""E1 -- helical-basis identities.

Verifies, on the helical basis h_pm = (e1 +/- i e2)/sqrt(2):
  (a) transversality      h_pm . k = 0
  (b) curl eigenvectors   i k x h_pm = +/- |k| h_pm
  (c) parity              h_+(-k) = conj(h_+(k)) = h_-(k),
      and that Hermitian symmetry u_hat(-k)=conj(u_hat(k)) sends
      a_+(k) -> conj(a_+(k)) at -k (the helicity bias is preserved).

Numeric check over many random wavevectors (numpy) + an exact symbolic
check in an abstract orthonormal frame (sympy, if available).
"""
import numpy as np


def frame(khat):
    """Right-handed transverse frame (e1, e2) with e1,e2 ⊥ khat, e1 x e2 = khat."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(khat[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, khat); e1 /= np.linalg.norm(e1)
    e2 = np.cross(khat, e1)                    # e1 x e2 = khat  (right-handed)
    return e1, e2


def numeric(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    r_trans = r_eig = r_par = r_amp = r_hand = 0.0
    for _ in range(n):
        k = rng.normal(size=3)
        kn = np.linalg.norm(k); khat = k / kn
        e1, e2 = frame(khat)
        hp = (e1 + 1j * e2) / np.sqrt(2)
        hm = (e1 - 1j * e2) / np.sqrt(2)
        # (a) transversality
        r_trans = max(r_trans, abs(hp @ k), abs(hm @ k))
        # (b) curl eigenvectors  i k x h_pm = +/- |k| h_pm
        r_eig = max(r_eig,
                    np.linalg.norm(1j * np.cross(k, hp) - kn * hp),
                    np.linalg.norm(1j * np.cross(k, hm) + kn * hm))
        # right-handedness
        r_hand = max(r_hand, abs(np.linalg.det(np.array([e1, e2, khat])) - 1.0))
        # (c) parity with the antipodal frame e1(-k)=e1(k), e2(-k)=-e2(k)
        e1m, e2m = e1, -e2
        hpm = (e1m + 1j * e2m) / np.sqrt(2)       # h_+(-k)
        r_par = max(r_par, np.linalg.norm(hpm - np.conj(hp)),
                            np.linalg.norm(hpm - hm))
        # amplitude map: u = a+ h+ + a- h- ; require u(-k)=conj(u(k)) -> b+ = conj(a+)
        ap, am = rng.normal() + 1j * rng.normal(), rng.normal() + 1j * rng.normal()
        u = ap * hp + am * hm
        uc = np.conj(u)
        # solve u(-k) = b+ h+(-k) + b- h-(-k) with h+(-k)=hm, h-(-k)=hp
        B = np.array([hm, hpm * 0 + hp]).T   # columns [h+(-k), h-(-k)] = [hm, hp]
        b = np.linalg.lstsq(B, uc, rcond=None)[0]
        r_amp = max(r_amp, abs(b[0] - np.conj(ap)), abs(b[1] - np.conj(am)))
    return dict(transverse=r_trans, curl_eig=r_eig, righthanded=r_hand,
               parity=r_par, amp_map=r_amp)


def symbolic():
    try:
        import sympy as sp
    except Exception:
        return None
    k = sp.symbols('k', positive=True)
    # abstract orthonormal frame; work with components in (e1,e2,khat)
    # h+ = (1, i, 0)/sqrt2, h- = (1,-i,0)/sqrt2, khat=(0,0,1), kvec = k*khat
    s2 = sp.sqrt(2)
    hp = sp.Matrix([1, sp.I, 0]) / s2
    hm = sp.Matrix([1, -sp.I, 0]) / s2
    kv = sp.Matrix([0, 0, k])
    def cross(a, b): return a.cross(b)
    res = {
        'hp.k': sp.simplify((hp.T * kv)[0]),
        'hm.k': sp.simplify((hm.T * kv)[0]),
        'curl+': sp.simplify((sp.I * cross(kv, hp) - k * hp)).norm(),
        'curl-': sp.simplify((sp.I * cross(kv, hm) + k * hm)).norm(),
    }
    return {kk: sp.nsimplify(vv) for kk, vv in res.items()}


if __name__ == '__main__':
    num = numeric()
    print("E1 helical identities -- numeric (2000 random k, seed 0)")
    for key, val in num.items():
        print(f"  max residual  {key:12s} = {val:.2e}")
    sym = symbolic()
    print("E1 helical identities -- symbolic (sympy, abstract frame)")
    if sym is None:
        print("  sympy not installed; skipped")
    else:
        for key, val in sym.items():
            print(f"  {key:8s} = {val}")

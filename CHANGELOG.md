# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/) and are published as GitHub releases,
each archived to Zenodo with a citable DOI.

## [1.0.5] — 2026-07-14

Correctness revision of the 3D phase-coherence construction, plus honesty and
scope fixes prompted by two independent referee passes.

- **Fixed the 3D phase-coherence builder** (`code/_helical.build_fft_3d`), which
  generates the teaser and advection figures. The previous builder baked
  *independent* random phases into the two helical channels *before* applying the
  reference-phase multiplier, so the field stayed a Gaussian haze at every
  `lambda` — the coherence knob was inoperative in 3D (2D was always correct). The
  builder now applies independent per-channel phases, each locking onto its own
  structured reference (two references, not one shared — a shared reference would
  collapse the two channels to the same phase at `lambda=1` and linearly polarize
  the `p=0` field, washing out its out-of-plane velocity). Reality is enforced by
  conjugate-copy on a spectral half-space (not an averaging projection, which had
  also distorted the helical polarization). Coherence now responds to `lambda`;
  energy, helicity, and the per-shell spectrum are frozen exactly; `rho(+1) ≈ 0.77`
  matches E2; `p=0` is isotropic at every `lambda`; `p=±1` net helicity is exactly
  antisymmetric. Verified by an 8-leg adversarial audit (bit-exact Hermitian
  symmetry, half-space partition, invariants frozen to machine precision, robust
  across even/odd/non-power-of-2 N).
- **New E7 validation** (`code/e7_phase_coherence_3d.py`, in `make reproduce`):
  the 3D analogue of E4, run on the figure-generating builder — the check whose
  absence let the broken builder ship.
- **Random phases** are now drawn i.i.d.-uniform on a spectral half-space in both
  2D and 3D (previously a triangular difference-of-uniforms); E4 numbers updated.
- **Curl-noise comparison** (Fig. 8, §Discussion): the featureless-vs-organized
  contrast is now reported via vorticity flatness (≈3.0 → ≈11.9), a far stronger
  discriminator than speed flatness; the baseline panel is labelled a random-phase
  spectral *surrogate*, not a specific curl-of-Perlin implementation.
- **Removed the singular "chord" phase scheme** (undefined at `lambda=1/2`,
  `theta=π`); the linear phase interpolation (9) is the single scheme.
- **Corrected the 2D sign convention** in Eq. (2) and the code (orientation-only).
- **Boundary section**: made the divergence-free claim precise (classical off the
  surface, distributional across it), noted the wall slip discontinuity and the
  smooth-SDF requirement, and stated continuous-flow containment as a theorem.
- **Novelty** narrowed and repositioned: added kinematic-simulation and
  divergence-free synthetic-turbulence prior art (Fung et al. 1992; Saad et al.
  2017); the contribution is the artist-facing controls, not spectral div-free
  synthesis per se.
- **Wording**: "no run-time cost" → "no added solver cost"; Beltrami qualified as
  mode-by-mode; `E(k)` clarified as an amplitude envelope; the mode-sum's lattice
  mode-density caveat noted.
- Bump version to 1.0.5.

## [1.0.4] — 2026-07-13

Solid-obstacle handling and fully script-generated figures.

- New **E6 solid-obstacle** experiment and paper section: ramping the vector
  potential before the curl keeps the field exactly divergence-free and tangent to
  the obstacle surface (`code/e6_obstacle.py`, Fig. 5). `make reproduce` now runs
  E1–E6.
- Every figure in the paper is now regenerated from source: `code/fig_*.py`
  scripts and a shared `code/_helical.py` field-construction module, wired to a new
  `make figures` target. The figure scripts import the E2/E3/E5 measurement
  routines directly, so the plotted validation numbers are the numbers those
  experiments print.
- Plain-language math on-ramp in the paper body; heavier formalism moved to an
  appendix. Figures relocated inline to their first-mention points; `main.pdf`
  rebuilt (smaller).
- `matplotlib` added to `code/requirements.txt`.
- Bump version to 1.0.4. (arXiv id still pending endorsement.)

## [1.0.3] — 2026-07-07

Cross-link the OSF project hub.

- Add OSF DOI `10.17605/OSF.IO/J8GFA` to `README.md`, `CITATION.cff`, and as a
  `related_identifiers` entry (`isVariantFormOf`) in `.zenodo.json`.
- Bump version to 1.0.3. (arXiv id still pending endorsement.)

## [1.0.2] — 2026-07-07

Bake the Zenodo DOI into the citable metadata and the PDF title page.

- Zenodo concept DOI `10.5281/zenodo.21246333` added to `README.md`,
  `CITATION.cff`, and the preprint title page (`main.tex`, `main_bibtex.tex`).
- Bump version to 1.0.2. (arXiv id still pending endorsement.)

## [1.0.1] — 2026-07-07

First Zenodo-archived release. No changes to the paper or the numerical results;
this release wires up citation/archival metadata and is the one that triggers the
Zenodo webhook to mint the DOI (see `RELEASING.md`).

- Add `CHANGELOG.md` and `RELEASING.md` (release + cross-linking procedure).
- Bump version to 1.0.1 in `CITATION.cff` and `.zenodo.json`.

## [1.0.0] — 2026-07-07

Initial public release.

- Preprint (`main.tex` / `main.pdf`) and BibTeX variant
  (`main_bibtex.tex` + `references.bib`, verified DOIs).
- Seven figures; reproduction code E1–E5 + the GLSL mode-sum shader.
- `CITATION.cff` and `.zenodo.json` citation/archival metadata. MIT-licensed.

# Beyond curl-noise: helicity and phase coherence controls for divergence-free flow fields

A standalone preprint with reproduction code. The paper describes a method to synthesize
divergence-free vector fields with three independent, artist-facing controls — **spectral
slope**, **helicity**, and **phase coherence** — for procedural animation of particles,
smoke, and other advected media, without running a fluid solver.

- `main.tex` — the paper source (single file, self-contained bibliography).
- `main.pdf` — the compiled paper.
- `main_original.tex` — the pre-figures version, kept for reference.
- `figures/` — the nine figures used by `main.tex` (`gallery.png` teaser, `slope.png`,
  `helix.png`, `ramp.png`, `obstacle.png`, `validation.png`, `performance.png`,
  `curlnoise.png`, `advection.png`).
- `code/` — reproduction scripts for every numerical claim, plus the real-time shader `modesum.frag`.

## Build the PDF

```
make pdf          # or: pdflatex main.tex && pdflatex main.tex
```

Requires a standard LaTeX distribution (TeX Live / MacTeX); only common packages
(`amsmath`, `amssymb`, `amsthm`, `mathtools`, `booktabs`, `graphicx`, `hyperref`,
`geometry`, `microtype`, `xcolor`, `enumitem`). The `figures/` PNGs must sit alongside
`main.tex` (the source sets `\graphicspath{{figures/}}`).

### Bibliography: two interchangeable variants

- **`main.tex`** ships a self-contained `thebibliography` block — no BibTeX run needed
  (`make pdf`). This is the default.
- **`main_bibtex.tex` + `references.bib`** are the BibTeX version: identical text, but the
  reference list is generated from the `.bib` (`make pdf-bibtex`, i.e.
  `pdflatex → bibtex → pdflatex → pdflatex`). All 18 entries carry verified DOIs. Pick
  whichever your submission workflow prefers; keep only one `main*.tex` when posting.

## Reproduce the numbers

```
pip install -r code/requirements.txt      # numpy, sympy
make reproduce                            # runs e1, e2, e3, e4, e5, e6
```

Each script prints exactly the numbers quoted in the paper's validation section (§7), at a
pinned random seed:

| script | paper | what it checks |
|---|---|---|
| `code/e1_helical_identities.py` | E1 | the helical vectors `h±` are transverse and are curl eigenvectors; the parity identity `h₊(−k)=conj(h₊(k))=h₋(k)`; and that Hermitian symmetry preserves the helicity bias (numeric over 2000 random `k` + a symbolic check). |
| `code/e2_fft_field.py` | E2 | a 32³ FFT field is real and divergence-free to machine precision and its relative helicity `ρ` is linear in the helicity parameter `p`. |
| `code/e3_mode_sum.py` | E3 | the analytic Beltrami mode-sum is exactly transverse (divergence-free), `ρ(p)` tracks the knob, and the `p=0` helicity residual decays as `N^{-1/2}`. |
| `code/e4_phase_coherence.py` | E4 | sweeping the phase-coherence knob `λ` holds total and per-shell energy fixed to machine precision while the vorticity flatness (organization) rises monotonically. |
| `code/e5_performance.py` | E5 | timing of the two paths: the `O(N³ log N)` FFT bake vs the `O(M)` per-point mode-sum (CPU numpy reference; absolute times are machine-dependent, the scaling is the point). |
| `code/e6_obstacle.py` | E6 | solid-obstacle handling: ramping the vector potential before the curl keeps the field exactly divergence-free (analytic complex-step `div` ≈ `1.2·10⁻¹⁵`) and tangent to the surface (`max‖u·n̂‖/rms‖u‖ ≈ 4.8·10⁻¹⁶`); RK4 advection puts `0.00%` of tracers inside the obstacle with the ramp vs `2.84%` without it. |

`code/modesum.frag` is a WebGL2 / GLSL ES 3.0 fragment shader implementing the mode-sum
(eq. for `u(x)`) per pixel; it is documentation of the real-time path, not run by `make`.

## Regenerate the figures

Every figure in the paper is produced by a script, so `figures/*.png` can be rebuilt from
source:

```
make figures                              # rebuilds all nine figures/*.png
```

| script | figure | what it draws |
|---|---|---|
| `code/fig_gallery.py` | Fig. 1 (teaser) | 3×3 grid: phase coherence `λ` across columns, helicity `p` down rows, on one 48³ field; prints the per-row net helicity (`+6.3·10⁻⁴`, `≈0`, `−6.4·10⁻⁴`). |
| `code/fig_slope.py` | Fig. 2 | the spectral-slope axis `s` on one 2D field at fixed seed (large-scale skeleton persists, fine detail added/removed). |
| `code/fig_helix.py` | Fig. 3 | a dye line advected by a single Beltrami mode winds into a helix; reversing `p` mirrors the handedness. Prints the 32³ multi-shell net helicity (`±8.5·10⁻⁴`). |
| `code/fig_ramp.py` | Fig. 4 | the Bridson free-slip quintic ramp `α(q)` (`α(0)=0`, `α′(0)=15/8`, `α(1)=1`, `α′(1)=α″(1)=0`). |
| `code/fig_validation.py` | Fig. 6 | the two validation results (E2 helicity linearity `ρ=0.796 p`; E3 residual exponent `−0.494`). |
| `code/fig_performance.py` | Fig. 7 | the two cost regimes (E5): `O(N³ log N)` FFT bake vs `O(1/M)` per-point mode-sum. |
| `code/fig_curlnoise.py` | Fig. 8 | curl-noise vs `λ=1` at identical spectrum and seed; prints speed flatness (`≈1.9` vs `≈4.0`). |
| `code/fig_advection.py` | Fig. 9 | a dye blob advected by the field: phase coherence winds it into a spiral (top), helicity sign sets the chirality (bottom). |
| `code/e6_obstacle.py` | Fig. 5 | the solid-obstacle panels (also part of `make reproduce`). |

`code/_helical.py` is the shared field-construction module: the FFT and mode-sum builders,
the phase-coherence multiplier, the helicity split, and the ramp — the *same* constructions
measured by `e1..e6`, so the figures and the validation numbers come from one implementation.
The figure scripts import the `e2`/`e3`/`e5` measurement routines directly, so the plotted
validation numbers are exactly the numbers those experiments print.

Runtime: a few seconds each (`e3`, `e4` ≈ 10 s).

## Citation

```bibtex
@misc{helicalfields,
  title  = {Beyond curl-noise: helicity and phase coherence controls
            for divergence-free flow fields},
  author = {Rifat Jumagulov},
  year   = {2026},
  note   = {Preprint},
  doi    = {10.5281/zenodo.21246333}
}
```

Archived on Zenodo — concept DOI (always the latest version):
[10.5281/zenodo.21246333](https://doi.org/10.5281/zenodo.21246333).
Project hub on OSF: [10.17605/OSF.IO/J8GFA](https://doi.org/10.17605/OSF.IO/J8GFA).
Machine-readable metadata lives in `CITATION.cff` (GitHub renders it as a
"Cite this repository" button) and `.zenodo.json` (read by Zenodo on release).

> **Pending arXiv:** once endorsement clears and the paper is posted, the arXiv id
> will be added here and as a `related_identifiers` entry in `.zenodo.json`.

## License

Code and text are released under the MIT License; see `LICENSE`.

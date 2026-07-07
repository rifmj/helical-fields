# Beyond curl-noise: helicity and phase coherence controls for divergence-free flow fields

A standalone preprint with reproduction code. The paper describes a method to synthesize
divergence-free vector fields with three independent, artist-facing controls — **spectral
slope**, **helicity**, and **phase coherence** — for procedural animation of particles,
smoke, and other advected media, without running a fluid solver.

- `main.tex` — the paper source (single file, self-contained bibliography).
- `main.pdf` — the compiled paper.
- `main_original.tex` — the pre-figures version, kept for reference.
- `figures/` — the seven figures used by `main.tex` (`gallery.png` teaser, `slope.png`,
  `helix.png`, `curlnoise.png`, `validation.png`, `advection.png`, `performance.png`).
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
make reproduce                            # runs e1, e2, e3, e4, e5
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

`code/modesum.frag` is a WebGL2 / GLSL ES 3.0 fragment shader implementing the mode-sum
(eq. for `u(x)`) per pixel; it is documentation of the real-time path, not run by `make`.

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

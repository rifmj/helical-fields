# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/) and are published as GitHub releases,
each archived to Zenodo with a citable DOI.

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

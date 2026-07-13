PY ?= python3

.PHONY: pdf pdf-bibtex reproduce figures clean

# Default: main.tex ships a self-contained thebibliography (no bibtex needed).
pdf:
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

# Alternative: main_bibtex.tex + references.bib (BibTeX, plain style).
pdf-bibtex:
	pdflatex -interaction=nonstopmode -halt-on-error main_bibtex.tex
	bibtex main_bibtex
	pdflatex -interaction=nonstopmode -halt-on-error main_bibtex.tex
	pdflatex -interaction=nonstopmode -halt-on-error main_bibtex.tex

reproduce:
	$(PY) code/e1_helical_identities.py
	$(PY) code/e2_fft_field.py
	$(PY) code/e3_mode_sum.py
	$(PY) code/e4_phase_coherence.py
	$(PY) code/e5_performance.py
	$(PY) code/e6_obstacle.py

# Regenerate every figure in the paper from code (writes figures/*.png).
figures:
	$(PY) code/fig_gallery.py
	$(PY) code/fig_slope.py
	$(PY) code/fig_helix.py
	$(PY) code/fig_ramp.py
	$(PY) code/fig_validation.py
	$(PY) code/fig_performance.py
	$(PY) code/fig_curlnoise.py
	$(PY) code/fig_advection.py
	$(PY) code/e6_obstacle.py

clean:
	rm -f main.aux main.log main.out main.fls main.fdb_latexmk main.synctex.gz main.toc
	rm -f main_bibtex.aux main_bibtex.log main_bibtex.out main_bibtex.bbl main_bibtex.blg

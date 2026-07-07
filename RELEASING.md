# Releasing & DOI cross-linking

The DOI and the arXiv id do not exist until *after* the acts that create them,
so cross-linking is a two-step dance. This file records the exact steps so the
metadata stays consistent.

## Step 1 — mint the Zenodo DOI (release v1.0.1)

1. On <https://zenodo.org> log in with GitHub.
2. **Settings → GitHub** (<https://zenodo.org/account/settings/github/>): flip the
   toggle for `rifmj/helical-fields` to **ON**.
3. Publish the **v1.0.1** GitHub release *after* the toggle is on — only releases
   published while the toggle is ON are archived. Zenodo then creates the record
   and assigns two DOIs:
   - a **concept DOI** (always resolves to the latest version — cite this one),
   - a **version DOI** (this specific release).
4. Copy the concept DOI (looks like `10.5281/zenodo.1234567`).

## Step 2 — bake the identifiers in (release v1.0.2)

Once you have the concept DOI (and, after arXiv endorsement, the arXiv id):

1. In `README.md`, replace `10.5281/zenodo.XXXXXXX` with the concept DOI.
2. In `.zenodo.json`, add the block below (fill in real values — do **not** commit
   placeholder ids; Zenodo rejects metadata with malformed identifiers):

   ```json
   "related_identifiers": [
     { "relation": "isSupplementTo",   "identifier": "arXiv:XXXX.XXXXX", "scheme": "arxiv" },
     { "relation": "isVersionOf",       "identifier": "10.5281/zenodo.XXXXXXX", "scheme": "doi" }
   ]
   ```

3. In `main.tex` / `main_bibtex.tex`, add the arXiv id and DOI to the title-page
   footnote and the BibTeX `note`/`doi` fields.
4. Commit, tag **v1.0.2**, publish the release — Zenodo archives it as a new
   version under the same concept DOI.

## arXiv side

When endorsement clears and the paper is posted, add the Zenodo concept DOI to the
arXiv submission's "DOI" / comments field so the two records point at each other.

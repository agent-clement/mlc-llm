# Qwen3.5 MLC Support Paper

This directory contains an arXiv-style paper draft for the MLC Qwen3.5 0.8B
vision-language support work.

## Files

- `paper.tex`: main LaTeX source.
- `references.bib`: bibliography.
- `generate_figures.py`: regenerates TikZ/PGFPlots figure fragments using only
  the Python standard library.
- `figures/*.tex`: generated figure fragments included by `paper.tex`.

## Regenerate Figures

```bash
/home/cwong/Projects/miniconda/envs/mlc/bin/python generate_figures.py
```

## Build

The current machine does not have a LaTeX toolchain installed. I used the
public TeXLive.net multipart API to compile the paper online:

```bash
./compile_online_texlive.sh
```

This writes `paper-online.pdf`. The script uploads only this paper package
(`paper.tex`, `references.bib`, and `figures/*.tex`) to TeXLive.net. It
flattens figure file names in a temporary directory because the API requires
the main file to be called `document.tex` and rejects subdirectory filenames.

On a machine with TeX Live, build locally with:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

or:

```bash
latexmk -pdf paper.tex
```

The paper uses `tikz` and `pgfplots`; install those packages with the TeX
distribution if they are not already present.

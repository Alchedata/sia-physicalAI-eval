# Self-Evolving Agents for Physical AI Evaluation - ACM LaTeX Package

This directory contains an ACM-ready LaTeX source package for the white paper.

## Files

- `main.tex` - ACM `acmart` manuscript source.
- `references.bib` - BibTeX references.
- `figures/system_architecture.png` - central architecture diagram.
- `Makefile` - convenience build commands.
- `main.pdf` - compiled preview PDF, if included.

## Build

```bash
make
```

or manually:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## ACM format notes

The source currently uses:

```latex
\documentclass[sigconf,review,anonymous]{acmart}
```

For an ACM journal-style submission, replace it with something like:

```latex
\documentclass[acmsmall,review,anonymous]{acmart}
```

Before final submission, update:

- author names, affiliations, emails;
- ACM conference or journal metadata;
- ACM copyright/permission settings;
- CCS concepts and keywords, if required by the venue;
- any anonymous-review settings required by the venue.

The current package suppresses the ACM reference block using `printacmref=false` so the draft is cleaner for internal review. Re-enable it if the venue requires the official ACM reference format.

PDF=main.pdf
TEX=main.tex

all: $(PDF)

$(PDF): $(TEX) references.bib figures/system_architecture.png
	pdflatex -interaction=nonstopmode $(TEX)
	@if command -v bibtex >/dev/null 2>&1; then bibtex main; else bibtex8 main; fi
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

clean:
	rm -f *.aux *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz

realclean: clean
	rm -f $(PDF) *.bbl

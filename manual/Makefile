# Build the KELP manual with latexmk. TeX Live must be on PATH, or pass the
# binary directory: `make TEXBIN=<TeX Live bin directory> pdf`.

TEXBIN ?=
export PATH := $(if $(TEXBIN),$(TEXBIN):,)$(PATH)

MAIN := main

.PHONY: all pdf quick clean veryclean
all: pdf

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error $(MAIN).tex

# faster, keeps going past the first error (for drafting)
quick:
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -c
	rm -f *.bbl *.run.xml

veryclean:
	latexmk -C
	rm -f *.bbl *.run.xml

#!/bin/bash

# Texfilename
texfilename=$1

# This script is used to compile a LaTeX file to PDF using pdflatex.
pdflatex --output-directory=garbage --interaction=nonstopmode ${texfilename}
biblatex ${texfilename}
pdflatex --output-directory=garbage --interaction=nonstopmode ${texfilename}

mv garbage/${texfilename%.tex}.pdf ${texfilename%.tex}.pdf

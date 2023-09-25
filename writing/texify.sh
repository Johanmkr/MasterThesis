#!/bin/bash

# Texfilename
#texfilename=$1
texfilename="masterthesis"
# This script is used to compile a LaTeX file to PDF using pdflatex.
#pdflatex --output-directory=garbage --interaction=nonstopmode ${texfilename}
pdflatex ${texfilename}.tex
biber ${texfilename}
pdflatex ${texfilename}.tex
pdflatex ${texfilename}.tex
#biblatex ${texfilename}
#pdflatex --output-directory=garbage --interaction=nonstopmode ${texfilename}

#mv garbage/${texfilename%.tex}.pdf ${texfilename%.tex}.pdf


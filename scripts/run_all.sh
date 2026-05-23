#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 1: Generate all CSV data files ==="
cd "$(dirname "$0")/../src"
python gen_data.py
echo ""

echo "=== Step 2: Generate all 17 figures ==="
python make_all_figs.py --all
echo ""

echo "=== Step 3: Copy figures to figures/ directory ==="
cp fig_*.pdf ../figures/
echo ""

echo "=== Step 4: Copy figures to paper/ for compilation ==="
cp fig_*.pdf ../paper/
echo ""

echo "=== Step 5: Compile paper ==="
cd ../paper
pdflatex -interaction=nonstopmode paper3_main.tex > /dev/null 2>&1
bibtex paper3_main > /dev/null 2>&1
pdflatex -interaction=nonstopmode paper3_main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode paper3_main.tex > /dev/null 2>&1
echo "Paper compiled: paper/paper3_main.pdf"

echo ""
echo "=== All done ==="
echo "  Data:    src/sim/*.csv"
echo "  Figures: figures/fig_*.pdf"
echo "  Paper:   paper/paper3_main.pdf"

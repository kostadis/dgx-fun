#!/usr/bin/env bash
# Regenerate every number/table/figure from the result files, then compile main.pdf in the
# texlive/texlive container on spark2 (no TeX on the workstation) and copy it back.
set -euo pipefail
cd "$(dirname "$0")"
python3 ../clef/eval/cascade_paper.py --out gen
ssh spark2 'rm -rf ~/paper-cascade-build && mkdir -p ~/paper-cascade-build'
scp -q -r main.tex gen spark2:paper-cascade-build/
ssh spark2 'cd ~/paper-cascade-build && docker run --rm -v $PWD:/w -w /w texlive/texlive:latest \
  latexmk -pdf -interaction=nonstopmode main.tex > build.log 2>&1; rc=$?;
  grep -E "^!|undefined|Overfull" main.log | sort | uniq -c | head -20; exit $rc'
scp -q spark2:paper-cascade-build/main.pdf main.pdf
echo "built main.pdf"

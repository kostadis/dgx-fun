#!/usr/bin/env bash
# Compile main.pdf in the texlive/texlive container on spark2 (no TeX on the workstation)
# and copy it back.
set -euo pipefail
cd "$(dirname "$0")"
ssh spark2 'rm -rf ~/paper-gateway-build && mkdir -p ~/paper-gateway-build'
scp -q main.tex gateway-not-platform-diagram.png spark2:paper-gateway-build/
ssh spark2 'cd ~/paper-gateway-build && docker run --rm -v $PWD:/w -w /w texlive/texlive:latest \
  latexmk -pdf -interaction=nonstopmode main.tex > build.log 2>&1; rc=$?;
  grep -E "^!|undefined|Overfull" main.log | sort | uniq -c | head -20; exit $rc'
scp -q spark2:paper-gateway-build/main.pdf main.pdf
echo "built main.pdf"

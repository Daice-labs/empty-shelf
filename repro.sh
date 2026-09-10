#!/bin/sh
set -e
OUT="${1:-$PWD/results_repro}"
python3 -m venv .venv 2>/dev/null || true
. .venv/bin/activate
pip -q install numpy pandas scipy matplotlib rapidfuzz nbconvert nbformat ipykernel
RPT11_RESULTS="$OUT" jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output executed_repro.ipynb code/RP-T11-empty-shelf-study.ipynb
echo "Run complete. Verifying shipped records:"
python3 verify_results.py

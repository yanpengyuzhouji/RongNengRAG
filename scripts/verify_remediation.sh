#!/usr/bin/env bash
set -euo pipefail

RAG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EXCEL_ROOT="${EXCEL_SERVICE_DIR:-${RAG_ROOT}/../excel-workbook-service}"
RAG_PYTHON="${RAG_PYTHON:-python3}"
EXCEL_PYTHON="${EXCEL_PYTHON:-python3}"
VERIFY_CACHE="$(mktemp -d "${TMPDIR:-/tmp}/rongneng-remediation.XXXXXX")"
trap 'rm -rf -- "$VERIFY_CACHE"' EXIT
export PYTHONPYCACHEPREFIX="${VERIFY_CACHE}/pycache"

if [[ ! -d "$EXCEL_ROOT/app" ]]; then
  echo "Excel service not found: $EXCEL_ROOT" >&2
  exit 2
fi

echo "[1/6] RAG Python syntax"
cd "$RAG_ROOT"
"$RAG_PYTHON" -m compileall -q -f \
  src/config.py src/api src/generation src/ingestion src/retrieval src/utils \
  scripts tests

echo "[2/6] RAG tests"
"$RAG_PYTHON" -m unittest discover -s tests -v

echo "[3/6] Excel Python syntax"
cd "$EXCEL_ROOT"
"$EXCEL_PYTHON" -m compileall -q -f app tests

echo "[4/6] Excel tests"
"$EXCEL_PYTHON" -m unittest discover -s tests -v

echo "[5/6] Frontend logic and SFC checks"
cd "$RAG_ROOT/src/ui-vue2"
npm run test:remediation

echo "[6/6] Frontend production build"
npm run build

echo "All remediation checks passed."

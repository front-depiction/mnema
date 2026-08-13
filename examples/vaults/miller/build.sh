#!/usr/bin/env bash
# Build the Miller vault: fetch each paper from its public mirror, ingest it
# dated by publication. The store lands OUTSIDE this repo; no paper text ships.
set -euo pipefail

VAULT_DIR="${1:-$HOME/.mnema-miller}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$SCRIPT_DIR/manifest.tsv"

command -v mnema >/dev/null 2>&1 || {
  echo "mnema is not on PATH — install it first (see the repo README)" >&2
  exit 1
}

# The paper format needs the optional anydoc converter — check the same
# Python environment mnema runs in, and fail closed with the remedy.
MNEMA_PY="$(sed -n '1s/^#! *//p' "$(command -v mnema)")"
[ -x "$MNEMA_PY" ] || MNEMA_PY="python3"
"$MNEMA_PY" -c "import anydoc" 2>/dev/null || {
  echo 'the paper format needs the anydoc converter — pip install "mnema[paper]"' >&2
  exit 1
}

if [ -e "$VAULT_DIR/config.json" ]; then
  echo "refusing: $VAULT_DIR already holds a store — choose another directory or remove it" >&2
  exit 1
fi
mkdir -p "$VAULT_DIR"
VAULT_DIR="$(cd "$VAULT_DIR" && pwd)"

mnema --store "$VAULT_DIR" init
mkdir -p "$VAULT_DIR/sources"

fetch() { # url dest -> 0 iff dest is a real PDF
  curl -sL --retry 3 --retry-delay 2 -o "$2" "$1" || return 1
  file -b "$2" | grep -q PDF
}

ingest() { # relative-source-path published-date
  # cd keeps topics (`<path>#<slug>`) relative and stable across rebuilds
  (cd "$VAULT_DIR" && mnema --store "$VAULT_DIR" ingest --format paper \
    --at "${2}T00:00:00.000Z" "$1")
}

while IFS=$'\t' read -r name published url tier; do
  [ "$tier" = "text" ] || continue
  echo "== $name ($published)"
  if ! fetch "$url" "$VAULT_DIR/sources/$name.pdf"; then
    echo "WARNING: $url did not return a PDF — skipping $name" >&2
    rm -f "$VAULT_DIR/sources/$name.pdf"
    continue
  fi
  ingest "sources/$name.pdf" "$published"
done < <(tail -n +2 "$MANIFEST")

# The 1988 papers are image-only scans (no text layer): OCR tier, optional.
if command -v tesseract >/dev/null 2>&1 && command -v pdftoppm >/dev/null 2>&1; then
  while IFS=$'\t' read -r name published url tier; do
    [ "$tier" = "scan" ] || continue
    echo "== $name ($published, OCR)"
    if ! fetch "$url" "$VAULT_DIR/sources/$name.pdf"; then
      echo "WARNING: $url did not return a PDF — skipping $name" >&2
      rm -f "$VAULT_DIR/sources/$name.pdf"
      continue
    fi
    tmp="$(mktemp -d)"
    pdftoppm -r 300 -gray -png "$VAULT_DIR/sources/$name.pdf" "$tmp/page"
    : > "$VAULT_DIR/sources/$name.ocr.txt"
    for page in "$tmp"/page-*.png; do
      tesseract "$page" stdout 2>/dev/null >> "$VAULT_DIR/sources/$name.ocr.txt"
    done
    rm -rf "$tmp"
    ingest "sources/$name.ocr.txt" "$published"
  done < <(tail -n +2 "$MANIFEST")
else
  echo "skipping the 3 scanned 1988 papers — OCR needs tesseract and pdftoppm (poppler) on PATH"
fi

echo
mnema --store "$VAULT_DIR" stats | head -3
echo
echo "next: mnema vault add \"$VAULT_DIR\" --name miller"
echo "      mnema ask \"what do capabilities solve that access control lists cannot?\""
